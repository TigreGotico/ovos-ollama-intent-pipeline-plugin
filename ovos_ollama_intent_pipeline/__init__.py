import collections
import os.path
import time
from typing import List, Optional, Union, Dict

import requests
from langcodes import closest_match
from ovos_bus_client.client import MessageBusClient
from ovos_bus_client.message import Message
from ovos_config.config import Configuration
from ovos_plugin_manager.templates.pipeline import IntentHandlerMatch, ConfidenceMatcherPipeline
from ovos_utils.fakebus import FakeBus
from ovos_utils.log import LOG
from ovos_utils.parse import match_one, MatchStrategy


class LLMIntentEngine:
    def __init__(self,
                 model: str,
                 base_url: str,
                 temperature: float = 0.0,
                 timeout: int = 5,
                 fuzzy: bool = True,
                 min_words: int = 2,
                 ignore_labels: Optional[List[str]] = None,
                 ignore_skills: Optional[List[str]] = None,
                 bus: Optional[MessageBusClient] = None):
        """
        Initializes the LLMIntentEngine for intent prediction using an LLM API.
         
        Args:
            model: Name of the LLM model to use.
            base_url: Base URL of the LLM API endpoint.
            temperature: Sampling temperature for the LLM (default 0.0).
            timeout: Timeout in seconds for LLM API requests (default 5).
            fuzzy: Whether to enable fuzzy matching for intent correction (default True).
            min_words: Minimum number of words required in an utterance to attempt prediction (default 2).
            ignore_labels: Optional list of intent labels to exclude from consideration.
            ignore_skills: Optional list of skill identifiers to exclude intents from.
            bus: Optional message bus client for intent synchronization and communication. If not provided, a new client is created.
         
        Loads locale-specific prompt templates, synchronizes or sets intent labels, and prepares the engine for intent prediction.
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        self.min_words = min_words
        self.fuzzy = fuzzy
        self.fuzzy_strategy = MatchStrategy.PARTIAL_TOKEN_SET_RATIO
        self.fuzzy_threshold = 0.55
        self.ignore_labels = ignore_labels or []
        self.prompts = collections.defaultdict(dict)
        self.mappings = {}
        # these skills are specialized/ambiguous enough to consistently throw off the LLMS
        self.ignore_skills = ignore_skills or []

        self.load_locale()

        if not bus:
            bus = MessageBusClient()
            bus.run_in_thread()

        self.bus = bus
        if not self.bus.connected_event.is_set():
            self.bus.connected_event.wait()

        self.sync_intents()

    def sync_intents(self, timeout=1):
        """
        Synchronizes and normalizes available intent labels from Adapt and Padatious services.
        
        Retrieves current intent labels from both Adapt and Padatious via the message bus, filters out ignored labels, and constructs a mapping of normalized label forms to their canonical names. Adds manual mappings for certain common or legacy intents to support consistent label normalization.
        """
        # TODO - persona/ocp/common_query/stop intents
        try:
            labels = self._get_adapt_intents(timeout) + self._get_padatious_intents(timeout)
        except Exception:
            LOG.error("Failed to sync intents via messagebus")
            return

        # mappings are there to help out the LLM a bit
        # normalizing labels to give them some more structure
        self.mappings = {self.normalize(l): l for l in labels if l not in self.ignore_labels}

        # HACK: manually maintained until exposed via bus api dynamically
        self.mappings["common-query:general_question"] = "common_query.question"
        self.mappings["common-play:play"] = "ovos.common_play.play_search"
        self.mappings["common-play:next"] = "ocp:next"
        self.mappings["common-play:prev"] = "ocp:prev"
        self.mappings["common-play:pause"] = "ocp:pause"
        self.mappings["common-play:resume"] = "ocp:resume"
        self.mappings["persona:release_persona"] = "persona:release"
        self.mappings["persona:summon"] = "persona:summon"
        self.mappings["persona:list_personas"] = "persona:list"
        self.mappings["persona:active_persona"] = "persona:check"
        self.mappings["persona:ask"] = "persona:query"
        self.mappings["stop:stop"] = "mycroft.stop"

    @property
    def labels(self):
        """
        Returns a list of intent labels, excluding those in the ignore list or associated with ignored skills.
        """
        return [l for l in self.mappings.values()
                if l not in self.ignore_labels
                and not any(s in l for s in self.ignore_skills)]

    def load_locale(self):
        """
        Loads language-specific prompt templates and examples from the locale directory.
        
        Scans each language subdirectory under the 'locale' folder and loads the contents of
        'system_prompt.txt', 'prompt_template.txt', and 'few_shot_examples.txt' files into the
        corresponding entries of the prompts dictionary for use in LLM prompting.
        """
        res_dir = os.path.join(os.path.dirname(__file__), 'locale')
        for lang in os.listdir(res_dir):
            prompt_file = os.path.join(res_dir, lang, 'system_prompt.txt')
            if os.path.isfile(prompt_file):
                with open(prompt_file) as f:
                    self.prompts[lang]["system"] = f.read()

            prompt_file = os.path.join(res_dir, lang, 'prompt_template.txt')
            if os.path.isfile(prompt_file):
                with open(prompt_file) as f:
                    self.prompts[lang]["prompt_template"] = f.read()

            prompt_file = os.path.join(res_dir, lang, 'few_shot_examples.txt')
            if os.path.isfile(prompt_file):
                with open(prompt_file) as f:
                    self.prompts[lang]["few_shot_examples"] = f.read()

    def _get_adapt_intents(self, timeout=1):
        """
        Retrieves Adapt intent names from the message bus, excluding ignored labels.
        
        Args:
            timeout: Maximum time in seconds to wait for the response.
        
        Returns:
            A list of Adapt intent names, or None if no response is received.
        """
        msg = Message("intent.service.adapt.manifest.get")
        res = self.bus.wait_for_response(msg, "intent.service.adapt.manifest", timeout=timeout)
        if not res:
            raise RuntimeError("Failed to retrieve intent names")
        return [i["name"] for i in res.data["intents"] if i["name"] not in self.ignore_labels]

    def _get_padatious_intents(self, timeout=1):
        """
        Retrieves Padatious intent names from the message bus, excluding ignored labels.
        
        Args:
            timeout: Maximum time in seconds to wait for the response.
        
        Returns:
            A list of Padatious intent names not present in the ignore list, or None if no response is received.
        """
        msg = Message("intent.service.padatious.manifest.get")
        res = self.bus.wait_for_response(msg, "intent.service.padatious.manifest", timeout=timeout)
        if not res:
            raise RuntimeError("Failed to retrieve intent names")
        return [i for i in res.data["intents"] if i not in self.ignore_labels]

    @staticmethod
    def normalize(text: str) -> str:
        # standardize labels as much as possible to reduce token usage + not confuse the LLM
        """
        Normalizes an intent label string by lowercasing and removing or replacing common substrings.
        
        This standardization reduces token usage and helps prevent confusion for the language model by producing concise, uniform intent labels.
        """
        norm = (text.lower().
                replace("", "").
                replace(".openvoiceos", "").
                replace(".intent", "").
                replace("skill-ovos-", "ovos-skill-").
                replace("ovos-skill-", "").
                replace("-pipeline", "").
                replace("ovos-", "").
                replace("intent", ""))
        if norm.endswith("alt"): # duplicate intents
            norm = norm[:-3]
        return norm

    def predict(self, utterance: str, lang: str) -> Optional[str]:
        """
        Predicts the most likely intent label for a given utterance using an LLM.
        
        Attempts to match the utterance to a known intent by constructing a prompt with language-specific or multilingual templates and sending it to the LLM API. The predicted label is normalized and validated against known intents, with optional fuzzy matching to correct minor mismatches. Returns the matched intent label or None if no valid intent is found.
        
        Args:
            utterance: The input text to classify.
            lang: The language code for prompt selection.
        
        Returns:
            The matched intent label, or None if no valid intent is identified.
        """
        if len(utterance.split()) < self.min_words:
            LOG.debug(f"Skipping LLM intent match, utterance too short (< {self.min_words} words)")
            return None

        lang, score = closest_match(lang, [l for l in self.prompts if l != "mul"], 10)

        # default to multilingual prompts
        system = self.prompts["mul"]["system"]
        prompt_template = self.prompts["mul"]["prompt_template"]
        examples = self.prompts["mul"]["few_shot_examples"]

        # optimized lang specific prompts
        if "system" in self.prompts[lang]:
            system = self.prompts[lang]["system"]
        if "prompt_template" in self.prompts[lang]:
            prompt_template = self.prompts[lang]["prompt_template"]
        if "few_shot_examples" in self.prompts[lang]:
            examples = self.prompts[lang]["few_shot_examples"]

        label_list = "\n- ".join([self.normalize(l) for l in self.labels])
        prompt = prompt_template.format(transcribed_text=utterance,
                                        language=lang,
                                        label_list=label_list,
                                        examples=examples)

        try:
            response = requests.post(f"{self.base_url}/api/generate",
                                     json={
                                         "model": self.model,
                                         "prompt": prompt,
                                         "system": system,
                                         "stream": False,
                                         "options": {
                                             "temperature": self.temperature,
                                             "num_predict": 25,
                                             "stop": ["\n"]
                                         }
                                     },
                                     timeout=self.timeout
                                     )
            result = response.json()["response"].strip()
        except Exception as e:
            LOG.error(f"⚠️ Error with model {self.model} and utterance '{utterance}': {e}")
            return None

        if not result or result == "None" or result in self.ignore_labels:
            #LOG.debug(f"⚠️ No intent for utterance: '{utterance}'")
            return None

        # ensure output is a valid intent, undo any normalization done to help the LLM
        result = self.mappings.get(result) or self.mappings.get(self.normalize(result)) or result

        if self.fuzzy and result not in self.labels:
            # force a valid label
            best, score = match_one(result, self.labels, strategy=self.fuzzy_strategy)

            if round(score, 2) >= self.fuzzy_threshold:
                result = best
            else:
                LOG.debug(f"⚠️ failed to fuzzy match hallucinated intent  ({score}) - {result} -> {best}")
        if result not in self.labels:
            #LOG.warning(f"⚠️ Error with model '{self.model}' and utterance '{utterance}': hallucinated intent - {result}")
            return None

        return result


class LLMIntentPipeline(ConfidenceMatcherPipeline):

    def __init__(self, bus: Optional[Union[MessageBusClient, FakeBus]] = None,
                 config: Optional[Dict] = None):
        config = config or Configuration().get('intents', {}).get("ovos_ollama_intent_pipeline") or dict()
        super().__init__(bus, config)

        self.llm = LLMIntentEngine(model=self.config["model"],
                                   base_url=self.config["base_url"],
                                   temperature=self.config.get("temperature", 0.0),
                                   timeout=self.config.get("timeout", 10),
                                   fuzzy=self.config.get("fuzzy", True),
                                   min_words=self.config.get("min_words", 2),
                                   ignore_labels=self.config.get("ignore_labels", []),
                                   ignore_skills=self.config.get("ignore_skills", []),
                                   bus=self.bus)
        LOG.info(f"Loaded Ollama Intents pipeline with model: '{self.llm.model}' and url: '{self.llm.base_url}'")
        self.bus.on("mycroft.ready", self.handle_sync_intents)
        self.bus.on("padatious:register_intent", self.handle_sync_intents)
        self.bus.on("register_intent", self.handle_sync_intents)
        self.bus.on("detach_intent", self.handle_sync_intents)
        self.bus.on("detach_skill", self.handle_sync_intents)
        self._syncing = False

    def handle_sync_intents(self, message):
        # sync newly (de)registered intents with debounce
        if self._syncing:
            return
        self._syncing = True
        time.sleep(5)
        self.llm.sync_intents()
        self._syncing = False
        LOG.debug(f"ollama registered intents: {len(self.llm.labels) - len(self.llm.ignore_labels)}")

    def match_low(self, utterances: List[str], lang: str, message: Message) -> Optional[IntentHandlerMatch]:
        LOG.debug("Matching intents via Ollama")
        match = self.llm.predict(utterances[0], lang)
        if match:
            return IntentHandlerMatch(
                match_type=match, match_data={}, skill_id="ovos-ollama-intent-pipeline", utterance=utterances[0]
            )
        return None
