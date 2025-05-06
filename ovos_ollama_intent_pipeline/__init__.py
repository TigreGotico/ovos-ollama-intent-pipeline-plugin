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
                 labels: Optional[Dict[str, str]] = None,
                 ignore_labels: Optional[List[str]] = None,
                 ignore_skills: Optional[List[str]] = None,
                 bus: Optional[MessageBusClient] = None):
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

        # these skills are specialized/ambiguous enough to consistently throw off the LLMS
        self.ignore_skills = ignore_skills or []

        self.load_locale()

        if not bus:
            bus = MessageBusClient()
            bus.run_in_thread()

        self.bus = bus
        if not self.bus.connected_event.is_set():
            self.bus.connected_event.wait()

        if not labels:
            self.sync_intents()
        else:
            self.mappings = {self.normalize(l): l for l in labels}

    def sync_intents(self, timeout=1):
        # TODO - persona/ocp/common_query/stop intents
        labels = self._get_adapt_intents(timeout) + self._get_padatious_intents(timeout)

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
        return [l for l in self.mappings.values()
                if l not in self.ignore_labels
                and not any(s in l for s in self.ignore_skills)]

    def load_locale(self):
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
        msg = Message("intent.service.adapt.manifest.get")
        res = self.bus.wait_for_response(msg, "intent.service.adapt.manifest", timeout=timeout)
        if not res:
            return None
        return [i["name"] for i in res.data["intents"] if i["name"] not in self.ignore_labels]

    def _get_padatious_intents(self, timeout=1):
        msg = Message("intent.service.padatious.manifest.get")
        res = self.bus.wait_for_response(msg, "intent.service.padatious.manifest", timeout=timeout)
        if not res:
            return None
        return [i for i in res.data["intents"] if i not in self.ignore_labels]

    @staticmethod
    def normalize(text: str) -> str:
        # standardize labels as much as possible to reduce token usage + not confuse the LLM
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
                                   labels=self.config.get("labels", []),
                                   ignore_labels=self.config.get("ignore_labels", []),
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
