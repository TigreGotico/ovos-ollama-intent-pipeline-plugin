import collections
import os.path
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
                 bus: Optional[MessageBusClient] = None):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        self.min_words = min_words
        self.fuzzy = fuzzy
        self.fuzzy_strategy = MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY
        self.fuzzy_threshold = 0.7
        self.ignore_labels = ignore_labels or []
        self.prompts = collections.defaultdict(dict)

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
            self.labels = labels
            self.mappings = {self.normalize(l): l for l in labels}

    def sync_intents(self, timeout=1):
        # TODO - allow retrieving these from bus just like adapt/padatious
        PERSONA = ["ovos-persona-pipeline:ask",
                   "ovos-persona-pipeline:summon",
                   "ovos-persona-pipeline:release_persona",
                   "ovos-persona-pipeline:list_personas",
                   "ovos-persona-pipeline:active_persona"]
        OCP = ["ovos-common-play-pipeline:play",
               "ovos-common-play-pipeline:open",
               "ovos-common-play-pipeline:media_stop",
               "ovos-common-play-pipeline:next",
               "ovos-common-play-pipeline:prev",
               "ovos-common-play-pipeline:pause",
               "ovos-common-play-pipeline:play_favorites",
               "ovos-common-play-pipeline:like_song",
               "ovos-common-play-pipeline:resume",
               "ovos-common-play-pipeline:save_game",
               "ovos-common-play-pipeline:load_game"]
        CQ = [
            'ovos-common-query-pipeline:general_question'
        ]
        STOP = [
            # "ovos-stop-pipeline:global_stop",
            "ovos-stop-pipeline:stop",
        ]

        self.labels = self._get_adapt_intents(timeout) + self._get_padatious_intents(timeout) + \
                      STOP + OCP + CQ + PERSONA

        # mappings are there to help out the LLM a bit
        # normalizing labels to give them some more structure
        self.mappings = {self.normalize(l): l for l in self.labels}


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
        return [i["name"] for i in res.data["intents"]]

    def _get_padatious_intents(self, timeout=1):
        msg = Message("intent.service.padatious.manifest.get")
        res = self.bus.wait_for_response(msg, "intent.service.padatious.manifest", timeout=timeout)
        if not res:
            return None
        return res.data["intents"]

    @staticmethod
    def normalize(text: str) -> str:
        norm = (text.lower().
                replace("", "").
                replace(".openvoiceos", "").
                replace("skill-ovos-", "ovos-skill-"))
        return norm

    def predict(self, utterance: str, lang: str) -> Optional[str]:
        if len(utterance.split()) < self.min_words:
            LOG.debug(f"Skipping LLM intent match, utterance too short (< {self.min_words} words)")
            return None

        lang, score = closest_match(lang, [l for l in self.prompts if l != "mul"], 10)

        if lang not in self.prompts:
            lang = "mul"  # use multilingual prompts

        system = self.prompts[lang]["system"]
        prompt_template = self.prompts[lang]["prompt_template"]
        examples = self.prompts[lang]["few_shot_examples"]

        prompt = prompt_template.format(transcribed_text=utterance,
                                        language=lang,
                                        label_list="\n- ".join(self.labels),
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

        if result == "None" or not result:
            LOG.debug(f"⚠️ No intent for utterance: '{utterance}'")
            return None

        mistakes = {
            # ".OpenVoiceOS": ".openvoiceos",
            "(intent)": "",
            "": ""
        }
        for k, v in mistakes.items():
            result = result.replace(k, v)

        # force a valid label
        if self.fuzzy and result not in self.labels:
            best, score = match_one(result, self.labels, strategy=self.fuzzy_strategy)
            if score >= self.fuzzy_threshold:
                LOG.debug(
                    f"⚠️ utterance '{utterance}' - fuzzy match hallucinated intent  ({score}) - {result} -> {best}")
                result = best
        if result not in self.labels:
            LOG.warning(f"⚠️ Error with model {self.model} and utterance '{utterance}': hallucinated intent - {result}")
            return None

        # ensure output is a valid intent, undo any normalization done to help the LLM
        return self.mappings.get(result) or result


class LLMIntentPipeline(ConfidenceMatcherPipeline):

    def __init__(self, bus: Optional[Union[MessageBusClient, FakeBus]] = None,
                 config: Optional[Dict] = None):
        config = config or Configuration().get('intents', {}).get("ovos_ollama_intent_pipeline") or dict()
        super().__init__(bus, config)

        self.llm = LLMIntentEngine(model=self.config.get("model"),
                                   base_url=self.config.get("base_url"),
                                   temperature=self.config.get("temperature", 0.0),
                                   timeout=self.config.get("timeout", 10),
                                   fuzzy=self.config.get("fuzzy", True),
                                   min_words=self.config.get("min_words", 2),
                                   labels=self.config.get("labels", []),
                                   ignore_labels=self.config.get("ignore_labels", []),
                                   bus=self.bus)

    def match_low(self, utterances: List[str], lang: str, message: Message) -> Optional[IntentHandlerMatch]:
        self.llm.sync_intents() # intents only known at runtime (dynamic)  TODO - optimize to only sync when needed
        match = self.llm.predict(utterances[0], lang)
        if match:
            return IntentHandlerMatch(
                match_type=match, match_data={}, skill_id="ovos-ollama-intent-pipeline", utterance=utterances[0]
            )
        return None
