import json
import os
from os.path import dirname

locale = f"{dirname(dirname(__file__))}/ovos_ollama_intent_pipeline/locale"
tx = f"{dirname(dirname(__file__))}/translations"

for lang in os.listdir(tx):
    prompts = f"{tx}/{lang}/prompts.json"

    os.makedirs(f"{locale}/{lang.lower()}", exist_ok=True)
    if os.path.isfile(prompts):
        with open(prompts) as f:
            data = json.load(f)
        for fid, txt in data.items():
            if fid.startswith("/"):
                p = f"{locale}/{lang.lower()}{fid}"
            else:
                p = f"{locale}/{lang.lower()}/{fid}"
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(f"{locale}/{lang.lower()}/{fid}", "w") as f:
                f.write(txt)
