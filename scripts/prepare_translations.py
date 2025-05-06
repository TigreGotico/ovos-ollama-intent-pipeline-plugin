import json
import os
from os.path import dirname

locale = f"{dirname(dirname(__file__))}/ovos_ollama_intent_pipeline/locale"
tx = f"{dirname(dirname(__file__))}/translations"

for lang in os.listdir(locale):

    prompts = {}
    for root, _, files in os.walk(f"{locale}/{lang}"):
        b = root.split(f"/{lang}")[-1]

        for f in files:
            if b:
                fid = f"{b}/{f}"
            else:
                fid = f
            if not fid.endswith(".txt"):
                continue

            with open(f"{root}/{f}") as fi:
                prompts[fid] = fi.read()

    os.makedirs(f"{tx}/{lang.lower()}", exist_ok=True)
    if prompts:
        with open(f"{tx}/{lang.lower()}/prompts.json", "w") as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)
