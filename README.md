# 🧠 OVOS Ollama Intent Pipeline

A intent matching pipeline for [OpenVoiceOS (OVOS)](https://openvoiceos.org), powered by local or remote LLMs via an HTTP API such as [Ollama](https://ollama.com/).

This plugin uses LLM-based reasoning to classify natural language utterances into intent labels registered with the system (Adapt, Padatious, and plugin-specific labels). It works well as a fallback pipeline when other deterministic engines don’t provide a high-confidence match.

---

## ✨ Features

* ✅ LLM-powered classification using prompt engineering
* ✅ Supports multiple languages with custom prompts
* ✅ Automatically syncs Adapt and Padatious intents at runtime
* ✅ Fuzzy matching to fix LLM hallucinations
* ✅ Plug-and-play integration with OVOS pipelines

---

## 📦 Installation

```bash
pip install ovos-ollama-intent-pipeline
```

---

## ⚙️ Configuration

In your `mycroft.conf` or OVOS config file:

```json
{
  "intents": {
    "ovos_ollama_intent_pipeline": {
      "model": "llama3",
      "base_url": "http://192.168.1.200:11434",
      "temperature": 0.0,
      "timeout": 5,
      "fuzzy": true,
      "min_words": 2,
      "ignore_labels": []
    }
  }
}
```

---

## 🧠 Usage

The `LLMIntentPipeline` class plugs into the OVOS intent system. It:

1. Receives an utterance (text).
2. Generates a prompt with few-shot examples and label list.
3. Sends it to an LLM (like LLaMA3, Mistral, etc.).
4. Interprets the LLM output as an intent label.
5. Optionally fuzzy matches it against known labels.


---

## 📁 Locale Support

This plugin supports multilingual prompts using prompt files located under:

```
locale/
  en-us/
    system_prompt.txt
    prompt_template.txt
    few_shot_examples.txt
  pt-pt/
    ...
```

> If no locale matches, it will fallback to a generic multilingual (`mul`) prompt.

---

## 🧪 Tips

* Tune `fuzzy_threshold` and `temperature` to control hallucination tolerance.
* Consider restricting the list of possible labels (`ignore_labels`) for more accurate results.
* Keep your few-shot examples short and relevant.
* Intent syncing uses the OVOS message bus and queries Adapt and Padatious services.
* LLM output is sanitized and optionally fuzzy-matched to prevent unregistered intent errors.

---

## 🛡 License

This project is licensed under the [Apache 2.0 License](LICENSE).
