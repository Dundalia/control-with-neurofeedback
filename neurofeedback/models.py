"""The models used in the paper.

Every script refers to a model by its key (e.g. ``qwen3.5-27b``). Each entry holds:

* ``hf_id``: the Hugging Face checkpoint loaded with vLLM.
* ``display_name``: the name used in the disclosed prompts of Experiment 1.
* ``rng_tag``: a string used only to seed the random directions of the readouts
  (``scripts/build``); it must not change, or rebuilt readouts will differ from the
  released ones.
* ``writer``: how the model is called when it writes for another subject in
  Experiment 1 (OpenRouter, or a self-served OpenAI-compatible endpoint for OLMo).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Model:
    key: str
    hf_id: str
    display_name: str
    rng_tag: str
    writer: dict = field(default_factory=dict)


_OPENROUTER_NO_REASONING = {"reasoning": {"enabled": False}}

MODELS: dict[str, Model] = {m.key: m for m in [
    Model("qwen3.5-27b", "Qwen/Qwen3.5-27B", "Qwen3.5-27B", "qwen3-5-27b",
          writer={"api": "openrouter", "model": "qwen/qwen3.5-27b", **_OPENROUTER_NO_REASONING,
                  "chat_template_kwargs": {"enable_thinking": False}}),
    Model("gemma-4-31b", "google/gemma-4-31B-it", "Gemma-4-31B", "gemma-4-31b-it",
          writer={"api": "openrouter", "model": "google/gemma-4-31b-it", **_OPENROUTER_NO_REASONING}),
    Model("gpt-oss-20b", "openai/gpt-oss-20b", "GPT-OSS-20B", "gpt-oss-20b",
          writer={"api": "openrouter", "model": "openai/gpt-oss-20b",
                  "reasoning": {"effort": "low"}}),
    Model("mistral-small-3.2-24b", "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
          "Mistral-Small-3.2-24B", "mistral-small-3-2-24b-instruct-2506",
          writer={"api": "openrouter", "model": "mistralai/mistral-small-3.2-24b-instruct",
                  **_OPENROUTER_NO_REASONING}),
    # OLMo is not on OpenRouter: it is served with `vllm serve` (scripts/serve_olmo.sh) at
    # $OLMO_URL. 100265 is <|im_end|>, its turn delimiter.
    Model("olmo-3.1-32b", "allenai/Olmo-3.1-32B-Instruct", "OLMo-3.1-32B", "olmo-3-1-32b-instruct",
          writer={"api": "endpoint", "model": "allenai/Olmo-3.1-32B-Instruct",
                  "base_url_env": "OLMO_URL", "base_url_default": "http://localhost:8000/v1",
                  "extra_body": {"stop_token_ids": [100265]}}),
    Model("qwen3-32b", "Qwen/Qwen3-32B", "Qwen3-32B", "qwen3-32b"),
    Model("qwen3-14b", "Qwen/Qwen3-14B", "Qwen3-14B", "qwen3-14b"),
    Model("gemma-4-12b", "google/gemma-4-12b-it", "Gemma-4-12B", "gemma-4-12b-it"),
    Model("qwen3.5-9b", "Qwen/Qwen3.5-9B", "Qwen3.5-9B", "qwen3-5-9b"),
    Model("qwen3.6-27b", "Qwen/Qwen3.6-27B", "Qwen3.6-27B", "qwen3-6-27b"),
]}

# Experiment 1: every model writes for every subject, itself included.
EXP1_MODELS = ["qwen3.5-27b", "gemma-4-31b", "gpt-oss-20b", "mistral-small-3.2-24b", "olmo-3.1-32b"]
# Experiment 2: the five models of Experiment 1 plus five more.
EXP2_MODELS = EXP1_MODELS + ["qwen3-32b", "qwen3-14b", "gemma-4-12b", "qwen3.5-9b", "qwen3.6-27b"]
# Experiments 3 and 4: the models of Experiment 2 that natively produce a chain of thought.
MONITOR_MODELS = ["gemma-4-12b", "gemma-4-31b", "qwen3-14b", "qwen3-32b", "qwen3.5-9b",
                  "qwen3.5-27b", "qwen3.6-27b", "gpt-oss-20b"]


def get_model(key: str) -> Model:
    if key not in MODELS:
        raise KeyError(f"unknown model {key!r}; known: {sorted(MODELS)}")
    return MODELS[key]
