"""A deterministic stand-in for the vLLM engine and a toy tokenizer, so that the session loops
run on CPU. Activations depend only on the token prefix, generations only on the prompt."""
import hashlib

import numpy as np
import torch

from neurofeedback.backend import VLLMBackend

WORDS = ("the quiet river carried old songs past every sleeping village under a pale autumn moon "
         "while distant bells answered lonely travelers searching for warm bread and honest friends").split()


def h(*xs):
    return int.from_bytes(hashlib.blake2b(repr(xs).encode(), digest_size=8).digest(), "little")


class ToyTokenizer:
    """One token per whitespace-separated piece; a ChatML-like template."""
    unk_token_id = 0

    def _encode(self, text):
        return [1 + h(w) % 50000 for w in text.split()]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self._encode(text)}

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kw):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            s += "<|im_start|>assistant\n"
        return self._encode(s) if tokenize else s

    def convert_tokens_to_ids(self, t):
        return 0


class SamplingParams:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def TokensPrompt(prompt_token_ids):
    return {"prompt_token_ids": list(prompt_token_ids)}


class _Completion:
    def __init__(self, text, finish):
        self.text, self.finish_reason = text, finish


class _Output:
    pass


class FakeLLM:
    def __init__(self, hidden, keep=64):
        self.hidden, self.keep = hidden, keep
        self.calls = []

    def _acts(self, ids, layers):
        hs, acc = [], 0
        for t in ids:
            acc = h(acc, t)
            hs.append(acc)
        hs = hs[-self.keep:]
        out = np.stack([np.stack([np.random.default_rng(h(x, layer)).standard_normal(self.hidden) + 0.3 * li
                                  for x in hs]) for li, layer in enumerate(layers)]).astype(np.float32)
        return torch.from_numpy(out)

    def _text(self, ids, sp):
        k = h(tuple(ids))
        if getattr(sp, "stop", None):
            return " ".join(WORDS[(k >> (i % 40)) % len(WORDS)] for i in range(5 + k % 20)), \
                ("stop" if k % 5 else "length")
        return " ".join(WORDS[(k // (i + 1)) % len(WORDS)] for i in range(4 + k % 22)).capitalize() + ".", "stop"

    def generate(self, prompts, sp):
        sps = sp if isinstance(sp, list) else [sp] * len(prompts)
        self.calls.append(sps)
        outs = []
        for p, s in zip(prompts, sps):
            o = _Output()
            ids = p["prompt_token_ids"]
            extra = getattr(s, "extra_args", None) or {}
            if "output_residual_stream" in extra:
                o.activations = {"residual_stream": self._acts(ids, sorted(extra["output_residual_stream"]))}
                o.outputs = [_Completion("", "length")]
            else:
                o.activations = None
                o.outputs = [_Completion(*self._text(ids, s))]
            outs.append(o)
        return outs


def fake_backend(model_name="Qwen/Qwen3.5-9B", num_layers=32, hidden=4096, prefix_cache_reads=False):
    """A VLLMBackend whose engine and tokenizer are the fakes above."""
    from neurofeedback.backend import BackendConfig
    b = VLLMBackend.__new__(VLLMBackend)
    b.config, b.prefix_cache_reads, b.model_name = BackendConfig(model_name), prefix_cache_reads, model_name
    b.family, b.tokenizer = "qwen", ToyTokenizer()
    b.num_layers, b.hidden_size = num_layers, hidden
    b.TokensPrompt, b.SamplingParams = TokensPrompt, SamplingParams
    b.stop_ids, b.gen_skip_special = [], True
    b.llm = FakeLLM(hidden)
    return b
