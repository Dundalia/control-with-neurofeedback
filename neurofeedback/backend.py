"""vLLM backend: greedy generation and teacher-forced activation reads.

Activations are captured with the ``vllm-lens`` plugin (``output_residual_stream``). A read
teacher-forces a sentence as the assistant reply after a conversation and returns the
residual stream on the sentence's tokens only (the *span*). The span is always the tail of
the token sequence, which every read relies on.

Each model family renders its assistant turn differently, and the span must contain only
the content tokens of the reply, never the template wrapper or a reasoning block:

* Qwen 3.x: ``enable_thinking=False`` prefills an empty think block; the reply follows.
* Gemma 4: no thinking by default; the reply follows the model turn opener.
* GPT-OSS (Harmony): an empty analysis channel and the final-channel opener are prefilled,
  so that the model replies directly in the final channel.
* Mistral-Small-3.x and OLMo 3.x: the reply follows the generation prompt.

vLLM is run with the prebuilt FlashAttention backend and ``enforce_eager`` (FlashInfer
JIT-compiles at start-up and needs nvcc), and with vLLM's V1 model runner, because the V2
runner lacks the ``input_batch`` attribute that the vllm-lens hook reads.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class FamilyConfig:
    name: str
    nothink_template_kwargs: dict = field(default_factory=dict)
    think_template_kwargs: dict = field(default_factory=dict)
    # Context between the generation prompt and the reply, excluded from the read span.
    assistant_prefix: str = ""
    # Turn delimiters to stop generation on, besides the default EOS.
    stop_tokens: list = field(default_factory=list)


FAMILY_CONFIGS = {
    "qwen": FamilyConfig("qwen", {"enable_thinking": False}, {"enable_thinking": True}),
    "gemma": FamilyConfig("gemma", {}, {"enable_thinking": True}),
    "gpt_oss": FamilyConfig(
        "gpt_oss", {"reasoning_effort": "low"}, {"reasoning_effort": "medium"},
        assistant_prefix="<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>",
        # Without these stops the model runs past its answer into a new reasoning turn.
        stop_tokens=["<|return|>", "<|end|>", "<|call|>"]),
    "mistral": FamilyConfig("mistral"),
    # In multi-turn conversations OLMo emits <|im_end|> but not its EOS <|endoftext|>.
    "olmo": FamilyConfig("olmo", stop_tokens=["<|im_end|>"]),
}


def detect_family(model_name: str) -> str:
    n = model_name.lower()
    if "gpt-oss" in n or "gpt_oss" in n:
        return "gpt_oss"
    if "gemma" in n:
        return "gemma"
    if "qwen" in n:
        return "qwen"
    if "olmo" in n:
        return "olmo"
    if "mistral" in n:
        return "mistral"
    raise ValueError(f"unknown model family for {model_name!r}")


def chat_prompt_ids(tokenizer, messages, template_kwargs) -> list[int]:
    """Render the chat prompt, with the generation prompt, to a flat list of token ids."""
    enc = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                        **template_kwargs)
    ids = enc["input_ids"] if (isinstance(enc, dict) or hasattr(enc, "input_ids")) else enc
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if len(ids) and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def build_teacher_forced_ids(tokenizer, messages, forced_response, family,
                             enable_thinking=False):
    """Return ``(input_ids, span_start, span_len)``, where the span is exactly the tokens of
    ``forced_response``, placed after the chat prompt and the family's assistant prefix."""
    cfg = FAMILY_CONFIGS[family]
    tkw = cfg.think_template_kwargs if enable_thinking else cfg.nothink_template_kwargs
    ctx_ids = chat_prompt_ids(tokenizer, messages, tkw)
    if cfg.assistant_prefix:
        ctx_ids += tokenizer(cfg.assistant_prefix, add_special_tokens=False)["input_ids"]
    resp_ids = tokenizer(forced_response, add_special_tokens=False)["input_ids"]
    if len(resp_ids) == 0:
        raise ValueError("forced_response tokenizes to 0 tokens")
    return ctx_ids + resp_ids, len(ctx_ids), len(resp_ids)


@dataclass
class BackendConfig:
    model_name: str
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 16384
    enable_prefix_caching: bool = True
    trust_remote_code: bool = True


class VLLMBackend:
    def __init__(self, config: BackendConfig, prefix_cache_reads: bool = False):
        self.config = config
        # Prefix-cached reads (Experiment 2 only, needs vendor/vllm_lens_span_prefix.patch):
        # a read reuses the cached KV of the conversation and recomputes only the span.
        self.prefix_cache_reads = prefix_cache_reads
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
        self._load()

    def _load(self):
        from transformers import AutoConfig, AutoTokenizer
        from vllm import LLM, SamplingParams, TokensPrompt

        cfg = self.config
        self.model_name = cfg.model_name
        self.family = detect_family(cfg.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name,
                                                       trust_remote_code=cfg.trust_remote_code)
        hf_cfg = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=cfg.trust_remote_code)
        text_cfg = getattr(hf_cfg, "text_config", hf_cfg)
        self.num_layers = int(text_cfg.num_hidden_layers)
        self.hidden_size = int(text_cfg.hidden_size)
        self.TokensPrompt = TokensPrompt
        self.SamplingParams = SamplingParams
        fam = FAMILY_CONFIGS[self.family]
        self.stop_ids = [i for t in fam.stop_tokens
                         for i in (self.tokenizer.convert_tokens_to_ids(t),)
                         if isinstance(i, int) and i >= 0 and i != self.tokenizer.unk_token_id]
        # Keep special tokens in the output only for families whose stops need extraction.
        self.gen_skip_special = not bool(self.stop_ids)
        llm_kwargs = {}
        if self.family == "mistral":
            # Mistral-Small-3.2 is a vision-language model used text-only: skip vision profiling.
            llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        self.llm = LLM(model=cfg.model_name, dtype=cfg.dtype,
                       gpu_memory_utilization=cfg.gpu_memory_utilization,
                       enforce_eager=True, max_model_len=cfg.max_model_len,
                       enable_prefix_caching=cfg.enable_prefix_caching,
                       trust_remote_code=cfg.trust_remote_code, **llm_kwargs)
        # The first capturing request registers the plugin hooks; pay that cost once here.
        try:
            self.read_residual_multi_batch([([{"role": "user", "content": "warm up"}], "ok")],
                                           [self.num_layers // 2])
        except Exception as e:  # pragma: no cover
            print(f"[backend] warmup skipped: {e}")

    def layer_for_depth(self, depth: float) -> int:
        return max(0, min(self.num_layers - 1, round(depth * (self.num_layers - 1))))

    def build_ids(self, messages, forced_response, enable_thinking=False):
        return build_teacher_forced_ids(self.tokenizer, messages, forced_response, self.family,
                                        enable_thinking)

    def _check_layers(self, layers):
        for layer in layers:
            if not 0 <= layer < self.num_layers:
                raise IndexError(f"layer {layer} out of range [0, {self.num_layers})")

    @staticmethod
    def _captured(output):
        """The [n_layers, seq, hidden] torch tensor that vllm-lens attaches to an output."""
        return output.activations["residual_stream"]

    def read_residual_multi_batch(self, items, layers, enable_thinking=False,
                                  allow_prefix_cache=False):
        """Span reads at several layers. Returns one [n_layers, span_len, hidden] tensor per
        item, with layers in ascending order (vllm-lens stacks them sorted).

        With ``allow_prefix_cache`` and ``self.prefix_cache_reads``, the cached context is
        reused and only the span is recomputed. If the span itself was cached (identical
        sequences, e.g. two control sessions of one sentence at turn 1), the capture is
        shorter than the span, and the item is re-read with full recomputation."""
        layers = sorted({int(x) for x in layers})
        self._check_layers(layers)
        built = [self.build_ids(m, r, enable_thinking) for (m, r) in items]
        use_pc = bool(self.prefix_cache_reads and allow_prefix_cache)
        if use_pc:
            # One SamplingParams per request: the patched plugin pops span_prefix_cache from
            # each request's extra_args, so a shared object would only opt in the first one.
            sp = [self.SamplingParams(temperature=0.0, max_tokens=1,
                                      extra_args={"output_residual_stream": list(layers),
                                                  "span_prefix_cache": True})
                  for _ in built]
        else:
            sp = self.SamplingParams(temperature=0.0, max_tokens=1,
                                     extra_args={"output_residual_stream": list(layers)})
        outs = self.llm.generate([self.TokensPrompt(prompt_token_ids=ids) for ids, _, _ in built], sp)
        reads, aliased = [None] * len(built), []
        for i, (o, (_, _, sl)) in enumerate(zip(outs, built)):
            a = self._captured(o)
            if a.dim() == 2:
                a = a.unsqueeze(0)
            if a.shape[0] != len(layers):
                raise RuntimeError(f"expected {len(layers)} captured layers, got {tuple(a.shape)}")
            o.activations = None
            if use_pc and a.shape[1] < sl:
                aliased.append(i)
                continue
            reads[i] = a[:, -sl:, :].float().cpu()
        if aliased:
            print(f"[backend] {len(aliased)}/{len(built)} spans hit the prefix cache, "
                  f"re-read with full recomputation", flush=True)
            sp_full = self.SamplingParams(temperature=0.0, max_tokens=1,
                                          extra_args={"output_residual_stream": list(layers)})
            outs2 = self.llm.generate([self.TokensPrompt(prompt_token_ids=built[i][0])
                                       for i in aliased], sp_full)
            for i, o in zip(aliased, outs2):
                sl = built[i][2]
                a = self._captured(o)
                if a.dim() == 2:
                    a = a.unsqueeze(0)
                if a.shape[0] != len(layers) or a.shape[1] < sl:
                    raise RuntimeError(f"full-recompute read still short: {tuple(a.shape)} vs {sl}")
                o.activations = None
                reads[i] = a[:, -sl:, :].float().cpu()
        return reads

    def read_residual_multi_ids(self, ids_list, layers):
        """Reads over caller-built token sequences. Returns one [n_layers, len(ids), hidden]
        tensor per sequence, layers ascending."""
        layers = sorted({int(x) for x in layers})
        self._check_layers(layers)
        sp = self.SamplingParams(temperature=0.0, max_tokens=1,
                                 extra_args={"output_residual_stream": list(layers)})
        outs = self.llm.generate([self.TokensPrompt(prompt_token_ids=list(ids)) for ids in ids_list], sp)
        reads = []
        for o, ids in zip(outs, ids_list):
            a = self._captured(o)
            if a.dim() == 2:
                a = a.unsqueeze(0)
            if a.shape[0] != len(layers):
                raise RuntimeError(f"expected {len(layers)} captured layers, got {tuple(a.shape)}")
            reads.append(a[:, -len(ids):, :].float().cpu())
            o.activations = None
        return reads

    def generate_batch(self, conversations, max_new_tokens: int = 64, seeds=None) -> list[str]:
        """Greedy replies to many conversations in one vLLM call. ``seeds`` are passed per
        request (they do not affect greedy decoding)."""
        cfg = FAMILY_CONFIGS[self.family]
        pre_ids = (self.tokenizer(cfg.assistant_prefix, add_special_tokens=False)["input_ids"]
                   if cfg.assistant_prefix else [])
        prompts = [self.TokensPrompt(prompt_token_ids=chat_prompt_ids(
            self.tokenizer, messages, cfg.nothink_template_kwargs) + pre_ids)
            for messages in conversations]
        seeds = seeds if seeds is not None else [None] * len(prompts)
        sps = [self.SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, seed=s,
                                   stop_token_ids=(self.stop_ids or None),
                                   skip_special_tokens=self.gen_skip_special) for s in seeds]
        outs = self.llm.generate(prompts, sps)
        return [self._extract_answer(o.outputs[0].text) for o in outs]

    def _extract_answer(self, text: str) -> str:
        if self.family == "gpt_oss" and "<|channel|>final<|message|>" in text:
            text = text.split("<|channel|>final<|message|>")[-1]
        if "</think>" in text:
            text = text.split("</think>")[-1]
        for t in ("<|im_end|>", "<|return|>", "<|end|>", "<turn|>", "<eos>", "</s>"):
            text = text.split(t)[0]
        return text.strip()
