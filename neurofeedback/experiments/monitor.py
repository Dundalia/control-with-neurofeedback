"""Experiments 3 and 4: in-context control of an activation monitor.

Each session opens with a system prompt (the scenario of the sentence and the instructions)
and the question of the sentence. At every turn the fixed answer is teacher-forced, the
monitor scores the answer tokens, and the next user message reports the score: the real one,
or an integer drawn uniformly from 0..100 with ``numpy.random.default_rng(seed)``.

* Experiment 3 (:func:`run_fixed`): the answer is the whole reply.
* Experiment 4 (:func:`run_cot`): at every turn the model first generates its reasoning
  (greedy, at most ``budget`` tokens) in its native reasoning channel, then the answer is
  teacher-forced after it. The reasoning, the answer and the score of every turn stay in
  context. Chat templates drop the reasoning of past turns, so the history is appended as raw
  tokens, with the turn delimiters taken from the template itself.

Seeds: the real session of sentence c has seed c (unused), control stream k has seed
c * 131 + k + 1.
"""
from __future__ import annotations

import re

import numpy as np

from ..backend import FAMILY_CONFIGS, chat_prompt_ids

TEMPLATES = ("hidden_min", "hidden_max", "disclosed_min", "disclosed_max")


def session_specs(sentences, k_control, system_fn):
    """One real and ``k_control`` control sessions per (sentence, template). ``system_fn(
    template, sentence)`` returns the system prompt."""
    specs = []
    for s in sentences:
        c = int(s["id"])
        for tmpl in TEMPLATES:
            base = dict(sentence=c, template=tmpl, question=s["question"], answer=s["answer"],
                        system=system_fn(tmpl, s))
            specs.append({**base, "arm": "real", "stream": 0, "seed": c})
            for k in range(k_control):
                specs.append({**base, "arm": "control", "stream": k, "seed": c * 131 + k + 1})
    return specs


def _messages(spec, history):
    msgs = [{"role": "system", "content": spec["system"]}, {"role": "user", "content": spec["question"]}]
    for answer, score in history:
        msgs.append({"role": "assistant", "content": answer})
        msgs.append({"role": "user", "content": f"Score: {int(round(score))}"})
    return msgs


def run_fixed(backend, monitor, specs, horizon, sub_batch=8, sink=None):
    """Experiment 3. Sessions are played ``sub_batch`` at a time; ``sink(records)`` is called
    after each sub-batch."""
    results = []
    for i0 in range(0, len(specs), sub_batch):
        chunk = [dict(s, history=[], scores=[], shown=[], rng=np.random.default_rng(s["seed"]))
                 for s in specs[i0:i0 + sub_batch]]
        for _ in range(horizon):
            built = [backend.build_ids(_messages(s, s["history"]), s["answer"]) for s in chunk]
            reads = backend.read_residual_multi_ids([ids for ids, _, _ in built], monitor.layers)
            for s, (_, _, alen), read in zip(chunk, built, reads):
                read = read.numpy() if hasattr(read, "numpy") else np.asarray(read)
                score = monitor.score(read[:, -alen:, :])
                s["scores"].append(round(float(score), 3))
                shown = score if s["arm"] == "real" else float(s["rng"].integers(0, 101))
                s["shown"].append(int(round(shown)))
                s["history"].append((s["answer"], shown))
        recs = [{k: s[k] for k in ("sentence", "template", "arm", "stream", "scores", "shown")}
                for s in chunk]
        results.extend(recs)
        if sink:
            sink(recs)
        print(f"[monitor] {min(i0 + sub_batch, len(specs))}/{len(specs)} sessions", flush=True)
    return results


# ---- Experiment 4 ---------------------------------------------------------------------
# (reasoning open, reasoning close, answer open). For Qwen the empty-reasoning frame must
# reproduce `<think>\n\n</think>\n\n{answer}`, the frame the monitors were trained on.
COT_MARKERS = {
    "gemma": ("<|channel>thought\n", "<channel|>", ""),
    "qwen": ("<think>\n", "\n</think>", "\n\n"),
    "gpt_oss": ("<|channel|>analysis<|message|>", "<|end|>",
                "<|start|>assistant<|channel|>final<|message|>"),
}
_GLUE = {}


def _turn_glue(tokenizer, family):
    """(close assistant + open user, close user + open assistant), from the chat template."""
    if family not in _GLUE:
        sa, su = "ZZASSISTANTZZ", "ZZUSERZZ"
        r = tokenizer.apply_chat_template(
            [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"},
             {"role": "assistant", "content": sa}, {"role": "user", "content": su}],
            add_generation_prompt=True, tokenize=False, **FAMILY_CONFIGS[family].think_template_kwargs)
        _, rest = r.split(sa, 1)
        _GLUE[family] = tuple(rest.split(su, 1))
    return _GLUE[family]


def cot_markers(tokenizer, family):
    """COT_MARKERS, without the reasoning opener when the generation prompt already opens the
    reasoning block (some Qwen templates do)."""
    r_open, r_stop, a_open = COT_MARKERS[family]
    _, post = _turn_glue(tokenizer, family)
    if r_open and post.endswith(r_open):
        r_open = ""
    return r_open, r_stop, a_open


def _prefix_and_history(tokenizer, family, base_msgs, history):
    r_open, r_stop, a_open = cot_markers(tokenizer, family)
    mid, post = _turn_glue(tokenizer, family)
    prefix = chat_prompt_ids(tokenizer, base_msgs, FAMILY_CONFIGS[family].think_template_kwargs)
    hist = "".join(f"{r_open}{cot}{r_stop}{a_open}{ans}{mid}Score: {int(round(sc))}{post}"
                   for (cot, ans, sc) in history)
    return prefix + (tokenizer(hist, add_special_tokens=False)["input_ids"] if hist else [])


def _clean_reasoning(text, r_stop):
    if r_stop and r_stop in text:
        text = text.split(r_stop)[0]
    return re.sub(r"<\|[^>]*\|>", "", text).strip()


def run_cot(backend, monitor, specs, horizon, budget=2048, gen_batch=48, read_batch=24):
    """Experiment 4. All sessions advance in lockstep: at each turn the reasoning of every
    session is generated in batches of ``gen_batch``, then the answers are read in batches
    of ``read_batch``."""
    tok, family = backend.tokenizer, backend.family
    r_open, r_stop, a_open = cot_markers(tok, family)
    gen_stop = r_stop.lstrip("\n")
    stop_ids = tok(gen_stop, add_special_tokens=False)["input_ids"]
    open_ids = tok(r_open, add_special_tokens=False)["input_ids"]
    S = [dict(s, base=[{"role": "system", "content": s["system"]}, {"role": "user", "content": s["question"]}],
              history=[], scores=[], shown=[], cots=[], cot_tokens=[], cot_closed=[],
              rng=np.random.default_rng(s["seed"]),
              ans_ids=tok(s["answer"], add_special_tokens=False)["input_ids"]) for s in specs]
    sp = backend.SamplingParams(temperature=0.0, max_tokens=budget, stop=[gen_stop],
                                stop_token_ids=(stop_ids if len(stop_ids) == 1 else None),
                                skip_special_tokens=backend.gen_skip_special)
    for t in range(horizon):
        prefixes = [_prefix_and_history(tok, family, s["base"], s["history"]) for s in S]
        gen_ids = [p + open_ids for p in prefixes]
        outs = []
        for i in range(0, len(gen_ids), gen_batch):
            outs += backend.llm.generate([backend.TokensPrompt(prompt_token_ids=ids)
                                          for ids in gen_ids[i:i + gen_batch]], sp)
        cots = [_clean_reasoning(o.outputs[0].text, gen_stop) for o in outs]
        closed = [o.outputs[0].finish_reason == "stop" for o in outs]
        read_ids = [p + tok(r_open + cot + r_stop + a_open, add_special_tokens=False)["input_ids"]
                    + s["ans_ids"] for s, p, cot in zip(S, prefixes, cots)]
        reads = []
        for i in range(0, len(read_ids), read_batch):
            reads += list(backend.read_residual_multi_ids(read_ids[i:i + read_batch], monitor.layers))
        for s, cot, cl, rd in zip(S, cots, closed, reads):
            rd = rd.numpy() if hasattr(rd, "numpy") else np.asarray(rd)
            score = float(monitor.score(rd[:, -len(s["ans_ids"]):, :]))
            s["scores"].append(round(score, 3))
            s["cots"].append(cot)
            s["cot_tokens"].append(len(tok(cot, add_special_tokens=False)["input_ids"]))
            s["cot_closed"].append(cl)
            shown = score if s["arm"] == "real" else float(s["rng"].integers(0, 101))
            s["shown"].append(int(round(shown)))
            s["history"].append((cot, s["answer"], shown))
        print(f"[monitor-cot] turn {t + 1}/{horizon} done ({len(S)} sessions)", flush=True)
    return [{k: s[k] for k in ("sentence", "template", "arm", "stream", "scores", "shown",
                               "cots", "cot_tokens", "cot_closed")} for s in S]
