#!/usr/bin/env python
"""Read the 20k corpus with one subject (1 GPU, a few hours).

Each sentence is read in the build context (scripts/build/common.py) at turn depths 0, 5 and
20, at 33% and 66% depth (both in one pass), and averaged over its tokens. For a fixed subsample
of 2000 sentences, the mean log-probability of the sentence tokens is also recorded at each of
``--logprob-depths``. Checkpointed; an interrupted run resumes.
-> build/<model>/corpus_reads.npz: X_<depth>_t<k> [20000, hidden] fp16, layer_<depth>, ids,
   lp_sub_idx, lp_t<k> [2000]

In the paper, Qwen3.5-27B, Gemma-4-31B and GPT-OSS-20B were read with --logprob-depths 0 5
(the turn-20 log-probabilities were never collected); the encoder-residual directions of these
three subjects are then screened without the log-probability criterion.

    python scripts/build/read_corpus.py --subject qwen3.5-9b
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CORPUS_NAMESPACE, DEPTHS, REPO, TURN_DEPTHS, build_conversation, corpus_objective,  # noqa: E402
                    feedback_scores, load_backend, model_dir)

N_LOGPROB_SUB = 2000
SAVE_EVERY = 16


def sequence_logprobs(backend, items, chunk=32):
    """Mean log-probability of the sentence tokens of each (messages, sentence)."""
    means = np.empty(len(items), np.float32)
    for i in range(0, len(items), chunk):
        built = [backend.build_ids(m, r) for (m, r) in items[i:i + chunk]]
        sp = backend.SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
        outs = backend.llm.generate([backend.TokensPrompt(prompt_token_ids=ids) for ids, _, _ in built], sp)
        for j, (o, (ids, ss, sl)) in enumerate(zip(outs, built)):
            lps = [o.prompt_logprobs[p][ids[p]].logprob for p in range(ss, ss + sl)]
            means[i + j] = sum(lps) / len(lps)
    return means


def save(path, store):
    np.savez(str(path) + ".tmp.npz", **store)
    os.replace(str(path) + ".tmp.npz", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--logprob-depths", type=int, nargs="*", default=TURN_DEPTHS)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                    help="lower it for large-vocabulary models (Gemma-4-12B: 0.70)")
    a = ap.parse_args()
    out = model_dir(a.subject) / "corpus_reads.npz"
    rows = sorted((json.loads(l) for l in open(REPO / "data" / "c4_corpus_20k.jsonl")), key=lambda r: r["id"])
    N = len(rows)
    sub_idx = np.sort(np.random.default_rng(0).permutation(N)[:N_LOGPROB_SUB]).astype(np.int32)
    backend = load_backend(a.subject, gpu_memory_utilization=a.gpu_memory_utilization)
    layers = {d: backend.layer_for_depth(depth) for d, depth in DEPTHS}
    order = sorted(layers, key=lambda d: layers[d])
    store = dict(np.load(out)) if out.exists() else {}
    store.update({"ids": np.array([r["id"] for r in rows], np.int32), "lp_sub_idx": sub_idx,
                  **{f"layer_{d}": np.array(l) for d, l in layers.items()}})

    def conv(i, k):
        return build_conversation(corpus_objective(rows[i]["id"]), rows[i]["text"],
                                  feedback_scores(CORPUS_NAMESPACE, rows[i]["id"]), k)

    for k in TURN_DEPTHS:
        for d in order:
            store.setdefault(f"X_{d}_t{k}", np.zeros((N, backend.hidden_size), np.float16))
        filled, since = int(store.get(f"filled_t{k}", 0)), 0
        while filled < N:
            hi = min(filled + a.chunk, N)
            reads = backend.read_residual_multi_batch([(conv(i, k), rows[i]["text"]) for i in range(filled, hi)],
                                                      sorted(layers.values()))
            for j, r in enumerate(reads):
                pooled = r.float().mean(1)                      # [2 layers, hidden], fp32 mean
                for li, d in enumerate(order):
                    store[f"X_{d}_t{k}"][filled + j] = pooled[li].numpy().astype(np.float16)
            filled = hi
            store[f"filled_t{k}"] = np.array(filled)
            since += 1
            if since >= SAVE_EVERY or filled >= N:
                save(out, store)
                since = 0
            print(f"[reads] {a.subject} t{k}: {filled}/{N}", flush=True)
    for k in a.logprob_depths:
        if f"lp_t{k}" not in store:
            store[f"lp_t{k}"] = sequence_logprobs(backend, [(conv(int(i), k), rows[int(i)]["text"]) for i in sub_idx],
                                                  chunk=min(a.chunk, 32))
            save(out, store)
            print(f"[reads] logprobs t{k} done", flush=True)
    print(f"[reads] wrote {out}")


if __name__ == "__main__":
    main()
