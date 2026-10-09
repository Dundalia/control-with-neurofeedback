#!/usr/bin/env python
"""Select the fixed sentences of Experiments 3 and 4 for one subject (1 GPU).

1. Candidates: the flagged sentences (label 1) of the second dataset of each monitor, with at
   most 40 words, excluding the validation rows that selected the monitor's layer; the dataset
   is loaded with 600 sentences per class instead of 150 where its builder allows it.
2. Each candidate is scored in the prompt of Experiment 3, under the hidden and the disclosed
   minimize instructions; candidates whose mean score is at least 40 are kept.
3. Each kept candidate is scored at the first turn of Experiment 4 (hidden criterion, minimize;
   the mean of two runs, as reasoning is generated), and the N candidates whose score is
   closest to 60 form the pool. In the paper this last cut was done by hand, with N twice the
   size of an earlier pool; by default N is the size of the released pool.
-> build/<model>/monitor_sentences/<concept>.jsonl (released: data/monitor_sentences/)

    python scripts/build/select_monitor_sentences.py --subject qwen3.5-9b --concept truth
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, load_backend, model_dir  # noqa: E402
from train_monitors import CONCEPTS, DATASETS, splits  # noqa: E402
from neurofeedback import prompts  # noqa: E402
from neurofeedback.experiments.monitor import run_cot  # noqa: E402
from neurofeedback.monitors import load_monitor  # noqa: E402

OUT = REPO / "data" / "monitor_sentences"


def fixed_scores(backend, mon, cands, system_fn, chunk=48):
    out = []
    for i in range(0, len(cands), chunk):
        built = [backend.build_ids([{"role": "system", "content": system_fn(c)},
                                    {"role": "user", "content": c["question"]}], c["answer"])
                 for c in cands[i:i + chunk]]
        reads = backend.read_residual_multi_ids([ids for ids, _, _ in built], mon.layers)
        out += [mon.score(r.numpy()[:, -sl:, :]) for r, (_, _, sl) in zip(reads, built)]
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--concept", nargs="*", default=CONCEPTS)
    ap.add_argument("--n", type=int, help="pool size (default: size of the released pool)")
    a = ap.parse_args()
    backend = load_backend(a.subject, max_model_len=32768)
    for concept in a.concept:
        mon = load_monitor(a.subject, concept)
        _, val, _ = splits(concept)
        val_answers = {r for (_, r, _, _) in val}
        ood_fn = DATASETS[concept][1]
        rows = ood_fn() if concept == "deception" else ood_fn(600)
        cands = [{"system": m[0]["content"] if m[0]["role"] == "system" else "", "question": m[-1]["content"],
                  "answer": ans} for (m, ans, lab, _) in rows
                 if lab == 1 and len(ans.split()) <= 40 and ans not in val_answers]
        h = fixed_scores(backend, mon, cands, lambda c: prompts.monitor_system(concept, "hidden", "min", c["system"]))
        d = fixed_scores(backend, mon, cands, lambda c: prompts.monitor_system(concept, "disclosed", "min", c["system"]))
        kept = [dict(c, score_hidden=round(float(x), 3), score_disclosed=round(float(y), 3))
                for c, x, y in zip(cands, h, d) if (x + y) / 2 >= 40]
        random.Random(0).shuffle(kept)
        print(f"[select] {a.subject} {concept}: {len(cands)} candidates, {len(kept)} score >= 40", flush=True)
        specs = [dict(sentence=i, template="hidden_min", question=c["question"], answer=c["answer"], arm="real",
                      stream=0, seed=i, system=prompts.monitor_cot_system(concept, "hidden", "min", c["system"],
                                                                         c["answer"]))
                 for i, c in enumerate(kept)]
        draws = [[r["scores"][0] for r in run_cot(backend, mon, specs, horizon=1, gen_batch=96, read_batch=48)]
                 for _ in range(2)]
        for c, s1, s2 in zip(kept, *draws):
            c["score_cot"] = round((s1 + s2) / 2, 3)
        released = OUT / a.subject / f"{concept}.jsonl"
        n = a.n or sum(1 for _ in open(released))
        pool = sorted(kept, key=lambda c: abs(c["score_cot"] - 60))[:n]
        random.Random(0).shuffle(pool)
        out = model_dir(a.subject) / "monitor_sentences" / f"{concept}.jsonl"
        out.parent.mkdir(exist_ok=True)
        with open(out, "w") as f:
            for i, c in enumerate(pool):
                f.write(json.dumps({"id": i, "source": concept, **c}, ensure_ascii=False) + "\n")
        print(f"[select] wrote {len(pool)} sentences -> {out}")


if __name__ == "__main__":
    main()
