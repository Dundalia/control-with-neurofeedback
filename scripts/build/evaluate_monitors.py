#!/usr/bin/env python
"""AUROC of the monitors on their test split, with a 95% bootstrap interval (1 GPU).

The test split is rebuilt exactly as in train_monitors.py and scored by the released monitors.
-> rows of results/monitor_auroc.csv (Table "AUROC of the 48 monitors")

    python scripts/build/evaluate_monitors.py --subject qwen3.5-9b
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, load_backend  # noqa: E402
from train_monitors import CONCEPTS, auroc, splits  # noqa: E402
from neurofeedback.monitors import load_monitor  # noqa: E402

FIELDS = ["model", "monitor", "layer", "n_test", "n_positive", "auroc", "ci_lo", "ci_hi"]


def bootstrap_ci(scores, y, n_boot=2000, seed=0):
    """Percentile interval of the AUROC over resamples of the test sentences (resamples with a
    single class are redrawn)."""
    rng, n, aucs = np.random.default_rng(seed), len(y), []
    while len(aucs) < n_boot:
        idx = rng.integers(0, n, n)
        if len(set(y[idx].tolist())) == 2:
            aucs.append(auroc(scores[idx], y[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--out", default=str(REPO / "results" / "monitor_auroc.csv"))
    a = ap.parse_args()
    backend = load_backend(a.subject)
    rows = [r for r in csv.DictReader(open(a.out))] if Path(a.out).exists() else []
    rows = [r for r in rows if r["model"] != a.subject]
    for concept in CONCEPTS:
        mon = load_monitor(a.subject, concept)
        _, _, test = splits(concept)
        y = np.array([lab for (_, _, lab, _) in test], np.int64)
        scores = np.array([mon.score(read.numpy()) for i in range(0, len(test), 48)
                           for read in backend.read_residual_multi_batch(
                               [(m, r) for (m, r, _, _) in test[i:i + 48]], mon.layers)])
        lo, hi = bootstrap_ci(scores, y)
        rows.append({"model": a.subject, "monitor": concept, "layer": mon.layer, "n_test": len(y),
                     "n_positive": int(y.sum()), "auroc": round(auroc(scores, y), 6),
                     "ci_lo": round(lo, 6), "ci_hi": round(hi, 6)})
        print(f"[evaluate] {a.subject} {concept}: AUROC {rows[-1]['auroc']:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
