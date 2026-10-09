#!/usr/bin/env python
"""Calibration of the readouts for Experiment 2 (1 GPU).

The readouts move with the context, and the instructions of Experiment 2 are a new context.
Each of the 256 reference sentences is read at turn depth 0 under the hidden-criterion
instructions of Experiment 2, with the sentence itself as the fixed sentence (objective by
index parity), through the read path of Experiment 2 (all readouts from one read at the four
anchor depths); the constants are the p1, p99 and median of each readout.
-> artifacts/readouts/<model>.exp2_calibration.npz

    python scripts/build/calibrate_exp2.py --subject qwen3.5-9b
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, corpus_objective, load_backend  # noqa: E402
from neurofeedback import prompts  # noqa: E402
from neurofeedback.readouts import ReadoutSet  # noqa: E402

CHUNK = 96


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    a = ap.parse_args()
    readouts = ReadoutSet(a.subject, calibration="exp1")       # only the raw values are used
    texts = [json.loads(line)["text"] for line in open(REPO / "data" / "c4_reference_256.jsonl")]
    items = [([{"role": "user", "content": prompts.exp2_instructions("hidden", corpus_objective(i), t,
                                                                     readouts.ids[0])}], t)
             for i, t in enumerate(texts)]
    backend = load_backend(a.subject)
    readouts.check_backend(backend)
    raws = np.full((len(readouts.ids), len(texts)), np.nan)
    n_tokens = np.zeros(len(texts), int)
    for i0 in range(0, len(items), CHUNK):
        for k, r in enumerate(readouts.raw_all(backend, items[i0:i0 + CHUNK])):
            raws[:, i0 + k] = [r[rid] for rid in readouts.ids]
            n_tokens[i0 + k] = backend.build_ids(*items[i0 + k])[2]
        print(f"[exp2 calibration] {min(i0 + CHUNK, len(items))}/{len(items)}", flush=True)
    lo, hi, med = (np.full(len(readouts.ids), np.nan) for _ in range(3))
    n_valid = np.zeros(len(readouts.ids), int)
    for i in range(len(readouts.ids)):
        v = raws[i][np.isfinite(raws[i])]
        n_valid[i] = len(v)
        if len(v):
            lo[i], hi[i] = float(np.percentile(v, 1)), float(np.percentile(v, 99))
            if hi[i] - lo[i] < 1e-6:
                hi[i] = lo[i] + 1.0
            med[i] = float(np.percentile(v, 50))
    out = REPO / "artifacts" / "readouts" / f"{a.subject}.exp2_calibration.npz"
    np.savez(out, readout_ids=np.array(readouts.ids), calib_lo=lo, calib_hi=hi, calib_median=med,
             calib_n_valid=n_valid, calib_raws=raws, n_tokens=n_tokens)
    print(f"[exp2 calibration] wrote {out}")


if __name__ == "__main__":
    main()
