#!/usr/bin/env python
"""Collect the session records written by scripts/run_exp*.py into the compact arrays of
results/ that the figure and table scripts read.

    python scripts/collect_results.py --exp 2 --outputs outputs/exp2 [--results results]

The released results/ were collected from the paper's runs. Units of Experiments 3 and 4 are
stored there in the order the paper's analysis met them; this script stores them sorted by
(model, monitor, sentence, condition), so bootstrap intervals computed from re-collected
records can differ from the paper's in the Monte Carlo noise.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from neurofeedback.analysis import READOUT_ORDER, ffill  # noqa: E402
from neurofeedback.experiments.free_generation import arm_path  # noqa: E402
from neurofeedback.io import read_jsonl  # noqa: E402
from neurofeedback.models import EXP1_MODELS, EXP2_MODELS  # noqa: E402
from neurofeedback.monitors import CONCEPTS  # noqa: E402

OBJECTIVES, CONDITIONS = ["max", "min"], ["hidden", "disclosed"]
MONITOR_ORDER = ["qwen3.5-9b", "qwen3.5-27b", "qwen3.6-27b", "qwen3-14b", "qwen3-32b", "gemma-4-12b",
                 "gemma-4-31b", "gpt-oss-20b"]
VALID, SHORT, LONG, EMPTY, FAILED, UNDEFINED, MISSING = 0, 1, 2, 3, 4, 5, -1


def turn_status(t):
    if t["invalid"] is None:
        return VALID
    if t["invalid"] == "undefined_readout":
        return UNDEFINED
    if t["n_words"] == 0:
        return FAILED if t.get("api_error") else EMPTY
    return SHORT if t["n_words"] < 10 else LONG


def collect_exp1(root, n_sessions=20, horizon=20):
    shape = (5, 5, len(READOUT_ORDER), 2, 2, 2, n_sessions, horizon)
    scores = np.full(shape, np.nan, np.float32)
    status = np.full(shape, MISSING, np.int8)
    repeat = np.zeros(shape, np.int8)
    missing = 0
    for si, S in enumerate(EXP1_MODELS):
        for ri, rid in enumerate(READOUT_ORDER):
            for oi, o in enumerate(OBJECTIVES):
                for ci, c in enumerate(CONDITIONS):
                    for wi, W in enumerate(EXP1_MODELS):
                        for ai, a in enumerate(("real", "control")):
                            recs = read_jsonl(arm_path(root, S, rid, o, c, W, a))
                            missing += int(not recs)
                            for r in recs:
                                idx = (si, wi, ri, oi, ci, ai, r["session"])
                                prev = None
                                for t, turn in enumerate(r["turns"][:horizon]):
                                    scores[idx + (t,)] = np.nan if turn["score"] is None else turn["score"]
                                    status[idx + (t,)] = st = turn_status(turn)
                                    text = turn["text"].strip()
                                    if t > 0 and st == VALID:
                                        repeat[idx + (t,)] = int(text == prev)
                                    if t == 0 or st == VALID:
                                        prev = text
    print(f"[exp1] {missing} (writer, arm) files missing")
    return dict(subjects=np.array(EXP1_MODELS), writers=np.array(EXP1_MODELS), readouts=np.array(READOUT_ORDER),
                objectives=np.array(OBJECTIVES), conditions=np.array(CONDITIONS), arms=np.array(["real", "control"]),
                scores=scores, status=status, repeat=repeat)


def collect_exp2(root, n_sentences=40, horizon=30):
    shape = (len(EXP2_MODELS), len(READOUT_ORDER), 2, 2, n_sentences, horizon)
    real = np.full(shape, np.nan, np.float32)
    control = np.full(shape, np.nan, np.float32)
    for si, S in enumerate(EXP2_MODELS):
        for oi, o in enumerate(OBJECTIVES):
            for ci, c in enumerate(CONDITIONS):
                recs = [r for f in sorted((Path(root) / S).glob(f"{o}.{c}.shard*.jsonl")) for r in read_jsonl(f)]
                if not recs:
                    print(f"[exp2] no records for {S} {o}.{c}")
                    continue
                ctrl = defaultdict(list)            # (readout, sentence) -> control trajectories
                for r in recs:
                    if r["arm"] == "real":
                        ri = READOUT_ORDER.index(r["readout"])
                        real[si, ri, oi, ci, r["sentence"]] = ffill(np.array(r["scores"][r["readout"]], float))
                    elif r["readout"] is None:      # hidden criterion: shared by all readouts
                        for rid in READOUT_ORDER:
                            ctrl[(rid, r["sentence"])].append(ffill(np.array(r["scores"][rid], float)))
                    else:
                        ctrl[(r["readout"], r["sentence"])].append(ffill(np.array(r["scores"][r["readout"]], float)))
                for (rid, sent), trajs in ctrl.items():
                    control[si, READOUT_ORDER.index(rid), oi, ci, sent] = np.nanmean(np.stack(trajs), axis=0)
    return dict(subjects=np.array(EXP2_MODELS), readouts=np.array(READOUT_ORDER), objectives=np.array(OBJECTIVES),
                conditions=np.array(CONDITIONS), real=real, control=control)


def collect_monitor(root, horizon, cot, k_control=3):
    um, umon, usent, ucond = [], [], [], []
    real, control, closed, tokens = [], [], [], []
    for m in MONITOR_ORDER:
        for mon in CONCEPTS:
            recs = [r for f in sorted((Path(root) / m).glob(f"{mon}.shard*.jsonl")) for r in read_jsonl(f)]
            units = defaultdict(dict)
            for r in recs:
                cond, obj = r["template"].split("_")
                units[(r["sentence"], cond)][(obj, r["arm"], r["stream"])] = r
            for (sent, cond) in sorted(units, key=lambda k: (k[0], CONDITIONS.index(k[1]))):
                cells = units[(sent, cond)]
                R = np.zeros((2, horizon), np.int32)
                C = np.zeros((2, k_control, horizon), np.int32)
                CL = np.zeros((2, 1 + k_control, horizon), bool)
                TK = np.zeros((2, 1 + k_control, horizon), np.int16)
                for oi, o in enumerate(OBJECTIVES):
                    sessions = [cells[(o, "real", 0)]] + [cells[(o, "control", k)] for k in range(k_control)]
                    for k, s in enumerate(sessions):
                        milli = np.round(np.array(s["scores"]) * 1000).astype(np.int32)
                        if k == 0:
                            R[oi] = milli
                        else:
                            C[oi, k - 1] = milli
                        if cot:
                            CL[oi, k], TK[oi, k] = s["cot_closed"], s["cot_tokens"]
                um.append(m), umon.append(mon), usent.append(sent), ucond.append(cond)
                real.append(R), control.append(C), closed.append(CL), tokens.append(TK)
        print(f"[monitor] {m}: {sum(1 for x in um if x == m)} units")
    out = dict(models=np.array(MONITOR_ORDER), monitors=np.array(CONCEPTS), unit_model=np.array(um),
               unit_monitor=np.array(umon), unit_sentence=np.array(usent, np.int32), unit_condition=np.array(ucond),
               real_milli=np.stack(real), control_milli=np.stack(control))
    if cot:
        out.update(cot_closed=np.stack(closed), cot_tokens=np.stack(tokens))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=int, choices=[1, 2, 3, 4], required=True)
    ap.add_argument("--outputs", required=True, help="the --out directory of scripts/run_exp<N>.py")
    ap.add_argument("--results", default=str(REPO / "results"))
    a = ap.parse_args()
    arrays = {1: lambda: collect_exp1(a.outputs), 2: lambda: collect_exp2(a.outputs),
              3: lambda: collect_monitor(a.outputs, 20, cot=False),
              4: lambda: collect_monitor(a.outputs, 10, cot=True)}[a.exp]()
    path = Path(a.results) / f"exp{a.exp}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
