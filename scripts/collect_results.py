#!/usr/bin/env python
"""Collect the session records written by scripts/run_exp*.py into the compact arrays of
results/ that the figure and table scripts read.

    python scripts/collect_results.py --exp 2 --outputs outputs/exp2 --results outputs/results

The default --results is outputs/results, so that the released results/ are not overwritten;
the figure and table scripts read another directory with their own --results.

The additional analyses of Experiment 2 have their own files, collected from the --out
directories of scripts/run_exp2.py (several directories are merged; a session found twice must
be identical):

    --exp 2-shuffled   exp2_shuffled.npz   shuffled control (arms shuffled and crossed)
    --exp 2-targeted   exp2_targeted.npz   targeted or broad modulation (hidden criterion: the
                                           real or replayed sessions with all 42 readouts, and
                                           the control sessions)
    --exp 2-depths     exp2_depths.npz     read depths (the depth-ablation real and control
                                           sessions, and their shuffled and crossed sessions)

The released results/ were collected from the paper's runs. Units of Experiments 3 and 4 are
stored there in the order the paper's analysis met them; this script stores them sorted by
(model, monitor, sentence, condition), so bootstrap intervals computed from re-collected
records can differ from the paper's in the Monte Carlo noise.
"""
import argparse
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from neurofeedback.analysis import READOUT_ORDER, ffill  # noqa: E402
from neurofeedback.experiments.free_generation import arm_path  # noqa: E402
from neurofeedback.io import read_jsonl  # noqa: E402
from neurofeedback.models import EXP1_MODELS, EXP2_ANALYSIS_MODELS, EXP2_MODELS  # noqa: E402
from neurofeedback.monitors import CONCEPTS  # noqa: E402
from neurofeedback.readouts import READOUTS_DIR, depth_readout_ids  # noqa: E402

OBJECTIVES, CONDITIONS, ARMS = ["max", "min"], ["hidden", "disclosed"], ["real", "control"]
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
    shape = (5, 5, len(READOUT_ORDER), 2, 2, n_sessions, horizon)
    scores = {a: np.full(shape, np.nan, np.float32) for a in ARMS}
    status = {a: np.full(shape, MISSING, np.int8) for a in ARMS}
    repeat = {a: np.zeros(shape, np.int8) for a in ARMS}
    missing = 0
    for si, S in enumerate(EXP1_MODELS):
        for ri, rid in enumerate(READOUT_ORDER):
            for oi, o in enumerate(OBJECTIVES):
                for ci, c in enumerate(CONDITIONS):
                    for wi, W in enumerate(EXP1_MODELS):
                        for a in ARMS:
                            recs = read_jsonl(arm_path(root, S, rid, o, c, W, a))
                            missing += int(not recs)
                            for r in recs:
                                idx = (si, wi, ri, oi, ci, r["session"])
                                prev = None
                                for t, turn in enumerate(r["turns"][:horizon]):
                                    scores[a][idx + (t,)] = np.nan if turn["score"] is None else turn["score"]
                                    status[a][idx + (t,)] = st = turn_status(turn)
                                    text = turn["text"].strip()
                                    if t > 0 and st == VALID:
                                        repeat[a][idx + (t,)] = int(text == prev)
                                    if t == 0 or st == VALID:
                                        prev = text
    print(f"[exp1] {missing} (writer, arm) files missing")
    return dict(subjects=np.array(EXP1_MODELS), writers=np.array(EXP1_MODELS), readouts=np.array(READOUT_ORDER),
                objectives=np.array(OBJECTIVES), conditions=np.array(CONDITIONS),
                real=scores["real"], control=scores["control"], status_real=status["real"],
                status_control=status["control"], repeat_real=repeat["real"], repeat_control=repeat["control"])


def collect_exp2(roots, n_sentences=40, horizon=30, subjects=EXP2_MODELS):
    roots = [roots] if isinstance(roots, (str, Path)) else roots
    shape = (len(subjects), len(READOUT_ORDER), 2, 2, n_sentences, horizon)
    real = np.full(shape, np.nan, np.float32)
    control = np.full(shape, np.nan, np.float32)
    for si, S in enumerate(subjects):
        for oi, o in enumerate(OBJECTIVES):
            for ci, c in enumerate(CONDITIONS):
                recs = [r for root in roots for f in sorted((Path(root) / S).glob(f"{o}.{c}.shard*.jsonl"))
                        for r in read_jsonl(f)]
                if not recs:
                    print(f"[exp2] no records for {S} {o}.{c}")
                    continue
                ctrl = defaultdict(list)            # (readout, sentence) -> control trajectories
                for r in recs:
                    if r["arm"] == "real":
                        ri = READOUT_ORDER.index(r["readout"])
                        real[si, ri, oi, ci, r["sentence"]] = ffill(np.array(r["scores"][r["readout"]], float))
                    elif r["arm"] != "control":     # the arms of the additional analyses
                        continue
                    elif r["readout"] is None:      # hidden criterion: shared by all readouts
                        for rid in READOUT_ORDER:
                            ctrl[(rid, r["sentence"])].append(ffill(np.array(r["scores"][rid], float)))
                    else:
                        ctrl[(r["readout"], r["sentence"])].append(ffill(np.array(r["scores"][r["readout"]], float)))
                for (rid, sent), trajs in ctrl.items():
                    control[si, READOUT_ORDER.index(rid), oi, ci, sent] = np.nanmean(np.stack(trajs), axis=0)
    return dict(subjects=np.array(subjects), readouts=np.array(READOUT_ORDER), objectives=np.array(OBJECTIVES),
                conditions=np.array(CONDITIONS), real=real, control=control)


# ---- additional analyses of Experiment 2 ------------------------------------------------

def exp2_records(roots, subject, arms):
    """The records of ``subject`` (``<root>/<subject>/*.jsonl``) whose arm is in ``arms``, each
    session once. Files are read one at a time and records are not kept."""
    seen = {}
    for root in roots:
        for f in sorted((Path(root) / subject).glob("*.jsonl")):
            for r in read_jsonl(f):
                if r.get("arm") not in arms:
                    continue
                key = (r["arm"], r["readout"], r["objective"], r["condition"], int(r["sentence"]), r["stream"])
                sig = hash((tuple(r["shown"]), repr(sorted(r["scores"].items()))))
                if key in seen:
                    if seen[key] != sig:
                        raise ValueError(f"{subject}: session {key} recorded twice with different scores ({f})")
                    continue
                seen[key] = sig
                yield r


def trajectory(scores, horizon):
    """Scores of one readout at turns 1..horizon, None as NaN, forward-filled."""
    return ffill(np.array([np.nan if v is None else v for v in scores[:horizon]], float))


def collect_exp2_shuffled(roots, subjects=EXP2_ANALYSIS_MODELS, n_sentences=40, k_shuffled=3, turn=30):
    """``shuffled`` and ``crossed[subject, readout, objective, condition, sentence, stream]``: the
    score of the session's readout at ``turn``."""
    shape = (len(subjects), len(READOUT_ORDER), 2, 2, n_sentences, k_shuffled)
    out = {a: np.full(shape, np.nan) for a in ("shuffled", "crossed")}
    for si, S in enumerate(subjects):
        n = 0
        for r in exp2_records(roots, S, ("shuffled", "crossed")):
            out[r["arm"]][si, READOUT_ORDER.index(r["readout"]), OBJECTIVES.index(r["objective"]),
                          CONDITIONS.index(r["condition"]), r["sentence"], r["stream"]] = \
                trajectory(r["scores"][r["readout"]], turn)[turn - 1]
            n += 1
        print(f"[exp2 shuffled] {S}: {n} sessions")
    return dict(subjects=np.array(subjects), readouts=np.array(READOUT_ORDER), objectives=np.array(OBJECTIVES),
                conditions=np.array(CONDITIONS), turn=np.array(turn), **out)


def collect_exp2_targeted(roots, subjects=EXP2_ANALYSIS_MODELS, n_sentences=40, n_control=8, horizon=30):
    """Hidden criterion. ``real[subject, target, readout, objective, sentence]``: the score of
    every readout at the last turn of the real (or replayed) session whose score is ``target``;
    ``control[subject, readout, objective, sentence]``: the mean of the control sessions at the
    last turn; ``level[subject, readout]``: the mean score of a readout over the turns before
    the last in its own sessions, both objectives (the level of the numbers it displays). Readouts
    in artifact order, the order the bootstraps resample them. Scores are not forward-filled."""
    ids = [str(x) for x in np.load(READOUTS_DIR / f"{subjects[0]}.npz")["readout_ids"]]
    R = len(ids)
    real = np.full((len(subjects), R, R, 2, n_sentences), np.nan)
    level = np.full((len(subjects), R), np.nan)
    control = np.full((len(subjects), R, 2, n_sentences), np.nan)
    for si, S in enumerate(subjects):
        assert [str(x) for x in np.load(READOUTS_DIR / f"{S}.npz")["readout_ids"]] == ids
        ctrl = np.full((n_control, R, 2, n_sentences, horizon), np.nan)
        target = np.full((R, 2, n_sentences, horizon), np.nan)
        for r in exp2_records(roots, S, ("real", "replay", "control")):
            if r["condition"] != "hidden":
                continue
            o, i = OBJECTIVES.index(r["objective"]), int(r["sentence"])
            sc = np.array([[np.nan if v is None else v for v in r["scores"][rid][:horizon]] for rid in ids], float)
            if r["arm"] == "control":
                ctrl[int(r["stream"]), :, o, i] = sc
            else:
                x = ids.index(r["readout"])
                real[si, x, :, o, i] = sc[:, horizon - 1]
                target[x, o, i] = sc[x]
        control[si] = np.nanmean(ctrl, axis=0)[..., horizon - 1]
        with warnings.catch_warnings():             # readouts without sessions (partial runs) stay NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            level[si] = np.nanmean(target[..., :horizon - 1], axis=(1, 2, 3))
        print(f"[exp2 targeted] {S}: {int(np.isfinite(real[si, :, 0, :, :]).sum())} real sessions")
    return dict(subjects=np.array(subjects), readouts=np.array(ids), objectives=np.array(OBJECTIVES),
                real=real, control=control, level=level)


def collect_exp2_depths(roots, subjects=EXP2_ANALYSIS_MODELS, n_sentences=40, n_control=8, k_shuffled=3,
                        horizon=30):
    """Hidden criterion, the 15 direction readouts at each extra depth as the score.
    ``real[subject, readout, objective, condition, sentence, turn]`` (one condition, hidden), as in
    exp2.npz; ``control[subject, readout, objective, condition, sentence]``: the mean of the 8
    control sessions at the last turn; ``shuffled`` and ``crossed[..., sentence, stream]`` at the
    last turn. Scores forward-filled."""
    ids = depth_readout_ids()
    R = len(ids)
    real = np.full((len(subjects), R, 2, 1, n_sentences, horizon), np.nan)
    control = np.full((len(subjects), R, 2, 1, n_sentences), np.nan)
    arms = {a: np.full((len(subjects), R, 2, 1, n_sentences, k_shuffled), np.nan) for a in ("shuffled", "crossed")}
    for si, S in enumerate(subjects):
        ctrl = np.full((n_control, R, 2, n_sentences, horizon), np.nan)
        for r in exp2_records(roots, S, ("real", "control", "shuffled", "crossed")):
            if r["condition"] != "hidden" or (r["readout"] is not None and r["readout"] not in ids):
                continue
            o, i = OBJECTIVES.index(r["objective"]), int(r["sentence"])
            if r["arm"] == "control":
                for x, rid in enumerate(ids):
                    ctrl[int(r["stream"]), x, o, i] = trajectory(r["scores"][rid], horizon)
            elif r["arm"] == "real":
                real[si, ids.index(r["readout"]), o, 0, i] = trajectory(r["scores"][r["readout"]], horizon)
            else:
                arms[r["arm"]][si, ids.index(r["readout"]), o, 0, i, int(r["stream"])] = \
                    trajectory(r["scores"][r["readout"]], horizon)[horizon - 1]
        if np.isnan(ctrl[..., -1]).any():
            raise ValueError(f"{S}: control sessions of the depth ablation incomplete")
        control[si, :, :, 0] = ctrl.mean(axis=0)[..., -1]
        print(f"[exp2 depths] {S}: {int(np.isfinite(real[si, ..., -1]).sum())} real sessions, "
              f"{int(np.isfinite(arms['shuffled'][si]).sum() + np.isfinite(arms['crossed'][si]).sum())} shuffled "
              f"and crossed")
    return dict(subjects=np.array(subjects), readouts=np.array(ids), objectives=np.array(OBJECTIVES),
                conditions=np.array(["hidden"]), turn=np.array(horizon), real=real, control=control, **arms)


def collect_monitor(root, horizon, cot, k_control=3):
    """One unit per (model, monitor, sentence, condition): ``real[unit, objective, turn]`` and
    ``control[unit, objective, stream, turn]``, and for Experiment 4 the same split of
    ``cot_closed`` and ``cot_tokens``."""
    fields = [("", "scores", np.float32)]
    if cot:
        fields += [("cot_closed_", "cot_closed", bool), ("cot_tokens_", "cot_tokens", np.int16)]
    um, umon, usent, ucond = [], [], [], []
    arrays = defaultdict(list)
    for m in MONITOR_ORDER:
        for mon in CONCEPTS:
            recs = [r for f in sorted((Path(root) / m).glob(f"{mon}.shard*.jsonl")) for r in read_jsonl(f)]
            units = defaultdict(dict)
            for r in recs:
                cond, obj = r["template"].split("_")
                units[(r["sentence"], cond)][(obj, r["arm"], r["stream"])] = r
            for (sent, cond) in sorted(units, key=lambda k: (k[0], CONDITIONS.index(k[1]))):
                cells = units[(sent, cond)]
                for prefix, field, dtype in fields:
                    arrays[prefix + "real"].append(np.array(
                        [cells[(o, "real", 0)][field][:horizon] for o in OBJECTIVES], dtype))
                    arrays[prefix + "control"].append(np.array(
                        [[cells[(o, "control", k)][field][:horizon] for k in range(k_control)] for o in OBJECTIVES],
                        dtype))
                um.append(m), umon.append(mon), usent.append(sent), ucond.append(cond)
        print(f"[monitor] {m}: {sum(1 for x in um if x == m)} units")
    return dict(models=np.array(MONITOR_ORDER), monitors=np.array(CONCEPTS), unit_model=np.array(um),
                unit_monitor=np.array(umon), unit_sentence=np.array(usent, np.int32), unit_condition=np.array(ucond),
                **{k: np.stack(v) for k, v in arrays.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", choices=["1", "2", "3", "4", "2-shuffled", "2-targeted", "2-depths"], required=True)
    ap.add_argument("--outputs", nargs="+", required=True, help="the --out directory of scripts/run_exp<N>.py "
                    "(Experiment 2: one or more, merged)")
    ap.add_argument("--results", default=str(REPO / "outputs" / "results"),
                    help="where to write the .npz (the released ones are in results/)")
    a = ap.parse_args()
    if len(a.outputs) > 1 and not a.exp.startswith("2"):
        ap.error("several --outputs only for Experiment 2")
    one = a.outputs[0]
    arrays = {"1": lambda: collect_exp1(one), "2": lambda: collect_exp2(a.outputs),
              "3": lambda: collect_monitor(one, 20, cot=False), "4": lambda: collect_monitor(one, 10, cot=True),
              "2-shuffled": lambda: collect_exp2_shuffled(a.outputs),
              "2-targeted": lambda: collect_exp2_targeted(a.outputs),
              "2-depths": lambda: collect_exp2_depths(a.outputs)}[a.exp]()
    path = Path(a.results) / f"exp{a.exp.replace('-', '_')}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
