#!/usr/bin/env python
"""Experiment 2 (fixed sentence): run one subject.

Loads the subject once and runs the four (objective, condition) blocks for the sentences
of ``--shard`` (``rank % num_shards == shard``). Interrupted runs resume where they stopped.

Prefix-cached reads (``vendor/``) recompute only the sentence at each turn. In the paper
they were used for the subjects in PREFIX_CACHE_READS, and full recomputation for the
others, after checking per subject that the two agree up to vLLM's numerical noise (run a
few sentences with ``--read-mode prefix_cache`` and ``--read-mode full`` and compare the
scores). Gemma-4-12B also ran with vLLM's automatic prefix caching disabled.
Qwen3.5-122B-A10B runs on two GPUs (``tensor_parallel_size`` in neurofeedback/models.py).

    python scripts/run_exp2.py --subject qwen3.5-27b --out outputs/exp2

Additional analyses of Experiment 2 (``neurofeedback/experiments/fixed_sentence.py`` describes
the arms). The planned arms take the numbers displayed by the real sessions from ``--source``
(default: the paper's, results/exp2.npz, rounded):

    # shuffled control: 3 shuffled and 3 crossed sessions per real session
    python scripts/run_exp2.py --subject qwen3.5-27b --arms shuffled crossed --gate-turns 8 --out outputs/exp2_shuffled
    # targeted or broad modulation: the real sessions under the hidden criterion, replayed in
    # order and scored with all 42 readouts, and their control sessions
    python scripts/run_exp2.py --subject qwen3.5-27b --conditions hidden --arms control replay --out outputs/exp2_replay
    # read depths: the 15 direction readouts at 45, 55, 75, 85 and 95% depth as the score
    # (artifacts/readouts/depths/), real and control sessions, then their shuffled control
    python scripts/run_exp2.py --subject qwen3.5-27b --depths 45 55 75 85 95 --conditions hidden --out outputs/exp2_depths
    python scripts/run_exp2.py --subject qwen3.5-27b --depths 45 55 75 85 95 --conditions hidden \\
        --arms shuffled crossed --source results/exp2_depths.npz --gate-turns 8 --out outputs/exp2_depths_shuffled

``--gate-turns T`` first replays the source's numbers in their original order on the shard's
first sentence (max, the first of ``--conditions``, every readout, T turns), compares the
scores with the source's, and stops if the median absolute difference exceeds
``--gate-max-median`` points.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from neurofeedback import prompts  # noqa: E402
from neurofeedback.backend import BackendConfig, VLLMBackend  # noqa: E402
from neurofeedback.experiments.fixed_sentence import (Exp2Config, play_chunk, replay_plans, run,  # noqa: E402
                                                      shuffled_plans)
from neurofeedback.io import load_sentences  # noqa: E402
from neurofeedback.models import EXP2_MODELS, get_model  # noqa: E402
from neurofeedback.readouts import DEPTH_READOUTS_DIR, EXTRA_DEPTHS, ReadoutSet, depth_readout_ids  # noqa: E402

PREFIX_CACHE_READS = {"qwen3.5-27b", "gpt-oss-20b", "mistral-small-3.2-24b", "olmo-3.1-32b",
                      "qwen3.5-9b", "qwen3.6-27b"}
NO_ENGINE_PREFIX_CACHING = {"gemma-4-12b"}
PLANNED = ("shuffled", "crossed", "replay")


def source_scores(path, subject, objective, condition, readout_ids, horizon):
    """(readout, sentence) -> the target-readout scores of the real session in ``path``
    (results/exp2.npz or results/exp2_depths.npz), turns 1..horizon."""
    z = np.load(path)
    si = list(z["subjects"]).index(subject)
    oi, ci = list(z["objectives"]).index(objective), list(z["conditions"]).index(condition)
    rids = list(z["readouts"])
    real = z["real"][si, :, oi, ci, :, :horizon]
    return {(rid, sent): [float(x) for x in real[rids.index(rid), sent]]
            for rid in readout_ids for sent in range(real.shape[1])}


def displayed(path, subject, objective, condition, readout_ids, horizon):
    """(readout, sentence) -> the numbers the real session displayed (its rounded scores)."""
    return {k: [int(round(x)) for x in v]
            for k, v in source_scores(path, subject, objective, condition, readout_ids, horizon).items()}


def replay_gate(a, sentences, backend, readouts, cfg, pc):
    """Replay the source's displayed numbers in order and compare the scores with the source's."""
    cond = a.conditions[0]
    s = [x for x in sentences[:cfg.n_sentences] if x["rank"] % a.num_shards == a.shard][0]
    ref = source_scores(a.source, a.subject, "max", cond, readouts.ids, a.gate_turns)
    units = [dict(arm="replay", readout=rid, sentence=s["rank"], stream=-1, text=s["text"], rng_seed=0,
                  plan=[int(round(x)) for x in ref[(rid, s["rank"])]],
                  instructions=prompts.exp2_instructions(cond, "max", s["text"], rid))
             for rid in readouts.ids]
    diffs = []
    for c0 in range(0, len(units), cfg.chunk):
        for r in play_chunk(units[c0:c0 + cfg.chunk], backend=backend, readouts=readouts, objective="max",
                            horizon=a.gate_turns, allow_prefix_cache=pc):
            new = np.array([np.nan if v is None else v for v in r["scores"][r["readout"]]], float)
            diffs.append(new - np.array(ref[(r["readout"], s["rank"])]))
    d = np.abs(np.stack(diffs))
    out = dict(subject=a.subject, shard=a.shard, sentence=s["rank"], condition=cond, turns=a.gate_turns,
               median_abs=float(np.nanmedian(d)), p95_abs=float(np.nanpercentile(d, 95)),
               max_abs=float(np.nanmax(d)), turn1_median_abs=float(np.nanmedian(d[:, 0])),
               median_abs_by_turn=[float(x) for x in np.nanmedian(d, axis=0)],
               passed=bool(np.nanmedian(d) <= a.gate_max_median))
    path = Path(a.out) / a.subject / f"gate.shard{a.shard}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"[gate] {json.dumps(out)}", flush=True)
    if not out["passed"]:
        raise SystemExit(f"[gate] FAILED: median |score - source| = {out['median_abs']:.2f} points")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True, choices=EXP2_MODELS)
    ap.add_argument("--out", default=str(REPO / "outputs" / "exp2"))
    ap.add_argument("--readouts", nargs="*", help="subset of readout ids (default: all 42)")
    ap.add_argument("--depths", nargs="*", type=int, choices=[int(d[1:]) for d in EXTRA_DEPTHS],
                    help="score the 15 direction readouts read at these depths (in %%) instead of the 42")
    ap.add_argument("--objectives", nargs="*", default=["max", "min"])
    ap.add_argument("--conditions", nargs="*", default=["hidden", "disclosed"])
    ap.add_argument("--read-mode", choices=["paper", "prefix_cache", "full"], default="paper",
                    help="'paper': the mode used for this subject in the paper")
    ap.add_argument("--n-sentences", type=int, default=40)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--arms", nargs="*", default=["control", "real"],
                    choices=["control", "real", *PLANNED], help="arms to play, in this order (the paper: control real)")
    ap.add_argument("--n-shuffled", type=int, default=3)
    ap.add_argument("--source", help="numbers displayed by the real sessions, for the planned arms and the gate "
                                     "(default: results/exp2.npz, or results/exp2_depths.npz with --depths)")
    ap.add_argument("--gate-turns", type=int, default=0)
    ap.add_argument("--gate-max-median", type=float, default=2.0)
    a = ap.parse_args()
    a.source = a.source or str(REPO / "results" / ("exp2_depths.npz" if a.depths else "exp2.npz"))

    pc = (a.subject in PREFIX_CACHE_READS) if a.read_mode == "paper" else a.read_mode == "prefix_cache"
    sentences = load_sentences(REPO / "data" / "exp2_fixed_sentences.jsonl")
    if a.depths:
        if a.readouts:
            raise SystemExit("--readouts and --depths are exclusive")
        readouts = ReadoutSet(a.subject, calibration="exp2", readouts_dir=DEPTH_READOUTS_DIR,
                              readout_ids=depth_readout_ids([f"d{d}" for d in a.depths]))
    else:
        readouts = ReadoutSet(a.subject, calibration="exp2", readout_ids=a.readouts)
    model = get_model(a.subject)
    backend = VLLMBackend(BackendConfig(model.hf_id, max_model_len=8192,
                                        enable_prefix_caching=a.subject not in NO_ENGINE_PREFIX_CACHING,
                                        tensor_parallel_size=model.tensor_parallel_size),
                          prefix_cache_reads=pc)
    readouts.check_backend(backend)
    print(f"[exp2] {a.subject}: {len(readouts.ids)} readouts, reads with "
          f"{'prefix cache' if pc else 'full recomputation'} at layers {readouts.read_layers}")
    cfg = Exp2Config(n_sentences=a.n_sentences, horizon=a.horizon, n_shuffled=a.n_shuffled)
    if a.gate_turns:
        replay_gate(a, sentences, backend, readouts, cfg, pc)
    for obj in a.objectives:
        for cond in a.conditions:
            def numbers(o):
                return displayed(a.source, a.subject, o, cond, readouts.ids, a.horizon)
            plans = {}
            if "shuffled" in a.arms:
                plans["shuffled"] = shuffled_plans(numbers(obj), condition=cond, n_shuffled=a.n_shuffled)
            if "crossed" in a.arms:
                other = {"max": "min", "min": "max"}[obj]
                plans["crossed"] = shuffled_plans(numbers(other), condition=cond, n_shuffled=a.n_shuffled)
            if "replay" in a.arms:
                plans["replay"] = replay_plans(numbers(obj))
            run(subject=a.subject, objective=obj, condition=cond, sentences=sentences, backend=backend,
                readouts=readouts, out_root=a.out, cfg=cfg, allow_prefix_cache=pc,
                shard_id=a.shard, num_shards=a.num_shards, arms=a.arms, plans=plans)


if __name__ == "__main__":
    main()
