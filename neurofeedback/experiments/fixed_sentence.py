"""Experiment 2: fixed sentence.

At every turn the fixed sentence is teacher-forced as the subject's reply, and all 42
readouts are scored from one read at the four anchor depths, with the Experiment 2
calibration. Only the displayed score differs between arms: the rounded target readout
(real) or an integer drawn uniformly from 0..100 (control).

For one (subject, objective, condition) and each of the 40 sentences:

* real: one session per readout (it is deterministic);
* control, hidden criterion: the prompt does not depend on the readout, so 8 control
  sessions are shared by all readouts (each is scored with all readouts);
* control, disclosure: the prompt names the readout, so 3 control sessions per readout.

Control stream j of sentence r is seeded with ``random.Random(seed + r * S + j)``, S = 8
(hidden) or 3 (disclosed). Sessions are played in chunks of 25 in lockstep, controls first.
"""
from __future__ import annotations

import math
import random
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from .. import prompts
from ..io import append_jsonl, finite, read_jsonl


@dataclass
class Exp2Config:
    horizon: int = 30
    n_sentences: int = 40
    n_control_hidden: int = 8
    n_control_disclosed: int = 3
    chunk: int = 25
    seed: int = 0


def run_path(out_root, subject, objective, condition, shard_id) -> Path:
    return Path(out_root) / subject / f"{objective}.{condition}.shard{shard_id}.jsonl"


def _key(arm, readout, sentence, stream):
    return (arm, readout, int(sentence), stream)


def play_chunk(units, *, backend, readouts, objective, horizon, allow_prefix_cache):
    """Play a chunk of sessions in lockstep. Each unit: dict(arm, readout, sentence, stream,
    text, instructions, rng_seed). Returns one record per unit."""
    rngs = [random.Random(int(u["rng_seed"])) for u in units]
    convs = [[{"role": "user", "content": u["instructions"]}] for u in units]
    shown = [[] for _ in units]
    scores = [{rid: [] for rid in readouts.ids} for _ in units]
    for t in range(horizon):
        items = [(convs[k], u["text"]) for k, u in enumerate(units)]
        meas = readouts.score_all(backend, items, allow_prefix_cache=allow_prefix_cache)
        for k, u in enumerate(units):
            if u["arm"] == "real":
                target = meas[k][u["readout"]]
                if not math.isfinite(target):
                    raise RuntimeError(f"undefined target readout {u['readout']} on sentence "
                                       f"{u['sentence']} at turn {t}")
                s = int(round(target))
            else:
                s = int(rngs[k].randint(0, 100))
            shown[k].append(s)
            for rid, v in meas[k].items():
                scores[k][rid].append(finite(v))
            convs[k].append({"role": "assistant", "content": u["text"]})
            convs[k].append({"role": "user", "content": prompts.feedback(s)})
    return [{"arm": u["arm"], "readout": u["readout"], "sentence": u["sentence"],
             "stream": u["stream"], "shown": shown[k], "scores": scores[k]}
            for k, u in enumerate(units)]


def run(*, subject, objective, condition, sentences, backend, readouts, out_root,
        cfg: Exp2Config, allow_prefix_cache=False, shard_id=0, num_shards=1):
    """Run the sessions of one (subject, objective, condition) for the sentences of this
    shard (``rank % num_shards == shard_id``) that are not on disk yet."""
    path = run_path(out_root, subject, objective, condition, shard_id)
    done = {_key(r["arm"], r["readout"], r["sentence"], r["stream"]) for r in read_jsonl(path)}
    mine = [s for s in sentences[:cfg.n_sentences] if s["rank"] % num_shards == shard_id]
    n_streams = cfg.n_control_hidden if condition == "hidden" else cfg.n_control_disclosed

    def unit(arm, readout, s, stream, rng_seed):
        return dict(arm=arm, readout=readout, sentence=s["rank"], stream=stream, text=s["text"],
                    rng_seed=rng_seed,
                    instructions=prompts.exp2_instructions(condition, objective, s["text"],
                                                           readout or readouts.ids[0]))

    controls = []
    if condition == "hidden":
        for s in mine:
            for j in range(n_streams):
                controls.append(unit("control", None, s, j, cfg.seed + s["rank"] * n_streams + j))
    else:
        for rid in readouts.ids:
            for s in mine:
                for j in range(n_streams):
                    controls.append(unit("control", rid, s, j, cfg.seed + s["rank"] * n_streams + j))
    real = [unit("real", rid, s, None, cfg.seed + s["rank"]) for rid in readouts.ids for s in mine]

    for phase, units in (("control", controls), ("real", real)):
        units = [u for u in units if _key(u["arm"], u["readout"], u["sentence"], u["stream"]) not in done]
        print(f"[exp2] {subject} {objective}.{condition} shard {shard_id}/{num_shards}: "
              f"{len(units)} {phase} sessions to play", flush=True)
        for c0 in range(0, len(units), cfg.chunk):
            chunk = units[c0:c0 + cfg.chunk]
            t0 = time.time()
            try:
                recs = play_chunk(chunk, backend=backend, readouts=readouts, objective=objective,
                                  horizon=cfg.horizon, allow_prefix_cache=allow_prefix_cache)
            except Exception:  # noqa: BLE001  (the chunk stays missing and is re-run later)
                traceback.print_exc()
                continue
            append_jsonl(path, [{"subject": subject, "objective": objective,
                                 "condition": condition, **r} for r in recs])
            print(f"[exp2] {phase}: {len(chunk)} sessions x {cfg.horizon} turns in "
                  f"{time.time() - t0:.0f}s", flush=True)
