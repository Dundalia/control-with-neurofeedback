"""Experiment 2: fixed sentence.

At every turn the fixed sentence is teacher-forced as the subject's reply, and all 42
readouts are scored from one read at the four anchor depths (and, for the readouts of the
depth ablation, at their own depths), with the Experiment 2 calibration. Only the displayed
score differs between arms: the rounded target readout (real) or an integer drawn uniformly
from 0..100 (control).

For one (subject, objective, condition) and each of the 40 sentences:

* real: one session per readout (it is deterministic);
* control, hidden criterion: the prompt does not depend on the readout, so 8 control
  sessions are shared by all readouts (each is scored with all readouts);
* control, disclosure: the prompt names the readout, so 3 control sessions per readout.

Control stream j of sentence r is seeded with ``random.Random(seed + r * S + j)``, S = 8
(hidden) or 3 (disclosed). Sessions are played in chunks of 25 in lockstep, controls first.
Every session records the scores of all the readouts of ``readouts`` at every turn.

Three more arms show numbers fixed in advance (a *plan*) and serve the additional analyses
of Experiment 2 (the appendix of the same name):

* shuffled (shuffled control): for every real session, 3 sessions that show the numbers the
  real session displayed after turns 1..29, in a random order (``shuffled_plans``). The order
  is drawn per (condition, readout, sentence, stream) and shared by the two objectives.
* crossed (shuffled control): as shuffled, but with the numbers of the real session of the
  other objective, in the same orders, so a crossed max session shows exactly the sequence of
  the shuffled min session and vice versa. Shuffled and crossed together show the two
  objectives the same sequences, as the random control does; the *shuffled control* is the
  mean of the 6 sessions.
* replay: for every real session, one session that shows its numbers in their original order
  (``replay_plans``). With the sentence fixed this reproduces the real session up to vLLM's
  numerical noise; it is how the real sessions were re-read with all 42 readouts for the
  analysis of targeted or broad modulation.
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
    n_shuffled: int = 3


def shuffled_plans(shown, *, condition, n_shuffled, seed=0):
    """Displayed numbers of the shuffled sessions. ``shown`` maps (readout, sentence) to the
    numbers the real session displayed (one per turn). Stream k of (readout, sentence) shows
    the first T-1 of them in a random order and then the last one, which no read follows.
    The order does not depend on the objective, so max and min share it, and the crossed arm
    is ``shuffled_plans`` of the other objective's numbers."""
    plans = {}
    for (rid, sent), seq in shown.items():
        seq = [int(x) for x in seq]
        for k in range(n_shuffled):
            rng = random.Random(f"{seed}/shuffled/{condition}/{rid}/{int(sent)}/{k}")
            perm = rng.sample(range(len(seq) - 1), len(seq) - 1)
            plans[(rid, int(sent), k)] = [seq[j] for j in perm] + seq[-1:]
    return plans


def replay_plans(shown):
    """Displayed numbers of the replay sessions: the real session's, in their order (stream 0)."""
    return {(rid, int(sent), 0): [int(x) for x in seq] for (rid, sent), seq in shown.items()}


PLANNED_ARMS = ("shuffled", "crossed", "replay")


def run_path(out_root, subject, objective, condition, shard_id) -> Path:
    return Path(out_root) / subject / f"{objective}.{condition}.shard{shard_id}.jsonl"


def _key(arm, readout, sentence, stream):
    return (arm, readout, int(sentence), stream)


def play_chunk(units, *, backend, readouts, objective, horizon, allow_prefix_cache):
    """Play a chunk of sessions in lockstep. Each unit: dict(arm, readout, sentence, stream,
    text, instructions, rng_seed, plan). Returns one record per unit."""
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
            elif u["arm"] in PLANNED_ARMS:
                s = int(u["plan"][t])
            elif u["arm"] == "control":
                s = int(rngs[k].randint(0, 100))
            else:
                raise ValueError(f"unknown arm {u['arm']}")
            shown[k].append(s)
            for rid, v in meas[k].items():
                scores[k][rid].append(finite(v))
            convs[k].append({"role": "assistant", "content": u["text"]})
            convs[k].append({"role": "user", "content": prompts.feedback(s)})
    return [{"arm": u["arm"], "readout": u["readout"], "sentence": u["sentence"],
             "stream": u["stream"], "shown": shown[k], "scores": scores[k]}
            for k, u in enumerate(units)]


def run(*, subject, objective, condition, sentences, backend, readouts, out_root,
        cfg: Exp2Config, allow_prefix_cache=False, shard_id=0, num_shards=1,
        arms=("control", "real"), plans=None):
    """Run the sessions of one (subject, objective, condition) for the sentences of this
    shard (``rank % num_shards == shard_id``) that are not on disk yet. ``arms`` are played
    in the given order. ``plans`` maps each planned arm to its plans, keyed (readout,
    sentence, stream): ``shuffled_plans`` of this objective's real numbers for the shuffled
    arm, of the other objective's for the crossed arm, ``replay_plans`` for the replay arm."""
    path = run_path(out_root, subject, objective, condition, shard_id)
    done = {_key(r["arm"], r["readout"], r["sentence"], r["stream"]) for r in read_jsonl(path)}
    mine = [s for s in sentences[:cfg.n_sentences] if s["rank"] % num_shards == shard_id]
    n_streams = cfg.n_control_hidden if condition == "hidden" else cfg.n_control_disclosed

    def unit(arm, readout, s, stream, rng_seed, plan=None):
        return dict(arm=arm, readout=readout, sentence=s["rank"], stream=stream, text=s["text"],
                    rng_seed=rng_seed, plan=plan,
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

    def planned(arm):
        pl = (plans or {}).get(arm)
        if pl is None:
            raise ValueError(f"the {arm} arm needs plans")
        n = 1 if arm == "replay" else cfg.n_shuffled
        out = []
        for rid in readouts.ids:
            for s in mine:
                for k in range(n):
                    plan = pl[(rid, s["rank"], k)]
                    if len(plan) < cfg.horizon:
                        raise ValueError(f"{arm} plan for {rid} sentence {s['rank']} is shorter than the horizon")
                    out.append(unit(arm, rid, s, k, 0, plan=plan))
        return out

    by_arm = {"control": controls, "real": real, **{a: planned(a) for a in PLANNED_ARMS if a in arms}}

    for phase in arms:
        units = by_arm[phase]
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
