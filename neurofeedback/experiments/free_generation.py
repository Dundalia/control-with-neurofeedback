"""Experiment 1: free generation.

A *cell* is (subject, readout, objective, condition). In each cell, every writer (the
subject itself and the four other models) plays 20 sessions with real feedback and 20 with
random feedback. All sessions of one (writer, arm) advance turn by turn in lockstep, so that
the subject's generations and reads are batched; API calls of different sessions overlap.

Session i opens with opening sentence i as the first assistant reply (turn 1). From turn 2
on the writer replies to the conversation; a reply with fewer than 10 or more than 20 words
receives the invalid-reply message instead of a score and has no readout (as does a reply
on which a geometry readout is undefined, being too short). The score shown is the rounded
readout (real arm) or an integer drawn uniformly from 0..100 with ``random.Random(seed + i)``
(control arm). Real and control sessions with the same index share the opening sentence.
"""
from __future__ import annotations

import asyncio
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .. import prompts
from ..io import finite, read_jsonl, write_jsonl
from ..models import EXP1_MODELS, get_model
from ..writers import SelfWriter, api_writer


@dataclass
class Exp1Config:
    horizon: int = 20
    n_sessions: int = 20
    word_band: tuple = (10, 20)
    self_max_new_tokens: int = 64
    api_max_tokens: int = 128
    api_timeout_s: float = 120.0     # bounds each API call, retries included
    seed: int = 0


def arm_path(out_root, subject, readout_id, objective, condition, writer, arm) -> Path:
    return (Path(out_root) / subject / readout_id / f"{objective}.{condition}"
            / f"{writer}.{arm}.jsonl")


async def run_arm(*, subject, writer_key, arm, readout_id, objective, condition, opening,
                  backend, readouts, cfg: Exp1Config, gpu_exec) -> list[dict]:
    """Play all sessions of one (writer, arm) of a cell. Returns one record per session."""
    n = cfg.n_sessions
    lo, hi = cfg.word_band
    is_self = writer_key == subject
    writer = SelfWriter(backend, cfg.self_max_new_tokens) if is_self else \
        api_writer(writer_key, cfg.api_max_tokens)
    loop = asyncio.get_running_loop()
    instructions = prompts.exp1_instructions(condition, objective,
                                             get_model(subject).display_name, readout_id)
    convs = [[{"role": "user", "content": instructions}] for _ in range(n)]
    rngs = [random.Random(cfg.seed + i) for i in range(n)]
    turns = [[] for _ in range(n)]

    for t in range(cfg.horizon):
        infos = [None] * n
        if t == 0:
            replies, forced = [opening[i] for i in range(n)], True
        elif is_self:
            seeds = [cfg.seed + i + t for i in range(n)]
            replies = await loop.run_in_executor(
                gpu_exec, lambda: writer.write_batch(convs, seeds))
            forced = False
        else:
            # A failed call (after retries, or a terminal error) becomes an empty reply.
            raw = await asyncio.gather(*[asyncio.wait_for(writer.write(convs[k]), cfg.api_timeout_s)
                                         for k in range(n)], return_exceptions=True)
            errs = [r for r in raw if isinstance(r, Exception)]
            if errs:
                print(f"[exp1] {writer_key}.{arm} t{t}: {len(errs)}/{n} API calls failed, e.g. "
                      f"{type(errs[0]).__name__}: {str(errs[0])[:140]}", flush=True)
            replies = ["" if isinstance(r, Exception) else r[0] for r in raw]
            infos = [{"api_error": True} if isinstance(r, Exception) else r[1] for r in raw]
            forced = False
        replies = [r or "" for r in replies]
        n_words = [len(r.split()) for r in replies]
        valid = [k for k in range(n) if forced or lo <= n_words[k] <= hi]
        items = [(convs[k], replies[k]) for k in valid]
        scores = (await loop.run_in_executor(
            gpu_exec, lambda: readouts.score_one(backend, readout_id, items))) if items else []
        by_k = dict(zip(valid, scores))
        for k in range(n):
            raw_v, score = by_k.get(k, (None, None))
            if score is not None and not math.isfinite(score):
                reason = "undefined_readout"
            elif k not in by_k:
                reason = "word_band"
            else:
                reason = None
            if reason is None:
                shown = int(round(score)) if arm == "real" else int(rngs[k].randint(0, 100))
                fb = prompts.feedback(shown)
            else:
                shown, fb = None, prompts.invalid_reply()
            info = infos[k] or {}
            turns[k].append({"text": replies[k], "n_words": n_words[k], "invalid": reason,
                             "raw": finite(raw_v), "score": finite(score) if reason is None else None,
                             "shown": shown, **{f: info[f] for f in
                                                ("provider", "completion_tokens", "finish_reason",
                                                 "api_error") if f in info}})
            # Some chat templates reject an empty assistant message.
            convs[k].append({"role": "assistant", "content": replies[k] if replies[k].strip() else " "})
            convs[k].append({"role": "user", "content": fb})

    return [{"subject": subject, "writer": writer_key, "arm": arm, "readout": readout_id,
             "objective": objective, "condition": condition, "session": i,
             "instructions": instructions, "writer_config": writer.describe(), "turns": turns[i]}
            for i in range(n)]


def run_cell(*, subject, readout_id, objective, condition, opening, backend, readouts,
             out_root, cfg: Exp1Config, writers=None):
    """Run the (writer, arm) pairs of a cell that are not on disk yet. Pairs run
    concurrently; calls to the subject's GPU are serialized on one thread."""
    writers = writers or EXP1_MODELS
    todo = [(w, a) for w in writers for a in ("real", "control")
            if not arm_path(out_root, subject, readout_id, objective, condition, w, a).exists()]
    if not todo:
        return
    gpu_exec = ThreadPoolExecutor(max_workers=1)

    async def one(w, a):
        t0 = time.time()
        try:
            recs = await run_arm(subject=subject, writer_key=w, arm=a, readout_id=readout_id,
                                 objective=objective, condition=condition, opening=opening,
                                 backend=backend, readouts=readouts, cfg=cfg, gpu_exec=gpu_exec)
        except Exception as e:  # noqa: BLE001  (the pair stays missing and is re-run later)
            print(f"[exp1] {readout_id}.{objective}.{condition} {w}.{a} FAILED: "
                  f"{type(e).__name__}: {e}", flush=True)
            return
        write_jsonl(arm_path(out_root, subject, readout_id, objective, condition, w, a), recs)
        print(f"[exp1] {readout_id}.{objective}.{condition} {w}.{a}: {len(recs)} sessions in "
              f"{time.time() - t0:.0f}s", flush=True)

    async def main():
        await asyncio.gather(*[one(w, a) for w, a in todo])

    asyncio.run(main())
    gpu_exec.shutdown(wait=True)


def cell_done(out_root, subject, readout_id, objective, condition, writers=None) -> bool:
    writers = writers or EXP1_MODELS
    return all(arm_path(out_root, subject, readout_id, objective, condition, w, a).exists()
               for w in writers for a in ("real", "control"))


def load_cell(out_root, subject, readout_id, objective, condition, writer, arm) -> list[dict]:
    return read_jsonl(arm_path(out_root, subject, readout_id, objective, condition, writer, arm))
