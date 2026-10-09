"""Session records: one JSON object per line, appended as sessions finish, so that an
interrupted run resumes by skipping the sessions already in its file."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path


def finite(x):
    """None for NaN/inf, so that records stay strict JSON."""
    return x if (x is not None and math.isfinite(x)) else None


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:   # a line cut by an interruption
                pass
    return rows


def append_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_jsonl(path, records):
    """Write a whole file atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_sentences(path, rank_key="rank") -> list[dict]:
    rows = sorted(read_jsonl(path), key=lambda r: r[rank_key])
    if [r[rank_key] for r in rows] != list(range(len(rows))):
        raise ValueError(f"{path}: {rank_key} is not 0..N-1")
    return rows
