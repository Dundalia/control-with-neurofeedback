#!/usr/bin/env python
"""Experiment 1 (free generation): run the cells of one subject.

Loads the subject once and runs its 168 cells (42 readouts x max/min x hidden/disclosed),
or the ``--shard``-th of ``--num-shards`` round-robin slices of them (one process per GPU).
Writers other than the subject are called through OpenRouter (OPENROUTER_API_KEY) and, for
OLMo, a vLLM server at $OLMO_URL (scripts/serve_olmo.sh). Cells already on disk are skipped.

    python scripts/run_exp1.py --subject gpt-oss-20b --out outputs/exp1
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from neurofeedback.backend import BackendConfig, VLLMBackend  # noqa: E402
from neurofeedback.experiments.free_generation import Exp1Config, cell_done, run_cell  # noqa: E402
from neurofeedback.io import load_sentences  # noqa: E402
from neurofeedback.models import EXP1_MODELS, get_model  # noqa: E402
from neurofeedback.readouts import ReadoutSet  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True, choices=EXP1_MODELS)
    ap.add_argument("--out", default=str(REPO / "outputs" / "exp1"))
    ap.add_argument("--readouts", nargs="*", help="subset of readout ids (default: all 42)")
    ap.add_argument("--objectives", nargs="*", default=["max", "min"])
    ap.add_argument("--conditions", nargs="*", default=["hidden", "disclosed"])
    ap.add_argument("--writers", nargs="*", default=EXP1_MODELS)
    ap.add_argument("--n-sessions", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()

    readouts = ReadoutSet(a.subject, calibration="exp1", readout_ids=a.readouts)
    cells = [(rid, obj, cond) for rid in readouts.ids for obj in a.objectives for cond in a.conditions]
    cells = cells[a.shard::a.num_shards]
    todo = [c for c in cells if not cell_done(a.out, a.subject, *c, writers=a.writers)]
    print(f"[exp1] {a.subject}: {len(todo)}/{len(cells)} cells to run")
    if not todo:
        return
    opening = [s["text"] for s in load_sentences(REPO / "data" / "exp1_opening_sentences.jsonl")]
    cfg = Exp1Config(n_sessions=a.n_sessions, horizon=a.horizon)
    backend = VLLMBackend(BackendConfig(get_model(a.subject).hf_id, max_model_len=16384))
    readouts.check_backend(backend)
    for rid, obj, cond in todo:
        run_cell(subject=a.subject, readout_id=rid, objective=obj, condition=cond, opening=opening,
                 backend=backend, readouts=readouts, out_root=a.out, cfg=cfg, writers=a.writers)


if __name__ == "__main__":
    main()
