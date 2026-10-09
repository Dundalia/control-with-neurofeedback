#!/usr/bin/env python
"""Experiment 3 (monitors, fixed sentence): run one subject against its monitors.

For every sentence of a monitor (``data/monitor_sentences/<model>/<concept>.jsonl``, or the
``--shard``-th stride of them), runs 4 templates (hidden/disclosed x min/max), each with one
real and 3 control sessions of 20 turns.

    python scripts/run_exp3.py --subject qwen3.5-27b --out outputs/exp3
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from neurofeedback import prompts  # noqa: E402
from neurofeedback.backend import BackendConfig, VLLMBackend  # noqa: E402
from neurofeedback.experiments.monitor import run_fixed, session_specs  # noqa: E402
from neurofeedback.io import append_jsonl, read_jsonl  # noqa: E402
from neurofeedback.models import MONITOR_MODELS, get_model  # noqa: E402
from neurofeedback.monitors import CONCEPTS, load_monitor  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True, choices=MONITOR_MODELS)
    ap.add_argument("--concepts", nargs="*", default=list(CONCEPTS))
    ap.add_argument("--out", default=str(REPO / "outputs" / "exp3"))
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--k-control", type=int, default=3)
    ap.add_argument("--sub-batch", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()

    backend = VLLMBackend(BackendConfig(get_model(a.subject).hf_id, max_model_len=16384))
    for concept in a.concepts:
        out = Path(a.out) / a.subject / f"{concept}.shard{a.shard}.jsonl"
        sentences = read_jsonl(REPO / "data" / "monitor_sentences" / a.subject / f"{concept}.jsonl")
        sentences = sentences[a.shard::a.num_shards]
        specs = session_specs(sentences, a.k_control, lambda tmpl, s: prompts.monitor_system(
            concept, *tmpl.split("_"), s["system"]))
        done = {(r["sentence"], r["template"], r["arm"], r["stream"]) for r in read_jsonl(out)}
        # Sessions are played in sub-batches; resume at the first sub-batch not on disk.
        todo = [s for s in specs if (s["sentence"], s["template"], s["arm"], s["stream"]) not in done]
        print(f"[exp3] {a.subject}/{concept}: {len(todo)}/{len(specs)} sessions to play", flush=True)
        if not todo:
            continue
        run_fixed(backend, load_monitor(a.subject, concept), todo, a.horizon, sub_batch=a.sub_batch,
                  sink=lambda recs: append_jsonl(out, [{"subject": a.subject, "monitor": concept, **r}
                                                       for r in recs]))


if __name__ == "__main__":
    main()
