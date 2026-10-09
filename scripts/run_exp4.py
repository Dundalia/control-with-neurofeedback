#!/usr/bin/env python
"""Experiment 4 (monitors, chain of thought): run one subject against its monitors.

As Experiment 3, but at every turn the subject first reasons (at most ``--budget`` tokens)
and the fixed sentence is teacher-forced after its reasoning; 10 turns. All sessions of a
monitor (or of its ``--shard``-th stride of sentences) advance in lockstep, and are written
when they finish.

    python scripts/run_exp4.py --subject qwen3.5-27b --out outputs/exp4
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from neurofeedback import prompts  # noqa: E402
from neurofeedback.backend import BackendConfig, VLLMBackend  # noqa: E402
from neurofeedback.experiments.monitor import run_cot, session_specs  # noqa: E402
from neurofeedback.io import read_jsonl, write_jsonl  # noqa: E402
from neurofeedback.models import MONITOR_MODELS, get_model  # noqa: E402
from neurofeedback.monitors import CONCEPTS, load_monitor  # noqa: E402

# Gemma-4-12B's 262k vocabulary needs more memory headroom for long generations.
GPU_MEMORY_UTILIZATION = {"gemma-4-12b": 0.70}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True, choices=MONITOR_MODELS)
    ap.add_argument("--concepts", nargs="*", default=list(CONCEPTS))
    ap.add_argument("--out", default=str(REPO / "outputs" / "exp4"))
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--k-control", type=int, default=3)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--gen-batch", type=int, default=48)
    ap.add_argument("--read-batch", type=int, default=24)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()

    # Each turn adds up to `budget` reasoning tokens to the context.
    backend = VLLMBackend(BackendConfig(get_model(a.subject).hf_id, max_model_len=32768,
                                        gpu_memory_utilization=GPU_MEMORY_UTILIZATION.get(a.subject, 0.9)))
    for concept in a.concepts:
        out = Path(a.out) / a.subject / f"{concept}.shard{a.shard}.jsonl"
        if out.exists():
            print(f"[exp4] {out} exists, skipping")
            continue
        sentences = read_jsonl(REPO / "data" / "monitor_sentences" / a.subject / f"{concept}.jsonl")
        sentences = sentences[a.shard::a.num_shards]
        specs = session_specs(sentences, a.k_control, lambda tmpl, s: prompts.monitor_cot_system(
            concept, *tmpl.split("_"), s["system"], s["answer"]))
        print(f"[exp4] {a.subject}/{concept}: {len(specs)} sessions", flush=True)
        recs = run_cot(backend, load_monitor(a.subject, concept), specs, a.horizon, budget=a.budget,
                       gen_batch=a.gen_batch, read_batch=a.read_batch)
        write_jsonl(out, [{"subject": a.subject, "monitor": concept, **r} for r in recs])


if __name__ == "__main__":
    main()
