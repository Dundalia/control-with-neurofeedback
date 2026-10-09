#!/usr/bin/env python
"""Experiment 2 (fixed sentence): run one subject.

Loads the subject once and runs the four (objective, condition) blocks for the sentences
of ``--shard`` (``rank % num_shards == shard``). Interrupted runs resume where they stopped.

Prefix-cached reads (``vendor/``) recompute only the sentence at each turn. In the paper
they were used for the subjects in PREFIX_CACHE_READS, and full recomputation for the
others, after checking per subject that the two agree up to vLLM's numerical noise (run a
few sentences with ``--read-mode prefix_cache`` and ``--read-mode full`` and compare the
scores). Gemma-4-12B also ran with vLLM's automatic prefix caching disabled.

    python scripts/run_exp2.py --subject qwen3.5-27b --out outputs/exp2
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from neurofeedback.backend import BackendConfig, VLLMBackend  # noqa: E402
from neurofeedback.experiments.fixed_sentence import Exp2Config, run  # noqa: E402
from neurofeedback.io import load_sentences  # noqa: E402
from neurofeedback.models import EXP2_MODELS, get_model  # noqa: E402
from neurofeedback.readouts import ReadoutSet  # noqa: E402

PREFIX_CACHE_READS = {"qwen3.5-27b", "gpt-oss-20b", "mistral-small-3.2-24b", "olmo-3.1-32b",
                      "qwen3.5-9b", "qwen3.6-27b"}
NO_ENGINE_PREFIX_CACHING = {"gemma-4-12b"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True, choices=EXP2_MODELS)
    ap.add_argument("--out", default=str(REPO / "outputs" / "exp2"))
    ap.add_argument("--readouts", nargs="*", help="subset of readout ids (default: all 42)")
    ap.add_argument("--objectives", nargs="*", default=["max", "min"])
    ap.add_argument("--conditions", nargs="*", default=["hidden", "disclosed"])
    ap.add_argument("--read-mode", choices=["paper", "prefix_cache", "full"], default="paper",
                    help="'paper': the mode used for this subject in the paper")
    ap.add_argument("--n-sentences", type=int, default=40)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()

    pc = (a.subject in PREFIX_CACHE_READS) if a.read_mode == "paper" else a.read_mode == "prefix_cache"
    sentences = load_sentences(REPO / "data" / "exp2_fixed_sentences.jsonl")
    readouts = ReadoutSet(a.subject, calibration="exp2", readout_ids=a.readouts)
    backend = VLLMBackend(BackendConfig(get_model(a.subject).hf_id, max_model_len=8192,
                                        enable_prefix_caching=a.subject not in NO_ENGINE_PREFIX_CACHING),
                          prefix_cache_reads=pc)
    readouts.check_backend(backend)
    print(f"[exp2] {a.subject}: reads with {'prefix cache' if pc else 'full recomputation'}")
    cfg = Exp2Config(n_sentences=a.n_sentences, horizon=a.horizon)
    for obj in a.objectives:
        for cond in a.conditions:
            run(subject=a.subject, objective=obj, condition=cond, sentences=sentences, backend=backend,
                readouts=readouts, out_root=a.out, cfg=cfg, allow_prefix_cache=pc,
                shard_id=a.shard, num_shards=a.num_shards)


if __name__ == "__main__":
    main()
