#!/usr/bin/env python
"""Raw readout values for calibration (1 GPU), after readouts.py --stage directions.

* The 256 reference sentences (data/c4_reference_256.jsonl), each read at turn depth 0 in the
  build context, at the four anchor depths: raw value of every readout, computed with the
  run-time code (neurofeedback.readouts) -> build/<model>/calibration_reads.npz
* The first 2000 corpus sentences, read the same way: the 12 geometry readouts, the number of
  tokens and the mean token log-probability -> build/<model>/structural_values.npz
  (the text-predictability diagnostics of readouts.py --stage finalize)

    python scripts/build/calibration_reads.py --subject qwen3.5-9b
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, build_conversation, corpus_objective, load_backend, model_dir  # noqa: E402
from read_corpus import sequence_logprobs  # noqa: E402
from readouts import N_PUBLIC, readout_table  # noqa: E402
from neurofeedback.readouts import ANCHOR_DEPTHS, Readout, _geometry_kind  # noqa: E402


def make_readouts(directions):
    """Readouts with placeholder calibration (only their raw values are used here)."""
    out = []
    for rid, fam, depth in readout_table():
        if fam == "geometry":
            out.append(Readout(rid, fam, depth, _geometry_kind(rid)[2], 0.0, 1.0, 0.0))
        else:
            out.append(Readout(rid, fam, depth, (depth,), 0.0, 1.0, 0.0,
                               mu=directions[f"mu_{depth}"].astype(np.float64),
                               vector=directions[f"vec_{rid}"].astype(np.float64)))
    return out


def read_spans(backend, items, layers):
    """Per item: anchor -> [T, hidden] float64 activations on the sentence."""
    lay = sorted(layers.values())
    out = []
    for r in backend.read_residual_multi_batch(items, lay):
        H = r.numpy().astype(np.float64)
        out.append({d: H[lay.index(l)] for d, l in layers.items()})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--chunk", type=int, default=96)
    a = ap.parse_args()
    out_dir = model_dir(a.subject)
    directions = np.load(out_dir / "directions.npz")
    readouts = make_readouts(directions)
    backend = load_backend(a.subject)
    layers = {d: backend.layer_for_depth(x) for d, x in ANCHOR_DEPTHS.items()}
    assert all(layers[d] == int(directions[f"layer_{d}"]) for d in ("d33", "d66"))

    ref = [json.loads(l)["text"] for l in open(REPO / "data" / "c4_reference_256.jsonl")]
    raws = np.full((len(readouts), len(ref)), np.nan)
    n_tokens = np.zeros(len(ref), np.int32)
    for i0 in range(0, len(ref), a.chunk):
        idx = list(range(i0, min(i0 + a.chunk, len(ref))))
        items = [(build_conversation(corpus_objective(i), ref[i], None, 0), ref[i]) for i in idx]
        for i, spans in zip(idx, read_spans(backend, items, layers)):
            n_tokens[i] = spans["d33"].shape[0]
            for k, r in enumerate(readouts):
                raws[k, i] = r.raw(spans)
        print(f"[calibration] {a.subject}: {idx[-1] + 1}/{len(ref)}", flush=True)
    np.savez(out_dir / "calibration_reads.npz", raws=raws, readout_ids=np.array([r.id for r in readouts]),
             n_tokens=n_tokens, num_layers=np.array(backend.num_layers),
             **{f"layer_{d}": np.array(l) for d, l in layers.items()})

    corpus = sorted((json.loads(l) for l in open(REPO / "data" / "c4_corpus_20k.jsonl")), key=lambda r: r["id"])
    corpus = corpus[:N_PUBLIC]
    geometry = [r for r in readouts if r.family == "geometry"]
    vals = {r.id: np.full(N_PUBLIC, np.nan) for r in geometry}
    lp, nt = np.full(N_PUBLIC, np.nan), np.zeros(N_PUBLIC, np.int32)
    chunk = min(a.chunk, 64)
    for i0 in range(0, N_PUBLIC, chunk):
        batch = corpus[i0:i0 + chunk]
        items = [(build_conversation(corpus_objective(s["id"]), s["text"], None, 0), s["text"]) for s in batch]
        for s, spans in zip(batch, read_spans(backend, items, layers)):
            nt[s["id"]] = spans["d33"].shape[0]
            for r in geometry:
                vals[r.id][s["id"]] = r.raw(spans)
        lp[[s["id"] for s in batch]] = sequence_logprobs(backend, items, chunk=len(items))
        print(f"[structural] {a.subject}: {i0 + len(batch)}/{N_PUBLIC}", flush=True)
    np.savez(out_dir / "structural_values.npz", logprob_mean=lp, n_tokens=nt, **vals)
    print(f"[calibration] wrote {out_dir}")


if __name__ == "__main__":
    main()
