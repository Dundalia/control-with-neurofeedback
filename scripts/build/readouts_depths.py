#!/usr/bin/env python
"""Build the 15 direction readouts of a subject at the extra depths of the depth ablation
(45, 55, 75, 85 and 95% of the layers), with the recipe of the 33% and 66% readouts, and
assemble the artifact artifacts/readouts/depths/<model>.npz: the 42 readouts of the paper plus
15 per extra depth, with its Experiment 2 calibration.

The functions of the other build scripts are reused; the per-depth steps that they hard-code for
33% and 66% are repeated here for any depth.

Stages
  embed       (GPU, once for all subjects) the 20k corpus under the sentence encoder, as
              embed_corpus.py -> build/c4_corpus_20k_embeddings.npz
  reads       (GPU) one block of the 20k corpus (--rows-shard i --rows-shards n), read in the
              build context at turn depths 0, 5 and 20, mean over the sentence tokens, at all
              the extra depths in one pass -> build/<model>/reads.rows<i>of<n>.npz
  aux         (GPU) the concept-probe sentences at the extra depths, and the mean token
              log-probability of the 2000-sentence screening subsample at the turn depths
              read_corpus.py used for this subject -> build/<model>/concept_reads.npz, lp.npz
  merge       (CPU) the row blocks -> build/<model>/corpus_reads.npz (as read_corpus.py)
  directions  (CPU) per extra depth: concept probes (fit as concept_probes.py), principal-
              component bands (as readouts.py --stage directions), encoder-residual directions
              (as encoder_residual.py) -> build/<model>/directions_<depth>.npz
  calibrate   (GPU) the artifact (the paper's 42 readouts, copied, plus the new ones) and its
              Experiment 2 calibration (as calibrate_exp2.py: the paper's constants for the 42,
              the 256 reference sentences read at the extra depths for the rest)
              -> artifacts/readouts/depths/<model>.npz, <model>.exp2_calibration.npz

In the paper the corpus reads were sharded over 8 GPUs (Qwen models), 10 (Gemma-4-31B) or 5
(Mistral-Small-3.2-24B), one block each:

    python scripts/build/readouts_depths.py embed
    python scripts/build/readouts_depths.py reads --subject qwen3.5-27b --rows-shard 0 --rows-shards 8
    python scripts/build/readouts_depths.py aux --subject qwen3.5-27b
    python scripts/build/readouts_depths.py merge --subject qwen3.5-27b
    python scripts/build/readouts_depths.py directions --subject qwen3.5-27b --depths d45
    python scripts/build/readouts_depths.py calibrate --subject qwen3.5-27b
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np  # noqa: E402

BUILD_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(BUILD_SRC))
from common import (BUILD, CORPUS_NAMESPACE, REPO, TURN_DEPTHS, build_conversation, corpus_objective,  # noqa: E402
                    feedback_scores, load_backend, model_dir, screen_random_directions)
from neurofeedback import prompts  # noqa: E402
from neurofeedback.models import get_model  # noqa: E402
from neurofeedback.readouts import DEPTH_READOUTS_DIR, EXTRA_DEPTHS, depth_readout_ids  # noqa: E402

NEW_DEPTHS = EXTRA_DEPTHS
ARTIFACTS = DEPTH_READOUTS_DIR


def new_readout_table(depths=tuple(NEW_DEPTHS)):
    """(id, family, depth) of the 15 direction readouts per extra depth, in artifact order."""
    return [(rid, rid.split("_", 1)[0], rid.rsplit("_", 1)[1]) for rid in depth_readout_ids(depths)]




def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, BUILD_SRC / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build_readouts = _load("build_readouts", "readouts.py")      # corpus_pcs, BANDS, CONCEPTS, ENCRES, SCREEN_TURN_DEPTHS
encoder_residual = _load("encoder_residual", "encoder_residual.py")
concept_probes = _load("concept_probes", "concept_probes.py")
read_corpus = _load("read_corpus", "read_corpus.py")

# Turn depths of the screening log-probabilities in the paper's build (read_corpus.py docstring):
# Qwen3.5-27B, Gemma-4-31B and GPT-OSS-20B were read with --logprob-depths 0 5.
LP_DEPTHS = {"qwen3.5-27b": [0, 5], "gemma-4-31b": [0, 5], "gpt-oss-20b": [0, 5]}
SAVE_EVERY = 16


def save(path, store):
    path = Path(path)
    np.savez(str(path) + ".tmp.npz", **store)
    os.replace(str(path) + ".tmp.npz", path)


def corpus_rows():
    return sorted((json.loads(line) for line in open(REPO / "data" / "c4_corpus_20k.jsonl")), key=lambda r: r["id"])


def depth_layers(backend, depths):
    layers = {d: backend.layer_for_depth(NEW_DEPTHS[d]) for d in depths}
    if len(set(layers.values())) != len(layers):
        raise ValueError(f"two depths map to one layer: {layers}")
    return layers


def pooled_reads(backend, items, layers, chunk, tag, store=None, key_fmt=None, offset=0, ckpt=None):
    """Mean over the sentence tokens at every layer of ``layers`` (dict depth -> layer), in one
    pass. Fills store[key_fmt.format(d=depth)][offset + i] (fp16); resumable via ``filled``."""
    order = sorted(layers, key=lambda d: layers[d])
    lay = [layers[d] for d in order]
    fkey = key_fmt.format(d="filled")
    filled, since = int(store.get(fkey, 0)), 0
    while filled < len(items):
        hi = min(filled + chunk, len(items))
        reads = backend.read_residual_multi_batch(items[filled:hi], lay)
        for j, r in enumerate(reads):
            pooled = r.float().mean(1)                                   # [n_layers, hidden]
            for li, d in enumerate(order):
                store[key_fmt.format(d=d)][offset + filled + j] = pooled[li].numpy().astype(np.float16)
        filled = hi
        store[fkey] = np.array(filled)
        since += 1
        if ckpt is not None and (since >= SAVE_EVERY or filled >= len(items)):
            save(ckpt, store)
            since = 0
        print(f"[reads] {tag}: {filled}/{len(items)}", flush=True)


# ---- stages ------------------------------------------------------------------------------

def stage_embed(a):
    out = BUILD / "c4_corpus_20k_embeddings.npz"
    if out.exists():
        print(f"[embed] {out} exists")
        return
    _load("embed_corpus", "embed_corpus.py").main()


def stage_reads(a):
    rows = corpus_rows()
    N = len(rows)
    blocks = np.array_split(np.arange(N), a.rows_shards)
    mine = blocks[a.rows_shard]
    if a.limit:
        mine = mine[:a.limit]
    depths = a.depths
    out = model_dir(a.subject) / f"reads.rows{a.rows_shard}of{a.rows_shards}.npz"
    backend = load_backend(a.subject, gpu_memory_utilization=a.gpu_memory_utilization)
    layers = depth_layers(backend, depths)
    store = dict(np.load(out)) if out.exists() else {}
    store.update({"rows": mine.astype(np.int32), "num_layers": np.array(backend.num_layers),
                  **{f"layer_{d}": np.array(l) for d, l in layers.items()}})
    t0 = time.time()
    for k in TURN_DEPTHS:
        for d in depths:
            store.setdefault(f"X_{d}_t{k}", np.zeros((len(mine), backend.hidden_size), np.float16))
        items = [(build_conversation(corpus_objective(rows[i]["id"]), rows[i]["text"],
                                     feedback_scores(CORPUS_NAMESPACE, rows[i]["id"]), k), rows[i]["text"])
                 for i in mine]
        pooled_reads(backend, items, layers, a.chunk, f"{a.subject} rows {a.rows_shard}/{a.rows_shards} t{k}",
                     store=store, key_fmt="X_{d}_t" + str(k), ckpt=out)
    store["done"] = np.array(1)
    save(out, store)
    print(f"[reads] wrote {out} in {time.time() - t0:.0f}s", flush=True)


def stage_aux(a):
    out_dir = model_dir(a.subject)
    data = json.load(open(concept_probes.SENTENCES_FILE))
    backend = load_backend(a.subject, gpu_memory_utilization=a.gpu_memory_utilization)
    layers = depth_layers(backend, a.depths)
    ck = out_dir / "concept_reads.npz"
    store = dict(np.load(ck)) if ck.exists() else {}
    store.update({f"layer_{d}": np.array(l) for d, l in layers.items()})
    for c in concept_probes.CONCEPTS:
        sents = data["concepts"][c]["sentences"]
        if a.probe_limit:                 # testing: the first n/2 sentences of each label
            sents = ([s for s in sents if s["label"] == 0][:a.probe_limit // 2]
                     + [s for s in sents if s["label"] == 1][:a.probe_limit // 2])
        items, si, td, lb = [], [], [], []
        for k in TURN_DEPTHS:
            for s in sents:
                obj = "max" if s["objective"] == "maximize" else "min"
                items.append((build_conversation(obj, s["text"], feedback_scores(c, s["idx"]), k), s["text"]))
                si.append(s["idx"]), td.append(k), lb.append(s["label"])
        for d in a.depths:
            store.setdefault(f"{c}_X_{d}", np.zeros((len(items), backend.hidden_size), np.float16))
        store.update({f"{c}_sent_idx": np.array(si), f"{c}_turn_depth": np.array(td), f"{c}_label": np.array(lb)})
        pooled_reads(backend, items, layers, a.chunk, f"{a.subject} {c}", store=store,
                     key_fmt=f"{c}_X_" + "{d}", ckpt=ck)
    save(ck, store)
    lp_path = out_dir / "lp.npz"
    lp = dict(np.load(lp_path)) if lp_path.exists() else {}
    rows = corpus_rows()
    N = len(rows)
    sub_idx = np.sort(np.random.default_rng(0).permutation(N)[:read_corpus.N_LOGPROB_SUB]).astype(np.int32)
    if a.limit:                       # testing: the subsample rows among the first ``limit`` corpus rows
        sub_idx = sub_idx[sub_idx < a.limit]
    lp["lp_sub_idx"] = sub_idx
    for k in LP_DEPTHS.get(a.subject, TURN_DEPTHS):
        if f"lp_t{k}" not in lp:
            items = [(build_conversation(corpus_objective(rows[int(i)]["id"]), rows[int(i)]["text"],
                                         feedback_scores(CORPUS_NAMESPACE, rows[int(i)]["id"]), k), rows[int(i)]["text"])
                     for i in sub_idx]
            lp[f"lp_t{k}"] = read_corpus.sequence_logprobs(backend, items, chunk=min(a.chunk, 32))
            save(lp_path, lp)
            print(f"[aux] logprobs t{k} done", flush=True)
    print(f"[aux] wrote {ck} and {lp_path}", flush=True)


def stage_merge(a):
    out_dir = model_dir(a.subject)
    parts = sorted(out_dir.glob("reads.rows*of*.npz"))
    parts = [p for p in parts if ".tmp" not in p.name]
    N = len(corpus_rows())
    rows = corpus_rows()
    first = np.load(parts[0])
    hid = first[f"X_{a.depths[0]}_t0"].shape[1]
    out = {"ids": np.array([r["id"] for r in rows], np.int32)}
    for d in a.depths:
        out[f"layer_{d}"] = first[f"layer_{d}"]
        for k in TURN_DEPTHS:
            out[f"X_{d}_t{k}"] = np.zeros((N, hid), np.float16)
    covered = np.zeros(N, bool)
    for p in parts:
        z = np.load(p)
        assert int(z["done"]) == 1, f"{p.name} is not complete"
        r = z["rows"]
        for d in a.depths:
            assert int(z[f"layer_{d}"]) == int(out[f"layer_{d}"])
            for k in TURN_DEPTHS:
                out[f"X_{d}_t{k}"][r] = z[f"X_{d}_t{k}"]
        covered[r] = True
    if not covered.all() and not a.allow_partial:
        raise SystemExit(f"[merge] {int((~covered).sum())} corpus rows missing")
    lp = np.load(out_dir / "lp.npz")
    out.update({k: lp[k] for k in lp.files})
    out["covered"] = covered
    save(out_dir / "corpus_reads.npz", out)
    print(f"[merge] {len(parts)} blocks, {int(covered.sum())}/{N} rows -> {out_dir / 'corpus_reads.npz'}", flush=True)


def directions_one_depth(subject, d, reads, conc, E, texts):
    """The 15 direction readouts at depth ``d``, as the paper's build computes them at 33/66%."""
    rng_tag = get_model(subject).rng_tag
    ids = reads["ids"]
    covered = reads["covered"] if "covered" in reads.files else np.ones(len(ids), bool)
    keep = np.flatnonzero(covered)
    N = len(keep)
    turn_depths = TURN_DEPTHS
    lp_depths = [k for k in turn_depths if f"lp_t{k}" in reads.files]
    npz, meta = {}, {"depth": d, "layer": int(reads[f"layer_{d}"])}

    # Encoder residual (encoder_residual.py main, for depth d).
    Y = np.concatenate([reads[f"X_{d}_t{k}"][keep].astype(np.float32) for k in turn_depths], 0)
    X = np.concatenate([E[ids[keep]]] * len(turn_depths), 0)
    sent = np.concatenate([np.arange(N)] * len(turn_depths))
    tdep = np.concatenate([np.full(N, k) for k in turn_depths])
    lam, fve = encoder_residual.ridge_cv(X, Y, sent, encoder_residual.LAMBDAS)
    half = np.random.default_rng(0).permutation(N)
    mA, mB = np.isin(sent, half[: N // 2]), np.isin(sent, half[N // 2:])
    W, xm, xs, ym = encoder_residual.ridge_fit(X[mA], Y[mA], lam)
    R_B = (Y[mB] - ym).astype(np.float32) - ((X[mB] - xm) / xs).astype(np.float32) @ W
    pcs_r, eigvals_r = encoder_residual.top_pcs(R_B, encoder_residual.TOP_RESID)
    sub = reads["lp_sub_idx"]
    pos = {int(i): j for j, i in enumerate(keep)}
    sub_k = np.array([pos[int(i)] for i in sub if int(i) in pos])
    lp_ok = all(k in lp_depths for k in turn_depths) and len(sub_k) == len(sub)
    if lp_ok:
        rows_s = np.concatenate([np.flatnonzero(np.isin(sent, sub_k) & (tdep == k)) for k in turn_depths])
        lps = np.concatenate([reads[f"lp_t{k}"] for k in turn_depths]).astype(np.float64)
        sub_texts = [texts[int(ids[keep][i])] for i in sub_k] * len(turn_depths)
    else:
        rows_s = np.random.default_rng(0).permutation(len(Y))[: 2000 * len(turn_depths)]
        rows_s = rows_s[rows_s < len(Y)]
        lps = None
        sub_texts = [texts[int(ids[keep][sent[i]])] for i in rows_s]
    rand_dirs, rand_meta, screen = screen_random_directions(
        pcs_r[:encoder_residual.TOP_RESID], Y[rows_s], lps, sub_texts, f"{rng_tag}:{d}:encres_dirs", n_keep=4)
    dirs = np.stack([pcs_r[0] / (np.linalg.norm(pcs_r[0]) + 1e-8), pcs_r[1] / (np.linalg.norm(pcs_r[1]) + 1e-8)]
                    + rand_dirs)
    mu = Y.mean(0).astype(np.float32)
    for k, kind in enumerate(build_readouts.ENCRES):
        npz[f"vec_encres_{kind}_{d}"] = dirs[k].astype(np.float32)
    meta["encres"] = {"lambda": lam, "fve_cv": float(fve.max()), "random_directions": rand_meta, "screen": screen}
    print(f"[directions] {subject} {d}: encoder residual lambda {lam:g}, FVE {fve.max():.3f}, screen {screen['status']}",
          flush=True)
    del X

    # Principal-component bands (readouts.py stage_directions, for depth d).
    sub_reads = {f"X_{d}_t{k}": reads[f"X_{d}_t{k}"][keep] for k in turn_depths}
    mu64, pcs, eigvals = build_readouts.corpus_pcs(sub_reads, d)
    assert np.linalg.norm(mu64 - mu) / np.linalg.norm(mu) < 5e-2
    S = build_readouts.SCREEN_TURN_DEPTHS
    Ysub = np.concatenate([reads[f"X_{d}_t{k}"][keep][sub_k].astype(np.float32) for k in S], 0)
    lps_s = np.concatenate([reads[f"lp_t{k}"][:len(sub_k)] for k in S]).astype(np.float64)
    texts_s = [texts[int(ids[keep][i])] for i in sub_k] * len(S)
    rank = int((eigvals > 0).sum())
    meta["pcband"] = {}
    for lo, hi in build_readouts.BANDS:
        hi_eff = min(hi, rank, pcs.shape[0])
        vecs, cmeta, scr = screen_random_directions(pcs[lo:hi_eff], Ysub, lps_s, texts_s,
                                                    f"{rng_tag}:{d}:pcaband_{lo}_{hi_eff}", n_keep=2)
        for r, v in enumerate(vecs, start=1):
            npz[f"vec_pcband_{lo}_{hi}_{r}_{d}"] = v.astype(np.float32)
        meta["pcband"][f"{lo}_{hi}"] = {"band": [lo, hi_eff], "directions": cmeta, "screen": scr}
        print(f"[directions] {subject} {d} band [{lo},{hi_eff}): {scr['n_pass']} pass ({scr['status']})", flush=True)

    # Concept probes (concept_probes.py fit, readouts.py direction).
    meta["probe"] = {}
    for c in build_readouts.CONCEPTS:
        fit = concept_probes.fit_probe(conc[f"{c}_X_{d}"], conc[f"{c}_label"], conc[f"{c}_sent_idx"],
                                       n_jobs=min(8, os.cpu_count() or 8))
        v = fit["lr_weight"].astype(np.float64) / (fit["feature_std"].astype(np.float64) + 1e-12)
        npz[f"vec_probe_{c}_{d}"] = (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)
        meta["probe"][c] = {k: v for k, v in fit.items() if not isinstance(v, np.ndarray)
                            and k not in ("lr_bias", "lr_weight_norm")}
        print(f"[directions] {subject} {d} probe {c}: held-out accuracy {fit['held_out_accuracy']:.3f}", flush=True)
    npz[f"mu_{d}"] = mu
    npz[f"layer_{d}"] = reads[f"layer_{d}"]
    return npz, meta


def stage_directions(a):
    out_dir = model_dir(a.subject)
    reads = np.load(out_dir / "corpus_reads.npz")
    conc = np.load(out_dir / "concept_reads.npz")
    E = np.load(BUILD / "c4_corpus_20k_embeddings.npz")["E"].astype(np.float32)
    texts = {r["id"]: r["text"] for r in corpus_rows()}
    for d in a.depths:
        path = out_dir / f"directions_{d}.npz"
        if path.exists():
            print(f"[directions] {path} exists")
            continue
        t0 = time.time()
        npz, meta = directions_one_depth(a.subject, d, reads, conc, E, texts)
        save(path, npz)
        json.dump(meta, open(out_dir / f"directions_{d}.json", "w"), indent=1, default=float)
        print(f"[directions] wrote {path} in {time.time() - t0:.0f}s", flush=True)


def stage_calibrate(a):
    from neurofeedback.readouts import Readout, centered_cosine
    out_dir = model_dir(a.subject)
    base = np.load(REPO / "artifacts" / "readouts" / f"{a.subject}.npz")
    base_cal = np.load(REPO / "artifacts" / "readouts" / f"{a.subject}.exp2_calibration.npz")
    base_ids = [str(x) for x in base["readout_ids"]]
    assert [str(x) for x in base_cal["readout_ids"]] == base_ids
    table = new_readout_table(a.depths)
    dirs = {d: np.load(out_dir / f"directions_{d}.npz") for d in a.depths}
    ids = base_ids + [r[0] for r in table]
    art = {"readout_ids": np.array(ids),
           "families": np.array([str(x) for x in base["families"]] + [r[1] for r in table]),
           "depths": np.array([str(x) for x in base["depths"]] + [r[2] for r in table]),
           "num_layers": base["num_layers"],
           **{k: base[k] for k in base.files if k.startswith(("layer_", "mu_", "vec_"))}}
    for d in a.depths:
        if f"layer_{d}" in art:           # d55 is also one of the paper's anchors
            assert int(art[f"layer_{d}"]) == int(dirs[d][f"layer_{d}"]), d
        art[f"layer_{d}"] = dirs[d][f"layer_{d}"]
        art[f"mu_{d}"] = dirs[d][f"mu_{d}"]
    for rid, fam, d in table:
        art[f"vec_{rid}"] = dirs[d][f"vec_{rid}"]
    # Experiment 1 constants are not used (Experiment 2 only): the paper's for the 42, NaN for the rest.
    for k in ("lo", "hi", "median"):
        art[f"calib_{k}"] = np.concatenate([base[f"calib_{k}"], np.full(len(table), np.nan)])

    # Experiment 2 calibration of the new readouts (calibrate_exp2.py, reads at the new depths only).
    texts = [json.loads(line)["text"] for line in open(REPO / "data" / "c4_reference_256.jsonl")]
    if a.limit:
        texts = texts[:a.limit]
    items = [([{"role": "user", "content": prompts.exp2_instructions("hidden", corpus_objective(i), t, ids[0])}], t)
             for i, t in enumerate(texts)]
    backend = load_backend(a.subject, gpu_memory_utilization=a.gpu_memory_utilization)
    for d in a.depths:
        assert backend.layer_for_depth(NEW_DEPTHS[d]) == int(art[f"layer_{d}"])
    order = sorted(a.depths, key=lambda d: int(art[f"layer_{d}"]))
    lay = [int(art[f"layer_{d}"]) for d in order]
    readouts = [Readout(rid, fam, d, (d,), 0.0, 1.0, 0.0, mu=art[f"mu_{d}"].astype(np.float64),
                        vector=art[f"vec_{rid}"].astype(np.float64)) for rid, fam, d in table]
    raws = np.full((len(table), len(texts)), np.nan)
    for i0 in range(0, len(items), 96):
        reads = backend.read_residual_multi_batch(items[i0:i0 + 96], lay)
        for k, r in enumerate(reads):
            H = np.asarray(r.numpy()).astype(np.float64)
            spans = {d: H[li] for li, d in enumerate(order)}
            raws[:, i0 + k] = [centered_cosine(spans[ro.depth], ro.mu, ro.vector) for ro in readouts]
        print(f"[calibrate] {min(i0 + 96, len(items))}/{len(items)}", flush=True)
    lo, hi, med = (np.full(len(table), np.nan) for _ in range(3))
    n_valid = np.zeros(len(table), int)
    for i in range(len(table)):
        v = raws[i][np.isfinite(raws[i])]
        n_valid[i] = len(v)
        if len(v):
            lo[i], hi[i] = float(np.percentile(v, 1)), float(np.percentile(v, 99))
            if hi[i] - lo[i] < 1e-6:
                hi[i] = lo[i] + 1.0
            med[i] = float(np.percentile(v, 50))
    cal = {"readout_ids": np.array(ids),
           "calib_lo": np.concatenate([base_cal["calib_lo"], lo]),
           "calib_hi": np.concatenate([base_cal["calib_hi"], hi]),
           "calib_median": np.concatenate([base_cal["calib_median"], med]),
           "calib_n_valid": np.concatenate([base_cal["calib_n_valid"], n_valid]),
           "new_calib_raws": raws}
    dest = Path(a.artifacts)
    dest.mkdir(parents=True, exist_ok=True)
    save(dest / f"{a.subject}.npz", art)
    save(dest / f"{a.subject}.exp2_calibration.npz", cal)
    print(f"[calibrate] wrote {dest / (a.subject + '.npz')} ({len(ids)} readouts) and its Exp 2 calibration", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["embed", "reads", "aux", "merge", "directions", "calibrate"])
    ap.add_argument("--subject")
    ap.add_argument("--depths", nargs="*", default=list(NEW_DEPTHS))
    ap.add_argument("--rows-shard", type=int, default=0)
    ap.add_argument("--rows-shards", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="testing: only the first n corpus rows / reference sentences")
    ap.add_argument("--probe-limit", type=int, default=0, help="testing: only n probe sentences per concept")
    ap.add_argument("--allow-partial", action="store_true", help="testing: merge with missing rows")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--artifacts", default=str(ARTIFACTS))
    a = ap.parse_args()
    {"embed": stage_embed, "reads": stage_reads, "aux": stage_aux, "merge": stage_merge,
     "directions": stage_directions, "calibrate": stage_calibrate}[a.stage](a)


if __name__ == "__main__":
    main()
