#!/usr/bin/env python
"""Assemble the 42 readouts of a subject (CPU), in two stages around the GPU step
scripts/build/calibration_reads.py.

  --stage directions   the 30 directions and the corpus means -> build/<model>/directions.npz
      probe     concept-probe weights divided by the feature std, unit norm (concept_probes.py)
      pcband    PCA of the corpus reads (20k sentences x turn depths 0/5/20); in each band of
                ranks [20,200), [200,1000), [1000,3000), 2 random directions screened by
                common.screen_random_directions on the 2000 log-probability sentences at
                turn depths 0 and 5
      encres    the 6 encoder-residual directions (encoder_residual.py)

  --stage finalize     -> artifacts/readouts/<model>.npz
      calibration: p1, p99 and median of each readout over the 256 reference sentences
      (calibration_reads.py); diagnostics of how predictable each readout is from the text,
      on the first 2000 corpus sentences (pub_*), used to select the opening sentences.

    python scripts/build/readouts.py --stage directions --subject qwen3.5-9b
    python scripts/build/readouts.py --stage finalize --subject qwen3.5-9b
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BUILD, REPO, cv_r2, model_dir, screen_random_directions, word_statistics  # noqa: E402
from neurofeedback.models import get_model  # noqa: E402

DEPTHS = ["d33", "d66"]
CONCEPTS = ["sentiment", "morality", "truth"]
BANDS = [(20, 200), (200, 1000), (1000, 3000)]
ENCRES = ["pc1", "pc2", "rand1", "rand2", "rand3", "rand4"]
GEOMETRY = ["curvature_step1", "curvature_step2", "curvature_step3", "step_change"]
DIFFS = ["curvature_diff_44_33", "curvature_diff_55_44", "curvature_diff_66_55", "curvature_diff_66_33"]
SCREEN_TURN_DEPTHS = [0, 5]
N_PUBLIC = 2000


def readout_table():
    """(id, family, depth) of the 42 readouts, in artifact order."""
    rows = []
    for d in DEPTHS:
        rows += [(f"probe_{c}_{d}", "probe", d) for c in CONCEPTS]
        rows += [(f"pcband_{lo}_{hi}_{r}_{d}", "pcband", d) for lo, hi in BANDS for r in (1, 2)]
        rows += [(f"encres_{k}_{d}", "encres", d) for k in ENCRES]
    rows += [(f"{g}_{d}", "geometry", d) for d in DEPTHS for g in GEOMETRY]
    rows += [(g, "geometry", "cross_depth") for g in DIFFS]
    return rows


def corpus_pcs(reads, d, turn_depths=(0, 5, 20)):
    """Mean and principal components (rows, decreasing eigenvalue, sign: largest coordinate
    positive) of the corpus reads at depth d."""
    hid = reads[f"X_{d}_t0"].shape[1]
    n, S, G = 0, np.zeros(hid), np.zeros((hid, hid))
    for k in turn_depths:
        X = reads[f"X_{d}_t{k}"].astype(np.float32)
        S += X.sum(0, dtype=np.float64)
        G += (X.T @ X).astype(np.float64)
        n += X.shape[0]
    mu = S / n
    cov = (G - n * np.outer(mu, mu)) / (n - 1)
    w, V = np.linalg.eigh((cov + cov.T) / 2.0)
    order = np.argsort(w)[::-1]
    pcs = V[:, order].T
    flip = pcs[np.arange(pcs.shape[0]), np.abs(pcs).argmax(1)] < 0
    pcs[flip] *= -1.0
    return mu, pcs.astype(np.float32), np.clip(w[order], 0.0, None)


def stage_directions(model):
    out_dir = model_dir(model)
    reads = np.load(out_dir / "corpus_reads.npz")
    enc = np.load(out_dir / "encoder_residual.npz")
    probes = np.load(out_dir / "concept_probes.npz")
    texts = {r["id"]: r["text"] for r in map(json.loads, open(REPO / "data" / "c4_corpus_20k.jsonl"))}
    sub = reads["lp_sub_idx"]
    npz, meta = {}, {}
    for d in DEPTHS:
        mu = enc[f"{d}_y_mean"].astype(np.float32)
        mu64, pcs, eigvals = corpus_pcs(reads, d)
        assert np.linalg.norm(mu64 - mu) / np.linalg.norm(mu) < 5e-2
        Ysub = np.concatenate([reads[f"X_{d}_t{k}"][sub].astype(np.float32) for k in SCREEN_TURN_DEPTHS], 0)
        lps = np.concatenate([reads[f"lp_t{k}"] for k in SCREEN_TURN_DEPTHS]).astype(np.float64)
        sub_texts = [texts[int(i)] for i in sub] * len(SCREEN_TURN_DEPTHS)
        for c in CONCEPTS:
            v = probes[f"{c}_{d}_lr_weight"].astype(np.float64) / (probes[f"{c}_{d}_feature_std"].astype(np.float64) + 1e-12)
            npz[f"vec_probe_{c}_{d}"] = (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)
        rank = int((eigvals > 0).sum())
        for lo, hi in BANDS:
            hi_eff = min(hi, rank, pcs.shape[0])        # 2880 for GPT-OSS-20B
            vecs, cmeta, screen = screen_random_directions(
                pcs[lo:hi_eff], Ysub, lps, sub_texts, f"{get_model(model).rng_tag}:{d}:pcaband_{lo}_{hi_eff}", n_keep=2)
            for r, v in enumerate(vecs, start=1):
                npz[f"vec_pcband_{lo}_{hi}_{r}_{d}"] = v.astype(np.float32)
            meta[f"pcband_{lo}_{hi}_{d}"] = {"band": [lo, hi_eff], "directions": cmeta, "screen": screen}
            print(f"[directions] {model} {d} band [{lo},{hi_eff}): {screen['n_pass']} pass ({screen['status']})")
        for k, kind in enumerate(ENCRES):
            npz[f"vec_encres_{kind}_{d}"] = enc[f"{d}_directions"][k].astype(np.float32)
        npz[f"mu_{d}"], npz[f"layer_{d}"] = mu, reads[f"layer_{d}"]
    np.savez(out_dir / "directions.npz", **npz)
    json.dump(meta, open(out_dir / "directions.json", "w"), indent=1)
    print(f"[directions] wrote {out_dir / 'directions.npz'}")


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def pca(X, k):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:max(1, min(k, Vt.shape[0]))].T


def stage_finalize(model, out_path=None):
    out_dir = model_dir(model)
    rows = readout_table()
    ids = [r[0] for r in rows]
    dirs = np.load(out_dir / "directions.npz")
    cal = np.load(out_dir / "calibration_reads.npz")
    assert [str(x) for x in cal["readout_ids"]] == ids
    raws = cal["raws"].astype(np.float64)                       # [42, 256]
    lo, hi, med = (np.full(42, np.nan) for _ in range(3))
    n_valid = np.zeros(42, int)
    for i in range(42):
        v = raws[i][np.isfinite(raws[i])]
        n_valid[i] = len(v)
        if len(v):
            lo[i], hi[i] = float(np.percentile(v, 1)), float(np.percentile(v, 99))
            if hi[i] - lo[i] < 1e-6:
                hi[i] = lo[i] + 1.0
            med[i] = float(np.percentile(v, 50))

    # How predictable each readout is from the text, on the first 2000 corpus sentences.
    reads = np.load(out_dir / "corpus_reads.npz")
    texts = {r["id"]: r["text"] for r in map(json.loads, open(REPO / "data" / "c4_corpus_20k.jsonl"))}
    texts = [texts[i] for i in range(N_PUBLIC)]
    emb = np.load(BUILD / "c4_corpus_20k_embeddings.npz")
    Epca = pca(emb["E"][:N_PUBLIC].astype(np.float64), 256)
    sv = np.load(out_dir / "structural_values.npz")             # geometry raws, logprob, n_tokens
    lp, ntok = sv["logprob_mean"].astype(np.float64), sv["n_tokens"].astype(np.float64)
    lex = word_statistics(texts, lp)
    pub = {k: np.full(42, np.nan) for k in ("abs_r_lp", "abs_r_ntok", "r2_encoder", "r2_lexical", "mean",
                                            "std", "p1", "p99", "spread_p99_p1")}
    pub_n = np.zeros(42, int)
    values = []
    for i, (rid, fam, d) in enumerate(rows):
        if fam == "geometry":
            y = sv[rid].astype(np.float64)
        else:
            X = reads[f"X_{d}_t0"][:N_PUBLIC].astype(np.float32)
            C = X - dirs[f"mu_{d}"].astype(np.float32)
            v = dirs[f"vec_{rid}"]
            y = ((C @ v) / (np.linalg.norm(C, axis=1) * np.linalg.norm(v) + 1e-8)).astype(np.float64)
        values.append(y)
        ok = np.isfinite(y)
        pub_n[i] = int(ok.sum())
        if pub_n[i] < 10:
            continue
        yv = y[ok]
        pub["mean"][i], pub["std"][i] = float(yv.mean()), float(yv.std())
        pub["p1"][i], pub["p99"][i] = np.percentile(yv, 1), np.percentile(yv, 99)
        pub["spread_p99_p1"][i] = pub["p99"][i] - pub["p1"][i]
        pub["abs_r_lp"][i] = abs(pearson(y[ok & np.isfinite(lp)], lp[ok & np.isfinite(lp)]))
        pub["abs_r_ntok"][i] = abs(pearson(y[ok], ntok[ok]))
        pub["r2_encoder"][i] = float(cv_r2(Epca[ok], yv[:, None])[0])
        pub["r2_lexical"][i] = float(cv_r2(lex[ok], yv[:, None])[0])

    out = {"readout_ids": np.array(ids), "families": np.array([r[1] for r in rows]),
           "depths": np.array([r[2] for r in rows]),
           "calib_lo": lo, "calib_hi": hi, "calib_median": med, "calib_n_valid": n_valid, "calib_raws": raws,
           "pub_n": pub_n, **{f"pub_{k}": v for k, v in pub.items()}, "pub_values": np.stack(values),
           "mu_d33": dirs["mu_d33"], "mu_d66": dirs["mu_d66"],
           **{f"vec_{rid}": dirs[f"vec_{rid}"] for rid, fam, _ in rows if fam != "geometry"},
           **{k: cal[k] for k in cal.files if k.startswith("layer_")}, "num_layers": cal["num_layers"]}
    path = Path(out_path or REPO / "artifacts" / "readouts" / f"{model}.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    print(f"[finalize] wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["directions", "finalize"])
    ap.add_argument("--subject", required=True)
    ap.add_argument("--out", help="finalize: output file (default artifacts/readouts/<model>.npz)")
    a = ap.parse_args()
    if a.stage == "directions":
        stage_directions(a.subject)
    else:
        stage_finalize(a.subject, a.out)


if __name__ == "__main__":
    main()
