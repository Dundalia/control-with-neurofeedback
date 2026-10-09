#!/usr/bin/env python
"""Encoder-residual directions and the corpus mean (CPU).

Per subject and depth: ridge regression of the subject's mean activation on a sentence (all
turn depths as rows) on the sentence's encoder embedding (z-scored), with one regularization
chosen by 5-fold cross-validation grouped by sentence. The regression is fit on a random half
of the sentences, and the residuals of the other half are decomposed by PCA. The directions
are the top two residual PCs and four random directions in the span of the top 100, screened
like the principal-component bands (common.screen_random_directions). The corpus mean of the
activations (mu of the centered cosine) is also stored here.
-> build/<model>/encoder_residual.{npz,json}

    python scripts/build/encoder_residual.py --subject qwen3.5-9b
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BUILD, REPO, model_dir, screen_random_directions  # noqa: E402
from neurofeedback.models import get_model  # noqa: E402

LAMBDAS = np.logspace(0, 6, 13)
N_FOLDS = 5
TOP_RESID = 100


def ridge_cv(X, Y, sent_ids, lambdas, seed=0):
    """Held-out fraction of variance explained per regularization, folds grouped by sentence.
    All regularizations share one eigendecomposition of the training Gram matrix."""
    perm = np.random.default_rng(seed).permutation(np.unique(sent_ids))
    folds = [np.isin(sent_ids, c) for c in np.array_split(perm, N_FOLDS)]
    ss_res, ss_tot = np.zeros(len(lambdas)), 0.0
    for fi, te in enumerate(folds):
        tr = ~te
        xm, xs, ym = X[tr].mean(0), X[tr].std(0) + 1e-8, Y[tr].mean(0)
        Xtr = ((X[tr] - xm) / xs).astype(np.float32)
        Xte = ((X[te] - xm) / xs).astype(np.float32)
        Ytr = (Y[tr] - ym).astype(np.float32)
        Yte = (Y[te] - ym).astype(np.float32)
        s, V = np.linalg.eigh((Xtr.T @ Xtr).astype(np.float64))
        s = np.clip(s, 0.0, None)
        A = V.T @ (Xtr.T @ Ytr).astype(np.float64)
        P = Xte @ V.astype(np.float32)
        q = np.einsum("ij,ij->i", A, (P.T @ Yte).astype(np.float64))
        B = (P.T @ P).astype(np.float64) * (A @ A.T)
        yte_ss = float((Yte.astype(np.float64) ** 2).sum())
        ss_tot += yte_ss
        for li, lam in enumerate(lambdas):
            d = 1.0 / (s + lam)
            ss_res[li] += yte_ss - 2.0 * float(q @ d) + float(d @ B @ d)
        print(f"[cv] fold {fi + 1}/{N_FOLDS}", flush=True)
    fve = 1.0 - ss_res / ss_tot
    return float(lambdas[int(np.argmax(fve))]), fve


def ridge_fit(X, Y, lam):
    xm, xs, ym = X.mean(0), X.std(0) + 1e-8, Y.mean(0)
    Xz = ((X - xm) / xs).astype(np.float32)
    G = (Xz.T @ Xz).astype(np.float64)
    G[np.diag_indices_from(G)] += lam
    W = np.linalg.solve(G, (Xz.T @ (Y - ym).astype(np.float32)).astype(np.float64)).astype(np.float32)
    return W, xm.astype(np.float32), xs.astype(np.float32), ym.astype(np.float32)


def top_pcs(R, k):
    """Top-k principal components (rows) and eigenvalues of the centered rows of R."""
    Rc = (R - R.mean(0)).astype(np.float32)
    G = (Rc.T @ Rc).astype(np.float64) / max(1, len(Rc) - 1)
    s, V = np.linalg.eigh(G)
    order = np.argsort(s)[::-1]
    k = min(k, len(s), np.linalg.matrix_rank(G, hermitian=True))
    return V[:, order[:k]].T.astype(np.float32), np.clip(s[order[:k]], 0, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    a = ap.parse_args()
    reads = np.load(model_dir(a.subject) / "corpus_reads.npz")
    ids = reads["ids"]
    N = len(ids)
    turn_depths = sorted(int(k.split("_t")[1]) for k in reads.files if k.startswith("X_d33_t"))
    E = np.load(BUILD / "c4_corpus_20k_embeddings.npz")["E"].astype(np.float32)[ids]
    texts = {r["id"]: r["text"] for r in map(json.loads, open(REPO / "data" / "c4_corpus_20k.jsonl"))}
    texts = [texts[int(i)] for i in ids]
    # The log-probability criterion of the screen needs the log-probabilities at every turn depth.
    lp_ok = all(f"lp_t{k}" in reads.files for k in turn_depths)
    npz, meta = {}, {}
    for d in ("d33", "d66"):
        Y = np.concatenate([reads[f"X_{d}_t{k}"].astype(np.float32) for k in turn_depths], 0)
        X = np.concatenate([E] * len(turn_depths), 0)
        sent = np.concatenate([np.arange(N)] * len(turn_depths))
        tdep = np.concatenate([np.full(N, k) for k in turn_depths])
        lam, fve = ridge_cv(X, Y, sent, LAMBDAS)
        half = np.random.default_rng(0).permutation(N)
        mA, mB = np.isin(sent, half[: N // 2]), np.isin(sent, half[N // 2:])
        W, xm, xs, ym = ridge_fit(X[mA], Y[mA], lam)
        R_B = (Y[mB] - ym).astype(np.float32) - ((X[mB] - xm) / xs).astype(np.float32) @ W
        pcs, eigvals = top_pcs(R_B, TOP_RESID)
        if lp_ok:
            sub = reads["lp_sub_idx"]
            rows = np.concatenate([np.flatnonzero(np.isin(sent, sub) & (tdep == k)) for k in turn_depths])
            lps = np.concatenate([reads[f"lp_t{k}"] for k in turn_depths]).astype(np.float64)
            sub_texts = [texts[int(i)] for i in sub] * len(turn_depths)
        else:
            rows = np.random.default_rng(0).permutation(len(Y))[: 2000 * len(turn_depths)]
            lps = None
            sub_texts = [texts[int(sent[i])] for i in rows]
        rand_dirs, rand_meta, screen = screen_random_directions(
            pcs[:TOP_RESID], Y[rows], lps, sub_texts, f"{get_model(a.subject).rng_tag}:{d}:encres_dirs", n_keep=4)
        dirs = np.stack([pcs[0] / (np.linalg.norm(pcs[0]) + 1e-8), pcs[1] / (np.linalg.norm(pcs[1]) + 1e-8)]
                        + rand_dirs)
        npz.update({f"{d}_y_mean": Y.mean(0).astype(np.float32), f"{d}_directions": dirs.astype(np.float32),
                    f"{d}_resid_eigvals": eigvals.astype(np.float32), f"{d}_layer": reads[f"layer_{d}"]})
        meta[d] = {"lambda": lam, "fve_cv": float(fve.max()), "random_directions": rand_meta, "screen": screen}
        print(f"[encoder] {a.subject} {d}: lambda {lam:g}, held-out FVE {fve.max():.3f}, screen {screen['status']}")
    out = model_dir(a.subject) / "encoder_residual"
    np.savez(str(out) + ".npz", **npz)
    json.dump({"model": a.subject, "turn_depths": turn_depths, "depths": meta}, open(str(out) + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
