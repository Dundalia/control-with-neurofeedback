"""Shared pieces of the readout build (scripts/build/*.py).

Every read of the build teacher-forces a sentence as the assistant reply after the build
instructions (``neurofeedback.prompts.build_instructions``), with k earlier turns in which the
same sentence received random scores. The scores of a sentence are drawn once from a seed
derived from a namespace and the sentence index, and the depth-k conversation uses the first k.
"""
import os
import sys
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from neurofeedback import prompts  # noqa: E402

BUILD = Path(os.environ.get("BUILD_DIR", REPO / "build"))   # intermediates (large, not released)
DEPTHS = [("d33", 0.33), ("d66", 0.66)]
TURN_DEPTHS = [0, 5, 20]
MAX_FEEDBACK_TURNS = 20
# Seed namespace of the corpus reads; it must not change, or the rebuilt directions differ.
CORPUS_NAMESPACE = "core_c4_v2_20k"


def model_dir(model: str) -> Path:
    d = BUILD / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def feedback_scores(namespace: str, idx: int) -> np.ndarray:
    seed = zlib.crc32(f"{namespace}:{idx}".encode())
    return np.random.default_rng(seed).integers(0, 101, size=MAX_FEEDBACK_TURNS)


def build_conversation(objective: str, sentence: str, scores, k: int) -> list[dict]:
    msgs = [{"role": "user", "content": prompts.build_instructions(objective)}]
    for i in range(k):
        msgs.append({"role": "assistant", "content": sentence})
        msgs.append({"role": "user", "content": prompts.feedback(int(scores[i]))})
    return msgs


def corpus_objective(idx: int) -> str:
    """Corpus sentences alternate objectives by index parity."""
    return "max" if idx % 2 == 0 else "min"


def load_backend(model: str, max_model_len: int = 8192, gpu_memory_utilization: float = 0.9):
    from neurofeedback.backend import BackendConfig, VLLMBackend
    from neurofeedback.models import get_model
    return VLLMBackend(BackendConfig(get_model(model).hf_id, max_model_len=max_model_len,
                                     gpu_memory_utilization=gpu_memory_utilization))


def mean_reads(backend, items, layer, chunk, tag=""):
    """Mean over the sentence tokens of the residual stream at one layer: [n, hidden] float32."""
    X = None
    for i in range(0, len(items), chunk):
        reads = backend.read_residual_multi_batch(items[i:i + chunk], [layer])
        for j, r in enumerate(reads):
            v = r[0].float().mean(0).cpu().numpy().astype(np.float32)
            if X is None:
                X = np.empty((len(items), v.shape[0]), np.float32)
            X[i + j] = v
        print(f"[reads] {tag} {min(i + chunk, len(items))}/{len(items)}", flush=True)
    return X


# ---- screening of random directions (principal-component bands, encoder residual) ----------

N_CANDIDATES = 100
MAX_LOGPROB_CORR = 0.05
MAX_R2_WORD_STATS = 0.10


def cv_r2(X, Y, alpha=1.0, folds=5, seed=0):
    """Held-out R^2 of a ridge regression of each column of Y on X (5 random folds)."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    N, p = X.shape
    folds = max(2, min(folds, N))
    chunks = np.array_split(np.random.default_rng(seed).permutation(N), folds)
    Yhat = np.empty_like(Y)
    for f in range(folds):
        te = chunks[f]
        tr = np.concatenate([chunks[g] for g in range(folds) if g != f])
        xm, ym = X[tr].mean(0), Y[tr].mean(0)
        Xtr = X[tr] - xm
        W = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(p), Xtr.T @ (Y[tr] - ym))
        Yhat[te] = (X[te] - xm) @ W + ym
    ss_res = ((Y - Yhat) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1.0 - ss_res / ss_tot, -1.0, 1.0)


def word_statistics(sentences, logprobs):
    """z-scored predictors of a readout from the text: characters, words, mean word length,
    type-token ratio, punctuation and uppercase fractions, and the subject's mean token
    log-probability of the sentence."""
    rows = []
    for s, lp in zip(sentences, logprobs):
        w = s.split()
        nw, nc = max(1, len(w)), max(1, len(s))
        rows.append([len(s), nw, (np.mean([len(t) for t in w]) if w else 0.0), len(set(w)) / nw,
                     sum(c in ".,;:!?-—\"'()" for c in s) / nc, sum(c.isupper() for c in s) / nc, float(lp)])
    F = np.asarray(rows, float)
    return (F - F.mean(0)) / (F.std(0) + 1e-8)


def screen_random_directions(span, Ysub, lps, texts, rng_seed, n_keep):
    """Draw 100 random unit directions in the row span of ``span`` and keep ``n_keep``.

    Each candidate scores the sentences of the screen by the centered cosine of their mean
    activations (``Ysub``, centered on their own mean). A candidate passes if its spread
    (p99 - p1) is at least the median spread, its correlation with the subject's
    log-probability is at most 0.05 in absolute value (skipped if ``lps`` is None), and its
    held-out R^2 from word statistics is at most 0.10; the passing candidates with the lowest
    R^2 are kept. If fewer than ``n_keep`` pass, the candidates with the median spread or more
    are ranked by |corr| + max(0, R^2) instead."""
    import zlib
    mu = Ysub.mean(0)
    Yc = (Ysub - mu).astype(np.float32)
    Ycn = Yc / (np.linalg.norm(Yc, axis=1, keepdims=True) + 1e-8)
    feats = word_statistics(texts, lps if lps is not None else np.zeros(len(texts)))
    if lps is None:
        feats = feats[:, :-1]

    def stats(v):
        raws = Ycn @ (v / (np.linalg.norm(v) + 1e-8))
        spread = float(np.percentile(raws, 99) - np.percentile(raws, 1))
        if lps is None:
            corr = None
        else:
            corr = float(np.corrcoef(raws, lps)[0, 1]) if raws.std() > 0 and lps.std() > 0 else 0.0
        r2 = max(float(x) for x in cv_r2(feats, raws[:, None])) if raws.std() > 1e-9 else 1.0
        return spread, corr, r2

    rng = np.random.default_rng(zlib.crc32(rng_seed.encode()))
    cands = []
    for j in range(N_CANDIDATES):
        v = rng.standard_normal(span.shape[0]).astype(np.float32) @ span
        v /= np.linalg.norm(v) + 1e-8
        spread, corr, r2 = stats(v)
        cands.append({"candidate": j, "v": v, "spread": spread, "logprob_corr": corr, "r2_word_stats": r2})
    floor = float(np.percentile([c["spread"] for c in cands], 50.0))
    ok = [c for c in cands if c["spread"] >= floor and c["r2_word_stats"] <= MAX_R2_WORD_STATS
          and (lps is None or abs(c["logprob_corr"]) <= MAX_LOGPROB_CORR)]
    status = "pass" if lps is not None else "no_logprob_criterion"
    n_pass = len(ok)
    if len(ok) < n_keep:
        alive = [c for c in cands if c["spread"] >= floor] or cands
        ok = sorted(alive, key=lambda c: abs(c["logprob_corr"] or 0) + max(0.0, c["r2_word_stats"]))[:n_keep]
        status = "relaxed" if lps is not None else status
    chosen = sorted(ok, key=lambda c: c["r2_word_stats"])[:n_keep]
    meta = [{k: c[k] for k in ("candidate", "spread", "logprob_corr", "r2_word_stats")} for c in chosen]
    return [c["v"] for c in chosen], meta, {"n_pass": n_pass, "spread_floor": floor, "status": status}
