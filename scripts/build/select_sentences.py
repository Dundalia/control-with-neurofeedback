#!/usr/bin/env python
"""Select the opening sentences of Experiment 1 and the fixed sentences of Experiment 2 (CPU),
from the readouts of the five subjects of Experiment 1 (artifacts/readouts/*.npz, pub_values:
every readout on the first 2000 corpus sentences) and the corpus embeddings.

Candidates: the first 2000 corpus sentences with 10 to 20 words and every readout defined.
Score: the mean over the 210 (subject, readout) pairs of |percentile - 50|, the percentile
being taken among the candidates (lower = closer to the median).
  Experiment 1: k-means (k = 20) on the embeddings of the 200 best candidates; the best
                candidate of each cluster -> data/exp1_opening_sentences.jsonl
  Experiment 2: the 20 sentences of Experiment 1, plus the best of each of 30 k-means clusters
                of the 300 best remaining candidates; of these 30, the best 20 are kept
                -> data/exp2_fixed_sentences.jsonl

    python scripts/build/select_sentences.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BUILD, REPO  # noqa: E402
from neurofeedback.models import EXP1_MODELS  # noqa: E402

BAND, N_PUBLIC = (10, 20), 2000


def kmeans_best(candidates, score, emb, k):
    from sklearn.cluster import KMeans
    E = emb[candidates].astype(np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit(E).labels_
    return [min((candidates[j] for j in range(len(candidates)) if labels[j] == c), key=lambda i: score[i])
            for c in range(k) if (labels == c).any()]


def main():
    texts = [json.loads(line)["text"] for line in open(REPO / "data" / "c4_corpus_20k.jsonl")]
    wc = np.array([len(texts[i].split()) for i in range(N_PUBLIC)])
    V = np.vstack([np.load(REPO / "artifacts" / "readouts" / f"{m}.npz")["pub_values"] for m in EXP1_MODELS])
    idx = np.where((wc >= BAND[0]) & (wc <= BAND[1]) & ~np.isnan(V).any(0))[0]
    pct = 100.0 * V[:, idx].argsort(1).argsort(1) / (len(idx) - 1)
    dev = np.abs(pct - 50.0)
    score = {int(i): float(dev.mean(0)[k]) for k, i in enumerate(idx)}
    n_extreme = {int(i): int((dev[:, k] > 40).sum()) for k, i in enumerate(idx)}
    emb = np.load(BUILD / "c4_corpus_20k_embeddings.npz")["E"]

    def row(rank, i):
        return {"rank": rank, "corpus_id": i, "n_words": int(wc[i]), "mean_pct_dev": round(score[i], 2),
                "n_extreme_readouts": n_extreme[i]}

    top = [int(i) for i in idx[np.argsort(dev.mean(0))[:200]]]
    exp1 = sorted(kmeans_best(top, score, emb, 20), key=lambda i: score[i])
    with open(REPO / "data" / "exp1_opening_sentences.jsonl", "w") as f:
        for rank, i in enumerate(exp1):
            f.write(json.dumps({**row(rank, i), "text": texts[i]}, ensure_ascii=False) + "\n")

    remaining = [int(i) for i in idx if int(i) not in set(exp1)]
    add = kmeans_best(sorted(remaining, key=lambda i: score[i])[:300], score, emb, 30)
    for i in sorted(remaining, key=lambda i: score[i]):          # if a cluster was empty
        if len(add) >= 30:
            break
        if i not in add:
            add.append(i)
    new = sorted(add, key=lambda i: (round(score[i], 2), i))[:20]
    exp2 = sorted(exp1 + new, key=lambda i: (round(score[i], 2), i))
    with open(REPO / "data" / "exp2_fixed_sentences.jsonl", "w") as f:
        for rank, i in enumerate(exp2):
            f.write(json.dumps({**row(rank, i), "in_exp1": i in set(exp1), "text": texts[i]},
                               ensure_ascii=False) + "\n")
    print(f"[select] {len(idx)} candidates; wrote 20 opening and 40 fixed sentences")


if __name__ == "__main__":
    main()
