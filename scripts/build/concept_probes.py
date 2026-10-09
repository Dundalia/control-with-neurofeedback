#!/usr/bin/env python
"""Concept probes: logistic-regression probes for sentiment (SST-2), commonsense morality
(ETHICS) and truthfulness (Azaria & Mitchell), per subject and depth (33%, 66%).

  --prepare            CPU: build the 600 balanced sentences of each concept
                       -> data/concept_probe_sentences.json (shared by all subjects)
  --subject <model>    GPU: read every sentence in the build context at turn depths 0, 5, 20,
                       mean over the sentence tokens (checkpointed in build/<model>/), then fit
                       -> build/<model>/concept_probes.{npz,json}

Fit: z-score with the training statistics, L2 logistic regression, regularization chosen by
5-fold cross-validation grouped by sentence over 2^-20..2^20 (every second power), on an 80/20
split grouped by sentence and stratified by label. The probe direction of the readouts is
lr_weight / feature_std (scripts/build/readouts.py).
"""
import argparse
import json
import os
import re
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEPTHS, REPO, TURN_DEPTHS, build_conversation, feedback_scores, load_backend, mean_reads, model_dir  # noqa: E402

CONCEPTS = ["sentiment", "morality", "truth"]
N_PER_LABEL = 300
LAMBDA_EXPS = list(range(-20, 21, 2))
SENTENCES_FILE = REPO / "data" / "concept_probe_sentences.json"
MIN_WORDS, MAX_WORDS = 5, 30
TRUTH_TOPICS = ["cities", "companies", "animals", "elements", "inventions", "facts"]
_FRAG_START = {"and", "or", "but", "of", "in", "on", "with", "as", "at", "by", "for", "from", "to",
               "than", "nor", "also"}


# ---- sentences ------------------------------------------------------------------------

def complete_sentence(s: str) -> bool:
    return MIN_WORDS <= len(s.split()) <= MAX_WORDS and bool(s) and s[0].isalpha() and s[-1] in ".!?"


def _detok_sst(s: str) -> str:
    """Undo SST-2's tokenized spacing."""
    s = s.strip().replace("``", '"').replace("''", '"')
    s = re.sub(r"\s+([.,;:!?%)\]])", r"\1", s)
    s = re.sub(r"([(\[])\s+", r"\1", s)
    s = re.sub(r"\s+('s|'re|'ve|'ll|'d|'m|n't)\b", r"\1", s)
    s = re.sub(r"(\w)\s+'(\s)", r"\1'\2", s)
    return re.sub(r"\s+", " ", s).strip()


def _balanced(pool, rng, n):
    out = []
    for lab in (0, 1):
        pick = rng.permutation(len(pool[lab]))[:n]
        out += [{"text": pool[lab][i], "label": lab} for i in sorted(pick)]
    return out


def build_sentiment(rng):
    """SST-2 train, label 1 = positive. SST-2 is lowercased: a sentence must start with a
    letter and not with a fragment word; its first letter is capitalized."""
    from datasets import load_dataset
    pool, seen = {0: [], 1: []}, set()
    for r in load_dataset("stanfordnlp/sst2", split="train"):
        s = _detok_sst(r["sentence"])
        if not s or not complete_sentence(s) or s.split()[0].lower() in _FRAG_START:
            continue
        s = s[0].upper() + s[1:]
        if s.lower() not in seen:
            seen.add(s.lower())
            pool[int(r["label"])].append(s)
    return _balanced(pool, rng, N_PER_LABEL), {"dataset": "stanfordnlp/sst2 (train)",
                                               "label_meaning": {"0": "negative", "1": "positive"}}


def build_morality(rng):
    """ETHICS commonsense train, label 1 = morally acceptable (1 - the dataset label)."""
    from datasets import load_dataset
    ds = load_dataset("parquet", split="train", data_files=(
        "hf://datasets/hendrycks/ethics@refs/convert/parquet/commonsense/train/0000.parquet"))
    pool, seen = {0: [], 1: []}, set()
    for r in ds:
        s = re.sub(r"\s+", " ", r["input"].strip())
        if complete_sentence(s) and s.lower() not in seen:
            seen.add(s.lower())
            pool[1 - int(r["label"])].append(s)
    return _balanced(pool, rng, N_PER_LABEL), {"dataset": "hendrycks/ethics commonsense (train)",
                                               "label_meaning": {"0": "morally unacceptable", "1": "morally acceptable"}}


def build_truth(rng):
    """Azaria & Mitchell true/false statements, 50 true and 50 false from each of six topics,
    label 1 = true."""
    from datasets import load_dataset
    out, seen = [], set()
    for topic in TRUTH_TOPICS:
        pool = {0: [], 1: []}
        for r in load_dataset("pminervini/true-false", split=topic):
            s = re.sub(r"\s+", " ", r["statement"].strip())
            if complete_sentence(s) and s.lower() not in seen:
                seen.add(s.lower())
                pool[int(r["label"])].append(s)
        for lab in (0, 1):
            pick = rng.permutation(len(pool[lab]))[:N_PER_LABEL // len(TRUTH_TOPICS)]
            out += [{"text": pool[lab][i], "label": lab, "topic": topic} for i in sorted(pick)]
    return out, {"dataset": "pminervini/true-false (Azaria & Mitchell)", "topics": TRUTH_TOPICS,
                 "label_meaning": {"0": "false", "1": "true"}}


def prepare():
    payload = {"seed": 0, "concepts": {}}
    for concept, build in (("sentiment", build_sentiment), ("morality", build_morality), ("truth", build_truth)):
        sents, meta = build(np.random.default_rng(0))
        assert len(sents) == 2 * N_PER_LABEL and sum(s["label"] for s in sents) == N_PER_LABEL
        rank = {0: 0, 1: 0}
        for i, s in enumerate(sents):
            s["idx"] = i
            # Within each label, sentences alternate between the two objectives.
            s["objective"] = "maximize" if rank[s["label"]] % 2 == 0 else "minimize"
            rank[s["label"]] += 1
            s["feedback_seed"] = zlib.crc32(f"{concept}:{i}".encode())
        payload["concepts"][concept] = {**meta, "n": len(sents), "sentences": sents}
        print(f"[prepare] {concept}: {len(sents)} sentences")
    with open(SENTENCES_FILE, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    print(f"[prepare] wrote {SENTENCES_FILE}")


# ---- fit --------------------------------------------------------------------------------

def fit_probe(X, y, groups, n_jobs=8, seed=0):
    import warnings
    from joblib import Parallel, delayed
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    uniq = np.unique(groups)
    lab_of = {int(g): int(y[groups == g][0]) for g in uniq}
    tr_ids, te_ids = train_test_split(uniq, test_size=0.2, stratify=np.array([lab_of[int(g)] for g in uniq]),
                                      random_state=seed)
    tr_ids, te_ids = set(int(i) for i in tr_ids), set(int(i) for i in te_ids)
    tr = np.array([int(g) in tr_ids for g in groups])
    Xtr_raw, ytr, gtr = X[tr].astype(np.float64), y[tr], groups[tr]
    Xte_raw, yte = X[~tr].astype(np.float64), y[~tr]
    mu, sd = Xtr_raw.mean(0), Xtr_raw.std(0) + 1e-8
    Xtr, Xte = (Xtr_raw - mu) / sd, (Xte_raw - mu) / sd
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(Xtr, ytr, gtr))

    def cv_one(e, fi):
        f_tr, f_va = folds[fi]
        clf = LogisticRegression(C=float(2.0 ** -e), solver="lbfgs", max_iter=1000).fit(Xtr[f_tr], ytr[f_tr])
        return e, clf.score(Xtr[f_va], ytr[f_va])

    res = Parallel(n_jobs=n_jobs)(delayed(cv_one)(e, fi) for e in LAMBDA_EXPS for fi in range(len(folds)))
    cv_acc = {e: float(np.mean([a for ee, a in res if ee == e])) for e in LAMBDA_EXPS}
    best_e = max(LAMBDA_EXPS, key=lambda e: (cv_acc[e], e))          # ties: more regularization
    clf = LogisticRegression(C=float(2.0 ** -best_e), solver="lbfgs", max_iter=2000).fit(Xtr, ytr)
    w = clf.coef_[0]
    w_norm = float(np.linalg.norm(w))
    return {"lr_weight": (w / (w_norm + 1e-12)).astype(np.float32), "lr_bias": np.float32(clf.intercept_[0]),
            "lr_weight_norm": np.float32(w_norm), "feature_mean": mu.astype(np.float32),
            "feature_std": sd.astype(np.float32), "held_out_accuracy": float(clf.score(Xte, yte)),
            "lambda_log2": int(best_e), "cv_accuracy": cv_acc[best_e],
            "n_train_sentences": len(tr_ids), "n_test_sentences": len(te_ids)}


def run_subject(model, chunk):
    out_dir = model_dir(model)
    if (out_dir / "concept_probes.json").exists():
        print(f"{out_dir / 'concept_probes.json'} exists; delete it to refit")
        return
    data = json.load(open(SENTENCES_FILE))
    backend, reads = None, {}
    for dname, depth in DEPTHS:
        ck = out_dir / f"concept_reads_{dname}.npz"
        store = dict(np.load(ck)) if ck.exists() else {}
        for c in CONCEPTS:
            if f"{c}_X" not in store:
                if backend is None:
                    backend = load_backend(model)
                layer = backend.layer_for_depth(depth)
                sents = data["concepts"][c]["sentences"]
                items, si, td, lb = [], [], [], []
                for k in TURN_DEPTHS:
                    for s in sents:
                        obj = "max" if s["objective"] == "maximize" else "min"
                        items.append((build_conversation(obj, s["text"], feedback_scores(c, s["idx"]), k), s["text"]))
                        si.append(s["idx"]), td.append(k), lb.append(s["label"])
                X = mean_reads(backend, items, layer, chunk, f"{model} {c} {dname}")
                store.update({f"{c}_X": X, f"{c}_sent_idx": np.array(si), f"{c}_turn_depth": np.array(td),
                              f"{c}_label": np.array(lb), f"{c}_layer": np.array(layer)})
                np.savez(str(ck) + ".tmp.npz", **store)
                os.replace(str(ck) + ".tmp.npz", ck)
            reads[(c, dname)] = {k: store[f"{c}_{k}"] for k in ("X", "sent_idx", "label", "layer")}
    npz, meta = {}, {}
    for c in CONCEPTS:
        for dname, depth in DEPTHS:
            r = reads[(c, dname)]
            fit = fit_probe(r["X"], r["label"], r["sent_idx"], n_jobs=min(8, os.cpu_count() or 8))
            for k in ("lr_weight", "lr_bias", "lr_weight_norm", "feature_mean", "feature_std"):
                npz[f"{c}_{dname}_{k}"] = fit[k]
            meta[f"{c}_{dname}"] = {"layer": int(r["layer"]), **{k: v for k, v in fit.items()
                                                                  if not isinstance(v, np.ndarray) and k not in
                                                                  ("lr_bias", "lr_weight_norm")}}
            print(f"[fit] {model} {c} {dname}: held-out accuracy {fit['held_out_accuracy']:.3f}", flush=True)
    np.savez(out_dir / "concept_probes.npz", **npz)
    json.dump({"model": model, "probes": meta}, open(out_dir / "concept_probes.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--subject")
    ap.add_argument("--chunk", type=int, default=128)
    a = ap.parse_args()
    if a.prepare:
        prepare()
    elif a.subject:
        run_subject(a.subject, a.chunk)
    else:
        ap.error("need --prepare or --subject")


if __name__ == "__main__":
    main()
