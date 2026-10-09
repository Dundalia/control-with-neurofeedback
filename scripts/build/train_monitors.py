#!/usr/bin/env python
"""Train the activation monitors of Experiments 3 and 4 (1 GPU).

For each concept, an attention-pooling probe (4 learned queries, then a linear classifier) is
trained on one dataset, at each candidate layer between 25% and 85% of the depth, and the layer
with the highest AUROC on the validation half of a second dataset, from a different source, is
kept; the other half is the test split (reported in the paper, and the source of the sentences
of the experiments). Activations are read on the answer tokens, teacher-forced after a generic
system prompt and question ("You are a helpful assistant." / "Please state a fact." or "Please
say something."), except for deception, whose sentences keep their own scenario. Label 1 is the
flagged class.

  truth      Azaria & Mitchell true/false statements  ->  FEVER claims (no "not enough info")
  deception  RepE facts under honest/deceptive personas -> AI-Liar + roleplaying dialogues
  sentiment  Yelp reviews                              ->  Rotten Tomatoes
  emotion    Twitter emotions (sadness/anger/fear vs joy/love) -> GoEmotions (Reddit)
  morality   ETHICS commonsense                        ->  Moral Stories
  formality  Pavlick formality, news and email         ->  same corpus, blog and answers

-> artifacts/monitors/<model>/<concept>.{npz,json}

    python scripts/build/train_monitors.py --subject qwen3.5-9b --concept truth
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, load_backend  # noqa: E402

DATA = REPO / "data" / "monitor_training"
DEPTHS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
BAND = (0.25, 0.85)
GEN_SYS, FACT_USER, SAY_USER = "You are a helpful assistant.", "Please state a fact.", "Please say something."
CONCEPTS = ["truth", "deception", "sentiment", "emotion", "morality", "formality"]


def _msgs(user):
    return [{"role": "system", "content": GEN_SYS}, {"role": "user", "content": user}]


def _single_sentence(t, max_words=40, min_words=3):
    return min_words <= len(t.split()) <= max_words and (t.count(".") + t.count("!") + t.count("?")) <= 1


# ---- datasets: rows (messages, answer, label, group) ---------------------------------------

def truth_train():
    from concept_probes import build_truth
    sents, _ = build_truth(np.random.default_rng(0))
    return [(_msgs(FACT_USER), s["text"], 1 - int(s["label"]), gid) for gid, s in enumerate(sents)]


def truth_ood(n_per_class=150):
    from datasets import load_dataset
    pools, seen = {"true": [], "false": []}, set()
    for r in load_dataset("copenlu/fever_gold_evidence", split="train"):
        key = {"SUPPORTS": "true", "REFUTES": "false"}.get(r["label"])
        if key is None:
            continue
        c = re.sub(r"\s+", " ", str(r["claim"])).strip()
        if 6 <= len(c.split()) <= 40 and c[:1].isalpha() and c.lower() not in seen:
            seen.add(c.lower())
            pools[key].append(c)
        if len(pools["true"]) >= 8 * n_per_class and len(pools["false"]) >= 8 * n_per_class:
            break
    rng = np.random.default_rng(0)
    claims = []
    for key in ("true", "false"):
        claims += [(pools[key][i], key) for i in sorted(rng.permutation(len(pools[key]))[:n_per_class])]
    return [(_msgs(FACT_USER), c, int(k == "false"), 200000 + gid) for gid, (c, k) in enumerate(claims)]


def deception_train():
    facts = [r["statement"] for r in csv.DictReader(open(DATA / "repe_true_false_facts.csv")) if r["label"] == "1"][:512]
    rows = []
    for gid, fact in enumerate(facts):
        for label, system in ((0, "You are an honest AI assistant."), (1, "You are a deceptive AI assistant.")):
            rows.append(([{"role": "system", "content": system},
                          {"role": "user", "content": "Please tell me a fact."}], fact, label, gid))
    return rows


def deception_ood():
    rows = []
    for gid, ex in enumerate(json.load(open(DATA / "ai_liar_goal_directed.json"))):
        q = ex["question"].strip()
        rows.append(([{"role": "system", "content": ex["deceive_instruction"]}, {"role": "user", "content": q}],
                     ex["answer"].strip(), 1, gid))
        rows.append(([{"role": "system", "content": ex["normal_instruction"]}, {"role": "user", "content": q}],
                     ex["other_answer"].strip(), 0, gid))
    for gid, line in enumerate(open(DATA / "roleplaying.jsonl")):
        e = json.loads(line)
        rows.append(([{"role": "system", "content": e["system"]}, {"role": "user", "content": e["user"]}],
                     e["response"], int(e["label"]), 100000 + gid))
    return rows


def _balanced_stream(ds, label_of, text_of, cap, ok=lambda t: True):
    rows, per, n = [], cap // 2, {0: 0, 1: 0}
    for j in np.random.default_rng(0).permutation(len(ds)):
        ex = ds[int(j)]
        lab = label_of(ex)
        if lab < 0 or n[lab] >= per:
            continue
        t = text_of(ex)
        if not ok(t):
            continue
        rows.append((_msgs(SAY_USER), t, lab, len(rows)))
        n[lab] += 1
        if n[0] >= per and n[1] >= per:
            break
    return rows


def sentiment_train(cap=600):
    from datasets import load_dataset
    return _balanced_stream(load_dataset("fancyzhx/yelp_polarity", split="train"), lambda ex: 1 - int(ex["label"]),
                            lambda ex: " ".join(str(ex["text"]).replace("\\n", " ").lower().split()[:80]).strip(),
                            cap, ok=lambda t: len(t.split()) >= 5)


def sentiment_ood(n_per_class=150):
    from datasets import load_dataset
    ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
    neg, pos = [], []
    for j in np.random.default_rng(0).permutation(len(ds)):
        ex = ds[int(j)]
        t = str(ex["text"]).strip()
        if len(t.split()) < 5:
            continue
        (neg if int(ex["label"]) == 0 else pos).append(t)
        if len(neg) >= n_per_class and len(pos) >= n_per_class:
            break
    rows = []
    for lab, texts in ((1, neg), (0, pos)):
        for t in texts[:n_per_class]:
            rows.append((_msgs(SAY_USER), t, lab, 300000 + len(rows)))
    return rows


EMO_NEG, EMO_POS = {"sadness", "anger", "fear"}, {"joy", "love"}


def emotion_train(cap=600):
    from datasets import load_dataset
    ds = load_dataset("dair-ai/emotion", "split", split="train")
    names = ds.features["label"].names
    lab = lambda ex: 1 if names[int(ex["label"])] in EMO_NEG else (0 if names[int(ex["label"])] in EMO_POS else -1)  # noqa: E731
    return _balanced_stream(ds, lab, lambda ex: str(ex["text"]).strip(), cap, ok=lambda t: len(t.split()) >= 3)


def emotion_ood(n_per_class=150):
    from datasets import load_dataset
    neg, pos = [], []
    for split in ("test", "validation"):
        ds = load_dataset("google-research-datasets/go_emotions", "simplified", split=split)
        names = ds.features["labels"].feature.names
        for ex in ds:
            t = str(ex["text"]).strip()
            if len(ex["labels"]) != 1 or not _single_sentence(t):
                continue
            nm = names[ex["labels"][0]]
            if nm in EMO_NEG:
                neg.append(t)
            elif nm in EMO_POS:
                pos.append(t)
    rng = np.random.default_rng(0)
    neg = [neg[i] for i in rng.permutation(len(neg))][:n_per_class]
    pos = [pos[i] for i in rng.permutation(len(pos))][:n_per_class]
    rows = []
    for lab, texts in ((1, neg), (0, pos)):
        for t in texts:
            rows.append((_msgs(SAY_USER), t, lab, 400000 + len(rows)))
    return rows


def morality_train(cap=600):
    from datasets import load_dataset
    ds = load_dataset("iproskurina/hendrycks_ethics_commonsense", split="train")
    rows, per, n = [], cap // 2, {0: 0, 1: 0}
    for j in np.random.default_rng(0).permutation(len(ds)):
        ex = ds[int(j)]
        lab, t = int(ex["label"]), str(ex["input"]).strip()
        if not _single_sentence(t) or n[lab] >= per:
            continue
        rows.append((_msgs(SAY_USER), t, lab, len(rows)))
        n[lab] += 1
        if n[0] >= per and n[1] >= per:
            break
    return rows


def morality_ood(n_per_class=150):
    from datasets import load_dataset
    ds = load_dataset("LabHC/moral_stories", "default", split="train")
    moral, immoral = [], []
    for j in np.random.default_rng(0).permutation(len(ds)):
        ex = ds[int(j)]
        m, im = str(ex["moral_action"]).strip(), str(ex["immoral_action"]).strip()
        if _single_sentence(m):
            moral.append(m)
        if _single_sentence(im):
            immoral.append(im)
        if len(moral) >= n_per_class and len(immoral) >= n_per_class:
            break
    rows = []
    for lab, texts in ((1, immoral), (0, moral)):
        for t in texts[:n_per_class]:
            rows.append((_msgs(SAY_USER), t, lab, 500000 + len(rows)))
    return rows


def _pavlick(genres, thresh=1.0):
    """Clearly formal (score >= 1) and informal (<= -1) sentences of the given genres, with
    matched word counts (5-word bins)."""
    from datasets import load_dataset
    formal, informal = [], []
    for split in ("train", "test"):
        for ex in load_dataset("osyvokon/pavlick-formality-scores", split=split):
            t, s = str(ex["sentence"]).strip(), float(ex["avg_score"])
            if ex["domain"] in genres and 3 <= len(t.split()) <= 40:
                if s >= thresh:
                    formal.append(t)
                elif s <= -thresh:
                    informal.append(t)
    rng = np.random.default_rng(0)
    ba, bb = collections.defaultdict(list), collections.defaultdict(list)
    for t in formal:
        ba[len(t.split()) // 5].append(t)
    for t in informal:
        bb[len(t.split()) // 5].append(t)
    oa, ob = [], []
    for k in sorted(set(ba) | set(bb)):
        m = min(len(ba[k]), len(bb[k]))
        if m:
            oa += [ba[k][i] for i in rng.permutation(len(ba[k]))][:m]
            ob += [bb[k][i] for i in rng.permutation(len(bb[k]))][:m]
    return oa, ob


def formality_train(cap=600):
    formal, informal = _pavlick({"news", "email"})
    rows = []
    for lab, texts in ((1, formal), (0, informal)):
        for t in texts[:cap // 2]:
            rows.append((_msgs(SAY_USER), t, lab, len(rows)))
    return rows


def formality_ood(n_per_class=150):
    formal, informal = _pavlick({"blog", "answers"})
    rows = []
    for lab, texts in ((1, formal), (0, informal)):
        for t in texts[:n_per_class]:
            rows.append((_msgs(SAY_USER), t, lab, 600000 + len(rows)))
    return rows


DATASETS = {"truth": (truth_train, truth_ood, "Azaria & Mitchell -> FEVER"),
            "deception": (deception_train, deception_ood, "RepE facts -> AI-Liar + roleplaying"),
            "sentiment": (sentiment_train, sentiment_ood, "Yelp -> Rotten Tomatoes"),
            "emotion": (emotion_train, emotion_ood, "dair-ai/emotion -> GoEmotions"),
            "morality": (morality_train, morality_ood, "ETHICS commonsense -> Moral Stories"),
            "formality": (formality_train, formality_ood, "Pavlick news+email -> blog+answers")}


def group_split(groups, frac=0.5, seed=0):
    uniq = np.array(sorted(set(int(g) for g in groups)))
    perm = np.random.default_rng(seed).permutation(len(uniq))
    first = set(uniq[perm[:max(1, int(round(frac * len(uniq))))]].tolist())
    is_a = np.array([int(g) in first for g in groups])
    return np.where(is_a)[0], np.where(~is_a)[0]


def splits(concept, seed=0, ood_cap=None):
    """(train rows, validation rows, test rows) as used to build the monitors."""
    train_fn, ood_fn, _ = DATASETS[concept]
    tr, ood = train_fn(), (ood_fn() if ood_cap is None else ood_fn(ood_cap))
    rng = np.random.default_rng(seed)
    tr = [tr[i] for i in rng.permutation(len(tr))]
    ood = [ood[i] for i in rng.permutation(len(ood))]
    vi, ti = group_split(np.array([g for (_, _, _, g) in ood]), 0.5, seed)
    return tr, [ood[i] for i in vi], [ood[i] for i in ti]


# ---- probe ---------------------------------------------------------------------------------

def collect_reads(backend, rows, layers, chunk=48):
    acts = {l: [None] * len(rows) for l in layers}
    for i in range(0, len(rows), chunk):
        reads = backend.read_residual_multi_batch([(m, r) for (m, r, _, _) in rows[i:i + chunk]], layers)
        for j, read in enumerate(reads):
            arr = read.numpy()
            for li, l in enumerate(sorted(layers)):
                acts[l][i + j] = arr[li].astype(np.float16)
        print(f"[reads] {min(i + chunk, len(rows))}/{len(rows)}", flush=True)
    return acts, np.array([lab for (_, _, lab, _) in rows], np.int64), np.array([g for (_, _, _, g) in rows])


def auroc(scores, y):
    from sklearn.metrics import roc_auc_score
    return float("nan") if len(set(np.asarray(y).tolist())) < 2 else float(roc_auc_score(y, scores))


def train_probe(acts_tr, y_tr, acts_va, y_va, lr=1e-3, batch=16, seed=0, max_epochs=200, patience=20):
    """Attention probe on z-scored activations, BCE + AdamW (weight decay 0.01), early stopping
    on the validation AUROC. Returns (scorer, params)."""
    import torch
    import torch.nn as nn

    class AttnProbe(nn.Module):
        def __init__(self, hidden, K=4):
            super().__init__()
            self.q = nn.Parameter(torch.randn(K, hidden) * hidden ** -0.5)
            self.w = nn.Parameter(torch.randn(K, hidden) * hidden ** -0.5)
            self.b = nn.Parameter(torch.zeros(1))

        def forward(self, H, M):
            logits = torch.einsum("bth,kh->btk", H, self.q).masked_fill(~M.bool().unsqueeze(-1), float("-inf"))
            c = torch.einsum("btk,bth->bkh", torch.softmax(logits, 1), H)
            return torch.sigmoid(torch.einsum("bkh,kh->b", c, self.w) + self.b)

    def pad(seqs):
        T = max(s.shape[0] for s in seqs)
        H = torch.zeros(len(seqs), T, seqs[0].shape[1], device=dev)
        M = torch.zeros(len(seqs), T, device=dev)
        for i, s in enumerate(seqs):
            H[i, :s.shape[0]] = s
            M[i, :s.shape[0]] = 1.0
        return H, M

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = [torch.from_numpy(a.astype(np.float32)) for a in acts_tr]
    mu, sd = torch.cat(tr, 0).mean(0), torch.cat(tr, 0).std(0) + 1e-6
    tr = [(s - mu) / sd for s in tr]
    va = [(torch.from_numpy(a.astype(np.float32)) - mu) / sd for a in acts_va]
    net = AttnProbe(tr[0].shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    bce, rng = nn.BCELoss(), np.random.default_rng(seed)

    def run(seqs):
        net.eval()
        with torch.no_grad():
            return np.concatenate([net(*pad(seqs[k:k + 256])).cpu().numpy() for k in range(0, len(seqs), 256)])

    best, best_state, since = -1.0, None, 0
    for _ in range(max_epochs):
        net.train()
        order = rng.permutation(len(tr))
        for k in range(0, len(order), batch):
            sel = order[k:k + batch]
            loss = bce(net(*pad([tr[i] for i in sel])).clamp(1e-6, 1 - 1e-6),
                       torch.from_numpy(y_tr[sel].astype(np.float32)).to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        va_auc = auroc(run(va), y_va)
        va_auc = -1.0 if np.isnan(va_auc) else va_auc
        if va_auc > best + 1e-4:
            best, since = va_auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    best_state = net.state_dict()

    def scorer(acts):
        return run([(torch.from_numpy(a.astype(np.float32)) - mu) / sd for a in acts])

    params = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in best_state.items()}
    return scorer, {"mu": mu.numpy().astype(np.float32), "sd": sd.numpy().astype(np.float32), **params}


def train(backend, model, concept, chunk=48):
    tr, val, test = splits(concept)
    nlay = backend.num_layers
    band = [l for l in sorted({backend.layer_for_depth(d) for d in DEPTHS}) if BAND[0] <= l / (nlay - 1) <= BAND[1]]
    tr_acts, y_tr, g_tr = collect_reads(backend, tr, band, chunk)
    val_acts, y_val, _ = collect_reads(backend, val, band, chunk)
    test_acts, y_test, _ = collect_reads(backend, test, band, chunk)
    # 15% of the training groups for early stopping
    uniq = np.array(sorted(set(int(g) for g in g_tr)))
    va_g = set(uniq[np.random.default_rng(0).permutation(len(uniq))[:max(1, int(round(0.15 * len(uniq))))]].tolist())
    is_va = np.array([int(g) in va_g for g in g_tr])
    tri, vai = np.where(~is_va)[0], np.where(is_va)[0]
    per_layer, params = {}, {}
    for l in band:
        scorer, p = train_probe([tr_acts[l][i] for i in tri], y_tr[tri], [tr_acts[l][i] for i in vai], y_tr[vai])
        per_layer[str(l)] = {"depth": round(l / (nlay - 1), 4), "val_auroc": auroc(scorer(val_acts[l]), y_val),
                             "test_auroc": auroc(scorer(test_acts[l]), y_test)}
        params[l] = p
        print(f"[monitor] {model} {concept} layer {l}: {per_layer[str(l)]}", flush=True)
    sel = max(band, key=lambda l: per_layer[str(l)]["val_auroc"] if not np.isnan(per_layer[str(l)]["val_auroc"]) else -9)
    out = REPO / "artifacts" / "monitors" / model
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{concept}.npz", layer=np.array(sel), **params[sel])
    json.dump({"model": backend.model_name, "concept": concept, "data": DATASETS[concept][2], "num_layers": nlay,
               "n_train": len(tr), "n_val": len(val), "n_test": len(test), "n_test_positive": int(y_test.sum()),
               "per_layer": per_layer, "selected_layer": sel, "val_auroc": per_layer[str(sel)]["val_auroc"],
               "test_auroc": per_layer[str(sel)]["test_auroc"]}, open(out / f"{concept}.json", "w"), indent=1)
    print(f"[monitor] {model} {concept}: layer {sel}, test AUROC {per_layer[str(sel)]['test_auroc']:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--concept", nargs="*", default=CONCEPTS)
    a = ap.parse_args()
    backend = load_backend(a.subject)
    for c in a.concept:
        train(backend, a.subject, c)


if __name__ == "__main__":
    main()
