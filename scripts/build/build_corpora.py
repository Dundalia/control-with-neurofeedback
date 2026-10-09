#!/usr/bin/env python
"""Build the two C4 sentence sets of the readouts (CPU, streams C4 from the Hugging Face Hub).

  reference  data/c4_reference_256.jsonl  256 low-entity declarative sentences, the reference
                                          set on which every readout is calibrated
  corpus     data/c4_corpus_20k.jsonl     20,000 sentences (at most one per document), on which
                                          the readout directions are built; disjoint from the
                                          reference set and from the concept-probe sentences

Both shuffle the C4 (en) train stream with seed 0 and stratify sentence lengths. The corpus
must be built after the reference set and after scripts/build/concept_probes.py --prepare.

    python scripts/build/build_corpora.py reference
    python scripts/build/build_corpora.py corpus
"""
import argparse
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
C4 = ("allenai/c4", "en", "1588ec454efa1a09f29cd18ddd04fe05fc8653a2")
SEED, SHUFFLE_BUF = 0, 5000
_WS = re.compile(r"\s+")
_MULTI = re.compile(r"[.!?][\"')\]]?\s+[A-Z0-9]")      # an embedded sentence boundary


def c4_sentences(head_sents):
    """(document index, first ``head_sents`` sentences of the document) from the C4 stream."""
    from blingfire import text_to_sentences
    from datasets import load_dataset
    ds = load_dataset(C4[0], C4[1], split="train", streaming=True, revision=C4[2]).shuffle(
        seed=SEED, buffer_size=SHUFFLE_BUF)
    for n, ex in enumerate(ds):
        sents = []
        for para in (ex.get("text", "") or "").split("\n"):
            para = para.strip()
            if len(para) < 25:
                continue
            sents.extend(text_to_sentences(para).split("\n"))
            if len(sents) >= head_sents:
                break
        yield n, sents[:head_sents]


# ---- reference set ----------------------------------------------------------------------
REF_BINS = {"short": (5, 11), "medium": (12, 20), "long": (21, 32)}
REF_QUOTA = {"short": 85, "medium": 85, "long": 86}
_CTX_START = re.compile(
    r"^(he|she|it|they|them|their|theirs|this|that|these|those|its|his|her|"
    r"however|therefore|thus|moreover|furthermore|additionally|also|then|"
    r"here|there|such|both|each|meanwhile|nonetheless|nevertheless|"
    r"consequently|hence|instead|otherwise|besides)\b", re.I)
_CITATION = re.compile(r"\[\d+\]|\[[a-z]\]|\[citation needed\]|\[clarification needed\]")
_BAD = re.compile(r"https?://|www\.|[|=#*<>{}\\@]|\bISBN\b|\bISSN\b")
_VERB = re.compile(r"\b(is|are|was|were|be|been|being|am|has|have|had|do|does|did|will|would|shall|"
                   r"should|can|could|may|might|must|became|become|becomes)\b", re.I)
_MASK_CAP = re.compile(r"\b[A-Z][\w'’-]*")
_MASK_NUM = re.compile(r"\d[\d,.\-/:]*")


def _has_verb(s):
    """A finite verb: an auxiliary, copula or modal, or a word ending in -ed."""
    return bool(_VERB.search(s)) or any(len(w) > 4 and w.lower().strip(".,!?;:").endswith("ed")
                                        for w in s.split())


def _low_entity(s):
    toks = s.split()
    if len(toks) < 2:
        return False
    caps_mid = sum(1 for t in toks[1:] if t[:1].isupper())
    nums = sum(1 for t in toks if any(c.isdigit() for c in t))
    return caps_mid <= 2 and nums <= 1 and (caps_mid + nums) / len(toks) <= 0.20


def reference_ok(s):
    if not s or s[-1] not in ".!?" or _MULTI.search(s[:-1]):
        return False
    if not 5 <= len(s.split()) <= 32 or not s[0].isupper() or _CTX_START.match(s) or _BAD.search(s):
        return False
    if ":" in s or ".." in s or " ." in s or "()" in s or "[]" in s:
        return False
    if s.count('"') % 2 or s.count("(") != s.count(")") or (s.count("(") + s.count("[")) > 2:
        return False
    if (s.count("!") + s.count("?")) > 1:
        return False
    if any(len(t) > 20 or (t.isalpha() and t.isupper() and len(t) >= 4) for t in s.split()):
        return False
    if sum(c.isalpha() or c.isspace() for c in s) / len(s) < 0.80:
        return False
    return _low_entity(s) and _has_verb(s)


def build_reference():
    rng = random.Random(SEED)
    buckets = {b: [] for b in REF_BINS}
    seen_text, seen_sig = set(), set()
    for n, sents in c4_sentences(head_sents=12):
        if all(len(buckets[b]) >= REF_QUOTA[b] for b in REF_BINS):
            break
        took = 0
        for raw in sents:
            if took >= 2:                                   # at most two per document
                break
            s = _WS.sub(" ", _CITATION.sub("", raw)).strip()
            if not reference_ok(s):
                continue
            sig = _WS.sub(" ", _MASK_NUM.sub("0", _MASK_CAP.sub("X", s)).lower()).strip()
            if s.lower() in seen_text or sig in seen_sig:   # exact and template near-duplicates
                continue
            b = next((k for k, (lo, hi) in REF_BINS.items() if lo <= len(s.split()) <= hi), None)
            if b is None or len(buckets[b]) >= REF_QUOTA[b]:
                continue
            seen_text.add(s.lower())
            seen_sig.add(sig)
            buckets[b].append(s)
            took += 1
    pool = [{"text": s, "n_words": len(s.split()), "length_bin": b} for b in REF_BINS for s in buckets[b]]
    rng.shuffle(pool)
    for i, r in enumerate(pool):
        r["id"] = i
    return pool


# ---- corpus -----------------------------------------------------------------------------
CORPUS_BINS = {"short": (5, 11), "medium": (12, 20), "long": (21, 30)}
CORPUS_QUOTA = {"short": 6666, "medium": 6667, "long": 6667}
_URL = re.compile(r"https?://|www\.|\b\w+\.(?:com|org|net|edu|gov|io|co|uk|info|biz)\b", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.\w")
_PHONE = re.compile(r"(?:\+?\d{1,3}[-.\s(]+)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b|\b\d{7,}\b")
_HTML = re.compile(r"<[^>]*>|&[a-z]+;|&#\d+;|[<>{}|\\^~`_]|\[/?\w+\]")
_ALLOWED_EXTRA = set("‘’“”–—…")
_PUNCT = set(".,;:!?\"'()[]-‘’“”–—…/&%$#@*+=")


def _sane_unicode(s):
    for c in s:
        if c.isascii():
            if not (c.isprintable() or c == " "):
                return False
        elif c not in _ALLOWED_EXTRA and not (ord(c) < 0x250 and unicodedata.category(c) in ("Ll", "Lu")):
            return False
    return True


def corpus_ok(s):
    if not 5 <= len(s.split()) <= 30 or not (s[0].isalpha() and s[0].isupper()) or s[-1] not in ".!?":
        return False
    if _MULTI.search(s[:-1]) or _URL.search(s) or _EMAIL.search(s) or _PHONE.search(s) or _HTML.search(s):
        return False
    if not _sane_unicode(s):
        return False
    letters = [c for c in s if c.isalpha()]
    if letters and (sum(c.isupper() for c in letters) / len(letters) > 0.50
                    or any(t.isalpha() and t.isupper() and len(t) >= 4 for t in s.split())):
        return False
    return sum(c.isdigit() for c in s) / len(s) < 0.20 and sum(c in _PUNCT for c in s) / len(s) <= 0.15


def build_corpus():
    excl = {json.loads(l)["text"].casefold() for l in open(DATA / "c4_reference_256.jsonl")}
    probes = json.load(open(DATA / "concept_probe_sentences.json"))
    excl |= {r["text"].casefold() for blk in probes["concepts"].values() for r in blk["sentences"]}
    rng = random.Random(SEED)
    buckets = {b: [] for b in CORPUS_BINS}
    seen = set()
    for n, sents in c4_sentences(head_sents=20):
        if all(len(buckets[b]) >= CORPUS_QUOTA[b] for b in CORPUS_BINS):
            break
        for raw in sents:                                   # the first accepted sentence wins
            s = _WS.sub(" ", raw).strip()
            if not s or not corpus_ok(s):
                continue
            key = s.casefold()
            if key in excl or key in seen:
                continue
            b = next((k for k, (lo, hi) in CORPUS_BINS.items() if lo <= len(s.split()) <= hi), None)
            if b is None or len(buckets[b]) >= CORPUS_QUOTA[b]:
                continue
            seen.add(key)
            buckets[b].append(s)
            break
        if n % 20000 == 0:
            print(f"[corpus] {n} documents: " + " ".join(f"{b}={len(v)}" for b, v in buckets.items()), flush=True)
    pool = [{"text": s, "n_words": len(s.split()), "length_bin": b} for b in CORPUS_BINS for s in buckets[b]]
    rng.shuffle(pool)
    for i, r in enumerate(pool):
        r["id"] = i
    return pool


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["reference", "corpus"])
    ap.add_argument("--out", help="output file (default: the file in data/)")
    a = ap.parse_args()
    pool = build_reference() if a.which == "reference" else build_corpus()
    out = Path(a.out or DATA / ("c4_reference_256.jsonl" if a.which == "reference" else "c4_corpus_20k.jsonl"))
    out.write_text("\n".join(json.dumps(r) for r in pool) + "\n")
    print(f"[{a.which}] {len(pool)} sentences, bins {dict(Counter(r['length_bin'] for r in pool))} -> {out}")
    sys.stdout.flush()
    os._exit(0)                     # the streaming dataset leaves a thread that crashes at exit


if __name__ == "__main__":
    main()
