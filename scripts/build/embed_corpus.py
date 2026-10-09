#!/usr/bin/env python
"""Embed the 20k corpus with the sentence encoder of the encoder-residual readouts (1 GPU).

Qwen3-Embedding-8B at a pinned revision, document mode (no instruction), embedding = hidden
state of the final EOS token (left padding), L2-normalized in fp32, stored in fp16.
-> build/c4_corpus_20k_embeddings.npz (E [20000, 4096], ids)

    python scripts/build/embed_corpus.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BUILD, REPO  # noqa: E402

ENCODER, REVISION = "Qwen/Qwen3-Embedding-8B", "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
MAX_LENGTH, BATCH = 128, 128
OUT = BUILD / "c4_corpus_20k_embeddings.npz"


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer

    rows = sorted((json.loads(line) for line in open(REPO / "data" / "c4_corpus_20k.jsonl")), key=lambda r: r["id"])
    texts = [r["text"] for r in rows]
    tok = AutoTokenizer.from_pretrained(ENCODER, revision=REVISION, padding_side="left")
    model = AutoModel.from_pretrained(ENCODER, revision=REVISION, dtype=torch.float16).cuda().eval()
    eos = tok.eos_token_id
    append_eos = tok(texts[0], truncation=True, max_length=MAX_LENGTH)["input_ids"][-1] != eos
    E = np.empty((len(texts), model.config.hidden_size), dtype=np.float16)
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            enc = tok(texts[i:i + BATCH], truncation=True, max_length=MAX_LENGTH - 1)["input_ids"]
            if append_eos:
                enc = [ids + [eos] for ids in enc]
            batch = tok.pad({"input_ids": enc}, padding=True, return_tensors="pt").to("cuda")
            pooled = model(**batch).last_hidden_state[:, -1, :].float()   # left padding: last = EOS
            E[i:i + BATCH] = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy().astype(np.float16)
            print(f"[embed] {min(i + BATCH, len(texts))}/{len(texts)}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, E=E, ids=np.array([r["id"] for r in rows], dtype=np.int32))
    print(f"[embed] wrote {OUT}")


if __name__ == "__main__":
    main()
