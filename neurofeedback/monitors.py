"""Activation monitors of Experiments 3 and 4.

A monitor is an attention-pooling probe at one layer: the activations of the answer tokens
are z-scored, K = 4 learned queries each pool them into a weighted average, and a linear
classifier on the K averages gives the probability that the sentence belongs to the
flagged class (false, deceptive, negative, immoral, formal). The score is 100 times that
probability. Monitors are trained by ``scripts/build/train_monitors.py`` and stored in
``artifacts/monitors/<model>/<concept>.npz``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

CONCEPTS = ("truth", "deception", "sentiment", "emotion", "morality", "formality")
MONITORS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "monitors"


@dataclass
class Monitor:
    concept: str
    layer: int
    mu: np.ndarray      # [hidden] z-score mean
    sd: np.ndarray      # [hidden] z-score std
    q: np.ndarray       # [K, hidden] pooling queries
    w: np.ndarray       # [K, hidden] classifier weights
    b: float

    @property
    def layers(self):
        return [self.layer]

    def score(self, span) -> float:
        """``span``: [1, T, hidden] (or [T, hidden]) activations of the answer tokens."""
        span = np.asarray(span, dtype=np.float64)
        if span.ndim == 3:
            span = span[0]
        Z = (span - self.mu) / self.sd
        s = Z @ self.q.T                                  # [T, K]
        s = s - s.max(0, keepdims=True)
        a = np.exp(s)
        a = a / a.sum(0, keepdims=True)
        c = a.T @ Z                                       # [K, hidden]
        p = float(1.0 / (1.0 + np.exp(-(float(np.sum(c * self.w)) + self.b))))
        return 100.0 * p


def load_monitor(model: str, concept: str, monitors_dir=MONITORS_DIR) -> Monitor:
    z = np.load(Path(monitors_dir) / model / f"{concept}.npz")
    return Monitor(concept=concept, layer=int(z["layer"]),
                   mu=z["mu"].astype(np.float64), sd=z["sd"].astype(np.float64),
                   q=z["q"].astype(np.float64), w=z["w"].astype(np.float64),
                   b=float(np.asarray(z["b"]).reshape(-1)[0]))
