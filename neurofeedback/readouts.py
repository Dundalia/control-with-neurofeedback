"""The 42 readouts of Experiments 1 and 2.

A readout maps the activations of the subject on the tokens of a sentence (the *span*) to
a scalar *raw* value, and the raw value to a *score* by an affine map fixed on a reference
set of sentences: score = 50 + 100 (raw - median) / (p99 - p1), unclipped.

Readouts are loaded from ``artifacts/readouts/<model>.npz`` (built by ``scripts/build``),
and come in four families:

* ``probe``    concept-probe directions (sentiment, morality, truthfulness);
* ``pcband``   random directions in a band of principal components;
* ``encres``   directions in the part of the activations a sentence encoder cannot predict;
* ``geometry`` trajectory geometry: curvature with step 1, 2, 3, step change, and the
               differences of step-1 curvature between depths.

The first three are a centered cosine, mean_j cos(h_j - mu, v), at 33% or 66% depth. Every
readout is computed from the residual stream at four anchor depths (33/44/55/66%).

The numerics (float64, the 1e-8 and 1e-12 guards, the NaN rules) are those the released
calibration constants were computed with; do not change them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ANCHOR_DEPTHS = {"d33": 0.33, "d44": 0.44, "d55": 0.55, "d66": 0.66}
FAMILIES = ("probe", "pcband", "encres", "geometry")
READOUTS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "readouts"


def _angles(A, B):
    """Angles (rad) between the rows A[i] and B[i]."""
    na = np.linalg.norm(A, axis=1)
    nb = np.linalg.norm(B, axis=1)
    cos = (A * B).sum(1) / (na * nb + 1e-12)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def curvature(H, step):
    """Mean angle between H[j+g] - H[j] and H[j] - H[j-g], g = ``step``. NaN if the span has
    fewer than 2g + 1 tokens."""
    T = H.shape[0]
    if T < 2 * step + 1:
        return np.nan
    fwd = H[2 * step:] - H[step:T - step]
    bwd = H[step:T - step] - H[:T - 2 * step]
    return float(_angles(fwd, bwd).mean())


def step_change(H):
    """Mean of ||d_{j+1} - d_j|| / ||d_j|| with d_j = H[j+1] - H[j]. NaN if T < 3."""
    V = np.diff(H, axis=0)
    if V.shape[0] < 2:
        return np.nan
    num = np.linalg.norm(V[1:] - V[:-1], axis=1)
    den = np.linalg.norm(V[:-1], axis=1) + 1e-8
    return float((num / den).mean())


def centered_cosine(H, mu, v):
    """Mean over tokens of cos(h_j - mu, v)."""
    C = H - mu
    return float(np.mean((C @ v) / (np.linalg.norm(C, axis=1) * np.linalg.norm(v) + 1e-8)))


def _geometry_kind(readout_id: str):
    """``curvature_step2_d33`` -> ("curvature", 2, ("d33",)); ``step_change_d66`` ->
    ("step_change", None, ("d66",)); ``curvature_diff_66_33`` -> ("diff", None, ("d66", "d33"))."""
    if readout_id.startswith("curvature_diff_"):
        hi, lo = readout_id.split("_")[2:4]
        return "diff", None, (f"d{hi}", f"d{lo}")
    if readout_id.startswith("curvature_step"):
        step, depth = readout_id[len("curvature_step"):].split("_")
        return "curvature", int(step), (depth,)
    if readout_id.startswith("step_change_"):
        return "step_change", None, (readout_id[len("step_change_"):],)
    raise ValueError(f"unknown geometry readout {readout_id!r}")


@dataclass
class Readout:
    id: str
    family: str
    depth: str                  # "d33", "d66", or "cross_depth" for curvature differences
    anchors: tuple              # anchor depths read, e.g. ("d33",) or ("d66", "d33")
    lo: float                   # p1 of the raws on the reference set
    hi: float                   # p99
    median: float
    mu: np.ndarray | None = None       # direction readouts: corpus mean at the readout depth
    vector: np.ndarray | None = None   # direction readouts: unit direction

    def raw(self, spans: dict) -> float:
        """``spans``: anchor depth -> [T, hidden] float64 activations on the sentence."""
        if self.family != "geometry":
            return centered_cosine(spans[self.depth], self.mu, self.vector)
        kind, step, anchors = _geometry_kind(self.id)
        if kind == "diff":
            return curvature(spans[anchors[0]], 1) - curvature(spans[anchors[1]], 1)
        if kind == "curvature":
            return curvature(spans[anchors[0]], step)
        return step_change(spans[anchors[0]])

    def score(self, raw: float) -> float:
        return 100.0 * float(0.5 + (raw - self.median) / (self.hi - self.lo))


class ReadoutSet:
    """All the readouts of one subject, scored from one read at the four anchor depths.

    ``calibration`` selects the affine maps: ``"exp1"`` (reference sentences read in the
    context of the battery build) or ``"exp2"`` (reference sentences read in the
    hidden-criterion prompt of Experiment 2, stored in ``<model>.exp2_calibration.npz``)."""

    def __init__(self, model: str, calibration: str = "exp1", readouts_dir=READOUTS_DIR,
                 readout_ids=None):
        path = Path(readouts_dir) / f"{model}.npz"
        z = np.load(path)
        ids = [str(x) for x in z["readout_ids"]]
        families = [str(x) for x in z["families"]]
        depths = [str(x) for x in z["depths"]]
        self.model = model
        self.num_layers = int(z["num_layers"])
        self.layers = {d: int(z[f"layer_{d}"]) for d in ANCHOR_DEPTHS}
        if calibration == "exp1":
            cal = {k: z[f"calib_{k}"] for k in ("lo", "hi", "median")}
        elif calibration == "exp2":
            c = np.load(Path(readouts_dir) / f"{model}.exp2_calibration.npz")
            if [str(x) for x in c["readout_ids"]] != ids:
                raise ValueError(f"{model}: exp2 calibration readouts differ from {path.name}")
            cal = {k: c[f"calib_{k}"] for k in ("lo", "hi", "median")}
        else:
            raise ValueError(f"calibration must be 'exp1' or 'exp2', got {calibration!r}")
        self.readouts = {}
        for i, (rid, fam, depth) in enumerate(zip(ids, families, depths)):
            if readout_ids is not None and rid not in readout_ids:
                continue
            lo, hi, med = (float(cal["lo"][i]), float(cal["hi"][i]), float(cal["median"][i]))
            if not all(math.isfinite(x) for x in (lo, hi, med)):
                raise ValueError(f"{model}/{rid}: calibration constants not finite")
            if fam == "geometry":
                anchors = _geometry_kind(rid)[2]
                self.readouts[rid] = Readout(rid, fam, depth, anchors, lo, hi, med)
            else:
                self.readouts[rid] = Readout(
                    rid, fam, depth, (depth,), lo, hi, med,
                    mu=z[f"mu_{depth}"].astype(np.float64), vector=z[f"vec_{rid}"].astype(np.float64))
        if readout_ids is not None and set(readout_ids) - set(self.readouts):
            raise KeyError(f"unknown readouts: {sorted(set(readout_ids) - set(self.readouts))}")
        self.ids = list(self.readouts)

    def check_backend(self, backend):
        if backend.num_layers != self.num_layers:
            raise ValueError(f"{self.model}: readouts built for {self.num_layers} layers, "
                             f"backend has {backend.num_layers}")
        for d, layer in self.layers.items():
            if backend.layer_for_depth(ANCHOR_DEPTHS[d]) != layer:
                raise ValueError(f"{self.model}: anchor {d} is layer {layer} in the readouts, "
                                 f"{backend.layer_for_depth(ANCHOR_DEPTHS[d])} in the backend")

    def layers_for(self, readout: Readout) -> list[int]:
        return sorted({self.layers[a] for a in readout.anchors})

    def _spans(self, read, layers):
        """[n_layers, T, hidden] read (layers ascending) -> anchor -> float64 [T, hidden]."""
        H = np.asarray(read.numpy() if hasattr(read, "numpy") else read).astype(np.float64)
        return {d: H[layers.index(lay)] for d, lay in self.layers.items() if lay in layers}

    def score_one(self, backend, readout_id, items):
        """Experiment 1: read only the layers ``readout_id`` needs. Returns (raw, score)
        per item."""
        r = self.readouts[readout_id]
        layers = self.layers_for(r)
        reads = backend.read_residual_multi_batch(items, layers, enable_thinking=False)
        out = []
        for read in reads:
            raw = r.raw(self._spans(read, layers))
            out.append((raw, r.score(raw)))
        return out

    def raw_all(self, backend, items, allow_prefix_cache=False):
        """One read at the four anchors; a dict readout id -> raw value per item."""
        layers = sorted(set(self.layers.values()))
        reads = backend.read_residual_multi_batch(items, layers, enable_thinking=False,
                                                  allow_prefix_cache=allow_prefix_cache)
        out = []
        for read in reads:
            spans = self._spans(read, layers)
            out.append({rid: r.raw(spans) for rid, r in self.readouts.items()})
        return out

    def score_all(self, backend, items, allow_prefix_cache=False):
        """Experiment 2: every readout from one read. A dict readout id -> score per item."""
        return [{rid: self.readouts[rid].score(v) for rid, v in raws.items()}
                for raws in self.raw_all(backend, items, allow_prefix_cache)]
