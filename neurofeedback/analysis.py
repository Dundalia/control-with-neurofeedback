"""Estimators and bootstraps of the paper (Section "Measuring control").

They read the processed results in ``results/`` (``scripts/collect_results.py``):

* ``exp1.npz``: scores[subject, writer, readout, objective, condition, arm, session, turn]
  (NaN where the turn has no readout), plus the compliance status of every turn;
* ``exp2.npz``: real and control[subject, readout, objective, condition, sentence, turn], the
  control being the mean over the control sessions of the sentence;
* ``exp3.npz``/``exp4.npz``: one *unit* per (model, monitor, sentence, condition), with the
  real session and the 3 control sessions of each objective, in milli-points.

Readouts and units are stored in the order the bootstraps resample them, so the figures
reproduce the paper's intervals exactly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
SIGN = {"max": 1.0, "min": -1.0}
# (objective, condition) blocks, in the order the bootstraps visit them.
BLOCKS = [("max", "hidden"), ("max", "disclosed"), ("min", "hidden"), ("min", "disclosed")]
# Order in which the readouts are stored in results/ (the bootstraps resample readouts by index).
READOUT_ORDER = [
    "probe_morality_d33", "probe_morality_d66", "probe_sentiment_d33", "probe_sentiment_d66",
    "probe_truth_d33", "probe_truth_d66",
    "pcband_1000_3000_1_d33", "pcband_1000_3000_1_d66", "pcband_1000_3000_2_d33", "pcband_1000_3000_2_d66",
    "pcband_200_1000_1_d33", "pcband_200_1000_1_d66", "pcband_200_1000_2_d33", "pcband_200_1000_2_d66",
    "pcband_20_200_1_d33", "pcband_20_200_1_d66", "pcband_20_200_2_d33", "pcband_20_200_2_d66",
    "encres_rand1_d33", "encres_rand1_d66", "encres_rand2_d33", "encres_rand2_d66",
    "encres_rand3_d33", "encres_rand3_d66", "encres_rand4_d33", "encres_rand4_d66",
    "encres_pc1_d33", "encres_pc1_d66", "encres_pc2_d33", "encres_pc2_d66",
    "step_change_d33", "step_change_d66", "curvature_step1_d33", "curvature_step1_d66",
    "curvature_step2_d33", "curvature_step2_d66", "curvature_step3_d33", "curvature_step3_d66",
    "curvature_diff_44_33", "curvature_diff_55_44", "curvature_diff_66_33", "curvature_diff_66_55",
]


def family_of(readout_id: str) -> str:
    for fam in ("probe", "pcband", "encres"):
        if readout_id.startswith(fam + "_"):
            return fam
    return "geometry"


def load(name: str, results_dir=RESULTS) -> dict:
    z = np.load(Path(results_dir) / f"{name}.npz")
    return {k: z[k] for k in z.files}


# ---- Experiment 1 -----------------------------------------------------------------------

def ffill(x):
    """Forward-fill NaNs along the last axis (a turn without readout keeps the last value)."""
    x = np.array(x, dtype=np.float64)
    last = np.full(x.shape[:-1], np.nan)
    for t in range(x.shape[-1]):
        v = x[..., t]
        last = np.where(np.isnan(v), last, v)
        x[..., t] = last
    return x


class Exp1:
    """Oriented control effects s_o (y_real - y_ctrl) of every session of Experiment 1, with
    missing readouts forward-filled within each session. ``E[(condition, objective, S)]`` is a
    dict readout -> {writer: float32[session, turn]}."""

    def __init__(self, results_dir=RESULTS):
        r = load("exp1", results_dir)
        self.models = [str(x) for x in r["subjects"]]
        self.readouts = [str(x) for x in r["readouts"]]
        objectives = [str(x) for x in r["objectives"]]
        conditions = [str(x) for x in r["conditions"]]
        self.status, self.repeat = r["status"], r["repeat"]
        real = ffill(r["scores"][..., 0, :, :])
        ctrl = ffill(r["scores"][..., 1, :, :])
        self.E = {}
        for si, S in enumerate(self.models):
            for oi, o in enumerate(objectives):
                for ci, c in enumerate(conditions):
                    self.E[(c, o, S)] = {
                        rid: {W: (SIGN[o] * (real[si, wi, ri, oi, ci] - ctrl[si, wi, ri, oi, ci])
                                  ).astype(np.float32) for wi, W in enumerate(self.models)}
                        for ri, rid in enumerate(self.readouts)}

    def readouts_in(self, family):
        return [r for r in self.readouts if family in (None, "all") or family_of(r) == family]

    def column(self, cond, obj, S, t, family=None):
        """[readout, writer, session] effects at turn index t for subject S."""
        col = self.E[(cond, obj, S)]
        rids = self.readouts_in(family)
        arr = np.full((len(rids), len(self.models), 20), np.nan)
        for ri, rid in enumerate(rids):
            for wi, W in enumerate(self.models):
                arr[ri, wi] = col[rid][W][:, t]
        return arr

    def grid(self, cond, obj, t, family=None):
        """[writer, subject] oriented control effect at turn index t, pooled over readouts and
        sessions."""
        V = np.full((5, 5), np.nan)
        for sj, S in enumerate(self.models):
            col = self.E[(cond, obj, S)]
            for wi, W in enumerate(self.models):
                d = np.concatenate([col[r][W][:, t] for r in self.readouts_in(family)])
                d = np.asarray(d, float)
                V[wi, sj] = d[~np.isnan(d)].mean()
        return V

    def objective_difference(self, t):
        """[writer, subject] tau^max_t - tau^min_t, averaged over the two conditions."""
        g = {(c, o): self.grid(c, o, t) for o, c in BLOCKS}
        return 0.5 * ((g[("hidden", "max")] + g[("hidden", "min")])
                      + (g[("disclosed", "max")] + g[("disclosed", "min")]))


def _design_row(i, j, n=5):
    x = np.zeros(1 + 2 * (n - 1))
    x[0] = 1
    if i < n - 1:
        x[1 + i] = 1
    else:
        x[1:n] -= 1
    if j < n - 1:
        x[n + j] = 1
    else:
        x[n:2 * n - 1] -= 1
    return x


def diagonal_excess(tau):
    """Fit tau[W, S] ~ m + a_W + b_S on the off-diagonal pairs (sum-to-zero a and b) and return
    (excess, excess with the prediction floored at 0, a, b, rms)."""
    n = tau.shape[0]
    X, y = [], []
    for i in range(n):
        for j in range(n):
            if i != j and not np.isnan(tau[i, j]):
                X.append(_design_row(i, j, n))
                y.append(tau[i, j])
    X, y = np.asarray(X), np.asarray(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rms = float(np.sqrt(np.mean((y - X @ beta) ** 2)))
    a = np.array([beta[1 + i] if i < n - 1 else -beta[1:n].sum() for i in range(n)])
    b = np.array([beta[n + j] if j < n - 1 else -beta[n:2 * n - 1].sum() for j in range(n)])
    pred = np.array([beta[0] + a[i] + b[i] for i in range(n)])
    exc = np.array([tau[i, i] - pred[i] if not np.isnan(tau[i, i]) else np.nan for i in range(n)])
    exc_floor = np.array([tau[i, i] - max(0.0, pred[i]) if not np.isnan(tau[i, i]) else np.nan
                          for i in range(n)])
    return exc, exc_floor, a, b, rms


def additive_prediction(tau):
    """Predictions of m + a_W + b_S for all pairs, fit on the off-diagonal pairs."""
    n = tau.shape[0]
    rows, y = [], []
    for w in range(n):
        for s in range(n):
            if w != s:
                x = np.zeros(1 + 2 * n)
                x[0] = x[1 + w] = x[1 + n + s] = 1.0
                rows.append(x)
                y.append(tau[w, s])
    beta, *_ = np.linalg.lstsq(np.array(rows), np.array(y), rcond=None)
    return np.array([[beta[0] + beta[1 + w] + beta[1 + n + s] for s in range(n)] for w in range(n)])


def _build_grid(cols, models, pick):
    """[writer, subject] grid from per-subject [readout, writer, session] arrays, optionally
    resampled: pick[S] = (readout indices, session indices)."""
    n = len(models)
    T = np.full((n, n), np.nan)
    with np.errstate(invalid="ignore"):
        for sj, S in enumerate(models):
            a = cols[S]
            sub = a if pick is None else a[pick[S][0]][:, :, pick[S][1]]
            T[:, sj] = np.nanmean(sub, axis=(0, 2))
    return T


def _draw(rng, cols, models):
    """One resample: readouts with replacement, then the 20 session indices (shared by all
    writers, so that a session stays paired with its control session)."""
    return {S: (rng.integers(0, cols[S].shape[0], cols[S].shape[0]), rng.integers(0, 20, 20))
            for S in models}


def exp1_self_bootstrap(e1: Exp1, n_boot=1000, horizon=20):
    """Per subject and turn: oriented control effect on itself and diagonal excess (prediction
    floored at 0), averaged over the four blocks, with 95% bootstrap intervals."""
    n = len(e1.models)
    out = {k: np.full((n, horizon), np.nan) for k in
           ("tau_m", "tau_lo", "tau_hi", "exc_m", "exc_lo", "exc_hi")}
    for t in range(horizon):
        tau_pts, exc_pts = np.full((4, n), np.nan), np.full((4, n), np.nan)
        tau_reps, exc_reps = np.full((4, n_boot, n), np.nan), np.full((4, n_boot, n), np.nan)
        for bi, (obj, cond) in enumerate(BLOCKS):
            cols = {S: e1.column(cond, obj, S, t) for S in e1.models}
            T0 = _build_grid(cols, e1.models, None)
            tau_pts[bi] = np.diag(T0)
            exc_pts[bi] = diagonal_excess(T0)[1]
            rng = np.random.default_rng(0)
            for it in range(n_boot):
                T = _build_grid(cols, e1.models, _draw(rng, cols, e1.models))
                tau_reps[bi, it] = np.diag(T)
                exc_reps[bi, it] = diagonal_excess(T)[1]
        out["tau_m"][:, t] = np.nanmean(tau_pts, axis=0)
        out["exc_m"][:, t] = np.nanmean(exc_pts, axis=0)
        tp, ep = np.nanmean(tau_reps, axis=0), np.nanmean(exc_reps, axis=0)
        out["tau_lo"][:, t], out["tau_hi"][:, t] = np.nanpercentile(tp, [2.5, 97.5], axis=0)
        out["exc_lo"][:, t], out["exc_hi"][:, t] = np.nanpercentile(ep, [2.5, 97.5], axis=0)
    return out


def exp1_grid(e1: Exp1, t, family=None):
    """[block, writer, subject] oriented control effect at turn index t."""
    return np.stack([_build_grid({S: e1.column(cond, obj, S, t, family) for S in e1.models},
                                 e1.models, None) for obj, cond in BLOCKS])


def exp1_objective_difference_ci(e1: Exp1, turn, n_boot=2000):
    """[writer, subject] 95% interval of tau^max - tau^min at a 1-indexed turn, averaged over
    conditions: readouts resampled, then sessions within each resampled readout."""
    lo = np.full((5, 5), np.nan)
    hi = lo.copy()
    for wi, W in enumerate(e1.models):
        for sj, S in enumerate(e1.models):
            arms = [[e1.E[(c, o, S)][r][W][:, turn - 1] for r in e1.readouts] for o, c in BLOCKS]
            R = min(len(a) for a in arms)
            rng = np.random.default_rng(1000 * turn + 10 * wi + sj)
            ri = rng.integers(0, R, (n_boot, R))
            tot = np.zeros(n_boot)
            for a in arms:
                A = np.array([np.pad(x, (0, 20 - len(x)), constant_values=np.nan)[:20] for x in a[:R]])
                si = rng.integers(0, 20, (n_boot, R, 20))
                tot += np.nanmean(A[ri[:, :, None], si].reshape(n_boot, -1), axis=1)
            lo[wi, sj], hi[wi, sj] = np.percentile(0.5 * tot, [2.5, 97.5])
    return lo, hi


# ---- Experiment 2 -----------------------------------------------------------------------

class Exp2:
    def __init__(self, results_dir=RESULTS):
        r = load("exp2", results_dir)
        self.models = [str(x) for x in r["subjects"]]
        self.readouts = [str(x) for x in r["readouts"]]
        self.objectives = [str(x) for x in r["objectives"]]
        self.conditions = [str(x) for x in r["conditions"]]
        self.real, self.control = r["real"], r["control"]

    def oriented(self, S, cond, obj):
        """[readout, sentence, turn] s_o (y_real - y_ctrl)."""
        si, oi, ci = self.models.index(S), self.objectives.index(obj), self.conditions.index(cond)
        return SIGN[obj] * (self.real[si, :, oi, ci] - self.control[si, :, oi, ci])

    def per_sentence(self, S, cond, obj, t):
        """[sentence] oriented effect at turn index t, averaged over readouts."""
        return np.nanmean(self.oriented(S, cond, obj)[:, :, t], axis=0)

    def objective_difference_matrix(self, S):
        """[sentence x condition, turn] tau^max - tau^min per sentence, averaged over readouts;
        rows are the 40 sentences under the hidden criterion, then under disclosure."""
        mats = []
        for cond in ("hidden", "disclosed"):
            blocks = {obj: np.nanmean(self.oriented(S, cond, obj), axis=0) for obj in ("max", "min")}
            mats.append(blocks["max"] + blocks["min"])
        return np.vstack(mats)

    def readout_vector(self, S, readout):
        """[sentence] oriented effect at the last turn for one readout, averaged over the two
        objectives and the two conditions."""
        ri = self.readouts.index(readout)
        parts = [self.oriented(S, c, o)[ri, :, -1] for c in ("hidden", "disclosed") for o in ("max", "min")]
        return np.nanmean(np.stack(parts), axis=0)


def bootstrap_mean(vec, n_boot=2000, seed=0):
    """(mean, 2.5th, 97.5th percentile) of the mean, resampling the entries of ``vec``."""
    rng = np.random.default_rng(seed)
    n = len(vec)
    bs = np.array([np.nanmean(vec[rng.integers(0, n, n)]) for _ in range(n_boot)])
    return float(np.nanmean(vec)), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def bootstrap_curve(mat, n_boot=2000, seed=0):
    """``mat`` [unit, turn]; units are resampled jointly across turns. Returns (mean, lo, hi)
    per turn."""
    rng = np.random.default_rng(seed)
    S, H = mat.shape
    reps = np.empty((n_boot, H))
    for b in range(n_boot):
        reps[b] = np.nanmean(mat[rng.integers(0, S, S)], axis=0)
    return np.nanmean(mat, 0), np.nanpercentile(reps, 2.5, 0), np.nanpercentile(reps, 97.5, 0)


# ---- Experiments 3 and 4 ----------------------------------------------------------------

class MonitorResults:
    """Units (model, monitor, sentence, condition) of Experiment 3 or 4."""

    def __init__(self, name, results_dir=RESULTS):
        r = load(name, results_dir)
        self.r = r
        self.models = [str(x) for x in r["models"]]
        self.monitors = [str(x) for x in r["monitors"]]
        self.unit_model = r["unit_model"].astype(str)
        self.unit_monitor = r["unit_monitor"].astype(str)
        self.unit_condition = r["unit_condition"].astype(str)
        self.real = r["real_milli"] / 1000.0         # [unit, objective (max, min), turn]
        self.control = r["control_milli"] / 1000.0   # [unit, objective, stream, turn]
        self.horizon = self.real.shape[-1]

    def units(self, model, monitor=None, condition=None):
        m = self.unit_model == model
        if monitor:
            m &= self.unit_monitor == monitor
        if condition:
            m &= self.unit_condition == condition
        return np.flatnonzero(m)

    def _effect(self, u, oi, t):
        return float(self.real[u, oi, t]) - float(np.mean([float(x) for x in self.control[u, oi, :, t]]))

    def objective_difference(self, model, t, monitor=None, condition=None):
        """Per unit: (y_real - y_ctrl | max) - (y_real - y_ctrl | min) at turn index t."""
        return np.array([self._effect(u, 0, t) - self._effect(u, 1, t)
                         for u in self.units(model, monitor, condition)])

    def control_effect(self, model, t, obj, condition=None):
        """Per unit: y_real - y_ctrl for one objective at turn index t (not oriented)."""
        oi = 0 if obj == "max" else 1
        return np.array([self._effect(u, oi, t) for u in self.units(model, condition=condition)])
