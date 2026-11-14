"""Estimators and bootstraps of the paper (Section "Measuring control").

They read the processed results in ``results/`` (``scripts/collect_results.py``):

* ``exp1.npz``: real and control[subject, writer, readout, objective, condition, session, turn]
  (NaN where the turn has no readout), plus the compliance status of every turn;
* ``exp2.npz``: real and control[subject, readout, objective, condition, sentence, turn], the
  control being the mean over the control sessions of the sentence;
* ``exp3.npz``/``exp4.npz``: one *unit* per (model, monitor, sentence, condition), with
  real[unit, objective, turn] and control[unit, objective, stream, turn] for its 3 control
  sessions.

All scores are in points.

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
        real, ctrl = ffill(r["real"]), ffill(r["control"])
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
        self.real = r["real"].astype(np.float64)        # [unit, objective (max, min), turn]
        self.control = r["control"].astype(np.float64)  # [unit, objective, stream, turn]
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


# ---- Experiment 2: bounds and additional analyses ---------------------------------------
#
# These analyses work on the objective difference at turn 30 per readout and sentence, in
# float64, and use the bootstraps below. Results files (scripts/collect_results.py):
#
# * ``exp2_shuffled.npz``: shuffled[model, readout, objective, condition, sentence, stream] and
#   crossed[...], the score at turn 30 of the shuffled and crossed sessions of every real session
#   of exp2.npz;
# * ``exp2_targeted.npz``: hidden criterion, real[model, target, readout, objective, sentence], every
#   readout at turn 30 of the session whose score is ``target``; control[model, readout,
#   objective, sentence], the mean of the 8 control sessions at turn 30; level[model, readout],
#   the mean score of a readout over turns 1-29 of its own sessions;
# * ``exp2_depths.npz``: hidden criterion, the 15 direction readouts at 45, 55, 75, 85 and 95% depth
#   as the score: real[model, readout, objective, condition, sentence, turn] as in exp2.npz,
#   control[..., sentence] (mean of the 8 control sessions), shuffled and crossed[..., sentence,
#   stream] at turn 30.

DIRECTION_PREFIXES = ("probe_", "pcband_", "encres_")


def direction_readouts(readouts, depth):
    """Indices of the 15 readouts defined as a direction and read at ``depth`` (e.g. "d66"), in
    stored order."""
    idx = [i for i, r in enumerate(readouts) if r.startswith(DIRECTION_PREFIXES) and r.endswith("_" + depth)]
    assert len(idx) == 15, (depth, len(idx))
    return idx


def bootstrap_sentences_readouts(o, n_boot=2000, seed=0):
    """``o`` [readout, condition, sentence]. Returns the mean and the replicates of the mean under
    two bootstraps that share their draws: resampling the sentences (jointly for all readouts
    and conditions), and resampling both the sentences and the readouts."""
    rng = np.random.default_rng(seed)
    nr, _, ns = o.shape
    b1, b2 = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, ns, ns)
        j = rng.integers(0, nr, nr)
        b1[b] = np.nanmean(o[:, :, i])
        b2[b] = np.nanmean(o[j][:, :, i])
    return float(np.nanmean(o)), b1, b2


def interval(mean, b1, b2, q=(2.5, 97.5)):
    return dict(mean=mean, sentence_ci=np.percentile(b1, q).tolist(), two_level_ci=np.percentile(b2, q).tolist())


def _objective_difference(real, ctrl):
    """[readout, objective (max, min), ...] -> (real - ctrl | max) - (real - ctrl | min)."""
    e = real - ctrl
    return e[:, 0] - e[:, 1]


def exp2_turn_objective_difference(e2: Exp2, S, readouts=None, conditions=("hidden", "disclosed"), t=-1):
    """[readout, condition, sentence] tau^max_t - tau^min_t against random scores, in float64."""
    assert e2.objectives == ["max", "min"]
    si = e2.models.index(S)
    od = _objective_difference(e2.real[si].astype(float), e2.control[si].astype(float))
    idx = list(range(len(e2.readouts))) if readouts is None else list(readouts)
    return od[idx][:, [e2.conditions.index(c) for c in conditions]][..., t]


def exp2_bound(e2: Exp2, S, n_boot=2000, seed=0):
    """Bound on tau^max_30 - tau^min_30 (all readouts, both conditions): the larger absolute end of
    the 90% interval of the bootstrap over sentences and readouts, an equivalence test at alpha
    = 0.05. Returns (mean, lo, hi, bound)."""
    m, _, b2 = bootstrap_sentences_readouts(exp2_turn_objective_difference(e2, S), n_boot, seed)
    lo, hi = np.percentile(b2, [5, 95])
    return m, float(lo), float(hi), float(max(abs(lo), abs(hi)))


class Exp2Shuffled:
    """Experiment 2 at turn 30 against three controls: the random control of the paper
    ("uniform"), the shuffled control (mean of the 3 shuffled and the 3 crossed sessions,
    "shuffled"), and the shuffled sessions alone ("own", the numbers of the session itself)."""

    def __init__(self, results_dir=RESULTS):
        r = load("exp2_shuffled", results_dir)
        self.e2 = Exp2(results_dir)
        self.models = [str(x) for x in r["subjects"]]
        self.readouts = [str(x) for x in r["readouts"]]
        assert self.readouts == self.e2.readouts
        assert [str(x) for x in r["objectives"]] == self.e2.objectives == ["max", "min"]
        assert [str(x) for x in r["conditions"]] == self.e2.conditions
        self.t = int(r["turn"]) - 1
        self.own = np.nanmean(r["shuffled"], axis=-1)      # [model, readout, objective, condition, sentence]
        self.cross = np.nanmean(r["crossed"], axis=-1)

    def arrays(self, S):
        """real, uniform, own, cross: [readout, objective, condition, sentence] at turn 30."""
        si, ei = self.models.index(S), self.e2.models.index(S)
        return (self.e2.real[ei][..., self.t].astype(float), self.e2.control[ei][..., self.t].astype(float),
                self.own[si], self.cross[si])

    def control(self, S, kind):
        real, unif, own, cross = self.arrays(S)
        return {"shuffled": (own + cross) / 2, "own": own, "uniform": unif}[kind]

    def objective_difference(self, S, kind="shuffled", readouts=None, conditions=("hidden", "disclosed")):
        """[readout, condition, sentence] tau^max_30 - tau^min_30 against the control ``kind``."""
        od = _objective_difference(self.arrays(S)[0], self.control(S, kind))
        idx = list(range(len(self.readouts))) if readouts is None else list(readouts)
        return od[idx][:, [self.e2.conditions.index(c) for c in conditions]]

    def two_by_two(self, S):
        """At turn 30, all readouts and conditions: real max - min, random-control max - min, and,
        for the shuffled sessions, the effect of the instruction with the numbers fixed (the
        shuffled control's max - min) and of the numbers with the instruction fixed."""
        real, unif, own, cross = self.arrays(S)
        sMM, smm, sMm, smM = own[:, 0], own[:, 1], cross[:, 0], cross[:, 1]
        return dict(real_max_minus_min=float(np.nanmean(real[:, 0] - real[:, 1])),
                    uniform_max_minus_min=float(np.nanmean(unif[:, 0] - unif[:, 1])),
                    instruction_effect_numbers_fixed=float(np.nanmean(((sMM - smM) + (sMm - smm)) / 2)),
                    number_effect_instruction_fixed=float(np.nanmean(((sMM - sMm) + (smM - smm)) / 2)))

    def per_objective(self, S, readouts=None):
        """Per objective, at turn 30: real - uniform and shuffled control - uniform, mean over
        ``readouts`` (default all), conditions and sentences."""
        real, unif, own, cross = self.arrays(S)
        idx = list(range(len(self.readouts))) if readouts is None else list(readouts)
        return {o: dict(real_minus_uniform=float(np.nanmean((real - unif)[idx, oi])),
                        shuffled_minus_uniform=float(np.nanmean(((own + cross) / 2 - unif)[idx, oi])))
                for oi, o in enumerate(self.e2.objectives)}


def bootstrap_readout_sentence_ci(X, rng, two_level, n_boot=2000):
    """``X`` [readout, sentence]: mean and 95% interval of the grand mean, resampling the
    sentences, and with ``two_level`` also the readouts; ``rng`` is shared across calls."""
    X = X[np.isfinite(X).any(1)][:, np.isfinite(X).any(0)]
    R, S = X.shape
    out = np.empty(n_boot)
    for b in range(n_boot):
        si = rng.integers(0, S, S)
        ri = rng.integers(0, R, R) if two_level else np.arange(R)
        out[b] = np.nanmean(X[np.ix_(ri, si)])
    return float(np.nanmean(X)), [float(x) for x in np.percentile(out, [2.5, 97.5])]


def signflip_p(v, rng, n=20000):
    """Two-sided p-value of the mean of ``v`` (one value per readout) under random sign flips."""
    v = v[np.isfinite(v)]
    obs = abs(v.mean())
    null = np.abs((rng.choice([-1.0, 1.0], (n, v.size)) * v).mean(1))
    return float((1 + (null >= obs).sum()) / (n + 1))


class Exp2Targeted:
    """Targeted or broad modulation (hidden criterion, turn 30). For readout r and sentence i,
    OD_x(r, i) is the objective difference of r in the sessions whose score is readout x:
    scored(r) = OD_r(r), other(r) = mean over x != r of OD_x(r), excess = scored - other (the
    control sessions, shared by all targets, cancel)."""

    def __init__(self, results_dir=RESULTS):
        r = load("exp2_targeted", results_dir)
        self.models = [str(x) for x in r["subjects"]]
        self.readouts = [str(x) for x in r["readouts"]]
        assert [str(x) for x in r["objectives"]] == ["max", "min"]
        self.real, self.control, self.level = r["real"], r["control"], r["level"]

    def contrasts(self, S):
        """scored, other, excess [readout, sentence], and OD [target, readout, sentence]."""
        si = self.models.index(S)
        R = len(self.readouts)
        real, ctrl = self.real[si], self.control[si]
        od = (real[:, :, 0] - real[:, :, 1]) - (ctrl[:, 0] - ctrl[:, 1])[None]
        diag = od[np.arange(R), np.arange(R)]
        other = np.nanmean(np.where(np.eye(R, dtype=bool)[:, :, None], np.nan, od), axis=0)
        return diag, other, diag - other, od

    def analysis(self, S, rng, d66):
        """Means, sentence and two-level intervals and readout sign-flip p-values, for all
        readouts and for the 15 direction readouts at 66% depth (``d66``), then the level check.
        ``rng`` is shared by the models, in the order of ``self.models``."""
        diag, other, excess, od = self.contrasts(S)
        R = len(self.readouts)
        res = {}
        for label, sel in (("all", list(range(R))), ("d66", d66)):
            res[label] = {}
            for name, X in (("scored", diag), ("other", other), ("excess", excess)):
                m, ci_s = bootstrap_readout_sentence_ci(X[sel], rng, False)
                _, ci_2 = bootstrap_readout_sentence_ci(X[sel], rng, True)
                res[label][name] = dict(mean=m, sentence_ci=ci_s, two_level_ci=ci_2,
                                        p_signflip=signflip_p(np.nanmean(X[sel], axis=1), rng))
        # Level check: targets show numbers at different distances from 50. Predict how much r moves
        # under target x from that distance (regression over the 41 other targets), evaluate it at
        # r's own distance, and subtract the prediction from what r does when it is the score.
        lev = np.abs(self.level[self.models.index(S)] - 50.0)
        od30 = np.nanmean(od, axis=2)
        adj = np.full(R, np.nan)
        for k in range(R):
            xs = [x for x in range(R) if x != k and np.isfinite(od30[x, k])]
            if len(xs) < 5 or not np.isfinite(od30[k, k]):
                continue
            b1, b0 = np.polyfit(lev[xs], od30[xs, k], 1)
            adj[k] = od30[k, k] - (b0 + b1 * lev[k])
        for label, sel in (("all", list(range(R))), ("d66", d66)):
            res[label]["excess_level_adjusted"] = dict(mean=float(np.nanmean(adj[sel])),
                                                       p_signflip=signflip_p(adj[sel], rng))
        res["per_readout"] = {rid: dict(scored=float(np.nanmean(diag[k])), other=float(np.nanmean(other[k])))
                              for k, rid in enumerate(self.readouts)}
        return res


class Exp2Depths:
    """Experiment 2 under the hidden criterion with the score read at seven depths, at turn 30:
    the 15 direction readouts at 33% and 66% (exp2.npz, exp2_shuffled.npz) and at 45, 55, 75, 85
    and 95% (exp2_depths.npz), against the random control and the shuffled control."""
    DEPTHS = ["d33", "d45", "d55", "d66", "d75", "d85", "d95"]

    def __init__(self, results_dir=RESULTS):
        r = load("exp2_depths", results_dir)
        self.shuffled = Exp2Shuffled(results_dir)
        self.models = [str(x) for x in r["subjects"]]
        self.readouts = [str(x) for x in r["readouts"]]
        assert [str(x) for x in r["conditions"]] == ["hidden"]
        self.t = int(r["turn"]) - 1
        self.real = r["real"][:, :, :, 0, :, self.t]          # [model, readout, objective, sentence]
        self.uniform = r["control"][:, :, :, 0]
        self.own = np.nanmean(r["shuffled"][:, :, :, 0], axis=-1)
        self.cross = np.nanmean(r["crossed"][:, :, :, 0], axis=-1)

    def objective_difference(self, S, depth, kind):
        """[readout, 1, sentence] tau^max_30 - tau^min_30 of the 15 direction readouts at ``depth``,
        each in the sessions where it is the score, against the control ``kind`` (uniform,
        shuffled, own)."""
        if depth in ("d33", "d66"):
            sh = self.shuffled
            return sh.objective_difference(S, kind, direction_readouts(sh.readouts, depth), ("hidden",))
        si = self.models.index(S)
        idx = direction_readouts(self.readouts, depth)
        real, unif, own, cross = (a[si][idx] for a in (self.real, self.uniform, self.own, self.cross))
        ctrl = {"shuffled": (own + cross) / 2, "own": own, "uniform": unif}[kind]
        return _objective_difference(real, ctrl)[:, None]


# ---- Experiment 1: additivity of the fit behind the diagonal excess -------------------------

def _additive_fit(T, cells):
    """Sum-to-zero fit m + a_W + b_S on the given (writer, subject) cells -> m, a, b."""
    n = T.shape[0]
    X = np.array([_design_row(w, s, n) for w, s in cells])
    y = np.array([T[w, s] for w, s in cells])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a = np.array([beta[1 + i] if i < n - 1 else -beta[1:n].sum() for i in range(n)])
    b = np.array([beta[n + j] if j < n - 1 else -beta[n:2 * n - 1].sum() for j in range(n)])
    return beta[0], a, b


def _additivity_stats(T):
    n = T.shape[0]
    off = [(w, s) for w in range(n) for s in range(n) if w != s]
    m, a, b = _additive_fit(T, off)
    loo = []
    for k, (w, s) in enumerate(off):
        mk, ak, bk = _additive_fit(T, off[:k] + off[k + 1:])
        loo.append(T[w, s] - (mk + ak[w] + bk[s]))
    diag = np.diag(T)
    others = np.array([np.mean([T[w, s] for s in range(n) if s != w]) for w in range(n)])
    return dict(b=b, loo=np.array(loo), excess=diag - np.maximum(0, m + a + b),
                excess_without_fit=diag - np.maximum(0, others))


def exp1_additivity(e1: Exp1, turn=20, n_boot=2000):
    """Test of the additive fit at a 1-indexed turn, for each objective and condition (the grid
    and the bootstrap of the diagonal excess, seed 0 per block): each pair with W != S predicted
    from the fit on the other 19, its error in units of its bootstrap standard deviation; the
    susceptibilities b_S; and the diagonal excess with the fit and without it (the effect of S on
    itself minus its average effect, as a writer, on the other four subjects, floored at 0 as in
    the diagonal excess), averaged over the four blocks, with 95% bootstrap intervals."""
    n = len(e1.models)
    out = dict(blocks={}, excess=[], excess_without_fit=[])
    reps = dict(excess=[], excess_without_fit=[])
    for obj, cond in BLOCKS:
        cols = {S: e1.column(cond, obj, S, turn - 1) for S in e1.models}
        T0 = _build_grid(cols, e1.models, None)
        s0 = _additivity_stats(T0)
        rng = np.random.default_rng(0)
        loo_b, rb = [], dict(excess=[], excess_without_fit=[])
        for _ in range(n_boot):
            sb = _additivity_stats(_build_grid(cols, e1.models, _draw(rng, cols, e1.models)))
            loo_b.append(sb["loo"])
            for k in rb:
                rb[k].append(sb[k])
        z = s0["loo"] / np.std(np.array(loo_b), axis=0, ddof=1)
        out["blocks"][(obj, cond)] = dict(loo=s0["loo"], loo_z=z, b=s0["b"])
        for k in reps:
            out[k].append(s0[k])
            reps[k].append(np.array(rb[k]))
    for k in reps:
        point = np.mean(out[k], 0)
        r = np.nanmean(np.stack(reps[k]), 0)
        out[k] = [(float(point[i]), *map(float, np.nanpercentile(r[:, i], [2.5, 97.5]))) for i in range(n)]
    return out
