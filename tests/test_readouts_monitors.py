"""Readout formulas and the released readout and monitor artifacts."""
import numpy as np
import pytest

from neurofeedback.analysis import READOUT_ORDER, family_of
from neurofeedback.models import EXP2_MODELS, MONITOR_MODELS
from neurofeedback.monitors import CONCEPTS, load_monitor
from neurofeedback.readouts import ReadoutSet, centered_cosine, curvature, step_change


def test_curvature_of_a_line_is_zero_and_of_a_zigzag_is_pi():
    t = np.arange(10, dtype=float)[:, None] * np.array([[1.0, 2.0, 0.5]])
    assert curvature(t, 1) == pytest.approx(0.0, abs=1e-6)
    zigzag = np.array([[0, 0], [1, 0], [0, 0], [1, 0], [0, 0]], dtype=float)
    assert curvature(zigzag, 1) == pytest.approx(np.pi)


def test_undefined_on_short_spans():
    H = np.random.default_rng(0).standard_normal((6, 8))
    assert np.isnan(curvature(H, 3)) and not np.isnan(curvature(H, 2))
    assert np.isnan(step_change(H[:2]))


def test_centered_cosine():
    mu, v = np.zeros(3), np.array([1.0, 0, 0])
    assert centered_cosine(np.array([[2.0, 0, 0], [0, 3.0, 0]]), mu, v) == pytest.approx(0.5, abs=1e-6)


@pytest.mark.parametrize("model", EXP2_MODELS)
@pytest.mark.parametrize("calibration", ["exp1", "exp2"])
def test_readout_artifacts(model, calibration):
    rs = ReadoutSet(model, calibration)
    assert sorted(rs.ids) == sorted(READOUT_ORDER)
    for rid, r in rs.readouts.items():
        assert r.family == family_of(rid)
        assert r.score(r.median) == pytest.approx(50.0)
        assert r.score(r.hi) - r.score(r.lo) == pytest.approx(100.0)
        if r.family != "geometry":
            assert np.linalg.norm(r.vector) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("model", MONITOR_MODELS)
def test_monitor_artifacts(model):
    for concept in CONCEPTS:
        m = load_monitor(model, concept)
        span = np.random.default_rng(0).standard_normal((1, 12, m.mu.shape[0])) * m.sd + m.mu
        assert 0.0 <= m.score(span) <= 100.0
