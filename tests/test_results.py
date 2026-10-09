"""The released results reproduce the numbers of the paper."""
import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO
from neurofeedback.analysis import Exp1, Exp2, MonitorResults, bootstrap_curve, diagonal_excess


@pytest.fixture(scope="module")
def e1():
    return Exp1()


def test_diagonal_excess_is_zero_on_an_additive_grid():
    a, b = np.array([3.0, -1, 0, 2, 1]), np.array([0.5, 1, -2, 0, 4])
    tau = 2.0 + a[:, None] + b[None, :]
    exc, exc_floor, a_hat, b_hat, rms = diagonal_excess(tau)
    assert np.allclose(exc, 0, atol=1e-9) and rms < 1e-9
    assert np.allclose(a_hat, a - a.mean()) and np.allclose(b_hat, b - b.mean())


def test_exp1_objective_difference(e1):
    """Figure "Experiment 1", observed tau^max_t - tau^min_t (writer rows, subject columns)."""
    assert np.round(e1.objective_difference(19)[0], 1).tolist() == [21.5, 36.7, 29.5, 22.9, 28.1]
    assert np.round(e1.objective_difference(19)[1], 1).tolist() == [44.4, 48.5, 37.1, 41.3, 40.7]
    assert np.round(e1.objective_difference(4)[:, 0], 1).tolist() == [5.4, 15.5, -0.3, 0.4, -0.3]


def test_exp1_self_effects(e1):
    """Text: by turn 20 the oriented effect on itself is 24 points for Gemma-4-31B, 11 for Qwen3.5-27B."""
    selfs = np.array([np.mean([e1.grid(c, o, 19)[i, i] for o, c in
                               [("max", "hidden"), ("max", "disclosed"), ("min", "hidden"), ("min", "disclosed")]])
                      for i in range(5)])
    assert np.round(selfs, 2).tolist()[:2] == [10.77, 24.27]


def test_exp2_objective_difference():
    """Text: by turn 30, 1.8 for Qwen3.5-27B, 1.1 for Qwen3.6-27B, 0.4 for Gemma-4-31B."""
    e2 = Exp2()
    end = {s: bootstrap_curve(e2.objective_difference_matrix(s))[0][-1] for s in e2.models}
    assert round(end["qwen3.5-27b"], 3) == 1.820
    assert round(end["qwen3.6-27b"], 3) == 1.070
    assert round(end["gemma-4-31b"], 3) == 0.376
    others = [v for s, v in end.items() if s not in ("qwen3.5-27b", "qwen3.6-27b", "gemma-4-31b")]
    assert max(abs(v) for v in others) < 0.2


def test_exp3_qwen35_27b_raises_the_monitors():
    """Text: about one point for Qwen3.5-27B from turn 6 to turn 20."""
    r = MonitorResults("exp3")
    m = [np.nanmean(r.objective_difference("qwen3.5-27b", t)) for t in range(20)]
    assert 0.5 < min(m[5:]) and max(m[5:]) < 1.6


def test_tables_reproduce_the_paper(tmp_path):
    subprocess.run([sys.executable, str(REPO / "scripts" / "tables.py")], check=True, capture_output=True)
    exp1 = (REPO / "tables" / "compliance_exp1.tex").read_text().splitlines()
    assert exp1[0] == "Qwen3.5-27B & 99.8 & 99.8 & 99.8 & 99.8 & 99.7 & 0.1 & 0.1 & 0.0 & 67.9 & 28.5 & 2.3 \\\\"
    assert exp1[-1] == "All writers & 96.5 & 96.6 & 96.4 & 96.9 & 96.1 & 2.9 & 0.4 & 0.2 & 74.3 & 13.6 & 1.4 \\\\"
    exp4 = (REPO / "tables" / "compliance_exp4.tex").read_text().splitlines()
    assert exp4[0] == "Qwen3.5-9B & 47.5 & 48.7 & 47.1 & 58.9 & 36.1 & 2047 \\\\"
    pre = (REPO / "tables" / "compliance_preflight.tex").read_text().splitlines()
    assert pre[-1] == "All models & 1251 & 85.9 & 86.2 \\\\"
    mon = (REPO / "tables" / "monitors.tex").read_text().splitlines()
    assert mon[0].startswith("Gemma-4-12B  & 0.79\\,{\\scriptsize[0.72,\\,0.86]}")
