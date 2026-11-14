"""The released results reproduce the numbers of the paper."""
import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO
from neurofeedback.analysis import (Exp1, Exp2, Exp2Depths, Exp2Shuffled, Exp2Targeted, MonitorResults,
                                    bootstrap_curve, bootstrap_sentences_readouts, diagonal_excess, direction_readouts,
                                    exp1_additivity, exp2_bound, exp2_turn_objective_difference, interval)


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
    assert len(others) == 8 and max(abs(v) for v in others) < 0.2
    assert round(end["qwen3.5-122b-a10b"], 2) == -0.13


def test_exp2_figure_depths():
    """Figure "Experiment 2", left: no model moves the readouts at 33% depth in the instructed
    direction, Qwen3.5-27B and Qwen3.6-27B move those at 66% by several points."""
    e2 = Exp2()
    c = {(s, d): interval(*bootstrap_sentences_readouts(
        exp2_turn_objective_difference(e2, s, direction_readouts(e2.readouts, d)))) for s in e2.models
         for d in ("d33", "d66")}
    assert all(c[(s, "d33")]["sentence_ci"][0] < 0 for s in e2.models)
    assert round(c[("qwen3.5-27b", "d66")]["mean"], 2) == 5.31 and round(c[("qwen3.6-27b", "d66")]["mean"], 2) == 2.81


def test_exp2_bounds():
    """Appendix, bounds on the effect, and the Discussion: 0.2 to 0.4 points for six of the eight."""
    e2 = Exp2()
    quoted = {"qwen3-32b": 0.20, "gemma-4-12b": 0.28, "qwen3.5-122b-a10b": 0.35, "qwen3-14b": 0.37,
              "gpt-oss-20b": 0.37, "qwen3.5-9b": 0.39, "olmo-3.1-32b": 0.72, "mistral-small-3.2-24b": 1.37}
    assert {s: round(exp2_bound(e2, s)[3], 2) for s in quoted} == quoted


def test_exp2_shuffled_control():
    """Appendix, shuffled control: about half the effect of the two Qwen models remains, none of Gemma-4-31B's;
    in Qwen3.5-27B the control sessions are 0.89 points apart with random scores, 1.79 with shuffled ones."""
    sh = Exp2Shuffled()
    od = {(s, k): interval(*bootstrap_sentences_readouts(sh.objective_difference(s, k)))
          for s in sh.models for k in ("shuffled", "uniform")}
    for s in ("qwen3.5-27b", "qwen3.6-27b"):
        assert od[(s, "shuffled")]["sentence_ci"][0] > 0
        assert 0.45 < od[(s, "shuffled")]["mean"] / od[(s, "uniform")]["mean"] < 0.55
    assert od[("gemma-4-31b", "shuffled")]["sentence_ci"][0] < 0 < od[("gemma-4-31b", "shuffled")]["sentence_ci"][1]
    t = sh.two_by_two("qwen3.5-27b")
    assert round(t["uniform_max_minus_min"], 2) == 0.89 and round(t["instruction_effect_numbers_fixed"], 2) == 1.79


def test_exp2_targeted():
    """Appendix, targeted or broad: in Qwen3.5-27B a readout moves by 1.7 points when it is the score and by
    -0.1 when another readout is (p = 0.008)."""
    tg = Exp2Targeted()
    res = tg.analysis("qwen3.5-27b", np.random.default_rng(0), direction_readouts(tg.readouts, "d66"))["all"]
    assert (round(res["scored"]["mean"], 1), round(res["other"]["mean"], 1)) == (1.7, -0.1)
    assert round(res["excess"]["p_signflip"], 3) == 0.008


def test_exp2_depths():
    """Appendix, read depths: in Qwen3.5-27B the effect against random scores is significant (both intervals)
    when the score is read at 55, 66 and 85% depth."""
    dp = Exp2Depths()
    sig = [d for d in dp.DEPTHS
           if interval(*bootstrap_sentences_readouts(dp.objective_difference("qwen3.5-27b", d, "uniform")))
           ["two_level_ci"][0] > 0]
    assert sig == ["d55", "d66", "d85"]


def test_exp1_additivity():
    """Appendix, additivity: 1 of the 80 held-out errors exceeds twice its standard deviation, and every b_S
    is within 2 points of zero."""
    r = exp1_additivity(Exp1())
    z = np.concatenate([b["loo_z"] for b in r["blocks"].values()])
    assert z.size == 80 and int((np.abs(z) > 2).sum()) == 1
    assert max(np.abs(b["b"]).max() for b in r["blocks"].values()) < 2


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
    add = (REPO / "tables" / "additivity_exp1.tex").read_text().splitlines()
    assert add[0] == "Qwen3.5-27B & $-$3.7\\,{\\scriptsize[$-$8.6,\\,2.5]} & $-$3.9\\,{\\scriptsize[$-$9.4,\\,3.1]} \\\\"
