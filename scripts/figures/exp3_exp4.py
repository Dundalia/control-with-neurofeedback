#!/usr/bin/env python
"""Figures of Experiments 3 and 4 (activation monitors), from results/exp3.npz and exp4.npz.

  figures/exp34_results.pdf    Figure "Experiments 3 and 4": objective difference against the turn
  figures/exp34_effects.pdf    appendix: control effect at the last turn per model and objective
  figures/exp3_monitors.pdf    appendix: Experiment 3, objective difference per monitor
  figures/exp4_monitors.pdf    appendix: Experiment 4, objective difference per monitor
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from neurofeedback.analysis import RESULTS, MonitorResults, bootstrap_mean  # noqa: E402
from style import (C_DISCLOSED, C_HIDDEN, DISPLAY, FIGURE_ORDER, GRID, LINE_COLOR, ZERO,  # noqa: E402
                   apply_style, clean_axes, end_labels, save)

OUT = REPO / "figures"
N_BOOT = 2000
MONITOR_NAME = {"truth": "truthfulness", "deception": "deceptiveness", "sentiment": "sentiment",
                "emotion": "emotion", "morality": "morality", "formality": "formality"}
MONITOR_COLOR = {"truth": "#1f5fa8", "deception": "#a8201a", "sentiment": "#0f8b8d",
                 "emotion": "#e39a1f", "morality": "#7b4ea3", "formality": "#8a6d3b"}
TITLE = {3: "Experiment 3 (fixed sentence)", 4: "Experiment 4 (reasoning, then fixed sentence)"}


def models_of(res):
    return [m for m in FIGURE_ORDER if m in res.models]


def turn_curve(res, model, monitor=None, condition=None, nanmean=True):
    """Objective difference per turn, units resampled jointly across turns."""
    H = res.horizon
    M = np.array([res.objective_difference(model, t, monitor, condition) for t in range(H)])  # [H, unit]
    n = M.shape[1]
    assert n > 0, (model, monitor, condition)
    rng = np.random.default_rng(0)
    mean_fn = np.nanmean if nanmean else np.mean
    pct_fn = np.nanpercentile if nanmean else np.percentile
    reps = np.empty((N_BOOT, H))
    for b in range(N_BOOT):
        reps[b] = mean_fn(M[:, rng.integers(0, n, n)], axis=1)
    return mean_fn(M, axis=1), pct_fn(reps, 2.5, axis=0), pct_fn(reps, 97.5, axis=0)


def turn_panel(ax, res):
    H = res.horizon
    turns = np.arange(1, H + 1)
    ends = []
    for s in models_of(res):
        mean, lo, hi = turn_curve(res, s)
        main = s == "qwen3.5-27b"
        ax.fill_between(turns, lo, hi, color=LINE_COLOR[s], alpha=0.13, lw=0, zorder=1)
        ax.plot(turns, mean, "-", color=LINE_COLOR[s], lw=2.4 if main else 1.8, zorder=3 if main else 2)
        ends.append((s, mean[-1]))
    yl = ax.get_ylim()
    end_labels(ax, ends, H, 0.045 * (yl[1] - yl[0]), LINE_COLOR, DISPLAY, dx_line=0.45, dx_text=0.6)
    clean_axes(ax)
    ax.axhline(0, color=ZERO, lw=2.0, ls=":")
    ax.set_xlim(1, H)
    ax.set_xticks([1, 5, 10, 15, 20] if H == 20 else range(1, H + 1))
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.tick_params(labelsize=16, length=0)
    ax.set_xlabel("turn", fontsize=18)
    ax.set_ylabel("$\\tau^{\\max}_t - \\tau^{\\min}_t$", fontsize=22)


def effects_panel(ax, res):
    H = res.horizon
    rows = [(s, o) for s in models_of(res) for o in ("max", "min")]
    ys = np.arange(len(rows))[::-1]
    for y, (s, o) in zip(ys, rows):
        for cond, col, dy in (("hidden", C_HIDDEN, +0.16), ("disclosed", C_DISCLOSED, -0.16)):
            m, lo, hi = bootstrap_mean(res.control_effect(s, H - 1, o, cond), N_BOOT)
            ax.plot([lo, hi], [y + dy, y + dy], color=col, lw=2.2, zorder=2)
            ax.plot(m, y + dy, "o", color=col, ms=7, zorder=3,
                    mfc=col if cond == "hidden" else "white", mec=col, mew=1.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{DISPLAY[s]} ({o})" for s, o in rows], fontsize=13)
    for i in range(0, len(rows), 2):
        ax.axhline(ys[i] - 1.0, color=GRID, lw=0.6)
    clean_axes(ax)
    ax.axvline(0, color=ZERO, lw=2.0, ls=":")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.yaxis.grid(False)
    ax.tick_params(labelsize=16, length=0)
    ax.set_xlabel(f"control effect  $\\tau^{{o}}_{{{H}}}$", fontsize=18)
    ax.legend(handles=[Line2D([], [], marker="o", color=C_HIDDEN, mfc=C_HIDDEN, ls="-", ms=7, label="no disclosure"),
                       Line2D([], [], marker="o", color=C_DISCLOSED, mfc="white", ls="-", ms=7,
                              label="disclosure")],
              fontsize=15, loc="lower left", framealpha=0.95)


def monitors_figure(res, exp):
    H = res.horizon
    turns = np.arange(1, H + 1)
    models = models_of(res)
    fig, axes = plt.subplots(len(models), 2, figsize=(13, 22), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.15, right=0.985, top=0.945, bottom=0.075, wspace=0.05, hspace=0.08)
    for r, s in enumerate(models):
        for c, (cond, cond_name) in enumerate((("hidden", "hidden"), ("disclosed", "disclosed"))):
            ax = axes[r][c]
            for mon in res.monitors:
                m, lo, hi = turn_curve(res, s, mon, cond, nanmean=False)
                ax.fill_between(turns, lo, hi, color=MONITOR_COLOR[mon], alpha=0.12, lw=0, zorder=1)
                ax.plot(turns, m, "-", color=MONITOR_COLOR[mon], lw=2.0, zorder=2)
            clean_axes(ax)
            ax.axhline(0, color=ZERO, lw=1.6, ls=":")
            ax.set_xlim(1, H)
            ax.set_xticks([1, 5, 10, 15, 20] if H == 20 else [1, 4, 7, 10])
            ax.grid(True, axis="y", color=GRID, lw=0.8)
            ax.xaxis.grid(False)
            ax.tick_params(labelsize=15, length=0)
            if r == 0:
                ax.set_title(cond_name, fontsize=20)
            if c == 0:
                ax.set_ylabel(DISPLAY[s], fontsize=17)
    lim = 0.9 * max(abs(v) for v in axes[0][0].get_ylim())
    step = next(q for q in (1, 2, 2.5, 5, 10, 12.5, 25, 50) if 2 * q >= lim)
    for row in axes:
        for ax in row:
            ax.set_ylim(-2 * step * 1.1, 2 * step * 1.1)
            ax.set_yticks([-2 * step, -step, 0, step, 2 * step])
    x0, x1 = axes[0][0].get_position().x0, axes[0][1].get_position().x1
    y0, y1 = axes[-1][0].get_position().y0, axes[0][0].get_position().y1
    xc = (x0 + x1) / 2
    fig.text(xc, 0.048, "turn", ha="center", va="center", fontsize=20)
    fig.text(0.058, (y0 + y1) / 2, "$\\tau_t^{\\max} - \\tau_t^{\\min}$", rotation=90,
             ha="center", va="center", fontsize=24)
    handles = [Line2D([], [], color=MONITOR_COLOR[k], lw=2.6, label=MONITOR_NAME[k]) for k in res.monitors]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(xc, 0.008), ncol=3, frameon=False,
               fontsize=18, handlelength=1.4, columnspacing=1.0)
    fig.suptitle(f"{TITLE[exp]}\n$\\tau^{{\\max}}_t - \\tau^{{\\min}}_t$ per monitor", x=xc, y=0.998, fontsize=22)
    save(fig, OUT / f"exp{exp}_monitors.pdf", bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(RESULTS), help="directory of exp3.npz and exp4.npz (default: results/)")
    a = ap.parse_args()
    apply_style()
    OUT.mkdir(exist_ok=True)
    r3, r4 = MonitorResults("exp3", a.results), MonitorResults("exp4", a.results)

    fig, (a3, a4) = plt.subplots(1, 2, figsize=(17, 6.0), constrained_layout=True)
    turn_panel(a3, r3)
    turn_panel(a4, r4)
    a3.set_title(TITLE[3], fontsize=21)
    a4.set_title(TITLE[4], fontsize=21)
    save(fig, OUT / "exp34_results.pdf")

    fig, (a3, a4) = plt.subplots(1, 2, figsize=(18, 8.5), constrained_layout=True)
    effects_panel(a3, r3)
    effects_panel(a4, r4)
    a3.set_xlim(-5, 5)
    a4.set_xlim(-8, 8)
    a4.set_xticks(range(-8, 9, 4))
    a3.set_title(f"{TITLE[3]}\ncontrol effect  $\\tau^{{o}}_{{20}}$", fontsize=20)
    a4.set_title(f"{TITLE[4]}\ncontrol effect  $\\tau^{{o}}_{{10}}$", fontsize=20)
    save(fig, OUT / "exp34_effects.pdf")

    monitors_figure(r3, 3)
    monitors_figure(r4, 4)
    print("wrote", *(OUT / f for f in ("exp34_results.pdf", "exp34_effects.pdf",
                                        "exp3_monitors.pdf", "exp4_monitors.pdf")))


if __name__ == "__main__":
    main()
