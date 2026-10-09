#!/usr/bin/env python
"""Figures of Experiment 2 (fixed sentence), from results/exp2.npz.

  figures/exp2_results.pdf    Figure "Experiment 2": objective difference against the turn
  figures/exp2_effects.pdf    appendix: control effect at turn 30 per model and objective
  figures/exp2_readouts.pdf   appendix: objective difference at turn 30 per readout
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from neurofeedback.analysis import SIGN, Exp2, bootstrap_curve, bootstrap_mean  # noqa: E402
from style import (C_DISCLOSED, C_HIDDEN, DISPLAY, FIGURE_ORDER, GRID, LINE_COLOR, ZERO,  # noqa: E402
                   apply_style, clean_axes, end_labels, save)

OUT = REPO / "figures"
H = 30
DEPTH_COLOR = {"d33": "#0f8b8d", "d66": "#ec9a29", "x": "#6c6f7f"}
DEPTH_LABEL = {"d33": "33% depth", "d66": "66% depth", "x": "between depths"}
READOUT_MODELS = ["qwen3.5-27b", "qwen3.6-27b", "gemma-4-31b", "qwen3-32b"]
# (family, [(readout id without depth, name)]), in the order of the paper.
READOUT_ROWS = [
    ("Concept probes", [("probe_sentiment", "Sentiment probe"), ("probe_morality", "Morality probe"),
                        ("probe_truth", "Truthfulness probe")]),
    ("Principal-component bands", [
        ("pcband_20_200_1", "PCs 20 to 200, direction 1"), ("pcband_20_200_2", "PCs 20 to 200, direction 2"),
        ("pcband_200_1000_1", "PCs 200 to 1000, direction 1"), ("pcband_200_1000_2", "PCs 200 to 1000, direction 2"),
        ("pcband_1000_3000_1", "PCs 1000 to 3000, direction 1"),
        ("pcband_1000_3000_2", "PCs 1000 to 3000, direction 2")]),
    ("Encoder-residual directions", [
        ("encres_pc1", "Residual PC 1"), ("encres_pc2", "Residual PC 2"),
        ("encres_rand1", "Random residual direction 1"), ("encres_rand2", "Random residual direction 2"),
        ("encres_rand3", "Random residual direction 3"), ("encres_rand4", "Random residual direction 4")]),
    ("Trajectory geometry", [
        ("curvature_step1", "Curvature"), ("curvature_step2", "Curvature, 2-token steps"),
        ("curvature_step3", "Curvature, 3-token steps"), ("step_change", "Step change"),
        ("curvature_diff_44_33", "Curvature change, 33% to 44%"),
        ("curvature_diff_55_44", "Curvature change, 44% to 55%"),
        ("curvature_diff_66_55", "Curvature change, 55% to 66%"),
        ("curvature_diff_66_33", "Curvature change, 33% to 66%")]),
]


def results_figure(e2):
    fig, ax = plt.subplots(figsize=(8.6, 6.2), constrained_layout=True)
    turns = np.arange(1, H + 1)
    ends = []
    for s in FIGURE_ORDER:
        mean, lo, hi = bootstrap_curve(e2.objective_difference_matrix(s))
        main = s == "qwen3.5-27b"
        ax.fill_between(turns, lo, hi, color=LINE_COLOR[s], alpha=0.13, lw=0, zorder=1)
        ax.plot(turns, mean, "-", color=LINE_COLOR[s], lw=2.4 if main else 1.8, zorder=3 if main else 2)
        ends.append((s, mean[-1]))
        print(f"  {s:22s} tau_max - tau_min at t=30: {mean[-1]:+.3f} [{lo[-1]:+.3f}, {hi[-1]:+.3f}]")
    vals = [v for _, v in ends]
    gap = max(max(max(vals) - min(vals), 0.6) / 12, 0.09)
    end_labels(ax, ends, H, gap, LINE_COLOR, DISPLAY)
    clean_axes(ax)
    ax.axhline(0, color=ZERO, lw=2.0, ls=":")
    ax.set_xlim(1, H)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.tick_params(labelsize=16, length=0)
    ax.set_xlabel("turn", fontsize=18)
    ax.set_ylabel("$\\tau^{\\max}_t - \\tau^{\\min}_t$", fontsize=24)
    ax.set_title("$\\tau^{\\max}_t - \\tau^{\\min}_t$ vs turn", fontsize=27)
    save(fig, OUT / "exp2_results.pdf")


def effects_figure(e2):
    fig, ax = plt.subplots(figsize=(9, 10), constrained_layout=True)
    rows = [(s, o) for s in FIGURE_ORDER for o in ("max", "min")]
    ys = np.arange(len(rows))[::-1]
    for y, (s, o) in zip(ys, rows):
        for cond, col, dy in (("hidden", C_HIDDEN, +0.16), ("disclosed", C_DISCLOSED, -0.16)):
            v = SIGN[o] * e2.per_sentence(s, cond, o, -1)     # raw y_real - y_ctrl
            m, lo, hi = bootstrap_mean(v)
            ax.plot([lo, hi], [y + dy, y + dy], color=col, lw=2.2, zorder=2)
            ax.plot(m, y + dy, "o", color=col, ms=7, zorder=3,
                    mfc=col if cond == "hidden" else "white", mec=col, mew=1.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{DISPLAY[s]} ({o})" for s, o in rows], fontsize=13)
    for i in range(0, len(rows), 2):
        ax.axhline(ys[i] - 1.0, color=GRID, lw=0.6)
    clean_axes(ax)
    ax.axvline(0, color=ZERO, lw=2.0, ls=":")
    ax.set_xlim(-2, 2)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.yaxis.grid(False)
    ax.tick_params(labelsize=16, length=0)
    ax.set_xlabel("control effect  $\\tau^{o}_{30}$", fontsize=18)
    ax.set_title("control effect  $\\tau^{o}_{30}$\nper model and objective", fontsize=20)
    ax.legend(handles=[Line2D([], [], marker="o", color=C_HIDDEN, mfc=C_HIDDEN, ls="-", ms=7, label="no disclosure"),
                       Line2D([], [], marker="o", color=C_DISCLOSED, mfc="white", ls="-", ms=7,
                              label="disclosure")],
              fontsize=15, loc="upper left", framealpha=0.95)
    save(fig, OUT / "exp2_effects.pdf")


def _depth_ids(readouts, base):
    if base + "_d33" in readouts:
        return [(base + "_d33", "d33"), (base + "_d66", "d66")]
    assert base in readouts, base
    return [(base, "x")]


def readouts_figure(e2):
    rows, y = [], 0.0
    for fam, items in READOUT_ROWS:
        rows.append((y, "fam", fam, None))
        y -= 1.0
        for base, name in items:
            rows.append((y, "ro", name, base))
            y -= 1.0
        y -= 0.4
    n_listed = sum(len(_depth_ids(e2.readouts, b)) for _, k, _, b in rows if k == "ro")
    assert n_listed == len(e2.readouts) == 42
    est = {}
    for s in READOUT_MODELS:
        for _, kind, _, base in rows:
            if kind == "ro":
                for rid, _ in _depth_ids(e2.readouts, base):
                    # twice the mean of the four oriented effects = tau^max - tau^min, averaged over conditions
                    est[(s, rid)] = tuple(2.0 * x for x in bootstrap_mean(e2.readout_vector(s, rid)))
    lim = np.ceil(max(max(abs(lo), abs(hi)) for _, lo, hi in est.values()) + 0.5)
    fig, axes = plt.subplots(1, len(READOUT_MODELS), figsize=(18, 15), sharey=True)
    fig.subplots_adjust(left=0.235, right=0.99, top=0.925, bottom=0.105, wspace=0.05)
    for ax, s in zip(axes, READOUT_MODELS):
        for y, kind, _, base in rows:
            if kind != "ro":
                continue
            for rid, d in _depth_ids(e2.readouts, base):
                dy = 0.0 if d == "x" else (0.17 if d == "d33" else -0.17)
                m, lo, hi = est[(s, rid)]
                ax.plot([lo, hi], [y + dy, y + dy], color=DEPTH_COLOR[d], lw=2.2, zorder=2)
                ax.plot(m, y + dy, "o", color=DEPTH_COLOR[d], ms=7, zorder=3)
        for y, kind, _, _ in rows:
            if kind == "fam" and y < 0:
                ax.axhline(y + 0.7, color=GRID, lw=1.0)
        clean_axes(ax)
        ax.axvline(0, color=ZERO, lw=2.0, ls=":")
        ax.set_xlim(-lim, lim)
        ax.set_xticks(np.arange(-10 * (lim // 10), lim + 1e-9, 10))
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.yaxis.grid(False)
        ax.tick_params(labelsize=18, length=0)
        ax.set_title(DISPLAY[s], fontsize=23)
    ax0 = axes[0]
    ax0.set_yticks([y for y, _, _, _ in rows])
    ax0.set_yticklabels([t for _, _, t, _ in rows])
    for lab, (_, kind, _, _) in zip(ax0.get_yticklabels(), rows):
        if kind == "fam":
            lab.set_fontweight("bold")
            lab.set_fontsize(19)
    ax0.set_ylim(rows[-1][0] - 0.8, 0.8)
    handles = [Line2D([], [], marker="o", ls="-", ms=7, lw=2.2, color=DEPTH_COLOR[d], label=DEPTH_LABEL[d])
               for d in ("d33", "d66", "x")]
    xc = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
    fig.text(xc, 0.052, "$\\tau^{\\max}_{30} - \\tau^{\\min}_{30}$", ha="center", va="center", fontsize=30)
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(xc, 0.018), ncol=3, frameon=False, fontsize=20)
    fig.suptitle("$\\tau^{\\max}_{30} - \\tau^{\\min}_{30}$ per readout", x=xc, y=0.985, fontsize=25)
    save(fig, OUT / "exp2_readouts.pdf", bbox_inches="tight")


def main():
    apply_style()
    OUT.mkdir(exist_ok=True)
    e2 = Exp2()
    results_figure(e2)
    effects_figure(e2)
    readouts_figure(e2)
    print("wrote", OUT / "exp2_results.pdf", OUT / "exp2_effects.pdf", OUT / "exp2_readouts.pdf")


if __name__ == "__main__":
    main()
