#!/usr/bin/env python
"""Figures of the additional analyses of Experiment 2 (appendix), from results/exp2.npz,
exp2_shuffled.npz, exp2_targeted.npz and exp2_depths.npz. Hidden criterion unless stated.

  figures/exp2_shuffled.pdf   shuffled control: objective difference at turn 30 against random
                              scores and against the shuffled control, both conditions, for all
                              readouts and the direction readouts at 33% and 66% depth
                              (sentence bootstrap)
  figures/exp2_targeted.pdf   targeted or broad modulation: objective difference at turn 30 of a
                              readout when it is the score and when another readout is
                              (bootstrap over readouts and sentences)
  figures/exp2_depths.pdf     read depths: objective difference at turn 30 of the direction
                              readouts at seven depths, against random scores and against the
                              shuffled control (bootstrap over readouts and sentences)

Also prints the numbers of the figures and of the text of the appendix. Every interval is a 95%
percentile interval over 2000 resamples. The shuffled control and the depths use a fresh
generator (seed 0) per cell; the targeted analysis one generator (seed 0) shared by the four
models in turn, as in the paper.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from neurofeedback.analysis import (RESULTS, Exp2Depths, Exp2Shuffled, Exp2Targeted,  # noqa: E402
                                    bootstrap_sentences_readouts, direction_readouts, interval)
from style import (C_FOCUS, C_REFERENCE, DISPLAY, GRID, SERIES_OFFSET, ZERO, DotStyle, apply_style,  # noqa: E402
                   clean_axes, save)

OUT = REPO / "figures"
CONDITIONS = {"both": ("hidden", "disclosed"), "hidden": ("hidden",)}


def fmt(c):
    return (f"{c['mean']:+.2f} [{c['sentence_ci'][0]:+.2f}, {c['sentence_ci'][1]:+.2f}] / "
            f"[{c['two_level_ci'][0]:+.2f}, {c['two_level_ci'][1]:+.2f}]")


def legend_below(fig, axes, st, series, y_label, label, y_legend, fontsize, label_size):
    xc = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
    fig.text(xc, y_label, label, ha="center", va="center", fontsize=label_size)
    handles = [st.handle(col, lab) for _, lab, col, _ in series]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(xc, y_legend), ncol=2, frameon=False, fontsize=fontsize)
    return xc


# ---- shuffled control -----------------------------------------------------------------------

SHUFFLED_PANELS = [("all", "all readouts"), ("d33", "readouts at 33% depth"), ("d66", "readouts at 66% depth")]
SHUFFLED_SERIES = [("uniform", "random scores", C_REFERENCE, +SERIES_OFFSET),
                   ("shuffled", "shuffled real scores", C_FOCUS, -SERIES_OFFSET)]


def shuffled_numbers(sh):
    """{model: {(control, conditions, readouts): interval}}, as in the figure."""
    out = {}
    for s in sh.models:
        out[s] = {}
        for kind in ("shuffled", "uniform"):
            for cond, conds in CONDITIONS.items():
                for key, _ in SHUFFLED_PANELS:
                    idx = None if key == "all" else direction_readouts(sh.readouts, key)
                    out[s][(kind, cond, key)] = interval(*bootstrap_sentences_readouts(
                        sh.objective_difference(s, kind, idx, conds)))
    return out


def shuffled_figure(sh):
    num = shuffled_numbers(sh)
    print("Shuffled control: tau_max - tau_min at t=30, mean [sentence 95% CI] / [readout x sentence 95% CI]")
    for s in sh.models:
        for cond in CONDITIONS:
            for key, _ in SHUFFLED_PANELS:
                print(f"  {s:22s} {cond:6s} {key:4s} random {fmt(num[s][('uniform', cond, key)])}   "
                      f"shuffled {fmt(num[s][('shuffled', cond, key)])}")
    print("  at t=30, all readouts, both conditions: max - min of the real sessions (R), of the random control "
          "(U), of the shuffled control (M, the instruction with the numbers fixed), and the effect of whose numbers "
          "are shown with the instruction fixed (N)")
    for s in sh.models:
        t = sh.two_by_two(s)
        ratio = num[s][("shuffled", "both", "all")]["mean"] / num[s][("uniform", "both", "all")]["mean"]
        print(f"  {s:22s} R {t['real_max_minus_min']:+.2f}  U {t['uniform_max_minus_min']:+.2f}  "
              f"M {t['instruction_effect_numbers_fixed']:+.2f}  N {t['number_effect_instruction_fixed']:+.2f}  "
              f"shuffled / random objective difference {ratio:.2f}")
    probes = [i for i, r in enumerate(sh.readouts) if r.startswith("probe_")]
    print("  concept probes at t=30, per objective: real - random control, shuffled control - random control")
    for s in sh.models:
        po = sh.per_objective(s, probes)
        print(f"  {s:22s} " + "   ".join(f"{o} {v['real_minus_uniform']:+.2f}, {v['shuffled_minus_uniform']:+.2f}"
                                          for o, v in po.items()))

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.4), sharey=True)
    fig.subplots_adjust(left=0.165, right=0.99, top=0.83, bottom=0.25, wspace=0.07)
    st = DotStyle(fig, 1.0)
    ys = np.arange(len(sh.models))[::-1]
    for ax, (key, title) in zip(axes, SHUFFLED_PANELS):
        for y, s in zip(ys, sh.models):
            for kind, _, col, dy in SHUFFLED_SERIES:
                c = num[s][(kind, "both", key)]
                st.dot(ax, c["mean"], *c["sentence_ci"], y + dy, col)
        clean_axes(ax)
        ax.axvline(0, color=ZERO, lw=2.0, ls=":")
        ax.set_xlim(-3.0, 6.5)
        ax.set_xticks([-2, 0, 2, 4, 6])
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.yaxis.grid(False)
        ax.tick_params(labelsize=17, length=0)
        ax.set_title(title, fontsize=21)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([DISPLAY[s] for s in sh.models], fontsize=18)
    axes[0].set_ylim(-0.6, len(sh.models) - 0.4)
    xc = legend_below(fig, axes, st, SHUFFLED_SERIES, 0.135, "$\\tau^{\\max}_{30} - \\tau^{\\min}_{30}$", 0.04, 19, 24)
    fig.suptitle("$\\tau^{\\max}_{30} - \\tau^{\\min}_{30}$ per control and read depth", x=xc, y=0.975, fontsize=25)
    save(fig, OUT / "exp2_shuffled.pdf")


# ---- targeted or broad modulation -----------------------------------------------------------

TARGETED_PANELS = [("all", "all readouts"), ("d66", "readouts at 66% depth")]
TARGETED_SERIES = [("other", "when another readout is the score", C_REFERENCE, +SERIES_OFFSET),
                   ("scored", "when it is the score", C_FOCUS, -SERIES_OFFSET)]


def targeted_figure(tg):
    rng = np.random.default_rng(0)
    res = {s: tg.analysis(s, rng, direction_readouts(tg.readouts, "d66")) for s in tg.models}
    print("Targeted or broad modulation: tau_max - tau_min at t=30 of a readout, hidden criterion, "
          "mean [sentence 95% CI] / [readout x sentence 95% CI], p from random sign flips of the readouts")
    for s in tg.models:
        for key, _ in TARGETED_PANELS:
            r = res[s][key]
            print(f"  {s:22s} {key:3s} " + "   ".join(f"{n} {fmt(r[n])} p={r[n]['p_signflip']:.4f}"
                                                     for n in ("scored", "other", "excess")))
            la = r["excess_level_adjusted"]
            print(f"  {'':22s} {'':3s} difference after the level check {la['mean']:+.2f} p={la['p_signflip']:.4f}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.985, top=0.83, bottom=0.25, wspace=0.07)
    st = DotStyle(fig, 0.85)
    ys = np.arange(len(tg.models))[::-1]
    lims = []
    for ax, (key, title) in zip(axes, TARGETED_PANELS):
        for y, s in zip(ys, tg.models):
            for name, _, col, dy in TARGETED_SERIES:
                c = res[s][key][name]
                lo, hi = c["two_level_ci"]
                lims += [lo, hi]
                st.dot(ax, c["mean"], lo, hi, y + dy, col)
        clean_axes(ax)
        ax.axvline(0, color=ZERO, lw=2.0, ls=":")
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.yaxis.grid(False)
        ax.tick_params(labelsize=17, length=0)
        ax.set_title(title, fontsize=21)
    lim = np.ceil(max(abs(min(lims)), abs(max(lims))) + 0.5)
    for ax in axes:
        ax.set_xlim(-lim, lim)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([DISPLAY[s] for s in tg.models], fontsize=18)
    axes[0].set_ylim(-0.6, len(tg.models) - 0.4)
    xc = legend_below(fig, axes, st, TARGETED_SERIES, 0.135, "$\\tau^{\\max}_{30} - \\tau^{\\min}_{30}$ of a readout",
                      0.04, 18, 22)
    fig.suptitle("$\\tau^{\\max}_{30} - \\tau^{\\min}_{30}$ of a readout, scored or not", x=xc, y=0.975, fontsize=24)
    save(fig, OUT / "exp2_targeted.pdf")


# ---- read depths ------------------------------------------------------------------------------

# Columns are read depths about 10 points apart: the series offset is SERIES_OFFSET of that spacing.
DEPTH_SPACING = 10
DEPTH_SERIES = [("uniform", "random scores", C_REFERENCE, -SERIES_OFFSET * DEPTH_SPACING),
                ("shuffled", "shuffled real scores", C_FOCUS, +SERIES_OFFSET * DEPTH_SPACING)]


def depths_figure(dp):
    num = {s: {d: {k: interval(*bootstrap_sentences_readouts(dp.objective_difference(s, d, k)))
                   for k in ("uniform", "shuffled", "own")} for d in dp.DEPTHS} for s in dp.models}
    print("Read depths: tau_max - tau_min at t=30, hidden criterion, the 15 direction readouts of each depth, "
          "mean [sentence 95% CI] / [readout x sentence 95% CI]")
    for s in dp.models:
        for d in dp.DEPTHS:
            c = num[s][d]
            print(f"  {s:22s} {d[1:]}%  random {fmt(c['uniform'])}   shuffled {fmt(c['shuffled'])}   "
                  f"own numbers only {fmt(c['own'])}")
    depths = [int(d[1:]) for d in dp.DEPTHS]
    fig, axgrid = plt.subplots(2, 2, figsize=(13, 10.5), sharey=True, sharex=True)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.88, bottom=0.15, wspace=0.06, hspace=0.18)
    st = DotStyle(fig, 0.85)
    axes = axgrid.ravel()
    lims = []
    for ax, s in zip(axes, dp.models):
        for kind, _, col, dx in DEPTH_SERIES:
            xs, ms = [], []
            for d, dep in zip(dp.DEPTHS, depths):
                c = num[s][d][kind]
                lo, hi = c["two_level_ci"]
                lims += [lo, hi]
                st.dot(ax, c["mean"], lo, hi, dep + dx, col, vertical=True)
                xs.append(dep + dx)
                ms.append(c["mean"])
            ax.plot(xs, ms, "-", color=col, lw=1.2, alpha=0.6, zorder=2)
        clean_axes(ax)
        ax.axhline(0, color=ZERO, lw=2.0, ls=":")
        ax.set_xticks(depths)
        ax.set_xticklabels([f"{x}%" for x in depths], fontsize=15)
        ax.set_xlim(28, 100)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.xaxis.grid(False)
        ax.tick_params(axis="y", labelsize=16, length=0)
        ax.tick_params(axis="x", length=0)
        ax.set_title(DISPLAY[s], fontsize=21)
    lim = np.ceil(max(abs(min(lims)), abs(max(lims))))
    axes[0].set_ylim(-lim, lim)
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("$\\tau^{\\max}_{30} - \\tau^{\\min}_{30}$", fontsize=21)
    xc = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
    fig.text(xc, 0.075, "depth at which the score is read", ha="center", va="center", fontsize=20)
    handles = [st.handle(col, lab) for _, lab, col, _ in DEPTH_SERIES]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(xc, 0.025), ncol=2, frameon=False, fontsize=19)
    fig.suptitle("$\\tau^{\\max}_{30} - \\tau^{\\min}_{30}$ per read depth", x=xc, y=0.975, fontsize=25)
    save(fig, OUT / "exp2_depths.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(RESULTS), help="directory of the results (default: results/)")
    a = ap.parse_args()
    apply_style()
    OUT.mkdir(exist_ok=True)
    shuffled_figure(Exp2Shuffled(a.results))
    targeted_figure(Exp2Targeted(a.results))
    depths_figure(Exp2Depths(a.results))
    print("wrote", *(OUT / f"exp2_{n}.pdf" for n in ("shuffled", "targeted", "depths")))


if __name__ == "__main__":
    main()
