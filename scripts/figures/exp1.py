#!/usr/bin/env python
"""Figures of Experiment 1 (free generation), from results/exp1.npz.

  figures/exp1_results.pdf   Figure "Experiment 1": writer x subject objective difference at
                             turns 5 and 20, and the effect of each subject on itself with
                             its diagonal excess over the turns
  figures/exp1_fit.pdf       appendix: observed vs predicted oriented control effect at turn 20
  figures/exp1_grid.pdf      appendix: every writer x subject pair, per block and family
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from neurofeedback.analysis import (BLOCKS, RESULTS, Exp1, additive_prediction, exp1_grid,  # noqa: E402
                                    exp1_objective_difference_ci, exp1_self_bootstrap)
from style import CMAP, GRID, SHORT, ZERO, apply_style, clean_axes, save  # noqa: E402

OUT = REPO / "figures"
H = 20
C_TAU, C_EXC = "#0f8b8d", "#a8201a"
FS_TICK, FS_LABEL, FS_TITLE, FS_VAL, FS_BUT = 22, 28, 30, 21, 30
FAMILIES = [("all", "All readouts"), ("probe", "Concept probes"),
            ("pcband", "Principal-component\nbands"), ("encres", "Encoder-residual\ndirections"),
            ("geometry", "Trajectory geometry")]
BLOCK_TITLE = {("max", "hidden"): "maximize, hidden", ("max", "disclosed"): "maximize, disclosed",
               ("min", "hidden"): "minimize, hidden", ("min", "disclosed"): "minimize, disclosed"}


def _text_color(norm, v):
    r, g, b, _ = CMAP(norm(v))
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else "#1a1a1e"


def draw_matrix(ax, M, names, norm, title, values, show_y):
    ax.imshow(M, cmap=CMAP, norm=norm, aspect="equal")
    n = M.shape[0]
    for i in range(n):
        box = ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#1a1a1e",
                                     linewidth=6.0))
        box.set_clip_path(Rectangle((i - 0.5, i - 0.5), 1, 1, transform=ax.transData))
        if values:
            for j in range(n):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=FS_VAL,
                        color=_text_color(norm, M[i, j]))
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, fontsize=FS_TICK, rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(names if show_y else [""] * n, fontsize=FS_TICK)
    ax.tick_params(length=0)
    ax.set_xlabel("subject S", fontsize=FS_LABEL)
    if show_y:
        ax.set_ylabel("writer W", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE, pad=22)
    ax.grid(False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def add_ci(ax, M, lo, hi, norm):
    """Values moved up, with the rounded 95% interval underneath."""
    for t in list(ax.texts):
        t.remove()
    n = M.shape[0]
    for i in range(n):
        for j in range(n):
            col = _text_color(norm, M[i, j])
            ax.text(j, i - 0.13, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=FS_VAL, color=col)
            ax.text(j, i + 0.25, f"[{round(lo[i, j]) + 0:.0f}, {round(hi[i, j]) + 0:.0f}]",
                    ha="center", va="center", fontsize=FS_VAL - 7, color=col)


def observed_title(ax, turn):
    ax.set_title("")
    parts = [TextArea("observed", textprops=dict(fontsize=FS_TITLE)),
             TextArea(f"$\\tau^{{\\max}}_{{{turn}}} - \\tau^{{\\min}}_{{{turn}}}$",
                      textprops=dict(fontsize=FS_TITLE + 8))]
    box = AnchoredOffsetbox(loc="lower center", child=HPacker(children=parts, align="baseline", pad=0, sep=14),
                            bbox_to_anchor=(0.5, 1.03), bbox_transform=ax.transAxes,
                            frameon=False, pad=0, borderpad=0)
    ax.add_artist(box)


def axis_arrow(ax, start, end, text, text_xy, rotation=0):
    ax.annotate("", xy=end, xytext=start, xycoords="axes fraction", annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>,head_length=0.6,head_width=0.3", lw=2.5, color="#555"))
    ax.text(*text_xy, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=FS_TICK + 6, color="#555", rotation=rotation, style="italic")


def hypothetical(n, seed=0):
    """Illustration: the matrix if being the subject gave the writer an advantage."""
    rng = np.random.default_rng(seed)
    M = rng.normal(3.0, 2.0, size=(n, n))
    M[np.diag_indices(n)] = rng.normal(35.0, 8.0, size=n)
    return M


def subject_panel(ax, boot, sj, title, show_y):
    turns = np.arange(1, H + 1)
    ax.fill_between(turns, boot["tau_lo"][sj], boot["tau_hi"][sj], color=C_TAU, alpha=0.18, lw=0)
    ax.plot(turns, boot["tau_m"][sj], "-o", color=C_TAU, lw=2.2, ms=4)
    ax.fill_between(turns, boot["exc_lo"][sj], boot["exc_hi"][sj], color=C_EXC, alpha=0.18, lw=0)
    ax.plot(turns, boot["exc_m"][sj], "-o", color=C_EXC, lw=2.2, ms=4)
    clean_axes(ax)
    ax.axhline(0, color=ZERO, lw=2.0, ls=":")
    ax.set_xlim(1, H)
    ax.set_xticks([5, 10, 15, 20])
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.tick_params(labelsize=FS_TICK, length=0, labelbottom=True, labelleft=show_y)
    ax.set_xlabel("turn", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE - 2)


def results_figure(e1):
    names = [SHORT[m] for m in e1.models]
    V20, V5 = e1.objective_difference(19), e1.objective_difference(4)
    M = hypothetical(len(names))
    vmax = max(np.abs(V20).max(), np.abs(V5).max(), np.abs(M).max()) * 1.05
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    boot = exp1_self_bootstrap(e1)

    fig = plt.figure(figsize=(26, 13.6), constrained_layout=True)
    sf_top, sf_bot = fig.subfigures(2, 1, height_ratios=[1.15, 0.62], hspace=0.07)
    gs = sf_top.add_gridspec(1, 5, width_ratios=[0.62, 0.26, 1, 0.12, 1])
    ax_l, ax_m, ax_5, ax_g, ax_r = (sf_top.add_subplot(gs[0, k]) for k in range(5))
    draw_matrix(ax_l, M, names, norm, "if access to\none's own activations\ngave an advantage", False, True)
    draw_matrix(ax_5, V5, names, norm, "", True, True)
    observed_title(ax_5, 5)
    ax_5.set_ylabel("")
    draw_matrix(ax_r, V20, names, norm, "", True, False)
    observed_title(ax_r, 20)
    add_ci(ax_5, V5, *exp1_objective_difference_ci(e1, 5), norm)
    add_ci(ax_r, V20, *exp1_objective_difference_ci(e1, 20), norm)
    ax_l.xaxis.labelpad = 14
    ax_l.yaxis.labelpad = 14
    axis_arrow(ax_l, (0.02, -0.43), (0.98, -0.43), "susceptibility $b_S$", (0.5, -0.53))
    axis_arrow(ax_l, (-0.53, 0.98), (-0.53, 0.02), "competence $a_W$", (-0.63, 0.5), rotation=90)
    for ax in (ax_m, ax_g):
        ax.axis("off")
    ax_g.text(0.5, 0.5, "$\\cdots$", ha="center", va="center", fontsize=FS_BUT + 10, transform=ax_g.transAxes)
    ax_m.annotate("", xy=(0.95, 0.5), xytext=(0.05, 0.5), xycoords="axes fraction",
                  arrowprops=dict(arrowstyle="-|>,head_length=0.8,head_width=0.45", lw=3.5, color="#1a1a1e"))
    ax_m.text(0.5, 0.56, "but", ha="center", va="bottom", fontsize=FS_BUT, transform=ax_m.transAxes)

    axs = sf_bot.subplots(1, len(e1.models), sharey=True)
    for k, (ax, S) in enumerate(zip(axs, e1.models)):
        subject_panel(ax, boot, k, SHORT[S], show_y=(k == 0))
    handles = [Line2D([], [], color=C_TAU, lw=2.6, marker="o", ms=6,
                      label="oriented control effect  $s_o\\,\\tau^{o}_t(S \\to S)$"),
               Line2D([], [], color=C_EXC, lw=2.6, marker="o", ms=6, label="diagonal excess  $\\Delta_t(S)$")]
    sf_bot.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2,
                  frameon=False, fontsize=FS_TICK + 6)
    save(fig, OUT / "exp1_results.pdf", bbox_inches="tight", pad_inches=0.15)
    for sj, S in enumerate(e1.models):
        print(f"  {S:22s} s*tau_20 = {boot['tau_m'][sj, -1]:+6.2f} [{boot['tau_lo'][sj, -1]:+6.2f}, "
              f"{boot['tau_hi'][sj, -1]:+6.2f}]  Delta_20 = {boot['exc_m'][sj, -1]:+6.2f} "
              f"[{boot['exc_lo'][sj, -1]:+6.2f}, {boot['exc_hi'][sj, -1]:+6.2f}]")


def fit_figure(e1):
    cross, self_ = [], []
    for obj, cond in BLOCKS:
        tau = e1.grid(cond, obj, H - 1)
        pred = additive_prediction(tau)
        for w in range(5):
            for s in range(5):
                (self_ if w == s else cross).append((pred[w, s], tau[w, s]))
    cross, self_ = np.array(cross), np.array(self_)
    lo = min(cross.min(), self_.min()) - 2.5
    hi = max(cross.max(), self_.max()) + 2.5
    fig = plt.figure(figsize=(8, 8.5), constrained_layout=True)
    ax = fig.subplots()
    ax.plot([lo, hi], [lo, hi], ls=":", color=ZERO, lw=1.3, zorder=1)
    ax.scatter(cross[:, 0], cross[:, 1], s=40, color="#0f8b8d", alpha=0.7, lw=0, zorder=2,
               label="pairs $W \\neq S$ (fit)")
    ax.scatter(self_[:, 0], self_[:, 1], s=120, facecolor="white", edgecolor="#a8201a", lw=2.2,
               zorder=3, label="pairs $W = S$ (predicted)")
    clean_axes(ax)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, color=GRID, lw=0.8)
    ax.tick_params(labelsize=16, length=0)
    ax.legend(fontsize=16, loc="upper left", framealpha=0.95)
    ax.set_xlabel("predicted  $s_o\\,\\tau^{o}_{20}$  ($m + a_W + b_S$)", fontsize=18)
    ax.set_ylabel("observed  $s_o\\,\\tau^{o}_{20}$", fontsize=18)
    ax.set_title("oriented control effect,\npredicted against observed", fontsize=20)
    save(fig, OUT / "exp1_fit.pdf")


def grid_figure(e1):
    n = len(e1.models)
    V = np.full((len(FAMILIES), 4, n, n), np.nan)
    for fi, (fam, _) in enumerate(FAMILIES):
        V[fi] = exp1_grid(e1, H - 1, family=None if fam == "all" else fam)
    vmax = float(np.nanmax(np.abs(V)))
    vmin = -vmax
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    nr, nc = len(BLOCKS), len(FAMILIES)
    fig, axes = plt.subplots(nr, nc, figsize=(3.7 * nc + 1.8, 3.9 * nr), constrained_layout=True)
    names = [SHORT[m] for m in e1.models]
    im = None
    for ri, block in enumerate(BLOCKS):
        for fi, (_, fam_name) in enumerate(FAMILIES):
            ax = axes[ri][fi]
            ax.grid(False)
            M = V[fi, ri]
            im = ax.imshow(M, cmap=CMAP, norm=norm, aspect="equal")
            for i in range(n):
                for j in range(n):
                    v = M[i, j]
                    col = "white" if (v > 0.55 * vmax or v < 0.55 * vmin) else "black"
                    ax.text(j, i, f"{v:+.1f}", ha="center", va="center", color=col, fontsize=12)
                    if i == j:
                        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=1.8))
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(names if ri == nr - 1 else [], fontsize=16, rotation=45, ha="right")
            ax.set_yticklabels(names if fi == 0 else [], fontsize=16)
            ax.tick_params(length=0)
            for side in ("top", "right", "bottom", "left"):
                ax.spines[side].set_visible(False)
            if ri == 0:
                ax.set_title(fam_name, fontsize=20)
            if fi == 0:
                ax.set_ylabel(BLOCK_TITLE[block] + "\nwriter", fontsize=18)
            if ri == nr - 1:
                ax.set_xlabel("subject", fontsize=18)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    cb.ax.tick_params(labelsize=16)
    cb.set_label("oriented points", fontsize=18)
    fig.suptitle("oriented control effect $s_o\\,\\tau^{o}_{20}(W{\\to}S)$ by readout family", fontsize=24)
    save(fig, OUT / "exp1_grid.pdf", dpi=165, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(RESULTS), help="directory of exp1.npz (default: results/)")
    a = ap.parse_args()
    apply_style()
    OUT.mkdir(exist_ok=True)
    e1 = Exp1(a.results)
    fit_figure(e1)
    grid_figure(e1)
    results_figure(e1)
    print("wrote", OUT / "exp1_results.pdf", OUT / "exp1_fit.pdf", OUT / "exp1_grid.pdf")


if __name__ == "__main__":
    main()
