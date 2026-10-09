"""Shared figure style and model display names."""
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

BACKGROUND = "#EFF1EA"
GRID = "#D6D8D0"
ZERO = "#555"
C_HIDDEN, C_DISCLOSED = "#0f8b8d", "#a8201a"
CMAP = LinearSegmentedColormap.from_list("teal_red", ["#0f8b8d", "#f5f4ef", "#a8201a"])
CMAP.set_bad("#ededed")

DISPLAY = {"qwen3.5-27b": "Qwen3.5-27B", "gemma-4-31b": "Gemma-4-31B", "gpt-oss-20b": "GPT-OSS-20B",
           "mistral-small-3.2-24b": "Mistral-Small-24B", "olmo-3.1-32b": "OLMo-3.1-32B",
           "qwen3-32b": "Qwen3-32B", "qwen3-14b": "Qwen3-14B", "gemma-4-12b": "Gemma-4-12B",
           "qwen3.5-9b": "Qwen3.5-9B", "qwen3.6-27b": "Qwen3.6-27B"}
SHORT = {"qwen3.5-27b": "Qwen", "gemma-4-31b": "Gemma", "gpt-oss-20b": "GPT-OSS",
         "mistral-small-3.2-24b": "Mistral", "olmo-3.1-32b": "OLMo"}
# One colour per model in the curves of Experiments 2-4.
LINE_COLOR = {"qwen3.6-27b": "#d1495b", "qwen3.5-27b": "#a8201a", "qwen3.5-9b": "#0f8b8d",
              "qwen3-14b": "#7a9e9f", "qwen3-32b": "#4d7ea8", "gemma-4-12b": "#b58900",
              "gemma-4-31b": "#8a6d3b", "gpt-oss-20b": "#6c6f7f", "mistral-small-3.2-24b": "#9b8ba5",
              "olmo-3.1-32b": "#708b75"}
# Order of the models in the figures of Experiments 2-4: families, small to large.
FIGURE_ORDER = ["qwen3.5-9b", "qwen3.5-27b", "qwen3.6-27b", "qwen3-14b", "qwen3-32b",
                "gemma-4-12b", "gemma-4-31b", "gpt-oss-20b", "mistral-small-3.2-24b", "olmo-3.1-32b"]


def apply_style() -> None:
    logging.getLogger("fontTools").setLevel(logging.ERROR)   # font-subsetting noise
    plt.rcParams.update({
        "figure.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia", "serif"],
        "mathtext.fontset": "cm",
        "axes.prop_cycle": matplotlib.cycler(color=["#143642", "#0f8b8d", "#ec9a29", "#a8201a", "#dad2d8"]),
        "axes.facecolor": BACKGROUND,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "axes.titlesize": 13, "axes.labelsize": 12.5,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 11,
    })


def clean_axes(ax):
    ax.set_facecolor(BACKGROUND)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)


def end_labels(ax, ends, x_end, gap, colors, names, dx_line=0.7, dx_text=0.9, fontsize=14):
    """Model names at the landing points of the curves, spread to avoid collisions."""
    order = sorted(range(len(ends)), key=lambda i: ends[i][1])
    label_y, prev = {}, -float("inf")
    for i in order:
        y = max(ends[i][1], prev + gap)
        label_y[ends[i][0]] = y
        prev = y
    for s, v in ends:
        ax.plot([x_end, x_end + dx_line], [v, label_y[s]], color=colors[s], lw=0.9, alpha=0.6,
                clip_on=False)
        ax.text(x_end + dx_text, label_y[s], names[s], color=colors[s], fontsize=fontsize,
                va="center", ha="left", clip_on=False)


def save(fig, path, dpi=150, **kwargs):
    """Write ``path`` as PNG, then as PDF (the PNG pass also settles the constrained layout)."""
    path = Path(path)
    fig.savefig(path.with_suffix(".png"), dpi=dpi, **kwargs)
    fig.savefig(path.with_suffix(".pdf"), **kwargs)
    plt.close(fig)
