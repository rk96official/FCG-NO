import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SINGLE_COLUMN = 3.5
DOUBLE_COLUMN = 7.16
TEXT_INK, MUTED_INK, GRID_INK = "#0b0b0b", "#52514e", "#d9d8d4"
NEUTRAL_BAR = "#b9b8b2"

# One fixed style per method across every figure: categorical slots of the validated reference palette in order, ours in slot 1 and drawn heaviest;
# line style separates our method (solid, thick), the paper's learned methods (solid) and classical solvers (dashed or dotted) in grayscale
METHODS = {
    "ours": {"label": "Ours (multi-level, trained in role)", "color": "#2a78d6", "marker": "o", "linestyle": "-", "linewidth": 2.8, "zorder": 10},
    "ours_two_level": {"label": "Ours (two-level)", "color": "#2a78d6", "marker": "o", "linestyle": "--", "linewidth": 2.2, "zorder": 9, "hollow": True},
    "ours_coef": {"label": r"Ours + $\log a$ input", "color": "#2a78d6", "marker": "o", "linestyle": "-.", "linewidth": 2.2, "zorder": 9, "hollow": True},
    "dilresnet": {"label": "FCG-DilResNet", "color": "#eb6834", "marker": "s", "linestyle": "-", "linewidth": 2.0, "zorder": 6},
    "redisc": {"label": "Setup-free multigrid", "color": "#1baf7a", "marker": "P", "linestyle": "--", "linewidth": 2.0, "zorder": 7},
    "unet": {"label": "FCG-U-Net", "color": "#eda100", "marker": "^", "linestyle": "-", "linewidth": 2.0, "zorder": 5},
    "sno": {"label": "FCG-NO (SNO)", "color": "#e87ba4", "marker": "D", "linestyle": "-", "linewidth": 2.0, "zorder": 4},
    "galerkin": {"label": "Galerkin GMG", "color": "#008300", "marker": "X", "linestyle": "--", "linewidth": 2.0, "zorder": 7},
    "fno": {"label": "FCG-FNO (fp64)", "color": "#6250d6", "marker": "v", "linestyle": "-", "linewidth": 2.0, "zorder": 4},
    "amgx": {"label": "AMGX classical AMG", "color": "#e34948", "marker": "h", "linestyle": "--", "linewidth": 2.0, "zorder": 6},
    "chebyshev": {"label": "Chebyshev(8)-PCG", "color": "#52514e", "marker": "*", "linestyle": ":", "linewidth": 2.0, "zorder": 3},
    "cg": {"label": "CG", "color": "#9a9994", "marker": "x", "linestyle": ":", "linewidth": 2.0, "zorder": 2},
}


# Matplotlib settings for IEEE print: 8-9 pt sans-serif text, recessive axes and grid, vector-friendly fonts
def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.edgecolor": MUTED_INK, "axes.labelcolor": TEXT_INK, "xtick.color": MUTED_INK, "ytick.color": MUTED_INK, "text.color": TEXT_INK,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "axes.grid": True, "grid.color": GRID_INK, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "lines.markersize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "mathtext.default": "regular",
    })


# Keyword arguments of a method's line and markers for matplotlib's plot; label overrides the method's legend text
def line_kwargs(method, with_label=True, label=None):
    style = METHODS[method]
    face = "white" if style.get("hollow") else style["color"]
    kwargs = {"color": style["color"], "marker": style["marker"], "linestyle": style["linestyle"], "linewidth": style["linewidth"],
              "zorder": style["zorder"], "markerfacecolor": face, "markeredgecolor": style["color"], "markeredgewidth": 1.4, "clip_on": False}
    return {**kwargs, "label": label or style["label"]} if with_label else kwargs


# Plot one method's medians against x; failed points (non-finite) are drawn hollow on the failure line, joined to the last finite point by a dotted segment
def plot_with_failures(axis, xs, values, method, fail_level, label=None):
    xs, values = np.asarray(xs, dtype=float), np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    axis.plot(xs[finite], values[finite], **line_kwargs(method, label=label))
    if finite.all():
        return
    style = METHODS[method]
    failed = np.flatnonzero(~finite)
    if finite.any():
        last = np.flatnonzero(finite)[-1]
        axis.plot([xs[last], xs[failed[0]]], [values[last], fail_level], color=style["color"], linestyle=":", linewidth=1.6, zorder=style["zorder"], clip_on=False)
    axis.plot(xs[failed], np.full(len(failed), fail_level), linestyle="none", marker=style["marker"], markersize=7, markerfacecolor="white",
              markeredgecolor=style["color"], markeredgewidth=1.6, zorder=style["zorder"], clip_on=False)


# Dashed failure line at the top of a log axis with its label, so failed methods are marked explicitly rather than dropped
def mark_failure_band(axis, fail_level, text="no convergence"):
    axis.axhline(fail_level, color=MUTED_INK, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    axis.text(0.01, fail_level, f" {text}", transform=axis.get_yaxis_transform(), va="bottom", ha="left", fontsize=7.5, color=MUTED_INK)


# Base-2 grid-size axis with n² tick labels
def grid_axis(axis, grids, cube=False):
    axis.set_xscale("log", base=2)
    axis.set_xticks(grids, [f"{grid}³" if cube else f"{grid}²" for grid in grids])
    axis.minorticks_off()
    axis.set_xlim(grids[0] / 1.25, grids[-1] * 1.25)


# Panel letter in the top-left corner, outside the plot area
def panel_label(axis, letter):
    axis.text(-0.02, 1.02, f"({letter})", transform=axis.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold")


# Save a figure as vector PDF and 600-dpi PNG
def save_figure(figure, output, name):
    for extension in ("pdf", "png"):
        figure.savefig(f"{output}/{name}.{extension}", dpi=600)
    plt.close(figure)
    print(f"wrote {output}/{name}.pdf and .png", flush=True)
