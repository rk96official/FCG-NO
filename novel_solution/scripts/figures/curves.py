import os

import numpy as np
import matplotlib.pyplot as plt

from scripts.figures.data import KINDS, KIND_TITLES
from scripts.figures.style import DOUBLE_COLUMN, MUTED_INK, line_kwargs, panel_label, save_figure

CURVE_GRIDS = (128, 1024)
FLOOR = 1e-12


# Stored residual curves (systems, steps) of every method for one kind and grid, from the f3_curves job; methods whose file is missing are left out
def method_curves(results, kind, grid):
    folder = os.path.join(results, "f3_curves")
    sources = {"cg": (f"classical_{kind}_{grid}", "cg"), "chebyshev": (f"classical_{kind}_{grid}", "chebyshev8"),
               "sno": (f"fcgno_sno_notebook_{kind}_128to{grid}_seed0", "plain_tf32"), "unet": (f"fcg_unet_{kind}_128to{grid}_seed0", "plain_tf32"),
               "dilresnet": (f"fcg_dilresnet_{kind}_128to{grid}_seed0", "plain_tf32"), "galerkin": (f"multigrid_{kind}_{grid}", "gmg_vcycle_fp32coarse"),
               "redisc": (f"multigrid_{kind}_{grid}", "redisc_dense16"), "ours": (f"final_2l2_{kind}_128to{grid}_seed0", "multi_b64_tf32")}
    curves = {}
    for method, (name, key) in sources.items():
        path = os.path.join(folder, f"{name}.npz")
        if os.path.exists(path) and f"{key}__curves" in np.load(path).files:
            curves[method] = np.load(path)[f"{key}__curves"]
    return curves


# Median over systems of the relative residual at each step, cut where it first reaches the floor (the solve has converged)
def median_curve(curves):
    median = np.median(np.clip(curves, FLOOR, None), axis=0)
    below = np.flatnonzero(median <= FLOOR)
    return median if len(below) == 0 else median[:below[0] + 1]


# Fig. 3: median relative residual vs iteration on the reproduction's 20 systems at 128² and 1024² (all models trained at 128²)
def figure_residual_curves(results, output):
    if not os.path.isdir(os.path.join(results, "f3_curves")):
        return print("fig3: results/f3_curves missing; submit slurm/f3_curves.slurm first")
    figure, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN, 4.6), sharey=True)
    for row, kind in enumerate(KINDS):
        for column, grid in enumerate(CURVE_GRIDS):
            axis = axes[row, column]
            for method, curves in method_curves(results, kind, grid).items():
                curve = median_curve(curves)
                style = line_kwargs(method)
                axis.plot(np.arange(1, len(curve) + 1), curve, **{**style, "markevery": np.unique(np.geomspace(1, len(curve), 8).astype(int)) - 1,
                                                                   "markersize": 5, "clip_on": True})
            axis.axhline(1e-6, color=MUTED_INK, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_ylim(FLOOR / 3, 3)
            axis.set_xlim(0.9, 6000)
            axis.set_title(f"{KIND_TITLES[kind]}, {grid}²", fontsize=9, loc="left", pad=3)
            panel_label(axis, "abcd"[2 * row + column])
    for axis in axes[1]:
        axis.set_xlabel("Iteration (log scale)")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$\Vert r_k\Vert_2 / \Vert r_0\Vert_2$ (median)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), handlelength=3.2, columnspacing=1.4)
    figure.subplots_adjust(bottom=0.17, hspace=0.32, wspace=0.08)
    save_figure(figure, output, "fig3_residual_curves")
