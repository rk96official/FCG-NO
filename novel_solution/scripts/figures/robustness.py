import os

import numpy as np
import matplotlib.pyplot as plt

from scripts.figures.data import KINDS, KIND_TITLES, json_median, median_iterations, per_system
from scripts.figures.style import DOUBLE_COLUMN, METHODS, grid_axis, line_kwargs, panel_label, plot_with_failures, save_figure

NEGATIVE_GRID = 512
CONTRASTS = (2, 4, 6)
NETWORK_LABELS = {"ours": "Ours (best network)", "ours_coef": r"Ours + $\log a$ input"}
THREE_D_GRIDS = (32, 64, 128)


# Median iterations to 1e-6 of every method on one coefficient family at 512² (E15 smooth GRF contrast or E20 jumps), one value per contrast exponent
def negative_rows(results, experiment, family):
    folder = {"E15": "e15_multigrid", "E20": "e20_multigrid"}[experiment]
    rows = {method: [] for method in ("chebyshev", "amgx", "galerkin", "redisc", "ours", "ours_coef")}
    for exponent in CONTRASTS:
        problem = f"{family}{exponent}_{NEGATIVE_GRID}"
        multigrid = os.path.join(results, folder, f"multigrid_{problem}.npz")
        rows["chebyshev"].append(json_median(os.path.join(results, "e0_classical", f"classical_{problem}.json"), "chebyshev8"))
        rows["amgx"].append(json_median(os.path.join(results, "e0_amgx", f"amgx_{problem}.json"), "amgx_classical"))
        rows["galerkin"].append(median_iterations(per_system(multigrid, "gmg_vcycle_fp32coarse")))
        rows["redisc"].append(median_iterations(per_system(multigrid, "redisc_dense16")))
        if experiment == "E15":
            rows["ours"].append(median_iterations(per_system(multigrid, "ml32_hc")))
            rows["ours_coef"].append(median_iterations(per_system(multigrid, "tl64_hc_coef")))
        else:
            rows["ours"].append(median_iterations(per_system(multigrid, "fine_jr")))
            rows["ours_coef"].append(median_iterations(np.concatenate([per_system(multigrid, f"fine_jr_coef_s{seed}") for seed in range(3)])))
    return rows


# Fig. 5: the negative results at 512²: iterations vs coefficient contrast for smooth fields (E15) and jumps (E20), ours against classical multigrid and AMG
def figure_negative_results(results, output):
    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.8), sharey=True)
    fail_level = 3e3
    panels = (("E15", "grf", "Smooth-field contrast"), ("E20", "jump", "Jump contrast"))
    for axis, (experiment, family, description), letter in zip(axes, panels, "ab"):
        for method, values in negative_rows(results, experiment, family).items():
            plot_with_failures(axis, CONTRASTS, values, method, fail_level, label=NETWORK_LABELS.get(method))
        axis.set_yscale("log")
        axis.set_ylim(4, fail_level)
        axis.set_xticks(CONTRASTS, [f"$10^{exponent}$" for exponent in CONTRASTS])
        axis.set_xlim(1.7, 6.3)
        axis.set_xlabel(f"{description} ({NEGATIVE_GRID}²)")
        panel_label(axis, letter)
    axes[0].set_ylabel("Iterations to $10^{-6}$ (median)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), handlelength=3.2, columnspacing=1.4)
    figure.subplots_adjust(bottom=0.32, wspace=0.06)
    save_figure(figure, output, "fig5_negative_results")


# Fig. 6: 3D Poisson and Diffusion: iterations vs grid size for our multi-level and two-level networks against setup-free multigrid and Chebyshev(8)-PCG
def figure_three_d(results, output):
    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.5), sharey=True)
    fail_level = 2000
    for axis, kind, letter in zip(axes, KINDS, "ab"):
        paths = [os.path.join(results, "e19_three_d", f"three_d_{kind}_{grid}.npz") for grid in THREE_D_GRIDS]
        series = {"chebyshev": "chebyshev8", "redisc": "mg_dense8", "ours": "net_ml16_s0"}
        for method, key in series.items():
            label = "Setup-free multigrid (dense 8³ bottom)" if method == "redisc" else None
            plot_with_failures(axis, THREE_D_GRIDS, [median_iterations(per_system(path, key)) for path in paths], method, fail_level, label=label)
        seeds = np.array([[median_iterations(per_system(path, f"net_tl16_s{seed}")) for seed in range(3)] for path in paths])
        pooled_two_level = [median_iterations(np.concatenate([per_system(path, f"net_tl16_s{seed}") for seed in range(3)])) for path in paths]
        axis.fill_between(THREE_D_GRIDS, seeds.min(axis=1), seeds.max(axis=1), color=METHODS["ours_two_level"]["color"], alpha=0.12, linewidth=0)
        axis.plot(THREE_D_GRIDS, pooled_two_level, **line_kwargs("ours_two_level", label="Ours (two-level, 3 seeds)"))
        axis.set_yscale("log")
        axis.set_ylim(4, 150)
        grid_axis(axis, THREE_D_GRIDS, cube=True)
        axis.set_xlabel(f"3D {KIND_TITLES[kind]}: grid size")
        panel_label(axis, letter)
    axes[0].set_ylabel("Iterations to $10^{-6}$ (median)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), handlelength=3.2, columnspacing=1.2)
    figure.subplots_adjust(bottom=0.3, wspace=0.08)
    save_figure(figure, output, "fig6_three_d")
