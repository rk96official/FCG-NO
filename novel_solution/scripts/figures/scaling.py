import os

import numpy as np
import matplotlib.pyplot as plt

from scripts.figures.data import (KINDS, KIND_TITLES, REPRODUCTION_SYSTEMS, SEEDS, amgx_entry, json_median, median_iterations, per_system, pooled,
                                  seconds_per_iteration, setup_seconds_per_system, solve_milliseconds)
from scripts.figures.style import (DOUBLE_COLUMN, METHODS, grid_axis, line_kwargs, mark_failure_band, panel_label, plot_with_failures, save_figure)

SCALING_GRIDS = (128, 256, 512, 1024)
PAPER_SCALING = (("sno", "fcgno_sno_notebook", "plain_tf32"), ("fno", "fcg_fno", "plain_fp64"), ("unet", "fcg_unet", "plain_tf32"),
                 ("dilresnet", "fcg_dilresnet", "plain_tf32"))
OURS_MODEL, OURS_VARIANT = "final_2l2", "multi_b64_tf32"


# Paths of our multi-level runs (E16, 5 seeds) for one kind and grid
def ours_paths(results, kind, grid):
    return [os.path.join(results, "e16_main", f"{OURS_MODEL}_{kind}_128to{grid}_seed{seed}.npz") for seed in SEEDS]


# Path of a paper model's U1 run (trained at 128, seed 0)
def paper_path(results, label, kind, grid):
    return os.path.join(results, "u1_paper_models", f"{label}_{kind}_128to{grid}_seed0.npz")


# Our per-seed values at each grid, as (pooled median, per-seed minimum, per-seed maximum) of a function of one seed's path
def ours_band(results, kind, value_of_seed, pooled_value):
    medians, lows, highs = [], [], []
    for grid in SCALING_GRIDS:
        paths = ours_paths(results, kind, grid)
        per_seed = np.array([value_of_seed(path) for path in paths])
        medians.append(pooled_value(paths))
        lows.append(np.nanmin(per_seed))
        highs.append(np.nanmax(per_seed))
    return np.array(medians), np.array(lows), np.array(highs)


# Shared legend under a row of panels, in the order the handles were drawn
def bottom_legend(figure, axis, columns):
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=columns, bbox_to_anchor=(0.5, -0.02), handlelength=3.2, columnspacing=1.4)


# Fig. 1: median iterations to 1e-6 vs grid size for the paper's learned preconditioners and ours, all trained at 128, on the reproduction's 20 systems
def figure_iterations_scaling(results, output):
    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.7), sharey=True)
    fail_level = 2e4
    for axis, kind, letter in zip(axes, KINDS, "ab"):
        for method, label, variant in PAPER_SCALING:
            values = [median_iterations(per_system(paper_path(results, label, kind, grid), variant, first=REPRODUCTION_SYSTEMS)) for grid in SCALING_GRIDS]
            plot_with_failures(axis, SCALING_GRIDS, values, method, fail_level)
        seed_median = lambda path: median_iterations(per_system(path, OURS_VARIANT, first=REPRODUCTION_SYSTEMS))
        medians, lows, highs = ours_band(results, kind, seed_median, lambda paths: median_iterations(pooled(paths, OURS_VARIANT, first=REPRODUCTION_SYSTEMS)))
        axis.fill_between(SCALING_GRIDS, lows, highs, color=METHODS["ours"]["color"], alpha=0.18, linewidth=0, zorder=8)
        plot_with_failures(axis, SCALING_GRIDS, medians, "ours", fail_level)
        axis.set_yscale("log")
        axis.set_ylim(3, fail_level)
        mark_failure_band(axis, fail_level)
        grid_axis(axis, SCALING_GRIDS)
        axis.set_xlabel(f"{KIND_TITLES[kind]}: grid size (trained at 128²)")
        panel_label(axis, letter)
    axes[0].set_ylabel("Iterations to $10^{-6}$ (median)")
    bottom_legend(figure, axes[0], 5)
    figure.subplots_adjust(bottom=0.3, wspace=0.08)
    save_figure(figure, output, "fig1_iterations_scaling")


# Milliseconds per system to 1e-6 at batch 1 of every method in the time comparison, for one kind and grid (first 20 systems)
def time_rows(results, kind, grid):
    first = REPRODUCTION_SYSTEMS
    rows = {}
    for method, label in (("dilresnet", "fcg_dilresnet"), ("unet", "fcg_unet")):
        path = paper_path(results, label, kind, grid)
        rows[method] = solve_milliseconds(per_system(path, "plain_tf32", first=first), seconds_per_iteration(path, "plain_tf32"))
    multigrid = os.path.join(results, "e16_multilevel", f"multigrid_{kind}_{grid}.npz")
    for method, key in (("redisc", "redisc_dense16"), ("galerkin", "gmg_vcycle_fp32coarse")):
        rows[method] = solve_milliseconds(per_system(multigrid, key, first=first), seconds_per_iteration(multigrid, key), setup=setup_seconds_per_system(multigrid, key))
    counts, setup, solve = amgx_entry(results, kind, grid, first=first)
    rows["amgx"] = setup + solve if counts is not None and np.isfinite(median_iterations(counts)) else np.nan
    classical = os.path.join(results, "e0_classical", f"classical_{kind}_{grid}.json")
    rows["chebyshev"] = 1e3 * json_median(classical, "chebyshev8") * seconds_per_iteration(classical, "chebyshev8")
    return rows


# Fig. 2: milliseconds per system to 1e-6 (one system at a time, setup included) vs grid size: ours, the paper's best learned methods and classical GPU solvers
def figure_time_scaling(results, output):
    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.8), sharey=True)
    order = ("chebyshev", "amgx", "galerkin", "unet", "dilresnet", "redisc")
    fail_level = 1e5
    for axis, kind, letter in zip(axes, KINDS, "ab"):
        table = [time_rows(results, kind, grid) for grid in SCALING_GRIDS]
        for method in order:
            plot_with_failures(axis, SCALING_GRIDS, [row[method] for row in table], method, fail_level)
        seed_time = lambda path: solve_milliseconds(per_system(path, OURS_VARIANT, first=REPRODUCTION_SYSTEMS), seconds_per_iteration(path, OURS_VARIANT))
        pooled_time = lambda paths: solve_milliseconds(pooled(paths, OURS_VARIANT, first=REPRODUCTION_SYSTEMS),
                                                       float(np.nanmean([seconds_per_iteration(path, OURS_VARIANT) for path in paths])))
        medians, lows, highs = ours_band(results, kind, seed_time, pooled_time)
        axis.fill_between(SCALING_GRIDS, lows, highs, color=METHODS["ours"]["color"], alpha=0.18, linewidth=0, zorder=8)
        axis.plot(SCALING_GRIDS, medians, **line_kwargs("ours"))
        axis.set_yscale("log")
        axis.set_ylim(0.5, fail_level)
        mark_failure_band(axis, fail_level)
        grid_axis(axis, SCALING_GRIDS)
        axis.set_xlabel(f"{KIND_TITLES[kind]}: grid size")
        panel_label(axis, letter)
    axes[0].set_ylabel("ms per system to $10^{-6}$\n(batch 1, setup included)")
    bottom_legend(figure, axes[0], 4)
    figure.subplots_adjust(bottom=0.33, wspace=0.08)
    save_figure(figure, output, "fig2_time_scaling")
