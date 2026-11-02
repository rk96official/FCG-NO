import os

import numpy as np
import matplotlib.pyplot as plt

from scripts.figures.data import KINDS, KIND_TITLES, SEEDS, median_iterations, per_system, pooled, upper_iterations
from scripts.figures.style import DOUBLE_COLUMN, METHODS, MUTED_INK, NEUTRAL_BAR, TEXT_INK, panel_label, save_figure

ABLATION_GRID = 1024
ABLATIONS = (("e16_main", "final_2l2", "Full recipe"), ("e9_ablation_main", "abl_m0", "No history input"),
             ("e9_ablation_main", "abl_notay", "Notay loss instead of step loss"), ("e9_ablation_main", "abl_cg", "CG data instead of on-policy"),
             ("e9_ablation_main", "abl_single", "Trained single-level (out of role)"))
THEORY_CASES = (("poisson", 128, "#86b6ef", "-"), ("poisson", 256, "#1c5cab", "--"), ("diffusion", 128, "#f2a07b", "-"), ("diffusion", 256, "#b4441b", "--"))


# Fig. 4: the E9 ablations, all deployed multi-level at 1024²: pooled median iterations (bar), p90 (whisker) and each seed's median (dots)
def figure_ablations(results, output):
    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.3), sharey=True)
    positions = np.arange(len(ABLATIONS))[::-1]
    for axis, kind, letter in zip(axes, KINDS, "ab"):
        for position, (folder, model, label) in zip(positions, ABLATIONS):
            paths = [os.path.join(results, folder, f"{model}_{kind}_128to{ABLATION_GRID}_seed{seed}.npz") for seed in SEEDS]
            counts = pooled(paths, "multi_b64_tf32")
            median, p90 = median_iterations(counts), upper_iterations(counts)
            color = METHODS["ours"]["color"] if model == "final_2l2" else NEUTRAL_BAR
            axis.barh(position, median, height=0.62, color=color, zorder=2)
            axis.plot([median, p90], [position, position], color=TEXT_INK, linewidth=1.2, zorder=3)
            axis.plot([p90, p90], [position - 0.15, position + 0.15], color=TEXT_INK, linewidth=1.2, zorder=3)
            seeds = [median_iterations(per_system(path, "multi_b64_tf32")) for path in paths]
            axis.scatter(seeds, position + np.linspace(-0.18, 0.18, len(seeds)), s=14, facecolor="white", edgecolor=TEXT_INK, linewidth=0.9, zorder=4)
            axis.text(max(p90, max(seeds)) + 2.5, position, f"{median:.0f}", va="center", ha="left", fontsize=8, color=TEXT_INK)
        axis.set_yticks(positions, [label for unused_folder, unused_model, label in ABLATIONS])
        axis.set_xlim(0, 105)
        axis.grid(axis="y", visible=False)
        axis.set_xlabel(f"{KIND_TITLES[kind]}: iterations to $10^{{-6}}$ at {ABLATION_GRID}²")
        panel_label(axis, letter)
    figure.subplots_adjust(wspace=0.06)
    save_figure(figure, output, "fig4_ablations")


# Empirical CDF of values as (sorted values, cumulative shares)
def empirical_cdf(values):
    ordered = np.sort(values)
    return ordered, np.arange(1, len(ordered) + 1) / len(ordered)


# Fig. 7: Theorem 3 on the E14 trajectories: learned coarse efficiency against (1 - delta) times the exact Galerkin one (the identity), and the distribution of delta
def figure_theorem_check(results, output):
    path = os.path.join(results, "f7_theory_records", "theory_records.npz")
    if not os.path.exists(path):
        return print("fig7: results/f7_theory_records/theory_records.npz missing; submit slurm/f3_curves.slurm (task 4) first")
    records = np.load(path)
    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.6))
    for kind, grid, color, linestyle in THEORY_CASES:
        name = f"{kind}_{grid}"
        delta, exact, learned = (records[f"{name}__{part}"] for part in ("delta", "exact", "learned"))
        label = f"{KIND_TITLES[kind]} {grid}²"
        axes[0].scatter((1 - delta) * exact, learned, s=4, color=color, alpha=0.5, linewidth=0, label=label, rasterized=True)
        values, shares = empirical_cdf(delta)
        axes[1].plot(values, shares, color=color, linestyle=linestyle, linewidth=2.0, label=label)
    axes[0].plot([0, 1], [0, 1], color=MUTED_INK, linewidth=1.0, linestyle=(0, (4, 3)), zorder=0)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel(r"Predicted: $(1-\delta)\,\eta_{\mathrm{exact}}$")
    axes[0].set_ylabel(r"Measured: $\eta_{\mathrm{learned}}$")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel(r"Coarse step loss $\delta$")
    axes[1].set_ylabel("Share of FCG steps")
    axes[1].legend(loc="lower right")
    for axis, letter in zip(axes, "ab"):
        panel_label(axis, letter)
    figure.subplots_adjust(wspace=0.35)
    save_figure(figure, output, "fig7_theorem3_check")
