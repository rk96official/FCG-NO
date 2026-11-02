import argparse
import glob
import json
import os

import numpy as np

from nfcg.eval.diagnostics import active_steps
from nfcg.eval.per_system import per_system_iterations, roc_auc

TOLERANCE = 1e-6
ACTIVE_FLOOR = 1e-10
SOLVER_COLUMNS = ("base", "full_orthogonalization", "residual_swap_0.9", "angle_cap_0.9", "angle_cap_0.75")
TRAINING_WINDOW = 100
PROBE_STEPS = (5, 20, 40)


# Every E1 run as (summary, arrays)
def load_runs(results_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "e1_*.json"))):
        with open(path) as handle:
            summary = json.load(handle)
        with np.load(path[:-len(".json")] + ".npz") as stored:
            runs.append((summary, {name: stored[name] for name in stored.files}))
    return runs


# Markdown table from a header and rows of cells
def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows])


# Short run label, e.g. "poisson 128 s0 paper"
def run_label(summary):
    return f"{summary['kind']} {summary['grid']} s{summary['seed']} {summary['tag']}"


# "failure% / median" at 1e-6 for one solver variant
def iteration_cell(summary, label):
    entry = summary["variants"].get(label)
    if entry is None:
        return "—"
    stats = entry["iterations"]["1e-06"]
    median = "—" if stats["median"] is None else f"{stats['median']:.0f}"
    return f"{100 * stats['failure_rate']:.0f}% / {median}"


# Systems that fail to reach 1e-6 or need more than slow_factor times the run's median of converged systems
def bad_systems(relative_residual, slow_factor):
    counts = per_system_iterations(relative_residual, TOLERANCE)
    finite = counts[np.isfinite(counts)]
    limit = slow_factor * np.median(finite) if len(finite) else np.inf
    return ~np.isfinite(counts) | (counts > limit)


# Mean of a per-step trace over the active steps among the first window steps; nan for systems with none
def window_mean(trace, active, window):
    values, mask = trace[:, :window].astype(float), active[:, :window]
    return np.where(mask.any(axis=1), (values * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1), np.nan)


# Maximum of a per-step trace over the active steps among the first window steps
def window_max(trace, active, window):
    values = np.where(active[:, :window], trace[:, :window], -np.inf)
    return np.where(active[:, :window].any(axis=1), values.max(axis=1), np.nan)


# Early-window statistics per system that might predict a stall
def early_statistics(arrays, active, window):
    sine, notay, probe = arrays["base__preconditioned_sine_squared"], arrays["base__notay"], arrays["base__error_probe"]
    return {
        "mean sin² of w": window_mean(sine, active, window),
        "max sin² of w": window_max(sine, active, window),
        "share sin² of w > 0.9": window_mean(sine > 0.9, active, window),
        "mean Notay ratio": window_mean(notay, active, window),
        "max Notay ratio": window_max(notay, active, window),
        "share Notay ratio > 1": window_mean(notay > 1.0, active, window),
        "mean error outside modes": window_mean(probe, active, window),
    }


# Section 1: failures and medians under each solver variant, i.e. the causal and truncation tests
def solver_section(runs):
    rows = [[run_label(summary), f"{summary['final_training_loss']:.3f}" + (" (collapsed)" if summary["collapsed"] else "")]
            + [iteration_cell(summary, label) for label in SOLVER_COLUMNS]
            + [f"{summary['variants'].get('angle_cap_0.9', {}).get('intervened_fraction', float('nan')):.3f}"] for summary, unused in runs]
    return "## Failures / median iterations to 1e-6 per solver variant\n\n" + markdown_table(
        ["run", "train loss", *SOLVER_COLUMNS, "share of steps capped at 0.9"], rows)


# Section 2 (H1): network angle on CG residuals (training distribution) vs on its own FCG residuals
def shift_section(runs, slow_factor):
    rows = []
    for summary, arrays in runs:
        if summary["collapsed"]:
            continue
        cg_active = active_steps(arrays["cg_observer__relative_residual"], ACTIVE_FLOOR)[:, :TRAINING_WINDOW]
        fcg_active = active_steps(arrays["base__relative_residual"], ACTIVE_FLOOR)
        on_cg = arrays["cg_observer__observer_sine_squared"][:, :TRAINING_WINDOW][cg_active]
        on_fcg = arrays["base__preconditioned_sine_squared"]
        bad = bad_systems(arrays["base__relative_residual"], slow_factor)
        cells = [np.median(on_cg), np.median(on_fcg[fcg_active]), np.median(on_fcg[fcg_active & ~bad[:, None]]) if (~bad).any() else np.nan,
                 np.median(on_fcg[fcg_active & bad[:, None]]) if bad.any() else np.nan]
        rows.append([run_label(summary)] + [f"{value:.3f}" for value in cells] + [int(bad.sum())])
    return "## H1 shift: median sin² of the network output on CG residuals vs its own FCG residuals\n\n" + markdown_table(
        ["run", "on CG (steps < 100)", "on FCG (all)", "on FCG (good systems)", "on FCG (bad systems)", "bad systems"], rows)


# Section 3 (H2, H3): how well early-window statistics separate bad from good systems (AUC), pooled per kind and grid
def prediction_section(runs, slow_factor, window):
    pools = {}
    for summary, arrays in runs:
        if summary["collapsed"]:
            continue
        active = active_steps(arrays["base__relative_residual"], ACTIVE_FLOOR)
        pool = pools.setdefault((summary["kind"], summary["grid"]), {"bad": [], "statistics": []})
        pool["bad"].append(bad_systems(arrays["base__relative_residual"], slow_factor))
        pool["statistics"].append(early_statistics(arrays, active, window))
    names = list(early_statistics({key: np.zeros((1, 1)) for key in ("base__preconditioned_sine_squared", "base__notay", "base__error_probe")},
                                  np.ones((1, 1), bool), 1))
    rows = []
    for (kind, grid), pool in sorted(pools.items()):
        bad = np.concatenate(pool["bad"])
        cells = []
        for name in names:
            scores = np.concatenate([statistics[name] for statistics in pool["statistics"]])
            keep = np.isfinite(scores)
            auc = roc_auc(scores[keep], bad[keep])
            cells.append("—" if auc is None else f"{auc:.2f}")
        rows.append([f"{kind} {grid}", f"{int(bad.sum())}/{len(bad)}"] + cells)
    return (f"## H2/H3: AUC of statistics over the first {window} steps for predicting bad systems (0.5 = no signal)\n\n"
            + markdown_table(["pool", "bad/all", *names], rows))


# Section 4 (H3): median share of the error outside the retained modes at fixed steps, bad vs good systems
def modes_section(runs, slow_factor):
    rows = []
    for summary, arrays in runs:
        if summary["collapsed"]:
            continue
        probe, active = arrays["base__error_probe"], active_steps(arrays["base__relative_residual"], ACTIVE_FLOOR)
        bad = bad_systems(arrays["base__relative_residual"], slow_factor)
        cells = []
        for step in PROBE_STEPS:
            for group in (~bad, bad):
                values = probe[group & active[:, step], step] if step < probe.shape[1] else np.array([])
                cells.append(f"{np.median(values):.3f}" if len(values) else "—")
        rows.append([run_label(summary)] + cells)
    header = ["run"] + [f"step {step} {group}" for step in PROBE_STEPS for group in ("good", "bad")]
    return "## H3: median share of error energy outside the 20 retained modes\n\n" + markdown_table(header, rows)


# Build the E1 report (markdown) from the saved runs and write it next to them
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/e1_diagnosis")
    parser.add_argument("--slow-factor", type=float, default=2.0)
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()
    runs = load_runs(args.results)
    sections = [solver_section(runs), shift_section(runs, args.slow_factor), prediction_section(runs, args.slow_factor, args.window),
                modes_section(runs, args.slow_factor)]
    report = "# E1 stall diagnosis\n\n" + "\n\n".join(sections) + "\n"
    with open(os.path.join(args.results, "report.md"), "w") as handle:
        handle.write(report)
    print(report)


if __name__ == "__main__":
    main()
