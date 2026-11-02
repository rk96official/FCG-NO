import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from nfcg.eval.per_system import paired_wilcoxon, quantile

TOLERANCE_KEY = "1e-06"
CLASSICAL_ROWS = ("cg", "jacobi4", "chebyshev8")


# Markdown table from a header and rows of cells
def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows])


# Number or dash
def cell(value, digits=0):
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


# Main-table runs (E6 or E8) grouped by (kind, test grid) and model label: list of (summary, {variant: per-system iterations at 1e-6})
def load_main(results_dir):
    groups = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as handle:
            summary = json.load(handle)
        with np.load(path[:-len(".json")] + ".npz") as stored:
            index = [f"{tol:g}" for tol in stored["tolerances"]].index(TOLERANCE_KEY)
            counts = {name[:-len("__iterations")]: stored[name][index] for name in stored.files if name.endswith("__iterations")}
        groups[(summary["kind"], summary["test_grid"])][summary["model"]["label"]].append((summary, counts))
    return groups


# Classical GPU rows (CG, Jacobi, Chebyshev) and AMG iterations for one kind and grid, from the baseline results
def baseline_rows(baselines_dir, amg_dir, kind, grid):
    rows = []
    path = os.path.join(baselines_dir, f"classical_{kind}_{grid}.json")
    if os.path.exists(path):
        with open(path) as handle:
            variants = json.load(handle)["variants"]
        for label in CLASSICAL_ROWS:
            stats = variants[label]["iterations"][TOLERANCE_KEY]
            per_system = None if stats["median"] is None else stats["median"] * variants[label]["seconds_per_iteration"]["16"] * 1e3 / 16
            rows.append([f"**{label}** (GPU)", "", f"{100 * stats['failure_rate']:.1f}", cell(stats["median"]), cell(stats["p90"]), "", cell(per_system, 2), ""])
    path = os.path.join(amg_dir, f"amg_{kind}_{grid}.json")
    if os.path.exists(path):
        with open(path) as handle:
            for label, entry in json.load(handle)["variants"].items():
                stats = entry["iterations"][TOLERANCE_KEY]
                rows.append([f"**AMG {label}** (CPU, 1 core)", "", f"{100 * stats['failure_rate']:.1f}", cell(stats["median"]), cell(stats["p90"]), "", "not comparable", ""])
    return rows


# Rows of one model: per variant pooled failures, median, p90, ms/it, ms per system to 1e-6 at batch 16, and Wilcoxon p against the notebook reference variant
def model_rows(label, runs, reference, cap):
    rows = []
    for variant in runs[0][0]["variants"]:
        pooled = np.concatenate([counts[variant] for unused_summary, counts in runs])
        b16 = np.median([summary["variants"][variant]["seconds_per_iteration"]["16"] for summary, unused_counts in runs]) * 1e3
        median = quantile(pooled, 0.5)
        per_system = median * b16 / 16 if np.isfinite(median) else None
        paired = [(counts[variant], reference[summary["seed"]]) for summary, counts in runs if summary["seed"] in reference]
        p_value = paired_wilcoxon(np.concatenate([pair[0] for pair in paired]), np.concatenate([pair[1] for pair in paired]), cap) if paired else None
        rows.append([label, variant, f"{100 * np.mean(~np.isfinite(pooled)):.1f}", cell(median), cell(quantile(pooled, 0.9)), f"{b16:.3f}",
                     cell(per_system, 2), cell(p_value, 4) if p_value is not None else "—"])
    return rows


# Build a main-table report (E6 or E8): per kind and grid, every model and composition next to the classical and AMG baselines
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/e6_main")
    parser.add_argument("--title", default="E6 main table")
    parser.add_argument("--classical", default="results/e0_classical")
    parser.add_argument("--amg", default="results/e0_amg")
    parser.add_argument("--reference", default="base", help="notebook variant the Wilcoxon test compares against")
    args = parser.parse_args()
    header = ["model", "variant", "fail %", "median its", "p90 its", "ms/it b16", "ms per system to 1e-6 (b16)", f"Wilcoxon p vs notebook {args.reference}"]
    sections = []
    for (kind, grid), models in sorted(load_main(args.results).items()):
        reference = {summary["seed"]: counts[args.reference] for summary, counts in models.get("notebook", [])}
        rows = baseline_rows(args.classical, args.amg, kind, grid)
        for label, runs in sorted(models.items()):
            rows += model_rows(label, sorted(runs, key=lambda run: run[0]["seed"]), reference, 4 * grid)
        sections.append(f"## {kind} {grid}² at 1e-6 (pooled over seeds × 100 systems)\n\n" + markdown_table(header, rows))
    report = f"# {args.title}\n\n" + "\n\n".join(sections) + "\n"
    with open(os.path.join(args.results, "report.md"), "w") as handle:
        handle.write(report)
    print(report)


if __name__ == "__main__":
    main()
