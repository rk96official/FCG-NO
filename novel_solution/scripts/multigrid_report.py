import argparse
import glob
import json
import os

from scripts.main_report import TOLERANCE_KEY, cell, markdown_table

BASELINE_ROWS = ("chebyshev8",)


# AMGX rows for one kind and grid: per-system setup and solve on the GPU (AMGX solves one system at a time, so there is no batch-16 column)
def amgx_rows(amgx_dir, kind, grid):
    path = os.path.join(amgx_dir, f"amgx_{kind}_{grid}.json")
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        variants = json.load(handle)["variants"]
    rows = []
    for label, entry in variants.items():
        stats = entry["iterations"][TOLERANCE_KEY]
        setup, solve = entry["setup_ms_per_system"], entry["solve_ms_per_system"]
        rows.append([f"**{label}** (GPU, per system)", f"{100 * stats['failure_rate']:.1f}", cell(stats["median"]), cell(stats["p90"]), "", f"{setup:.2f}",
                     f"{solve:.2f}", f"{setup + solve:.2f}"])
    return rows


# Chebyshev(8)-PCG and AMG rows for one kind and grid, in the multigrid table's columns
def baseline_rows(classical_dir, amg_dir, kind, grid):
    rows = []
    path = os.path.join(classical_dir, f"classical_{kind}_{grid}.json")
    if os.path.exists(path):
        with open(path) as handle:
            variants = json.load(handle)["variants"]
        for label in BASELINE_ROWS:
            stats = variants[label]["iterations"][TOLERANCE_KEY]
            per_iteration = variants[label]["seconds_per_iteration"]["16"] * 1e3
            solve = None if stats["median"] is None else stats["median"] * per_iteration / 16
            rows.append([f"**{label}** (GPU)", f"{100 * stats['failure_rate']:.1f}", cell(stats["median"]), cell(stats["p90"]), f"{per_iteration:.3f}", "0",
                         cell(solve, 2), cell(solve, 2)])
    path = os.path.join(amg_dir, f"amg_{kind}_{grid}.json")
    if os.path.exists(path):
        with open(path) as handle:
            for label, entry in json.load(handle)["variants"].items():
                stats = entry["iterations"][TOLERANCE_KEY]
                rows.append([f"**AMG {label}** (CPU, 1 core)", f"{100 * stats['failure_rate']:.1f}", cell(stats["median"]), cell(stats["p90"]), "", "", "", "not comparable"])
    return rows


# One row per variant: failures, median and p90 iterations, ms per iteration and setup ms per system at batch 16, solve and total (setup + solve) ms per system
def variant_rows(summary):
    rows = []
    for label, entry in summary["variants"].items():
        stats = entry["iterations"][TOLERANCE_KEY]
        per_iteration = entry["seconds_per_iteration"]["16"] * 1e3
        setup = entry["setup_seconds"]["16"] * 1e3 / 16
        solve = None if stats["median"] is None else stats["median"] * per_iteration / 16
        rows.append([label, f"{100 * stats['failure_rate']:.1f}", cell(stats["median"]), cell(stats["p90"]), f"{per_iteration:.3f}", f"{setup:.2f}",
                     cell(solve, 2), cell(None if solve is None else solve + setup, 2)])
    return rows


# Multigrid-format report (E13, E15, E16, NeuralIF): per kind and grid, the Chebyshev / AMG baselines and every variant with setup, solve and total time
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/e13_multigrid")
    parser.add_argument("--classical", default="results/e0_classical")
    parser.add_argument("--amg", default="results/e0_amg")
    parser.add_argument("--amgx", default="results/e0_amgx")
    parser.add_argument("--pattern", default="multigrid_*.json")
    parser.add_argument("--title", default="E13 classical multigrid vs the learned coarse solver")
    args = parser.parse_args()
    header = ["method", "fail %", "median its", "p90 its", "ms/it b16", "setup ms/system", "solve ms/system", "total ms/system"]
    sections = []
    summaries = []
    for path in glob.glob(os.path.join(args.results, args.pattern)):
        with open(path) as handle:
            summaries.append(json.load(handle))
    for summary in sorted(summaries, key=lambda item: (item["kind"], item["grid"])):
        rows = baseline_rows(args.classical, args.amg, summary["kind"], summary["grid"]) + amgx_rows(args.amgx, summary["kind"], summary["grid"]) + variant_rows(summary)
        sections.append(f"## {summary['kind']} {summary['grid']}² at 1e-6 (100 systems; batch 16 timings)\n\n" + markdown_table(header, rows))
    report = f"# {args.title}\n\n" + "\n\n".join(sections) + "\n"
    with open(os.path.join(args.results, "report.md"), "w") as handle:
        handle.write(report)
    print(report)


if __name__ == "__main__":
    main()
