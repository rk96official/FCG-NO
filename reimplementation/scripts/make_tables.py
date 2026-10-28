import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import yaml


# All result rows saved by one experiment
def load_rows(results_root, experiment):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_root, experiment, "*.json"))):
        with open(path) as handle:
            rows.extend(json.load(handle))
    return rows


# Our value at a tolerance: NO+FCG mean-curve iterations or a timing saving field in percent; NaN when not reached or missing
def row_metric(row, tolerance, metric):
    if metric == "iterations":
        value = row["fcg"][tolerance]["iterations_mean_curve"]
        return np.nan if value < 0 else float(value)
    timing = (row.get("timing") or {}).get(tolerance)
    return np.nan if not timing or metric not in timing else timing[metric]


# Key matching a paper entry: the training grid, or "train-test" for cross-grid tables
def grid_key(row, cross_grid):
    return f"{row['grid']}-{row['test_grid']}" if cross_grid else row["grid"]


# Long-format comparison of our values (mean and std over seeds) with the paper for one table spec
def build_table(spec, tolerances, results_root):
    rows = [row for source in spec["sources"] for row in load_rows(results_root, source)
            if all(row.get(key) == value for key, value in spec["filter"].items())]
    rows = [row for row in rows if spec.get("cross_grid") or row["test_grid"] == row["grid"]]
    records = []
    for column_value, by_kind in spec["paper"].items():
        for kind, by_grid in by_kind.items():
            for grid, paper_values in by_grid.items():
                matching = [row for row in rows if row[spec["column"]] == column_value and row["kind"] == kind and grid_key(row, spec.get("cross_grid")) == grid]
                for tolerance, paper_value in zip(tolerances, paper_values):
                    ours = np.array([row_metric(row, tolerance, spec["metric"]) for row in matching], dtype=float)
                    records.append({
                        "kind": kind, "grid": grid, "column": column_value, "tolerance": tolerance, "runs": len(matching),
                        "ours_mean": np.nanmean(ours) if np.any(np.isfinite(ours)) else np.nan,
                        "ours_std": np.nanstd(ours) if np.sum(np.isfinite(ours)) > 1 else np.nan,
                        "paper": np.nan if paper_value is None else paper_value,
                    })
    return pd.DataFrame(records)


# One markdown cell "ours (paper)", with "—" for values that are missing or never reached
def format_cell(record, metric):
    show = lambda value: "—" if not np.isfinite(value) else (f"{value:.0f}%" if metric.startswith("saving") else f"{value:.0f}")
    ours = "not run" if record["runs"] == 0 else show(record["ours_mean"])
    if np.isfinite(record["ours_std"]):
        ours += f" ± {record['ours_std']:.1f}"
    return f"{ours} ({show(record['paper'])})"


# Markdown table with rows (kind, grid) and columns (column value, tolerance)
def to_markdown(spec, table):
    table = table.assign(cell=[format_cell(record, spec["metric"]) for record in table.to_dict("records")])
    pivot = table.pivot_table(index=["kind", "grid"], columns=["column", "tolerance"], values="cell", aggfunc="first", sort=False)
    header = "| kind | grid | " + " | ".join(f"{column} {tolerance}" for column, tolerance in pivot.columns) + " |"
    lines = [f"### {spec['title']}", "", "Cells: ours (paper).", "", header, "|" + "---|" * (len(pivot.columns) + 2)]
    lines += [f"| {kind} | {grid} | " + " | ".join(values) + " |" for (kind, grid), values in zip(pivot.index, pivot.values.tolist())]
    return "\n".join(lines)


# Flat CSV of every preconditioner result row with NO+FCG and CG iterations per tolerance
def all_runs_table(results_root, tolerances):
    records = []
    for path in sorted(glob.glob(os.path.join(results_root, "*", "*.json"))):
        with open(path) as handle:
            for row in json.load(handle):
                if "fcg" not in row:
                    continue
                record = {key: row[key] for key in ("name", "model", "kind", "grid", "test_grid", "loss", "sampling", "normalization", "seed")}
                for tolerance in tolerances:
                    record[f"fcg_{tolerance}"] = row["fcg"][tolerance]["iterations_mean_curve"]
                    record[f"cg_{tolerance}"] = row["cg"][tolerance]["iterations_mean_curve"]
                    record[f"saving_{tolerance}"] = row_metric(row, tolerance, "saving_percent")
                    record[f"saving_vs_identity_fcg_{tolerance}"] = row_metric(row, tolerance, "saving_vs_identity_fcg_percent")
                    record[f"saving_eager_{tolerance}"] = row_metric(row, tolerance, "saving_eager_percent")
                records.append(record)
    return pd.DataFrame(records)


# Build every paper comparison table that has results, plus a flat CSV of all runs
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="configs/paper_targets.yaml")
    parser.add_argument("--results", default="results")
    parser.add_argument("--output", default="results/tables")
    args = parser.parse_args()
    with open(args.targets) as handle:
        targets = yaml.safe_load(handle)
    os.makedirs(args.output, exist_ok=True)
    sections = []
    for name, spec in targets["tables"].items():
        table = build_table(spec, targets["tolerances"], args.results)
        table.to_csv(os.path.join(args.output, f"{name}.csv"), index=False)
        sections.append(to_markdown(spec, table))
    report = "\n\n".join(sections)
    with open(os.path.join(args.output, "comparison.md"), "w") as handle:
        handle.write(report + "\n")
    all_runs_table(args.results, targets["tolerances"]).to_csv(os.path.join(args.output, "all_runs.csv"), index=False)
    print(report)


if __name__ == "__main__":
    main()
