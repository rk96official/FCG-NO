import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

TOLERANCE_KEY = "1e-06"


# Markdown table from a header and rows of cells
def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows])


# Every E2 run summary grouped by (kind, case), with runs sorted by seed
def load_groups(results_dir):
    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(results_dir, "e2_*.json"))):
        with open(path) as handle:
            summary = json.load(handle)
        case = summary["case"]
        groups[(summary["kind"], f"{case['train_grid']}→{case['test_grid']} {case['tag']}")].append(summary)
    return {key: sorted(runs, key=lambda run: run["seed"]) for key, runs in sorted(groups.items())}


# Number or dash for a possibly missing value
def cell(value, digits=0):
    return "—" if value is None else f"{value:.{digits}f}"


# Extra statistics of a variant pooled over seeds: fallback share of the safeguard, or delta and the Lemma 3 check for precision variants
def extras(entries):
    if "fallback_share" in entries[0]:
        return f"fallback share {np.mean([entry['fallback_share'] for entry in entries]):.3f}"
    if "deviation_median" in entries[0]:
        return (f"δ median {np.median([entry['deviation_median'] for entry in entries]):.2e}, max {max(entry['deviation_max'] for entry in entries):.2e}, "
                f"Lemma 3 holds on {100 * min(entry['lemma_share'] for entry in entries):.1f}% of steps")
    return ""


# One row per solver variant of a group: failures and medians per seed, cost per iteration, estimated time to 1e-6 per system at batch 16
def group_rows(runs):
    rows = []
    for label in runs[0]["variants"]:
        entries = [run["variants"][label] for run in runs if label in run["variants"]]
        stats = [entry["iterations"][TOLERANCE_KEY] for entry in entries]
        b1 = np.median([entry["seconds_per_iteration"]["1"] for entry in entries]) * 1e3
        b16 = np.median([entry["seconds_per_iteration"]["16"] for entry in entries]) * 1e3
        medians = [stat["median"] for stat in stats]
        finite = [value for value in medians if value is not None]
        estimate = np.median(finite) * b16 / 16 if len(finite) == len(medians) else None
        rows.append([label, " / ".join(f"{100 * stat['failure_rate']:.0f}" for stat in stats), " / ".join(cell(value) for value in medians),
                     " / ".join(cell(stat["p90"]) for stat in stats), f"{b1:.3f}", f"{b16:.3f}", cell(estimate, 3), extras(entries)])
    return rows


# Build the E2 report (markdown): one table per (kind, case) over the solver variants
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/e2_supporting")
    args = parser.parse_args()
    header = ["variant", "fail % (s0/s1/s2)", "median its", "p90 its", "ms/it b1", "ms/it b16", "est. ms to 1e-6 per system (b16)", "notes"]
    sections = [f"## {kind}, {case} ({len(runs)} seeds)\n\n" + markdown_table(header, group_rows(runs)) for (kind, case), runs in load_groups(args.results).items()]
    report = "# E2: safeguard, smoothing wrapper, precision (no retraining), at 1e-6\n\n" + "\n\n".join(sections) + "\n"
    with open(os.path.join(args.results, "report.md"), "w") as handle:
        handle.write(report)
    print(report)


if __name__ == "__main__":
    main()
