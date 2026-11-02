import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import yaml
from scipy import stats

from fcgno.experiments.settings import resolve_iterations
from nfcg.data.coefficient_features import coefficient_contrast
from nfcg.data.test_sets import load_test_set
from nfcg.eval.per_system import paired_wilcoxon, quantile, roc_auc

TOLERANCE_KEY = "1e-06"


# Markdown table from a header and rows of cells
def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows])


# Number or dash
def cell(value, digits=0):
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


# Training runs grouped by (kind, grid) and variant label: list of (summary, per-system iterations at 1e-6) sorted by seed
def load_runs(results_dir):
    groups = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(path) == "report.json":
            continue
        with open(path) as handle:
            summary = json.load(handle)
        with np.load(path[:-len(".json")] + ".npz") as stored:
            tolerances = [f"{tol:g}" for tol in stored["tolerances"]]
            counts = stored["iterations"][tolerances.index(TOLERANCE_KEY)]
        groups[(summary["kind"], summary["grid"])][summary["variant"]["label"]].append((summary, counts))
    return {key: {label: sorted(runs, key=lambda run: run[0]["seed"]) for label, runs in variants.items()} for key, variants in sorted(groups.items())}


# One row per variant: final-round failures and medians per seed, pooled median/p90, rounds progression, Wilcoxon p against the baseline, ms/it
def group_rows(variants, baseline, cap):
    rows = []
    reference = {run[0]["seed"]: run[1] for run in variants.get(baseline, [])}
    for label, runs in variants.items():
        finals = [run[0]["rounds"][-1]["iterations"][TOLERANCE_KEY] for run in runs]
        pooled = np.concatenate([run[1] for run in runs])
        progression = []
        for index in range(len(runs[0][0]["rounds"])):
            per_round = [run[0]["rounds"][index]["iterations"][TOLERANCE_KEY] for run in runs if index < len(run[0]["rounds"])]
            progression.append(f"r{index}: {100 * np.mean([entry['failure_rate'] for entry in per_round]):.0f}%/"
                               f"{cell(np.median([entry['median'] if entry['median'] is not None else np.inf for entry in per_round]))}")
        paired = [(counts, reference[run["seed"]]) for run, counts in runs if run["seed"] in reference]
        p_value = paired_wilcoxon(np.concatenate([pair[0] for pair in paired]), np.concatenate([pair[1] for pair in paired]), cap) if paired and label != baseline else None
        b16 = np.median([run[0]["seconds_per_iteration"]["16"] for run in runs]) * 1e3
        rows.append([label, len(runs), " / ".join(f"{100 * entry['failure_rate']:.0f}" for entry in finals),
                     " / ".join(cell(entry["median"]) for entry in finals), f"{100 * np.mean(~np.isfinite(pooled)):.1f}",
                     cell(quantile(pooled, 0.5)), cell(quantile(pooled, 0.9)), "; ".join(progression), cell(p_value, 4) if p_value is not None else "—",
                     f"{b16:.3f}"])
    return rows


# Contrast check for non-smooth kinds: AUC of the system's contrast for predicting failure, and Spearman correlation with iterations
def contrast_rows(variants, contrast, cap):
    rows = []
    for label, runs in variants.items():
        counts = np.concatenate([run[1] for run in runs])
        repeated = np.tile(contrast, len(runs))
        failed = ~np.isfinite(counts)
        auc = roc_auc(repeated, failed)
        rho = stats.spearmanr(repeated, np.where(failed, cap + 1, counts)).statistic
        rows.append([label, f"{int(failed.sum())}/{len(counts)}", cell(auc, 2) if auc is not None else "—", cell(rho, 2)])
    return rows


# Build a training-experiment report (E3, E4, E7) from its results and config; written next to the results
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline", default="m0_cg_notay")
    args = parser.parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    header = ["variant", "seeds", "fail % per seed", "median per seed", "pooled fail %", "pooled median", "pooled p90", "rounds: fail/median",
              f"Wilcoxon p vs {args.baseline}", "ms/it b16"]
    sections = []
    for (kind, grid), variants in load_runs(config["results_dir"]).items():
        cap = resolve_iterations(config["test"]["cap"], grid)
        sections.append(f"## {kind} {grid}² at 1e-6 (cap {cap})\n\n" + markdown_table(header, group_rows(variants, args.baseline, cap)))
        if kind in ("a1", "a2"):
            test_set = load_test_set(config["paper_cache_dir"], config["cache_dir"], kind, grid, config["test"]["n_test"], config["normalization"])
            sections.append(f"### {kind} {grid}²: do failures coincide with high contrast?\n\n"
                            + markdown_table(["variant", "failed/all", "AUC of contrast", "Spearman(contrast, iterations)"],
                                             contrast_rows(variants, coefficient_contrast(test_set), cap)))
    report = f"# {config['experiment']}\n\n" + "\n\n".join(sections) + "\n"
    with open(os.path.join(config["results_dir"], "report.md"), "w") as handle:
        handle.write(report)
    print(report)


if __name__ == "__main__":
    main()
