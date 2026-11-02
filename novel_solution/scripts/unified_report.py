import argparse
import csv
import json
import os

import numpy as np

from nfcg.eval.per_system import per_system_iterations

KINDS = ("poisson", "diffusion")
GRIDS = (32, 64, 128)
TOLERANCES = (1e-3, 1e-6, 1e-12)
SEEDS = (0, 1, 2, 3, 4)
TIMING_BATCH = "1"
CLASSICAL_CPU = (("CG", "identity"), ("Jacobi(4)", "jacobi_4"), ("SGS(1)", "sgs_1"), ("SGS(4)", "sgs_4"), ("ILU(1)", "ilu_1"), ("ILU(8)", "ilu_8"))
ARCHITECTURES = (("FCG-FNO", "fno"), ("FCG-U-Net", "unet"), ("FCG-DilResNet", "dilresnet"))
CLASSICAL_GPU = (("CG (GPU)", "cg"), ("Jacobi(4)-PCG (GPU)", "jacobi4"), ("Chebyshev(8)-PCG (GPU)", "chebyshev8"))
SCALING_GRIDS = (128, 256, 512, 1024)
PAPER_MODELS = (("fcgno_sno_paper", "FCG-NO (SNO), paper settings"), ("fcgno_sno_notebook", "FCG-NO (SNO), notebook settings"), ("fcg_fno", "FCG-FNO"),
                ("fcg_unet", "FCG-U-Net"), ("fcg_dilresnet", "FCG-DilResNet"))
OURS_SCALING = (("e16_main", "final_2l2", "multi_b64_tf32", "HC-NO multi-level"), ("e16_main", "rec_b32", "recursive_b32_max64_tf32", "HC-NO recursive"),
                ("e9_ablation_main", "abl_m0", "multi_b64_tf32", "NO in-role, no history, multi-level"))
OURS = (("final_2l2", "two_level_c2_tf32", "HC-NO two-level"), ("abl_m0", "two_level_c2_tf32", "NO in-role, no history, two-level"))


# Iterations per system at each tolerance from a stack of relative residual curves (systems, steps)
def counts_from_curves(curves):
    return np.stack([per_system_iterations(curves, tol) for tol in TOLERANCES])


# Pooled counts (tolerances, systems over all files) of the stored curves under one key, or None when no file exists
def pooled_curves(paths, key):
    found = [np.load(path)[key] for path in paths if os.path.exists(path)]
    return None if not found else np.concatenate([counts_from_curves(curves) for curves in found], axis=1)


# Pooled counts from per-system iteration arrays saved by the nfcg executors (rows ordered like their tolerances), with the mean seconds per iteration at batch 1
def pooled_arrays(paths, key, json_key):
    counts, seconds = [], []
    for path in paths:
        if not os.path.exists(path):
            continue
        stored = np.load(path)
        order = [int(np.argmin(np.abs(np.log(stored["tolerances"]) - np.log(tol)))) for tol in TOLERANCES]
        counts.append(stored[key][order])
        variants = json.load(open(path[:-len(".npz")] + ".json"))["variants"]
        seconds.append(variants[json_key]["seconds_per_iteration"].get(TIMING_BATCH, np.nan))
    if not counts:
        return None, None
    return np.concatenate(counts, axis=1), (float(np.mean(seconds)) if np.all(np.isfinite(seconds)) else None)


# Pooled counts at 1e-6 over the first 20 systems (the reproduction's) of per-system arrays, with the mean seconds per iteration at batch 1
def pooled_first20(paths, key, json_key):
    counts, seconds = [], []
    for path in paths:
        if not os.path.exists(path):
            continue
        stored = np.load(path)
        counts.append(stored[key][int(np.argmin(np.abs(stored["tolerances"] - 1e-6)))][:20])
        seconds.append(json.load(open(path[:-len(".npz")] + ".json"))["variants"][json_key]["seconds_per_iteration"].get(TIMING_BATCH, np.nan))
    return (None, None) if not counts else (np.concatenate(counts), float(np.mean(seconds)))


# Median iterations and ms per system to 1e-6 at batch 1 as one cell; failures shown instead of a time
def scaling_cell(counts, seconds):
    if counts is None:
        return "—"
    text = cell(counts)
    if np.all(np.isfinite(counts)) and seconds is not None and np.isfinite(seconds):
        text += f" / {1e3 * float(np.quantile(counts, 0.5, method='inverted_cdf')) * seconds:.2f} ms"
    return text


# Scaling table: the paper's learned preconditioners and ours, trained at 128, iterations and time per system to 1e-6 at 128-1024 on the reproduction's 20 systems
def scaling_lines(kind, paper_models, results):
    lines = [f"\n## {kind}: trained at 128, tested at 128-1024 (median iterations to 1e-6 / ms per system at batch 1)\n",
             "| method | " + " | ".join(f"{grid}²" for grid in SCALING_GRIDS) + " |", "|---|" + "---|" * len(SCALING_GRIDS)]
    for label, name in PAPER_MODELS:
        for variant in ("plain_fp64", "plain_tf32"):
            cells = [scaling_cell(*pooled_first20([f"{paper_models}/{label}_{kind}_128to{grid}_seed0.npz"], f"{variant}__iterations", variant)) for grid in SCALING_GRIDS]
            lines.append(f"| {name} ({variant.split('_')[1].upper()}) | " + " | ".join(cells) + " |")
    for directory, model, variant, name in OURS_SCALING:
        cells = [scaling_cell(*pooled_first20([f"{results}/{directory}/{model}_{kind}_128to{grid}_seed{seed}.npz" for seed in SEEDS], f"{variant}__iterations", variant))
                 for grid in SCALING_GRIDS]
        lines.append(f"| {name} (ours, TF32, 5 seeds) | " + " | ".join(cells) + " |")
    return lines


# Table cell: pooled median iterations, with the share of systems that never reach the tolerance
def cell(counts):
    if counts is None:
        return "—"
    failures = float(np.mean(~np.isfinite(counts)))
    median = float(np.quantile(counts, 0.5, method="inverted_cdf"))
    text = "fail" if not np.isfinite(median) else f"{median:.0f}"
    return text if failures == 0 else f"{text} ({100 * failures:.0f}% fail)"


# Every method's pooled counts (tolerances, systems) and GPU seconds per iteration at batch 1, for one kind and grid
def method_rows(kind, grid, reproduction, unified_main, unified_classical):
    rows = []
    for label, key in CLASSICAL_CPU:
        rows.append((label, "paper", pooled_curves([f"{reproduction}/r1_classical/curves_{kind}_{grid}.npz"], key), None))
    sno = lambda directory, suffix: [f"{reproduction}/{directory}/sno_{kind}_{grid}_notay_krylov_n{grid}_upstream_seed{seed}{suffix}_test{grid}.npz" for seed in (0, 1, 2)]
    rows.append(("FCG-NO (SNO), paper settings", "paper", pooled_curves(sno("r2_losses", ""), "fcg_residual"), None))
    rows.append(("FCG-NO (SNO), authors' notebook settings", "paper", pooled_curves(sno("r2_notebook", "_notebook"), "fcg_residual"), None))
    for label, name in ARCHITECTURES:
        path = f"{reproduction}/r5_architectures/{name}_{kind}_{grid}_notay_krylov_n{grid}_upstream_seed0_test{grid}.npz"
        rows.append((label, "paper", pooled_curves([path], "fcg_residual"), None))
    if kind == "poisson":
        rows.append(("PCG-GNN (Li et al. 2023)", "paper", pooled_curves([f"{reproduction}/r7_li2023/pcg_gnn_poisson_{grid}_test{grid}.npz"], "pcg_residual"), None))
    for label, key in CLASSICAL_GPU:
        counts, seconds = pooled_arrays([f"{unified_classical}/classical_{kind}_{grid}.npz"], f"{key}__iterations", key)
        rows.append((label, "baseline", counts, seconds))
    for model, variant, label in OURS:
        for train_grid, suffix in ((grid, "trained per grid"), (128, "trained at 128")):
            if suffix == "trained at 128" and grid == 128:
                continue
            paths = [f"{unified_main}/{model}_{kind}_{train_grid}to{grid}_seed{seed}.npz" for seed in SEEDS]
            counts, seconds = pooled_arrays(paths, f"{variant}__iterations", variant)
            rows.append((f"{label} ({suffix})", "ours", counts, seconds))
    return rows


# Unified comparison on the reproduction's 20 test systems: one table per kind and tolerance (grids as columns) and the 128 time per system
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reproduction", default="../reimplementation/results")
    parser.add_argument("--unified-main", default="results/u1_unified_main")
    parser.add_argument("--unified-classical", default="results/u1_unified_classical")
    parser.add_argument("--paper-models", default="results/u1_paper_models")
    parser.add_argument("--results", default="results")
    parser.add_argument("--output", default="results/u1_unified")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    lines = ["# Unified comparison with the paper's methods\n",
             "Same 20 test systems and initial guesses as the reproduction (seed 12 / guess seed 1). Cells: median iterations to the relative residual "
             "tolerance pooled over seeds and systems (SNO 3 seeds, ours 5 seeds, others 1), with the share of systems that never reach it. Paper methods "
             "come from the reproduction runs (classical ones on the CPU); GPU time per system (batch 1, one system at a time) is to 1e-6.\n"]
    records = []
    for kind in KINDS:
        table = {grid: method_rows(kind, grid, args.reproduction, args.unified_main, args.unified_classical) for grid in GRIDS}
        labels = [row[0] for row in table[GRIDS[-1]]]
        for index, tol in enumerate(TOLERANCES):
            lines.append(f"\n## {kind}, tolerance {tol:g}\n")
            lines.append("| method | source | " + " | ".join(f"{grid}²" for grid in GRIDS) + (" | ms per system at 128² (batch 1) |" if tol == 1e-6 else " |"))
            lines.append("|---|---|" + "---|" * len(GRIDS) + ("---|" if tol == 1e-6 else ""))
            for label in labels:
                cells, source, timing = [], "", ""
                for grid in GRIDS:
                    match = next((row for row in table[grid] if row[0] == label), None)
                    counts = None if match is None or match[2] is None else match[2][index]
                    cells.append(cell(counts))
                    if match is not None:
                        source = match[1]
                        records.append({"kind": kind, "grid": grid, "tolerance": tol, "method": label, "source": source, "cell": cell(counts)})
                        if grid == GRIDS[-1] and tol == 1e-6 and match[3] is not None and counts is not None and np.all(np.isfinite(counts)):
                            timing = f"{1e3 * float(np.quantile(counts, 0.5, method='inverted_cdf')) * match[3]:.3f}"
                lines.append(f"| {label} | {source} | " + " | ".join(cells) + (f" | {timing or '—'} |" if tol == 1e-6 else " |"))
    for kind in KINDS:
        lines.extend(scaling_lines(kind, args.paper_models, args.results))
    with open(os.path.join(args.output, "report.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    with open(os.path.join(args.output, "unified.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "grid", "tolerance", "method", "source", "cell"])
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {args.output}/report.md and unified.csv ({len(records)} cells)")


if __name__ == "__main__":
    main()
