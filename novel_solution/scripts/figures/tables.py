import os

import numpy as np

from scripts.figures.analysis import ABLATIONS, ABLATION_GRID
from scripts.figures.data import (KINDS, SEEDS, amgx_entry, json_median, median_iterations, per_system, pooled, seconds_per_iteration,
                                  setup_seconds_per_system, solve_milliseconds, upper_iterations)
from scripts.figures.robustness import CONTRASTS, NEGATIVE_GRID, NETWORK_LABELS, THREE_D_GRIDS, negative_rows
from scripts.figures.style import METHODS
from scripts.unified_report import GRIDS as UNIFIED_GRIDS, method_rows

LARGE_GRID = 1024
REPORT_NAMES = {"HC-NO two-level (trained per grid)": "Ours, two-level (trained per grid)",
                "NO in-role, no history, two-level (trained per grid)": "Ours, two-level, no history (trained per grid)"}
SHORT_ABLATIONS = {"final_2l2": "Full recipe", "abl_m0": "No history", "abl_notay": "Notay loss", "abl_cg": "CG data (off-policy)", "abl_single": "Single-level (out of role)"}


# Iteration count as text: an integer, or "fail" when the median system never reaches the tolerance
def iterations_text(value):
    return "fail" if not np.isfinite(value) else f"{value:.0f}"


# Milliseconds as text with about three significant digits, or a dash when unavailable
def milliseconds_text(value):
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:,.0f}" if value >= 100 else (f"{value:.1f}" if value >= 10 else f"{value:.2f}")


# Iteration cell with the failure share, e.g. "187 (50\% fail)"
def iterations_with_failures(counts):
    if counts is None:
        return "--"
    failures = float(np.mean(~np.isfinite(counts)))
    text = iterations_text(median_iterations(counts))
    return text if failures == 0 else f"{text} ({100 * failures:.0f}\\% fail)"


# A booktabs LaTeX table (table* when wide) with a caption, a label, a column spec, header lines and body rows (lists of cells; a string row is a raw line)
def latex_table(caption, label, columns, header, rows, wide=False):
    environment = "table*" if wide else "table"
    spacing = [] if wide else ["\\setlength{\\tabcolsep}{3pt}"]
    lines = [f"\\begin{{{environment}}}[t]", "\\centering", "\\footnotesize", *spacing, f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{columns}}}", "\\toprule", *header, "\\midrule"]
    lines += [row if isinstance(row, str) else " & ".join(row) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}", f"\\end{{{environment}}}"]
    return "\n".join(lines) + "\n"


# Table 1: the unified comparison with the paper's methods on the reproduction's 20 systems at 1e-6 (U1), with GPU ms per system at 128² (batch 1)
def table_unified(results):
    reproduction = os.path.join(os.path.dirname(os.path.abspath(results)), "..", "reimplementation", "results")
    per_kind = {kind: {grid: method_rows(kind, grid, reproduction, os.path.join(results, "u1_unified_main"), os.path.join(results, "u1_unified_classical"))
                       for grid in UNIFIED_GRIDS} for kind in KINDS}
    labels = [row[0] for row in per_kind["poisson"][UNIFIED_GRIDS[-1]]]
    rows = []
    for label in labels:
        cells = [REPORT_NAMES.get(label, label).replace("&", "\\&")]
        times = []
        for kind in KINDS:
            for grid in UNIFIED_GRIDS:
                match = next((row for row in per_kind[kind][grid] if row[0] == label), None)
                cells.append("--" if match is None or match[2] is None else iterations_with_failures(match[2][1]))
            match = next((row for row in per_kind[kind][UNIFIED_GRIDS[-1]] if row[0] == label), None)
            times.append(np.nan if match is None or match[3] is None or match[2] is None else solve_milliseconds(match[2][1], match[3]))
        rows.append(cells + [milliseconds_text(value) for value in times])
    header = ["& \\multicolumn{3}{c}{Poisson} & \\multicolumn{3}{c}{Diffusion} & \\multicolumn{2}{c}{ms / system, 128$^2$} \\\\",
              "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}",
              "Method & " + " & ".join(f"{grid}$^2$" for grid in UNIFIED_GRIDS * 2) + " & Poisson & Diffusion \\\\"]
    caption = ("Median FCG iterations to a relative residual of $10^{-6}$ on the reproduction's 20 test systems, each model trained at its test grid "
               "(ours: 5 seeds; SNO: 3 seeds; others: 1). Paper rows come from the reproduction runs (classical ones on the CPU, no time); GPU times are per "
               "system, one system at a time (batch 1).")
    return latex_table(caption, "tab:unified", "l" + "c" * 8, header, rows, wide=True)


# Iterations and ms per system at batch 1 and 16 for one 1024² method: counts pooled over its files, seconds per iteration averaged, setup per system added
def large_grid_cells(paths, key, setup_key=None):
    counts = pooled(paths, key)
    cells = [iterations_text(median_iterations(counts))]
    for batch in ("1", "16"):
        per_iteration = float(np.nanmean([seconds_per_iteration(path, key, batch) for path in paths]))
        setup = 0.0 if setup_key is None else float(np.nanmean([setup_seconds_per_system(path, setup_key, batch) for path in paths]))
        cells.append(milliseconds_text(solve_milliseconds(counts, per_iteration, batch, setup)))
    return cells


# Table 2: ours against classical GPU solvers at 1024² (100 systems): iterations and ms per system including setup, at batch 1 and 16
def table_classical(results):
    rows = []
    entries = (("Ours, multi-level (5 seeds)", "e16_main", "final_2l2", "multi_b64_tf32"), ("Ours, recursive (5 seeds)", "e16_main", "rec_b32", "recursive_b32_max64_tf32"),
               ("Ours, no history input (5 seeds)", "e9_ablation_main", "abl_m0", "multi_b64_tf32"))
    for label, folder, model, key in entries:
        cells = [label]
        for kind in KINDS:
            cells += large_grid_cells([os.path.join(results, folder, f"{model}_{kind}_128to{LARGE_GRID}_seed{seed}.npz") for seed in SEEDS], key)
        rows.append(cells)
    for label, key in ((METHODS["redisc"]["label"] + " (dense 16$^2$ bottom)", "redisc_dense16"), (METHODS["galerkin"]["label"] + " V-cycle", "gmg_vcycle_fp32coarse")):
        cells = [label]
        for kind in KINDS:
            cells += large_grid_cells([os.path.join(results, "e16_multilevel", f"multigrid_{kind}_{LARGE_GRID}.npz")], key, key)
        rows.append(cells)
    cells = [METHODS["amgx"]["label"]]
    for kind in KINDS:
        counts, setup, solve = amgx_entry(results, kind, LARGE_GRID)
        cells += [iterations_text(median_iterations(counts)), milliseconds_text(setup + solve), "--"]
    rows.append(cells)
    cells = [METHODS["chebyshev"]["label"]]
    for kind in KINDS:
        path = os.path.join(results, "e0_classical", f"classical_{kind}_{LARGE_GRID}.json")
        median = json_median(path, "chebyshev8")
        cells += [iterations_text(median)] + [milliseconds_text(1e3 * median * seconds_per_iteration(path, "chebyshev8", batch) / int(batch)) for batch in ("1", "16")]
    rows.append(cells)
    header = ["& \\multicolumn{3}{c}{Poisson} & \\multicolumn{3}{c}{Diffusion} \\\\", "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
              "Method & its & ms (b1) & ms (b16) & its & ms (b1) & ms (b16) \\\\"]
    caption = (f"Ours against classical GPU solvers at {LARGE_GRID}$^2$ (100 test systems): median iterations to $10^{{-6}}$ and milliseconds per system, setup "
               "included, solving one system at a time (b1) or 16 at once (b16). AMGX solves one system at a time.")
    return latex_table(caption, "tab:classical", "lcccccc", header, rows, wide=True)


# Table 3: the ablations deployed multi-level at 1024²: median and p90 iterations over 5 seeds x 100 systems, and ms per system at batch 16
def table_ablations(results):
    rows = []
    for folder, model, unused_label in ABLATIONS:
        cells = [SHORT_ABLATIONS[model]]
        for kind in KINDS:
            paths = [os.path.join(results, folder, f"{model}_{kind}_128to{ABLATION_GRID}_seed{seed}.npz") for seed in SEEDS]
            counts = pooled(paths, "multi_b64_tf32")
            per_iteration = float(np.nanmean([seconds_per_iteration(path, "multi_b64_tf32", "16") for path in paths]))
            cells += [iterations_text(median_iterations(counts)), iterations_text(upper_iterations(counts)),
                      milliseconds_text(solve_milliseconds(counts, per_iteration, "16"))]
        rows.append(cells)
    header = ["& \\multicolumn{3}{c}{Poisson} & \\multicolumn{3}{c}{Diffusion} \\\\", "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
              "Variant & median & p90 & ms & median & p90 & ms \\\\"]
    caption = (f"Ablations, all deployed in the multi-level scheme at {ABLATION_GRID}$^2$: iterations to $10^{{-6}}$ over 5 seeds $\\times$ 100 systems and "
               "milliseconds per system at batch 16. Each variant changes one component of the full recipe: no history input, the Notay loss "
               "instead of the step loss, CG trajectories instead of on-policy data, or single-level training deployed multi-level.")
    return latex_table(caption, "tab:ablations", "lcccccc", header, rows)


# Table 4: 3D iterations at 32³-128³ and ms per system at 128³ (batch 1 and 16) for our networks against setup-free multigrid and Chebyshev(8)-PCG
def table_three_d(results):
    entries = (("Ours, multi-level (seed 0)", ("net_ml16_s0",)), ("Ours, multi-level, no history (seed 0)", ("net_m0_ml16_s0",)),
               ("Ours, two-level (3 seeds)", ("net_tl16_s0", "net_tl16_s1", "net_tl16_s2")), ("Setup-free multigrid (dense 8$^3$ bottom)", ("mg_dense8",)),
               ("Setup-free multigrid (Chebyshev 16$^3$ bottom)", ("mg_cheb16",)), (METHODS["chebyshev"]["label"], ("chebyshev8",)))
    rows = []
    for label, keys in entries:
        cells = [label]
        for kind in KINDS:
            paths = {grid: os.path.join(results, "e19_three_d", f"three_d_{kind}_{grid}.npz") for grid in THREE_D_GRIDS}
            counts = {grid: np.concatenate([per_system(paths[grid], key) for key in keys]) for grid in THREE_D_GRIDS}
            cells += [iterations_text(median_iterations(counts[grid])) for grid in THREE_D_GRIDS]
            for batch in ("1", "16"):
                per_iteration = float(np.mean([seconds_per_iteration(paths[THREE_D_GRIDS[-1]], key, batch) for key in keys]))
                cells.append(milliseconds_text(solve_milliseconds(counts[THREE_D_GRIDS[-1]], per_iteration, batch)))
        rows.append(cells)
    columns = " & ".join([f"{grid}$^3$" for grid in THREE_D_GRIDS] + ["ms (b1)", "ms (b16)"])
    header = ["& \\multicolumn{5}{c}{Poisson} & \\multicolumn{5}{c}{Diffusion} \\\\", "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}",
              f"Method & {columns} & {columns} \\\\"]
    caption = (f"3D problems: median iterations to $10^{{-6}}$ (32 test systems) and milliseconds per system at {THREE_D_GRIDS[-1]}$^3$, one system at a time "
               "(b1) or 16 at once (b16). No method has a setup phase.")
    return latex_table(caption, "tab:three_d", "l" + "c" * 10, header, rows, wide=True)


# Table 5: the negative results at 512²: median iterations against coefficient contrast for smooth fields (E15) and jumps (E20)
def table_negative(results):
    families = {"E15": negative_rows(results, "E15", "grf"), "E20": negative_rows(results, "E20", "jump")}
    rows = []
    for method in families["E15"]:
        rows.append([NETWORK_LABELS.get(method, METHODS[method]["label"])] + [iterations_text(value) for experiment in ("E15", "E20") for value in families[experiment][method]])
    header = ["& \\multicolumn{3}{c}{Smooth field (E15)} & \\multicolumn{3}{c}{Jumps (E20)} \\\\", "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
              "Method & " + " & ".join(f"$10^{exponent}$" for exponent in CONTRASTS * 2) + " \\\\"]
    caption = (f"Negative results at {NEGATIVE_GRID}$^2$: median iterations to $10^{{-6}}$ (100 systems) against the coefficient contrast. Our best network is "
               "the multi-level model on smooth fields and the model with a network also on the fine grid on jumps.")
    return latex_table(caption, "tab:negative", "lcccccc", header, rows)


TABLES = {"table1_unified": table_unified, "table2_classical": table_classical, "table3_ablations": table_ablations, "table4_three_d": table_three_d,
          "table5_negative": table_negative}


# Write every table as a standalone .tex file for \input in the report
def write_tables(results, output):
    os.makedirs(output, exist_ok=True)
    for name, build in TABLES.items():
        with open(os.path.join(output, f"{name}.tex"), "w") as handle:
            handle.write(build(results))
        print(f"wrote {output}/{name}.tex", flush=True)
