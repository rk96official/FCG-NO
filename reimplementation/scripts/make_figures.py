import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from scripts.paper_style import (NO_FCG_COLOR, CG_COLOR, PCG_COLOR, NOTAY_LABEL, ITERATIONS_LABEL, apply_paper_style,
                                 open_axes, plot_curve, residual_axes, split_log_scale, save_figure)

PAPER_ITERATIONS = 81


# Saved curves of the first run matching a pattern (seed 0 when the pattern names it), or None
def load_curves(results_root, experiment, pattern):
    matches = sorted(glob.glob(os.path.join(results_root, experiment, pattern)))
    return np.load(matches[0]) if matches else None


# Exact 80-iteration Notay and residual curves written by scripts/notay_curves.py, or an empty dict
def load_exact_curves(results_root):
    path = os.path.join(results_root, "paper", "notay_curves.npz")
    return dict(np.load(path)) if os.path.exists(path) else {}


# Mean relative-residual (or Notay) curve over test samples, truncated to a number of iterations
def mean_curve(curves, length=None):
    return np.mean(curves, axis=0)[:length]


# Cached dataset written by fcgno for a kind, grid, sample count and seed, or None
def load_dataset(results_root, kind, grid, n_samples, seed, normalization):
    path = os.path.join(results_root, "datasets", f"{kind}_n{grid}_samples{n_samples}_seed{seed}_mode4_alpha2_{normalization}.npz")
    return np.load(path) if os.path.exists(path) else None


# Fig. 1: mean relative test error vs resolution for NO, U-Net and NO+FCG, split log scale and an h^2 segment
def figure_1(results_root, output):
    paths = glob.glob(os.path.join(results_root, "r9_surrogates", "*.json"))
    rows = sorted((json.load(open(path))[0] for path in paths), key=lambda row: row["grid"])
    if not rows:
        return print("fig1: no r9_surrogates results")
    grids = np.array([row["grid"] for row in rows])
    figure, axis = plt.subplots(figsize=(4.0, 3.8))
    axis.plot(grids, [np.mean(row["sno"]) for row in rows], color="black", linestyle="--", linewidth=1.5, label="NO")
    axis.plot(grids, [np.mean(row["unet"]) for row in rows], color="black", marker="x", markersize=4, linewidth=1, label="U-Net")
    axis.plot(grids, [np.mean(row["no_fcg"]) for row in rows], color="black", marker=".", markersize=5, linewidth=1, label="NO+FCG")
    tail = grids[grids >= 128]
    if len(tail) > 1:
        anchor = 1.8 * np.mean(rows[list(grids).index(tail[0])]["no_fcg"])
        axis.plot(tail, anchor * (tail[0] / tail) ** 2, color="black", linewidth=1.5, label=r"$h^2$")
    axis.set_xscale("log", base=2)
    axis.set_yscale("function", functions=split_log_scale())
    axis.set_xticks(grids, [str(grid) for grid in grids])
    axis.minorticks_off()
    errors = np.concatenate([np.ravel([np.mean(row[key]) for row in rows]) for key in ("sno", "unet", "no_fcg")])
    axis.set_ylim(min(6e-6, errors.min() / 2), max(0.13, errors.max() * 1.3))
    axis.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1], [r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", "1%", "3%", "10%"])
    axis.set_xlabel("resolution")
    axis.set_ylabel("mean relative test error")
    axis.legend(loc="lower left")
    save_figure(figure, output, "figure_1")


# Sparsity pattern (rows, cols) of the 5-point operator on an n x n grid, for the Fig. 2 sketch
def stencil_pattern(n):
    index = np.arange(n * n).reshape(n, n)
    pairs = [(index, index), (index[1:, :], index[:-1, :]), (index[:-1, :], index[1:, :]), (index[:, 1:], index[:, :-1]), (index[:, :-1], index[:, 1:])]
    return np.concatenate([rows.ravel() for rows, _ in pairs]), np.concatenate([cols.ravel() for _, cols in pairs])


# Draw a stack of three offset field images (contour plots) at (x, y) in data units of the sketch axes
def field_stack(axis, fields, x, y, size, colormap):
    for offset, field in zip((4, 2, 0), fields[:3]):
        inset = axis.inset_axes([x + offset, y + offset, size, size], transform=axis.transData)
        inset.contourf(field.T, levels=12, cmap=colormap)
        inset.set_xticks([])
        inset.set_yticks([])


# Draw the sparse matrix, rhs stack and labels for one input dataset of the Fig. 2 sketch
def input_block(axis, title, rhs, top):
    rows, cols = stencil_pattern(6)
    spy = axis.inset_axes([2, top - 17, 13, 13], transform=axis.transData)
    spy.scatter(cols, rows, s=2, marker="s", color="#3b2a7a")
    spy.set_xlim(-1, 36)
    spy.set_ylim(36, -1)
    spy.axis("off")
    field_stack(axis, rhs, 18, top - 17, 11, "viridis")
    axis.text(15, top, title, ha="center", fontsize=10)
    axis.text(8.5, top - 22, "Sparse matrix,\n$A$", ha="center", fontsize=10)
    axis.text(24, top - 21, "$f$", ha="center", fontsize=11)


# Arrow with a label below or beside it in the Fig. 2 sketch
def labelled_arrow(axis, start, end, label, label_at):
    axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "color": "black", "linewidth": 0.8})
    axis.text(*label_at, label, ha="center", va="center", fontsize=10)


# Boxed text centred at (x, y) in the Fig. 2 sketch
def boxed_text(axis, x, y, text, width):
    axis.add_patch(FancyBboxPatch((x - width / 2, y - 3), width, 6, boxstyle="square,pad=0", fill=False, linewidth=0.8))
    axis.text(x, y, text, ha="center", va="center", fontsize=10)


# Fig. 2: scheme of the approach, drawn from our Poisson data (sparsity pattern, right-hand sides, solutions)
def figure_2(results_root, output):
    data = load_dataset(results_root, "poisson", 32, 20, 12, "upstream")
    if data is None:
        return print("fig2: no cached Poisson 32 test dataset")
    solutions = np.array([field if field[16, 16] > 0 else -field for field in data["solution"]])
    with plt.rc_context({"font.family": "serif", "font.weight": "normal", "mathtext.default": "it", "mathtext.fontset": "dejavuserif"}):
        figure, axis = plt.subplots(figsize=(9, 5.2))
        axis.set_xlim(0, 100)
        axis.set_ylim(0, 61)
        axis.axis("off")
        input_block(axis, r"Input train dataset, $\mathcal{D}_{\mathrm{train}}$", data["rhs"][:3], 58)
        input_block(axis, r"Input test dataset, $\mathcal{D}_{\mathrm{test}}$", data["rhs"][3:6], 27)
        labelled_arrow(axis, (35, 50), (47, 50), "(a)", (41, 48))
        boxed_text(axis, 60, 50, r"FCG($A,\ f,\ \mathcal{B}=I,\ldots$)", 24)
        labelled_arrow(axis, (60, 47), (60, 39.5), "(b)", (57, 43))
        axis.text(62, 43, r"$A,\, r_{\mathrm{iter}},\ A^{-1}f-u_{\mathrm{iter}}$", va="center", fontsize=10)
        boxed_text(axis, 60, 36.5, "NO", 6)
        labelled_arrow(axis, (60, 33.5), (60, 22), "(c)", (57, 28))
        labelled_arrow(axis, (35, 19), (47, 19), "(c)", (41, 17))
        boxed_text(axis, 60, 19, r"FCG($A,\ f,\ \mathcal{B}=$NO$,\ldots$)", 24)
        labelled_arrow(axis, (73, 19), (83, 19), "(d)", (78, 17))
        field_stack(axis, solutions, 84, 11, 11, "viridis_r")
        axis.text(91, 6, r"$u_{\mathrm{test}}$", ha="center", fontsize=11)
        save_figure(figure, output, "figure_2")


# Fig. 3: L_Notay (top) and residual decline (bottom) per iteration for CG and NO+FCG, Poisson grid 32
def figure_3(results_root, output):
    exact = load_exact_curves(results_root)
    if "poisson32_sno_notay" not in exact or "poisson32_cg_notay" not in exact:
        return print("fig3: run scripts/notay_curves.py first")
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(4.0, 5.4))
    plot_curve(top, mean_curve(exact["poisson32_sno_notay"]), NO_FCG_COLOR, "NO+FCG", marker=".")
    plot_curve(top, mean_curve(exact["poisson32_cg_notay"]), CG_COLOR, "CG", marker=".")
    top.set_yscale("log")
    top.set_ylabel(NOTAY_LABEL)
    open_axes(top)
    top.legend(loc="lower right")
    plot_curve(bottom, mean_curve(exact["poisson32_sno_residual"]), NO_FCG_COLOR, "NO+FCG", marker=".")
    plot_curve(bottom, mean_curve(exact["poisson32_cg_residual"]), CG_COLOR, "CG", marker=".")
    residual_axes(bottom)
    bottom.legend(loc="lower left")
    save_figure(figure, output, "figure_3")


# Fig. 4: L_Notay per iteration for SNO trained with the Notay and the L2 loss, Diffusion grid 32
def figure_4(results_root, output):
    exact = load_exact_curves(results_root)
    if "diffusion32_sno_notay" not in exact or "diffusion32_sno_l2_notay" not in exact:
        return print("fig4: run scripts/notay_curves.py first")
    figure, axis = plt.subplots(figsize=(4.0, 3.0))
    plot_curve(axis, mean_curve(exact["diffusion32_sno_notay"]), NO_FCG_COLOR, NOTAY_LABEL)
    plot_curve(axis, mean_curve(exact["diffusion32_sno_l2_notay"]), CG_COLOR, r"$L_2$")
    axis.set_yscale("log")
    axis.set_xlabel(ITERATIONS_LABEL)
    open_axes(axis)
    axis.legend(loc="lower right")
    save_figure(figure, output, "figure_4")


# Fig. 5: residual decline for FCG with SNO, CG and PCG with the Li et al. 2023 preconditioner, Poisson grid 32
def figure_5(results_root, output):
    exact = load_exact_curves(results_root)
    pcg = load_curves(results_root, "r7_li2023", "pcg_gnn_poisson_32_test32.npz")
    if "poisson32_sno_residual" not in exact or pcg is None:
        return print("fig5: run scripts/notay_curves.py and r7_li2023 first")
    figure, axis = plt.subplots(figsize=(4.6, 3.2))
    plot_curve(axis, mean_curve(exact["poisson32_sno_residual"]), NO_FCG_COLOR, "NO+FCG")
    plot_curve(axis, mean_curve(exact["poisson32_cg_residual"]), CG_COLOR, "CG")
    plot_curve(axis, mean_curve(pcg["pcg_residual"], PAPER_ITERATIONS), PCG_COLOR, "PCG")
    residual_axes(axis)
    axis.legend(loc="lower left")
    save_figure(figure, output, "figure_5")


# Fig. 6: the non-smooth coefficients a1(x) and a2(x) along x on a grid of 32 (one sample, middle row)
def figure_6(results_root, output):
    samples = [load_dataset(results_root, kind, 32, 20, 12, "upstream") for kind in ("a1", "a2")]
    if any(sample is None for sample in samples):
        return print("fig6: missing cached a1/a2 grid 32 test datasets")
    nodes = np.arange(1, 33) / 33
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for axis, sample in zip(axes, samples):
        axis.plot(nodes, sample["coefficient"][0][:, 16], color="black", marker=".", markersize=5, linewidth=1)
        axis.set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$a(x)$")
    save_figure(figure, output, "figure_6")


# Figs. 7 and 8: NO+FCG vs CG residual decline for a1 and a2 on grids 32 and 64
def figures_7_8(results_root, output):
    for number, grid in ((7, 32), (8, 64)):
        curves = {kind: load_curves(results_root, "r8_nonsmooth", f"sno_{kind}_{grid}_notay_krylov_*_upstream_seed0_test{grid}.npz") for kind in ("a1", "a2")}
        if any(data is None for data in curves.values()):
            print(f"fig{number}: missing r8_nonsmooth curves for grid {grid}")
            continue
        figure, axes = plt.subplots(1, 2, figsize=(9, 2.9))
        for index, (axis, (kind, data)) in enumerate(zip(axes, curves.items())):
            length = data["fcg_residual"].shape[1]
            plot_curve(axis, mean_curve(data["fcg_residual"]), NO_FCG_COLOR, "NO+FCG")
            plot_curve(axis, mean_curve(data["cg_residual"], length), CG_COLOR, "CG")
            residual_axes(axis, ylabel=index == 0)
            axis.set_title(rf"$-\mathrm{{div}}(a_{kind[1]}(x)\cdot\mathrm{{grad}}(u)) = f(x)$, grid = {grid}")
            axis.legend(loc="lower left")
        save_figure(figure, output, f"figure_{number}")


# Figs. 9-11: residual decline of SNO + FCG for the cases the paper reports as not converging
def figures_9_11(results_root, output):
    cases = [(9, "r2_losses", "sno_diffusion_128_l2_krylov_*_upstream_seed0_test128.npz"),
             (10, "r3_sampling", "sno_diffusion_64_notay_random_*_upstream_seed0_test64.npz"),
             (11, "r3_sampling", "sno_diffusion_128_notay_random_*_upstream_seed0_test128.npz")]
    for number, experiment, pattern in cases:
        data = load_curves(results_root, experiment, pattern)
        if data is None:
            print(f"fig{number}: no curves for {pattern}")
            continue
        figure, axis = plt.subplots(figsize=(4.4, 3.2))
        plot_curve(axis, mean_curve(data["fcg_residual"]), "black", "SNO + FCG", marker=".", markersize=3, linewidth=1.6)
        residual_axes(axis)
        axis.legend(loc="upper right")
        save_figure(figure, output, f"figure_{number}")


# Fig. S1: the stalled SNO + FCG run, Poisson 128^2 with paper settings: residual and exact Notay ratio per test problem
def figure_stall(results_root, output):
    exact = load_exact_curves(results_root)
    if "poisson128_sno_residual" not in exact:
        return print("figS1: run scripts/notay_curves.py first")
    residuals, notay = exact["poisson128_sno_residual"], exact["poisson128_sno_notay"]
    stalled = residuals[:, -1] > 1e-6
    figure, (left, right) = plt.subplots(1, 2, figsize=(9, 3.2))
    for axis, curves in ((left, residuals), (right, notay)):
        for index, curve in enumerate(curves):
            color, label = (NO_FCG_COLOR, f"stalled ({stalled.sum()})") if stalled[index] else (PCG_COLOR, f"converged ({(~stalled).sum()})")
            first = index == int(np.argmax(stalled == stalled[index]))
            axis.plot(np.arange(len(curve)), curve, color=color, linewidth=0.7, alpha=0.8, label=label if first else None)
    residual_axes(left)
    left.legend(loc="lower left")
    right.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Notay ratio = 1")
    right.set_yscale("log")
    right.set_xlabel(ITERATIONS_LABEL)
    right.set_ylabel(NOTAY_LABEL)
    right.set_xlim(0, 80)
    open_axes(right)
    right.legend(loc="lower right")
    save_figure(figure, output, "figure_S1")


# Draw every reproduced figure in the paper's style whose results exist
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--output", default="results/paper/figures")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    apply_paper_style()
    for draw in (figure_1, figure_2, figure_3, figure_4, figure_5, figure_6, figures_7_8, figures_9_11, figure_stall):
        draw(args.results, args.output)


if __name__ == "__main__":
    main()
