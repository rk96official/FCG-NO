import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NO_FCG_COLOR, CG_COLOR, PCG_COLOR = "red", "green", "navy"
RESIDUAL_LABEL = r"$\frac{\Vert r_i\Vert_2}{\Vert r_0\Vert_2}$"
ITERATIONS_LABEL = "# its."
NOTAY_LABEL = r"$L_{Notay}$"


# Matplotlib settings matching the paper's plots: bold sans-serif text, bold math, framed legends, no grid
def apply_paper_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
        "mathtext.default": "bf", "axes.labelsize": 11, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 9, "legend.frameon": True, "axes.grid": False, "savefig.bbox": "tight",
    })


# Hide the top and right spines, as in the paper's convergence plots
def open_axes(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


# Thin line with small markers on about 40 iterations (the paper's convergence-curve style)
def plot_curve(axis, values, color, label, marker="s", markersize=1.6, linewidth=0.8):
    axis.plot(np.arange(len(values)), values, color=color, marker=marker, markersize=markersize, linewidth=linewidth, label=label,
              markevery=max(1, len(values) // 40))


# Residual-decline axes: log scale, fraction label on the y axis, "# its." on the x axis
def residual_axes(axis, ylabel=True):
    axis.set_yscale("log")
    axis.set_xlabel(ITERATIONS_LABEL)
    if ylabel:
        axis.set_ylabel(RESIDUAL_LABEL, fontsize=14)
    open_axes(axis)


# Fig. 1 vertical scale: log10 below 1% and a stretched log10 above it, the paper's two scales on one axis
def split_log_scale(stretch=2.9, pivot=-2.0):
    # Data value to axis position
    def forward(values):
        logs = np.log10(np.clip(values, 1e-300, None))
        return np.where(logs >= pivot, pivot + (logs - pivot) * stretch, logs)

    # Axis position to data value
    def inverse(positions):
        return np.power(10.0, np.where(positions >= pivot, pivot + (positions - pivot) / stretch, positions))

    return forward, inverse


# Save a figure as vector PDF and PNG
def save_figure(figure, output, name):
    for extension in ("pdf", "png"):
        figure.savefig(f"{output}/{name}.{extension}", dpi=200)
    plt.close(figure)
    print(f"wrote {name}", flush=True)
