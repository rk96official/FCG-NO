import argparse
import glob
import os
import re

import numpy as np

from fcgno.eval.evaluate import iteration_table
from fcgno.experiments.runner import write_json

TOLERANCES = [1e-3, 1e-6, 1e-12]
KINDS = {"Poisson": "poisson", "Elliptic": "diffusion"}
SAMPLING = {"FCG": "krylov", "random": "random"}
FIXED = re.compile(r"notay_loss_SNO_(FCG|random)_(\d+)_(\d+)_(\d+)_(\d+)_data\.npz")
CROSS = re.compile(r"diff_grid_SNO_(FCG|random)_(\d+)-(\d+)_(\d+)_(\d+)_(\d+)_data\.npz")


# Result row in our format from one upstream output file, or None for files we do not recognise
def upstream_row(path, cross_grid_train):
    folder, name = os.path.basename(os.path.dirname(os.path.dirname(path))), os.path.basename(path)
    fixed, cross = FIXED.fullmatch(name), CROSS.fullmatch(name)
    if not (fixed or cross) or folder not in KINDS:
        return None
    kind = KINDS[folder]
    if fixed:
        gentype, grid, samples_div = fixed.group(1), int(fixed.group(2)), int(fixed.group(3))
        train, test, n_train = grid, grid, grid // samples_div
    else:
        gentype, train, test = cross.group(1), int(cross.group(2)), int(cross.group(3))
        n_train = cross_grid_train[kind] or train
    with np.load(path) as data:
        residuals = np.linalg.norm(data["R_model"], axis=1)
    curves = residuals / residuals[:, :1]
    unavailable = {f"{tol:g}": {"iterations_mean_curve": -1} for tol in TOLERANCES}
    return {"name": f"original_code_{kind}_{gentype}_{train}-{test}", "model": "sno", "kind": kind, "grid": train, "test_grid": test,
            "loss": "notay", "sampling": SAMPLING[gentype], "normalization": "upstream", "seed": 0, "tag": "original_code",
            "n_train": n_train, "n_iter": curves.shape[1] - 1, "source_file": path, "fcg": iteration_table(curves, TOLERANCES), "cg": unavailable}


# Convert every upstream pipeline output found under the work directory into result rows and curves
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="results/upstream_code/raw")
    parser.add_argument("--output", default="results/upstream_code")
    parser.add_argument("--poisson-cross-train", type=int, default=100, help="--nsamp passed to the Poisson cross-grid pipeline")
    args = parser.parse_args()
    cross_grid_train = {"poisson": args.poisson_cross_train, "diffusion": None}
    for path in sorted(glob.glob(os.path.join(args.workdir, "*", "SNO", "*_data.npz"))):
        row = upstream_row(path, cross_grid_train)
        if row is None:
            continue
        write_json(os.path.join(args.output, f"{row['name']}.json"), [row])
        print(row["name"], {tol: summary["iterations_mean_curve"] for tol, summary in row["fcg"].items()}, flush=True)


if __name__ == "__main__":
    main()
