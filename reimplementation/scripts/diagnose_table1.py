import argparse

import numpy as np
import pandas as pd
import yaml

from fcgno.data.assembly import assemble_csr
from fcgno.data.generate import load_or_make_dataset
from fcgno.linalg.classical import build_classical
from fcgno.linalg.reference_krylov import fcg_numpy
from fcgno.eval.metrics import iterations_to_tolerance

VARIANTS = {
    "upstream A/N^2, u0~N(0,1)": {"matrix_scale": lambda n: 1.0, "random_guess": True},
    "u0 = 0": {"matrix_scale": lambda n: 1.0, "random_guess": False},
    "physical A (x N^2), u0~N(0,1)": {"matrix_scale": lambda n: float(n ** 2), "random_guess": True},
    "A x 1/N^2, u0~N(0,1)": {"matrix_scale": lambda n: 1.0 / n ** 2, "random_guess": True},
}


# Iterations to 1e-6 from the mean relative residual curve for one variant, preconditioner and test set
def count_iterations(dataset, spec, variant, guesses, n_iter, m_max):
    curves = []
    for face_x, face_y, rhs, guess in zip(dataset["face_x"], dataset["face_y"], dataset["rhs"], guesses):
        matrix = assemble_csr(face_x, face_y, normalization="upstream") * variant["matrix_scale"](rhs.shape[0])
        initial = guess.ravel() if variant["random_guess"] else np.zeros(rhs.size)
        preconditioner, _ = build_classical(spec, matrix)
        _, norms = fcg_numpy(matrix, preconditioner, rhs.ravel(), initial, n_iter, m_max)
        curves.append(norms / norms[0])
    return int(iterations_to_tolerance(np.mean(curves, axis=0), 1e-6))


# Compare setup variants against the paper's Table 1 to find how its classical baselines were run
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/r1_classical.yaml")
    parser.add_argument("--kind", default="poisson")
    parser.add_argument("--preconditioners", default="identity,jacobi:4,sgs:1,sgs:4")
    args = parser.parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    rows = []
    for grid in config["grids"]:
        dataset = load_or_make_dataset(config["cache_dir"], args.kind, grid, config["n_test"], config["test_seed"], normalization="upstream")
        guesses = np.random.default_rng(config["initial_guess_seed"]).standard_normal((config["n_test"], grid, grid))
        for spec in args.preconditioners.split(","):
            row = {"grid": grid, "preconditioner": config["paper_labels"][spec], "paper": config["paper_iterations_1e-6"][args.kind][grid][spec]}
            for name, variant in VARIANTS.items():
                row[name] = count_iterations(dataset, spec, variant, guesses, config["max_iter_factor"] * grid, config["m_max"])
            rows.append(row)
            print(row, flush=True)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
