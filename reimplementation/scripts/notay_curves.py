import argparse
import os

import numpy as np

from fcgno.data.generate import load_or_make_dataset
from fcgno.training.checkpoint import load_model
from fcgno.eval.evaluate import exact_notay_curves

CASES = {
    "poisson32_sno": ("poisson", 32, "sno_poisson_32_notay_krylov_n32_upstream_seed0", None),
    "diffusion32_sno": ("diffusion", 32, "sno_diffusion_32_notay_krylov_n32_upstream_seed0", None),
    "diffusion32_sno_l2": ("diffusion", 32, "sno_diffusion_32_l2_krylov_n32_upstream_seed0", None),
    "poisson32_cg": ("poisson", 32, None, None),
    "diffusion32_cg": ("diffusion", 32, None, None),
    "poisson128_sno": ("poisson", 128, "sno_poisson_128_notay_krylov_n128_upstream_seed0", 256),
}


# Exact Notay and residual curves on the R2 test sets: 80 iterations for Figs. 3-5, 2 * grid for the stalled 128^2 run (Fig. S1)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--output", default="results/paper")
    parser.add_argument("--n-iter", type=int, default=80)
    args = parser.parse_args()
    curves = {}
    for key, (kind, grid, checkpoint, n_iter) in CASES.items():
        loaded = load_model(os.path.join(args.results, "checkpoints", checkpoint)) if checkpoint else (None, None)
        if loaded is None:
            print(f"{key}: checkpoint {checkpoint} not found", flush=True)
            continue
        dataset = load_or_make_dataset(os.path.join(args.results, "datasets"), kind, grid, 20, 12, normalization="upstream")
        guesses = np.random.default_rng(1).standard_normal(dataset["rhs"].shape)
        result = exact_notay_curves(loaded[0], dataset, guesses, n_iter or args.n_iter, 20, "upstream")
        curves[f"{key}_notay"], curves[f"{key}_residual"] = result["notay"], result["relative_residual"]
        print(f"{key}: Notay at iterations 0, 10, 40: {np.round(result['notay'].mean(0)[[0, 10, 40]], 3)}", flush=True)
    os.makedirs(args.output, exist_ok=True)
    np.savez(os.path.join(args.output, "notay_curves.npz"), **curves)


if __name__ == "__main__":
    main()
