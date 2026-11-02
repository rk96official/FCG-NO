import argparse
import json
import os

import numpy as np
import jax.numpy as jnp
import yaml

from fcgno.data.assembly import apply_stencil
from fcgno.linalg.krylov import batch_dot
from nfcg.data.test_sets import initial_guesses, load_test_set
from nfcg.experiments.main_table import load_run_model
from nfcg.linalg.multigrid import apply_stencil_operator, dense_factor, dense_solve, galerkin_stencil
from nfcg.linalg.precision import low_precision_model
from nfcg.linalg.smoothers import make_smoother, stencil_diagonal
from nfcg.linalg.two_level import interpolation_matrix, prolongate, restrict
from nfcg.training.records import collect_records, model_preconditioner

COARSE = 64
CHUNK = 16


# Squared cosine between batched fields in the inner product of a batched operator
def cosine_squared(apply_operator, first, second):
    return batch_dot(first, apply_operator(second)) ** 2 / (batch_dot(first, apply_operator(first)) * batch_dot(second, apply_operator(second)))


# Theorem 3 quantities for one chunk of records of a model in the two-level scheme (Chebyshev(2) pre-smoothing, network on the coarse grid)
def chunk_quantities(precondition_coarse, records, problems, indices, normalization):
    owner = records["owner"][indices]
    face_x, face_y = jnp.asarray(problems["face_x"][owner]), jnp.asarray(problems["face_y"][owner])
    residual, error = jnp.asarray(records["residual"][indices]), jnp.asarray(records["error"][indices])
    updates, changes = (jnp.moveaxis(jnp.asarray(records[name][indices]), 1, 0) for name in ("updates", "changes"))
    operator = lambda u: apply_stencil(face_x, face_y, u, normalization)
    smoothing = make_smoother("chebyshev", operator, stencil_diagonal(face_x, face_y, normalization), 2)
    correction = smoothing(residual)
    remainder_error, remainder = error - correction, residual - operator(correction)
    interpolation = interpolation_matrix(residual.shape[-1], COARSE)
    coarse_residual = restrict(remainder, interpolation)
    learned = precondition_coarse(coarse_residual, restrict(updates, interpolation), restrict(changes, interpolation)).astype(residual.dtype)
    stencil = galerkin_stencil(operator, residual.shape[0], COARSE, interpolation, residual.dtype)
    exact = dense_solve(dense_factor(stencil), coarse_residual)
    coarse_operator = lambda u: apply_stencil_operator(stencil, u)
    delta = 1.0 - cosine_squared(coarse_operator, learned, exact)
    exact_efficiency = cosine_squared(operator, prolongate(exact, interpolation), remainder_error)
    learned_efficiency = cosine_squared(operator, prolongate(learned, interpolation), remainder_error)
    deviation = jnp.abs(learned_efficiency - (1.0 - delta) * exact_efficiency) / exact_efficiency
    return [np.asarray(values) for values in (delta, exact_efficiency, learned_efficiency, deviation)]


# Measure Theorem 3 on the E14 models: along their own two-level FCG trajectories, the coarse loss delta against the dense Galerkin coarse solve
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e14_main.yaml")
    parser.add_argument("--label", default="final_2l2")
    parser.add_argument("--kinds", nargs="+", default=["poisson", "diffusion"])
    parser.add_argument("--grids", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--systems", type=int, default=20)
    parser.add_argument("--results", default="results/t1_theory")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))
    normalization = config["normalization"]
    rows, summary = [], {}
    for kind in args.kinds:
        for grid in args.grids:
            dataset = load_test_set(config["paper_cache_dir"], config["cache_dir"], kind, grid, args.systems, normalization)
            guesses = initial_guesses(args.systems, grid)
            pooled = [[], [], [], []]
            for seed in args.seeds:
                run = {"kind": kind, "seed": seed, "model": {"label": args.label, "source": "training", "train_grid": 128, "round": 3}}
                model, history, unused_coefficient = load_run_model(run, config)
                options = {"coarsen": grid // COARSE, "smoother": "chebyshev", "sweeps": 2}
                records = collect_records(model, history, dataset, guesses, 4 * grid, 20, normalization, 4, 1e-10, two_level=options)
                precondition_coarse = model_preconditioner(low_precision_model(model, "float32"), history)
                problems = {"face_x": dataset["face_x"], "face_y": dataset["face_y"]}
                for start in range(0, len(records["owner"]), CHUNK):
                    for target, values in zip(pooled, chunk_quantities(precondition_coarse, records, problems, np.arange(start, min(start + CHUNK, len(records["owner"]))),
                                                                        normalization)):
                        target.append(values)
            delta, exact, learned, deviation = (np.concatenate(values) for values in pooled)
            entry = {"records": int(len(delta)), "delta_median": float(np.median(delta)), "delta_p90": float(np.quantile(delta, 0.9)),
                     "exact_efficiency_median": float(np.median(exact)), "learned_efficiency_median": float(np.median(learned)),
                     "identity_max_relative_deviation": float(deviation.max())}
            summary[f"{kind}_{grid}"] = entry
            rows.append(f"| {kind} | {grid} | {entry['records']} | {entry['delta_median']:.4f} | {entry['delta_p90']:.4f} | {entry['exact_efficiency_median']:.3f} | "
                        f"{entry['learned_efficiency_median']:.3f} | {entry['identity_max_relative_deviation']:.1e} |")
            print(rows[-1], flush=True)
    os.makedirs(args.results, exist_ok=True)
    with open(os.path.join(args.results, "theory_check.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    header = ("# Theorem 3 check: learned coarse solve vs the exact Galerkin coarse solve\n\n"
              f"E14 `{args.label}` models (seeds {args.seeds}), network on the {COARSE} grid, records along their own two-level FCG trajectories "
              f"({args.systems} test systems each). delta = sin^2 in the Galerkin coarse norm; efficiency = cos^2_A of the prolongated coarse direction with the "
              "remainder error (the share of the energy the coarse step removes). Theorem 3 predicts learned = (1 - delta) * exact; the last column is the "
              "largest relative deviation.\n\n"
              "| kind | grid | records | median delta | p90 delta | median exact efficiency | median learned efficiency | identity max rel. dev. |\n"
              "|---|---|---|---|---|---|---|---|\n")
    with open(os.path.join(args.results, "report.md"), "w") as handle:
        handle.write(header + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
