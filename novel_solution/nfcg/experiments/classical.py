import os

import numpy as np
import jax.numpy as jnp
import equinox as eqx

from fcgno.data.assembly import apply_stencil
from fcgno.experiments.runner import write_json
from fcgno.experiments.settings import resolve_iterations
from fcgno.linalg.krylov import fcg
from nfcg.data.test_sets import initial_guesses, load_test_set
from nfcg.eval.per_system import per_system_iterations, summarize_systems
from nfcg.eval.timing import seconds_per_iteration
from nfcg.linalg.smoothers import make_smoother, stencil_diagonal

CLASSICAL_AXES = ("kind", "grid")


# Name of one classical-baseline run
def classical_name(run):
    return f"classical_{run['kind']}_{run['grid']}"


# Label of a classical preconditioner, e.g. "chebyshev4"
def smoother_label(variant):
    return "cg" if variant["smoother"] == "identity" else f"{variant['smoother']}{variant['sweeps']}"


# Jitted batched PCG (FCG with a fixed linear SPD preconditioner) with Jacobi or Chebyshev sweeps, or CG for identity; returns residual norms
@eqx.filter_jit
def classical_chunk(face_x, face_y, rhs, guess, n_iter, m_max, normalization, smoother, sweeps):
    operator = lambda u: apply_stencil(face_x, face_y, u, normalization)
    return fcg(operator, make_smoother(smoother, operator, stencil_diagonal(face_x, face_y, normalization), sweeps), rhs, guess, n_iter, m_max)["residual_norm"]


# GPU classical baselines (CG, Jacobi-PCG, Chebyshev-PCG) on the 100-system test set: per-system iterations and time per iteration
def execute_classical_run(run, config, log=print):
    path = os.path.join(config["results_dir"], f"{classical_name(run)}.json")
    if os.path.exists(path):
        log(f"skip {classical_name(run)}: result exists")
        return
    test, grid, normalization = config["test"], run["grid"], config["normalization"]
    dataset = load_test_set(config["paper_cache_dir"], config["cache_dir"], run["kind"], grid, test["n_test"], normalization)
    guesses = initial_guesses(test["n_test"], grid)
    n_iter = resolve_iterations(test["cap"], grid)
    summaries, arrays = {}, {}
    for variant in config["variants"]:
        curves = []
        for start in range(0, len(guesses), test["chunk"]):
            part = [jnp.asarray(dataset[name][start:start + test["chunk"]]) for name in ("face_x", "face_y", "rhs")]
            norms = np.asarray(classical_chunk(*part, jnp.asarray(guesses[start:start + test["chunk"]]), n_iter, test["m_max"], normalization,
                                               variant["smoother"], variant["sweeps"]))
            curves.append(norms / norms[:, :1])
        curves = np.concatenate(curves)
        timings = {}
        for batch in test["timing_batches"]:
            arguments = [jnp.asarray(dataset[name][:batch]) for name in ("face_x", "face_y", "rhs")] + [jnp.asarray(guesses[:batch])]
            solve = lambda *values: classical_chunk(*values, test["timing_iterations"], test["m_max"], normalization, variant["smoother"], variant["sweeps"])
            timings[str(batch)] = seconds_per_iteration(solve, arguments, test["timing_iterations"])
        label = smoother_label(variant)
        summaries[label] = {"iterations": {f"{tol:g}": summarize_systems(curves, tol) for tol in test["tolerances"]}, "seconds_per_iteration": timings}
        arrays[f"{label}__iterations"] = np.stack([per_system_iterations(curves, tol) for tol in test["tolerances"]])
        if test.get("save_curves"):
            arrays[f"{label}__curves"] = curves.astype(np.float32)
        stats = summaries[label]["iterations"]["1e-06"]
        log(f"{classical_name(run)} {label}: 1e-6 median {stats['median']} p90 {stats['p90']} failures {stats['failure_rate']:.2f} | "
            f"ms/it {', '.join(f'b{batch} {1e3 * value:.3f}' for batch, value in timings.items())}")
    os.makedirs(config["results_dir"], exist_ok=True)
    np.savez_compressed(path[:-len(".json")] + ".npz", tolerances=np.array(test["tolerances"]), **arrays)
    write_json(path, {**run, "variants": summaries})
