import os

import numpy as np
import jax.numpy as jnp
import equinox as eqx

from fcgno.data.assembly import apply_stencil
from fcgno.experiments.runner import write_json
from fcgno.experiments.settings import resolve_iterations
from fcgno.linalg.krylov import fcg
from fcgno.models.registry import as_preconditioner
from nfcg.data.test_sets import initial_guesses, load_test_set
from nfcg.eval.diagnostics import active_steps
from nfcg.eval.per_system import summarize_systems
from nfcg.eval.timing import seconds_per_iteration
from nfcg.experiments.checkpoints import load_checkpoint, sno_checkpoint_name
from nfcg.experiments.supporting import SUPPORTING_AXES, supporting_name
from nfcg.linalg.precision import low_precision_preconditioner
from nfcg.linalg.safeguarded_fcg import safeguarded_fcg
from nfcg.linalg.smoothers import make_smoother, smoothed_preconditioner, stencil_diagonal

COMPOSED_AXES = SUPPORTING_AXES


# Name of one composed run, e.g. "e2b_poisson_128to128_seed0_paper"
def composed_name(run):
    return "e2b" + supporting_name(run)[len("e2"):]


# Hashable form of a composed variant: (precision, wrapper, safeguard) option tuples, each None when that fix is off
def variant_key(variant):
    freeze = lambda part: None if part is None else tuple(sorted(part.items()))
    return freeze(variant.get("precision")), freeze(variant.get("wrapper")), freeze(variant.get("safeguard"))


# Solve one chunk with a composition of fixes: network precision, then the smoothing wrapper around it, then plain or safeguarded FCG
@eqx.filter_jit
def solve_composed(model, face_x, face_y, rhs, guess, n_iter, m_max, normalization, key):
    precision, wrapper, safeguard = [None if part is None else dict(part) for part in key]
    operator = lambda u: apply_stencil(face_x, face_y, u, normalization)
    diagonal = stencil_diagonal(face_x, face_y, normalization)
    network = as_preconditioner(model) if precision is None else low_precision_preconditioner(model, precision["dtype"], precision.get("matmul"))
    if wrapper is not None:
        network = smoothed_preconditioner(network, make_smoother(wrapper["smoother"], operator, diagonal, wrapper["sweeps"]), operator)
    if safeguard is None:
        return fcg(operator, network, rhs, guess, n_iter, m_max)
    fallback = make_smoother(safeguard["fallback"], operator, diagonal, safeguard.get("sweeps", 1))
    return safeguarded_fcg(operator, network, fallback, rhs, guess, n_iter, m_max, safeguard.get("every", 1))


# Relative residual curves and fallback shares of a composed variant over the test set, in chunks
def composed_test_set(model, dataset, guesses, n_iter, m_max, normalization, key, chunk):
    curves, shares = [], []
    for start in range(0, len(guesses), chunk):
        piece = [jnp.asarray(dataset[name][start:start + chunk]) for name in ("face_x", "face_y", "rhs")]
        outputs = solve_composed(model, *piece, jnp.asarray(guesses[start:start + chunk]), n_iter, m_max, normalization, key)
        norms = np.asarray(outputs["residual_norm"])
        curves.append(norms / norms[:, :1])
        if "fallback_share" in outputs:
            shares.append(np.asarray(outputs["fallback_share"]))
    return np.concatenate(curves), (np.concatenate(shares) if shares else None)


# Run every composed variant (wrapper, safeguard, precision combined) on one existing checkpoint; per-system iterations and GPU time per iteration
def execute_composed_run(run, config, log=print):
    path = os.path.join(config["results_dir"], f"{composed_name(run)}.json")
    if os.path.exists(path):
        log(f"skip {composed_name(run)}: result exists")
        return
    case, test, normalization = run["case"], config["test"], config["normalization"]
    name = sno_checkpoint_name(run["kind"], case["train_grid"], run["seed"], case["tag"])
    model, unused_spec = load_checkpoint(config["checkpoint_dir"], name)
    grid = case["test_grid"]
    dataset = load_test_set(config["paper_cache_dir"], config["cache_dir"], run["kind"], grid, test["n_test"], normalization)
    guesses = initial_guesses(test["n_test"], grid)
    n_iter = resolve_iterations(test["cap"], grid)
    summaries = {}
    for variant in config["variants"]:
        key = variant_key(variant)
        curves, shares = composed_test_set(model, dataset, guesses, n_iter, test["m_max"], normalization, key, test["chunk"])
        timings = {}
        for batch in test["timing_batches"]:
            arguments = [jnp.asarray(dataset[field][:batch]) for field in ("face_x", "face_y", "rhs")] + [jnp.asarray(guesses[:batch])]
            solve = lambda *values: solve_composed(model, *values, test["timing_iterations"], test["m_max"], normalization, key)
            timings[str(batch)] = seconds_per_iteration(solve, arguments, test["timing_iterations"])
        summary = {"iterations": {f"{tol:g}": summarize_systems(curves, tol) for tol in test["tolerances"]}, "seconds_per_iteration": timings}
        if shares is not None:
            summary["fallback_share"] = float(shares[active_steps(curves, test["active_floor"])].mean())
        summaries[variant["label"]] = summary
        stats = summary["iterations"]["1e-06"]
        log(f"{composed_name(run)} {variant['label']}: 1e-6 failures {stats['failure_rate']:.2f} median {stats['median']} p90 {stats['p90']} | "
            f"ms/it {', '.join(f'b{batch} {1e3 * value:.3f}' for batch, value in timings.items())}")
    write_json(path, {**run, "checkpoint": name, "variants": summaries})
