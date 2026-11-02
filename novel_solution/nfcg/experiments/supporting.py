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
from nfcg.linalg.precision import low_precision_preconditioner
from nfcg.linalg.safeguarded_fcg import safeguarded_fcg
from nfcg.linalg.smoothers import make_smoother, smoothed_preconditioner, stencil_diagonal
from nfcg.linalg.traced_fcg import fcg_traced

SUPPORTING_AXES = ("kind", "seed", "case")
SAVED_KEYS = ("notay", "preconditioned_sine_squared", "fallback_share", "observer_sine_squared", "observer_deviation")
LEMMA_TOLERANCE = 1e-6


# Name of one E2 run, e.g. "e2_poisson_32to128_seed0_transfer"
def supporting_name(run):
    case = run["case"]
    return f"e2_{run['kind']}_{case['train_grid']}to{case['test_grid']}_seed{run['seed']}_{case['tag']}"


# Label of a solver variant from its options, e.g. "safeguard_every2_fallbackchebyshev_sweeps2"
def variant_label(variant):
    return "_".join([variant["name"]] + [f"{key}{value}" for key, value in sorted(variant.items()) if key != "name"])


# Solve one chunk with a variant (options as a sorted tuple of pairs); traced runs record diagnostics against the exact solution
@eqx.filter_jit
def solve_variant(model, face_x, face_y, rhs, guess, solution, n_iter, m_max, normalization, variant, traced):
    options = dict(variant)
    operator = lambda u: apply_stencil(face_x, face_y, u, normalization)
    diagonal = stencil_diagonal(face_x, face_y, normalization)
    network = as_preconditioner(model)
    if options["name"] == "safeguard":
        fallback = make_smoother(options.get("fallback", "identity"), operator, diagonal, options.get("sweeps", 1))
        return safeguarded_fcg(operator, network, fallback, rhs, guess, n_iter, m_max, options.get("every", 1))
    observer = None
    if options["name"] == "cg":
        precondition = lambda residual: residual
    elif options["name"] == "base":
        precondition = network
    elif options["name"] == "wrapper":
        precondition = smoothed_preconditioner(network, make_smoother(options["smoother"], operator, diagonal, options["sweeps"]), operator)
    elif options["name"] == "precision":
        precondition, observer = low_precision_preconditioner(model, options["dtype"], options.get("matmul")), network
    else:
        raise ValueError(f"unknown variant: {options['name']}")
    if traced:
        return fcg_traced(operator, precondition, rhs, guess, solution, n_iter, m_max, observer=observer)
    return fcg(operator, precondition, rhs, guess, n_iter, m_max)


# Outputs of a variant over the whole test set, solved in chunks; arrays of shape (systems, steps)
def solve_test_set(model, dataset, guesses, n_iter, m_max, normalization, variant, chunk):
    options = tuple(sorted(variant.items()))
    parts = []
    for start in range(0, len(guesses), chunk):
        piece = [jnp.asarray(dataset[name][start:start + chunk]) for name in ("face_x", "face_y", "rhs", "solution")]
        outputs = solve_variant(model, *piece[:3], jnp.asarray(guesses[start:start + chunk]), piece[3], n_iter, m_max, normalization, options, True)
        parts.append({name: np.asarray(values) for name, values in outputs.items() if name != "iterate"})
    return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}


# Seconds per iteration of a variant on the first `batch` systems, without diagnostics
def variant_timing(model, dataset, guesses, m_max, normalization, variant, batch, n_iter):
    options = tuple(sorted(variant.items()))
    arguments = [jnp.asarray(dataset[name][:batch]) for name in ("face_x", "face_y", "rhs")]
    arguments += [jnp.asarray(guesses[:batch]), jnp.asarray(dataset["solution"][:batch])]
    solve = lambda *values: solve_variant(model, *values, n_iter, m_max, normalization, options, False)
    return seconds_per_iteration(solve, arguments, n_iter)


# Share of active steps where Lemma 3 holds: angle of the low-precision output <= angle of the float64 output + arcsin(delta)
def lemma_share(outputs, active):
    low = np.arcsin(np.sqrt(outputs["preconditioned_sine_squared"]))
    full = np.arcsin(np.sqrt(outputs["observer_sine_squared"]))
    bound = full + np.arcsin(np.minimum(outputs["observer_deviation"], 1.0))
    return float(np.mean((low <= bound + LEMMA_TOLERANCE)[active]))


# Variant summary: per-system iterations, time per iteration, and the fallback share or precision statistics when present
def summarize_variant(outputs, relative, active, test, timings):
    summary = {"iterations": {f"{tol:g}": summarize_systems(relative, tol) for tol in test["tolerances"]}, "seconds_per_iteration": timings}
    if "fallback_share" in outputs:
        summary["fallback_share"] = float(outputs["fallback_share"][active].mean())
    if "observer_deviation" in outputs:
        deviation = outputs["observer_deviation"][active]
        summary.update(deviation_median=float(np.median(deviation)), deviation_max=float(deviation.max()), lemma_share=lemma_share(outputs, active))
    return summary


# Run every E2 solver variant (safeguard, smoothing wrapper, precision) on one existing checkpoint, on its own or a finer test grid
def execute_supporting_run(run, config, log=print):
    path = os.path.join(config["results_dir"], f"{supporting_name(run)}.json")
    if os.path.exists(path):
        log(f"skip {supporting_name(run)}: result exists")
        return
    case, test, normalization = run["case"], config["test"], config["normalization"]
    name = sno_checkpoint_name(run["kind"], case["train_grid"], run["seed"], case["tag"])
    model, spec = load_checkpoint(config["checkpoint_dir"], name)
    grid = case["test_grid"]
    dataset = load_test_set(config["paper_cache_dir"], config["cache_dir"], run["kind"], grid, test["n_test"], normalization)
    guesses = initial_guesses(test["n_test"], grid)
    n_iter = resolve_iterations(test["cap"], grid)
    arrays, summaries = {}, {}
    for variant in config["variants"]:
        label = variant_label(variant)
        outputs = solve_test_set(model, dataset, guesses, n_iter, test["m_max"], normalization, variant, test["chunk"])
        relative = outputs["residual_norm"] / outputs["residual_norm"][:, :1]
        active = active_steps(relative, test["active_floor"])
        arrays[f"{label}__relative_residual"] = relative.astype(np.float32)
        arrays.update({f"{label}__{key}": outputs[key].astype(np.float32) for key in SAVED_KEYS if key in outputs})
        timings = {str(batch): variant_timing(model, dataset, guesses, test["m_max"], normalization, variant, batch, test["timing_iterations"])
                   for batch in test["timing_batches"]}
        summaries[label] = summarize_variant(outputs, relative, active, test, timings)
        stats = summaries[label]["iterations"]["1e-06"]
        log(f"{supporting_name(run)} {label}: 1e-6 failures {stats['failure_rate']:.2f} median {stats['median']} p90 {stats['p90']} | "
            f"ms/it {', '.join(f'b{batch} {1e3 * value:.3f}' for batch, value in timings.items())}")
    result = {**run, "checkpoint": name, "final_training_loss": spec["history"][-1], "variants": summaries}
    os.makedirs(config["results_dir"], exist_ok=True)
    np.savez_compressed(path[:-len(".json")] + ".npz", **arrays)
    write_json(path, result)
