import os

import numpy as np
import jax.numpy as jnp
import equinox as eqx

from fcgno.data.assembly import apply_stencil
from fcgno.experiments.runner import write_json
from fcgno.experiments.settings import resolve_iterations
from nfcg.data.coefficient_features import log_coefficient
from nfcg.data.test_sets import initial_guesses, load_test_set
from nfcg.eval.diagnostics import active_steps
from nfcg.eval.per_system import per_system_iterations, summarize_systems
from nfcg.eval.timing import seconds_per_iteration
from nfcg.experiments.checkpoints import load_checkpoint, reproduction_checkpoint_name, sno_checkpoint_name
from nfcg.linalg.history_fcg import history_fcg, history_safeguarded_fcg
from nfcg.linalg.precision import low_precision_model
from nfcg.linalg.smoothers import make_smoother, smoothed_history_preconditioner, stencil_diagonal
from nfcg.models.checkpoint import load_history_model
from nfcg.training.records import deployed_preconditioner

MAIN_AXES = ("kind", "seed", "model", "test_grid")


# Name of one main-table run (E6 and E8 share this runner and keep separate result directories), e.g. "h1_on_step_poisson_128to256_seed0"
def main_name(run):
    model = run["model"]
    return f"{model['label']}_{run['kind']}_{model['train_grid']}to{run['test_grid']}_seed{run['seed']}"


# Model of a run with its inputs: an fcgno notebook or reproduction checkpoint (memoryless) or an nfcg training checkpoint (history, coefficient); returns
# (model, history, coefficient)
def load_run_model(run, config):
    source = run["model"]
    if source["source"] == "notebook":
        model, unused_spec = load_checkpoint(config["notebook_checkpoint_dir"], sno_checkpoint_name(run["kind"], source["train_grid"], run["seed"], "notebook"))
        return model, 0, False
    if source["source"] == "reproduction":
        name = reproduction_checkpoint_name(source["architecture"], run["kind"], source["train_grid"], run["seed"], source.get("tag", "paper"))
        model, unused_spec = load_checkpoint(config["notebook_checkpoint_dir"], name)
        return model, 0, False
    path = os.path.join(config["checkpoint_dir"], f"{source['label']}_{run['kind']}_{source['train_grid']}_seed{run['seed']}_round{source['round']}")
    loaded = load_history_model(path)
    if loaded is None:
        raise FileNotFoundError(f"missing checkpoint {path}; its training experiment must finish first")
    model, spec = loaded
    return model, spec["history"], spec.get("coefficient", False)


# Hashable form of a variant: (precision, wrapper, safeguard, two_level) option tuples, each None when that fix is off
def main_key(variant):
    freeze = lambda part: None if part is None else tuple(sorted(part.items()))
    return tuple(freeze(variant.get(part)) for part in ("precision", "wrapper", "safeguard", "two_level"))


# Solve one chunk with any model (memoryless or history-aware) under a composition of fixes: network precision, then either the fine-grid
# smoothing wrapper or the two- or multi-level scheme (network on coarser grids), then plain or safeguarded FCG
@eqx.filter_jit
def solve_main(model, face_x, face_y, rhs, guess, log_coefficient_field, n_iter, m_max, normalization, history, key):
    precision, wrapper, safeguard, two_level = [None if part is None else dict(part) for part in key]
    operator = lambda u: apply_stencil(face_x, face_y, u, normalization)
    diagonal = stencil_diagonal(face_x, face_y, normalization)
    network = model if precision is None else low_precision_model(model, precision["dtype"], precision.get("matmul"))
    precondition = deployed_preconditioner(network, history, face_x, face_y, normalization, rhs.shape[-1], log_coefficient_field,
                                           None if two_level is None else tuple(sorted(two_level.items())))
    if wrapper is not None:
        precondition = smoothed_history_preconditioner(precondition, make_smoother(wrapper["smoother"], operator, diagonal, wrapper["sweeps"]), operator)
    stored = max(history, 1)
    if safeguard is None:
        return history_fcg(operator, precondition, rhs, guess, n_iter, m_max, stored)
    fallback = make_smoother(safeguard["fallback"], operator, diagonal, safeguard.get("sweeps", 1))
    return history_safeguarded_fcg(operator, precondition, fallback, rhs, guess, n_iter, m_max, stored, safeguard.get("every", 1))


# Relative residual curves and fallback shares of a variant over the test set, in chunks
def main_test_set(model, history, dataset, fields, guesses, n_iter, m_max, normalization, key, chunk):
    curves, shares = [], []
    for start in range(0, len(guesses), chunk):
        piece = [jnp.asarray(dataset[name][start:start + chunk]) for name in ("face_x", "face_y", "rhs")]
        field = None if fields is None else jnp.asarray(fields[start:start + chunk])
        outputs = solve_main(model, *piece, jnp.asarray(guesses[start:start + chunk]), field, n_iter, m_max, normalization, history, key)
        norms = np.asarray(outputs["residual_norm"])
        curves.append(norms / norms[:, :1])
        if "fallback_share" in outputs:
            shares.append(np.asarray(outputs["fallback_share"]))
    return np.concatenate(curves), (np.concatenate(shares) if shares else None)


# Seconds per iteration of a variant at one batch size
def main_timing(model, history, dataset, fields, guesses, m_max, normalization, key, batch, n_iter):
    arguments = [jnp.asarray(dataset[name][:batch]) for name in ("face_x", "face_y", "rhs")] + [jnp.asarray(guesses[:batch])]
    field = None if fields is None else jnp.asarray(fields[:batch])
    solve = lambda *values: solve_main(model, *values, field, n_iter, m_max, normalization, history, key)
    return seconds_per_iteration(solve, arguments, n_iter)


# Main table: one model (notebook baseline, HC-NO, memoryless on-policy) under every composition of fixes, on its own grid or a finer one
def execute_main_run(run, config, log=print):
    name = main_name(run)
    path = os.path.join(config["results_dir"], f"{name}.json")
    if os.path.exists(path):
        log(f"skip {name}: result exists")
        return
    test, normalization, grid = config["test"], config["normalization"], run["test_grid"]
    model, history, coefficient = load_run_model(run, config)
    dataset = load_test_set(config["paper_cache_dir"], config["cache_dir"], run["kind"], grid, test["n_test"], normalization)
    fields = log_coefficient(dataset) if coefficient else None
    guesses = initial_guesses(test["n_test"], grid)
    n_iter = resolve_iterations(test["cap"], grid)
    summaries, arrays = {}, {}
    for variant in config["variants"]:
        key = main_key(variant)
        curves, shares = main_test_set(model, history, dataset, fields, guesses, n_iter, test["m_max"], normalization, key, test["chunk"])
        timings = {str(batch): main_timing(model, history, dataset, fields, guesses, test["m_max"], normalization, key, batch, test["timing_iterations"])
                   for batch in test["timing_batches"]}
        summary = {"iterations": {f"{tol:g}": summarize_systems(curves, tol) for tol in test["tolerances"]}, "seconds_per_iteration": timings}
        if shares is not None:
            summary["fallback_share"] = float(shares[active_steps(curves, test["active_floor"])].mean())
        summaries[variant["label"]] = summary
        arrays[f"{variant['label']}__iterations"] = np.stack([per_system_iterations(curves, tol) for tol in test["tolerances"]])
        stats = summary["iterations"]["1e-06"]
        log(f"{name} {variant['label']}: 1e-6 failures {stats['failure_rate']:.2f} median {stats['median']} p90 {stats['p90']} | "
            f"ms/it {', '.join(f'b{batch} {1e3 * value:.3f}' for batch, value in timings.items())}")
    os.makedirs(config["results_dir"], exist_ok=True)
    np.savez_compressed(path[:-len(".json")] + ".npz", tolerances=np.array(test["tolerances"]), **arrays)
    write_json(path, {**run, "name": name, "history": history, "coefficient": coefficient, "variants": summaries})
