import os

import numpy as np
import jax.numpy as jnp
import equinox as eqx

from fcgno.data.assembly import apply_stencil
from fcgno.experiments.runner import write_json
from fcgno.experiments.settings import resolve_iterations
from fcgno.models.registry import as_preconditioner
from nfcg.data.test_sets import initial_guesses, load_test_set
from nfcg.eval.diagnostics import active_steps, energy_outside_modes, retained_basis
from nfcg.eval.interventions import INTERVENTIONS
from nfcg.eval.per_system import summarize_systems
from nfcg.experiments.checkpoints import load_checkpoint, sno_checkpoint_name
from nfcg.linalg.traced_fcg import fcg_traced

DIAGNOSIS_AXES = ("kind", "grid", "seed", "tag")
TRACE_KEYS = ("notay", "preconditioned_sine_squared", "direction_sine_squared", "contraction", "error_probe", "intervened",
              "observer_notay", "observer_sine_squared")


# Name of one E1 diagnosis run
def diagnosis_name(run):
    return f"e1_{run['kind']}_{run['grid']}_seed{run['seed']}_{run['tag']}"


# Label of a solver variant, e.g. "angle_cap_0.9"
def variant_label(variant):
    return variant["name"] if variant.get("threshold") is None else f"{variant['name']}_{variant['threshold']:g}"


# Jitted traced solve of one chunk; variant = (name, threshold) where "cg_observer" runs CG and only evaluates the network on CG residuals
@eqx.filter_jit
def traced_chunk(model, face_x, face_y, rhs, guess, solution, basis, n_iter, m_max, normalization, variant):
    name, threshold = variant
    operator = lambda u: apply_stencil(face_x, face_y, u, normalization)
    network = as_preconditioner(model)
    probe = lambda error: energy_outside_modes(error, basis)
    if name == "cg_observer":
        return fcg_traced(operator, lambda r: r, rhs, guess, solution, n_iter, m_max, probe, observer=network)
    window = n_iter if name == "full_orthogonalization" else m_max
    intervention = INTERVENTIONS[name](threshold) if name in INTERVENTIONS else None
    return fcg_traced(operator, network, rhs, guess, solution, n_iter, window, probe, intervention)


# Traces of one variant over the whole test set, solved in chunks of fixed size
def traced_test_set(model, dataset, guesses, n_iter, m_max, normalization, variant, n_modes, chunk):
    basis = retained_basis(guesses.shape[-1], n_modes)
    parts = []
    for start in range(0, len(guesses), chunk):
        piece = [jnp.asarray(dataset[name][start:start + chunk]) for name in ("face_x", "face_y", "rhs", "solution")]
        traces = traced_chunk(model, *piece[:3], jnp.asarray(guesses[start:start + chunk]), piece[3], basis, n_iter, m_max, normalization,
                              (variant["name"], variant.get("threshold")))
        parts.append({name: np.asarray(values) for name, values in traces.items()})
    return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}


# Run every solver variant of E1 on one existing checkpoint and save per-step traces and per-system summaries
def execute_diagnosis_run(run, config, log=print):
    path = os.path.join(config["results_dir"], f"{diagnosis_name(run)}.json")
    if os.path.exists(path):
        log(f"skip {diagnosis_name(run)}: result exists")
        return
    test, grid, normalization = config["test"], run["grid"], config["normalization"]
    name = sno_checkpoint_name(run["kind"], grid, run["seed"], run["tag"])
    model, spec = load_checkpoint(config["checkpoint_dir"], name)
    dataset = load_test_set(config["paper_cache_dir"], config["cache_dir"], run["kind"], grid, test["n_test"], normalization)
    guesses = initial_guesses(test["n_test"], grid)
    n_iter = resolve_iterations(test["cap"], grid)
    arrays, summaries = {}, {}
    for variant in config["variants"]:
        label = variant_label(variant)
        traces = traced_test_set(model, dataset, guesses, n_iter, test["m_max"], normalization, variant, config["n_modes"], test["chunk"])
        relative = traces["residual_norm"] / traces["residual_norm"][:, :1]
        mask = active_steps(relative, test["active_floor"])
        arrays[f"{label}__relative_residual"] = relative.astype(np.float32)
        arrays.update({f"{label}__{key}": traces[key] if key == "intervened" else traces[key].astype(np.float32) for key in TRACE_KEYS if key in traces})
        summaries[label] = {"iterations": {f"{tol:g}": summarize_systems(relative, tol) for tol in test["tolerances"]},
                            "intervened_fraction": float(traces["intervened"][mask].mean()) if mask.any() else 0.0}
        summary = summaries[label]["iterations"]["1e-06"]
        log(f"{diagnosis_name(run)} {label}: 1e-6 failures {summary['failure_rate']:.2f} median {summary['median']} "
            f"p90 {summary['p90']} | intervened {summaries[label]['intervened_fraction']:.3f}")
    final_loss = spec["history"][-1]
    result = {**run, "checkpoint": name, "final_training_loss": final_loss, "collapsed": final_loss > config["collapse_loss"], "variants": summaries}
    os.makedirs(config["results_dir"], exist_ok=True)
    np.savez_compressed(path[:-len(".json")] + ".npz", **arrays)
    write_json(path, result)
