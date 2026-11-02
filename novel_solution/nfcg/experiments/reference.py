import json
import os
import time

import numpy as np

from fcgno.eval.metrics import iterations_to_tolerance
from fcgno.experiments.runner import write_json
from fcgno.experiments.settings import resolve_iterations
from nfcg.data.test_sets import PAPER_TEST_SIZE, initial_guesses, load_test_set
from nfcg.eval.per_system import per_system_iterations, summarize_systems
from nfcg.experiments.checkpoints import load_checkpoint, sno_checkpoint_name
from nfcg.experiments.solve import relative_residuals

REFERENCE_AXES = ("kind", "grid", "seed", "tag")


# Name of one E0 reference run
def reference_name(run):
    return f"sno_{run['kind']}_{run['grid']}_seed{run['seed']}_{run['tag']}"


# Mean-curve iterations of the first 20 systems within the stored run's budget, next to the stored fcgno value
def reproduction_check(curves, stored_row, tolerances):
    ours = curves[:PAPER_TEST_SIZE, :stored_row["n_iter"] + 1].mean(axis=0)
    rows = {}
    for tol in tolerances:
        key = f"{tol:g}"
        value = int(iterations_to_tolerance(ours, tol))
        stored = stored_row["fcg"][key]["iterations_mean_curve"]
        rows[key] = {"ours": value, "stored": stored, "match": value == stored}
    return rows


# Stored fcgno result row of a checkpoint on its own training grid
def stored_result(results_dir, name, grid):
    with open(os.path.join(results_dir, f"{name}.json")) as handle:
        return next(row for row in json.load(handle) if row["test_grid"] == grid)


# Evaluate one existing checkpoint on the extended test set: reproduction check, per-system FCG-NO and CG statistics, training cost
def execute_reference_run(run, config, log=print):
    path = os.path.join(config["results_dir"], f"{reference_name(run)}.json")
    if os.path.exists(path):
        log(f"skip {reference_name(run)}: result exists")
        return
    test, grid, normalization = config["test"], run["grid"], config["normalization"]
    name = sno_checkpoint_name(run["kind"], grid, run["seed"], run["tag"])
    model, spec = load_checkpoint(config["checkpoint_dir"], name)
    dataset = load_test_set(config["paper_cache_dir"], config["cache_dir"], run["kind"], grid, test["n_test"], normalization)
    guesses = initial_guesses(test["n_test"], grid)
    start = time.perf_counter()
    fcg_curves = relative_residuals(model, dataset, guesses, resolve_iterations(test["cap"], grid), test["m_max"], normalization, test["chunk"])
    fcg_seconds = time.perf_counter() - start
    cg_curves = relative_residuals(None, dataset, guesses, resolve_iterations(test["cg_cap"], grid), test["m_max"], normalization, test["chunk"])
    tolerances = test["tolerances"]
    result = {**run, "checkpoint": name, "train_minutes": spec["train_seconds"] / 60, "parameters": spec["parameters"],
              "evaluation_seconds": fcg_seconds,
              "reproduction": reproduction_check(fcg_curves, stored_result(config["reference_results"][run["tag"]], name, grid), test["reproduce_tolerances"]),
              "fcg": {f"{tol:g}": summarize_systems(fcg_curves, tol) for tol in tolerances},
              "cg": {f"{tol:g}": summarize_systems(cg_curves, tol) for tol in tolerances}}
    os.makedirs(config["results_dir"], exist_ok=True)
    np.savez(path[:-len(".json")] + ".npz", tolerances=np.array(tolerances), fcg_residual=fcg_curves.astype(np.float32),
             fcg_iterations=np.stack([per_system_iterations(fcg_curves, tol) for tol in tolerances]),
             cg_iterations=np.stack([per_system_iterations(cg_curves, tol) for tol in tolerances]))
    write_json(path, result)
    matches = all(row["match"] for row in result["reproduction"].values())
    summary = result["fcg"]["1e-06"]
    log(f"{reference_name(run)}: reproduction {'OK' if matches else 'MISMATCH ' + str(result['reproduction'])} | "
        f"FCG-NO 1e-6 failures {summary['failure_rate']:.2f} median {summary['median']} p90 {summary['p90']} | "
        f"CG median {result['cg']['1e-06']['median']} | train {result['train_minutes']:.1f} min")
