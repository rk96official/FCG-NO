import os
import time

import numpy as np
import pyamg

from fcgno.data.assembly import assemble_csr
from fcgno.experiments.runner import write_json
from fcgno.linalg.reference_krylov import fcg_numpy
from nfcg.data.test_sets import initial_guesses, load_test_set
from nfcg.eval.per_system import summarize_systems

AMG_AXES = ("kind", "grid")
SOLVERS = {"smoothed_aggregation": pyamg.smoothed_aggregation_solver, "ruge_stuben": pyamg.ruge_stuben_solver}


# Name of one AMG-baseline run
def amg_name(run):
    return f"amg_{run['kind']}_{run['grid']}"


# AMG-preconditioned FCG (one V-cycle per application, same FCG and stopping rule as the neural methods) on one system; CPU seconds for setup and solve
def solve_system(matrix, rhs, guess, solver, n_iter, m_max):
    start = time.perf_counter()
    hierarchy = SOLVERS[solver](matrix)
    setup_seconds = time.perf_counter() - start
    cycle = hierarchy.aspreconditioner(cycle="V")
    start = time.perf_counter()
    unused_iterate, norms = fcg_numpy(matrix, cycle.matvec, rhs, guess, n_iter, m_max)
    solve_seconds = time.perf_counter() - start
    return norms / norms[0], setup_seconds, solve_seconds / n_iter, hierarchy


# CPU AMG baselines (pyamg smoothed aggregation and Ruge-Stuben) on the 100-system test set: per-system iterations, setup and per-iteration time
def execute_amg_run(run, config, log=print):
    path = os.path.join(config["results_dir"], f"{amg_name(run)}.json")
    if os.path.exists(path):
        log(f"skip {amg_name(run)}: result exists")
        return
    test, grid, normalization = config["test"], run["grid"], config["normalization"]
    dataset = load_test_set(config["paper_cache_dir"], config["cache_dir"], run["kind"], grid, test["n_test"], normalization)
    guesses = initial_guesses(test["n_test"], grid)
    summaries = {}
    for solver in config["solvers"]:
        curves, setups, per_iteration = [], [], []
        for index in range(test["n_test"]):
            matrix = assemble_csr(dataset["face_x"][index], dataset["face_y"][index], normalization)
            curve, setup_seconds, iteration_seconds, hierarchy = solve_system(matrix, dataset["rhs"][index].ravel(), guesses[index].ravel(),
                                                                               solver, test["n_iter"], test["m_max"])
            curves.append(curve)
            setups.append(setup_seconds)
            per_iteration.append(iteration_seconds)
        curves = np.stack(curves)
        summaries[solver] = {"iterations": {f"{tol:g}": summarize_systems(curves, tol) for tol in test["tolerances"]},
                             "setup_seconds_median": float(np.median(setups)), "seconds_per_iteration_median": float(np.median(per_iteration)),
                             "levels": len(hierarchy.levels), "operator_complexity": float(hierarchy.operator_complexity())}
        stats = summaries[solver]["iterations"]["1e-06"]
        log(f"{amg_name(run)} {solver}: 1e-6 median {stats['median']} p90 {stats['p90']} failures {stats['failure_rate']:.2f} | "
            f"setup {1e3 * summaries[solver]['setup_seconds_median']:.2f} ms, {1e3 * summaries[solver]['seconds_per_iteration_median']:.3f} ms/it, "
            f"{summaries[solver]['levels']} levels")
    write_json(path, {**run, "pyamg_version": pyamg.__version__, "variants": summaries})
