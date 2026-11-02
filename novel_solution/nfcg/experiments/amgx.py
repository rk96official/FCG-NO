import os
import sys
import time
import traceback

import numpy as np

from fcgno.data.assembly import assemble_csr
from fcgno.experiments.runner import write_json
from nfcg.data.test_sets import initial_guesses, load_test_set
from nfcg.eval.per_system import per_system_iterations, summarize_systems

AMGX_AXES = ("kind", "grid")
HISTORY_TOLERANCE = 1e-10
TIMING_TOLERANCE = 1e-6


# Name of one AMGX run, e.g. "amgx_diffusion_512"
def amgx_name(run):
    return f"amgx_{run['kind']}_{run['grid']}"


# AMGX solver config (layout of AMGX's src/configs/PCG_CLASSICAL_V_JACOBI.json): PCG with one AMG V-cycle as preconditioner, relative tolerance on the
# initial residual, residual history stored
def solver_config(preconditioner, tolerance, max_iterations):
    amg = {**preconditioner, "solver": "AMG", "scope": "amg", "max_iters": 1, "cycle": "V", "monitor_residual": 0, "store_res_history": 0, "print_solve_stats": 0}
    return {"config_version": 2, "solver": {"solver": "PCG", "scope": "main", "preconditioner": amg, "max_iters": max_iterations, "monitor_residual": 1,
                                            "convergence": "RELATIVE_INI_CORE", "tolerance": tolerance, "norm": "L2", "store_res_history": 1,
                                            "print_solve_stats": 0, "obtain_timings": 0}}


# CSR matrix with the contiguous int32 indices and values in the mode's matrix precision that AMGX reads
def amgx_csr(matrix, dtype):
    matrix = matrix.tocsr().astype(dtype)
    matrix.indptr = np.ascontiguousarray(matrix.indptr, dtype=np.int32)
    matrix.indices = np.ascontiguousarray(matrix.indices, dtype=np.int32)
    matrix.data = np.ascontiguousarray(matrix.data, dtype=dtype)
    return matrix


# Matrix and vector precisions of an AMGX mode such as "dDDI" (double matrix and vectors) or "dFFI" (single)
def mode_dtypes(mode):
    return tuple(np.float32 if mode[index] == "F" else np.float64 for index in (1, 2))


# One AMGX session per process, as in pyamgx's own tests: initialize, a single Config and Resources for the whole run, finalize on close. AMGX keeps one
# device memory pool per process, so a second live Resources object makes its destruction free memory still in use and abort the process.
class AmgxSession:
    # Initialize AMGX and create the session's Resources
    def __init__(self, pyamgx):
        self.pyamgx = pyamgx
        pyamgx.initialize()
        self.config = pyamgx.Config().create("")
        self.resources = pyamgx.Resources().create_simple(self.config)

    # Destroy the Resources and Config, then finalize AMGX
    def close(self):
        self.resources.destroy()
        self.config.destroy()
        self.pyamgx.finalize()


# Solve the systems one after another with one solver configuration; every AMGX object is created and destroyed inside the call (reverse creation
# order), so no solver, matrix or vector outlives it even on error. system(index) gives (csr matrix, rhs, guess) in the mode's precisions.
def solve_systems(session, variant, system, count, tolerance, max_iterations):
    pyamgx, mode, created, results = session.pyamgx, variant["mode"], [], []
    try:
        config = pyamgx.Config().create_from_dict(solver_config(variant["preconditioner"], tolerance, max_iterations))
        created.append(config)
        operator = pyamgx.Matrix().create(session.resources, mode)
        created.append(operator)
        right = pyamgx.Vector().create(session.resources, mode)
        created.append(right)
        solution = pyamgx.Vector().create(session.resources, mode)
        created.append(solution)
        solver = pyamgx.Solver().create(session.resources, config, mode)
        created.append(solver)
        for index in range(count):
            matrix, rhs, guess = system(index)
            operator.upload(matrix.indptr, matrix.indices, matrix.data, shape=list(matrix.shape))
            right.upload(rhs)
            solution.upload(guess)
            start = time.perf_counter()
            solver.setup(operator)
            middle = time.perf_counter()
            solver.solve(right, solution, zero_initial_guess=False)
            end = time.perf_counter()
            history = np.array([solver.get_residual(step) for step in range(solver.iterations_number + 1)])
            results.append({"history": history, "setup": middle - start, "solve": end - middle})
    finally:
        for item in reversed(created):
            item.destroy()
    return results


# One AMGX configuration on the test set (mode dDDI double, dFFI single precision): residual curves from a pass to 1e-10, and per-system setup and solve
# seconds from a separate pass to 1e-6
def amgx_variant(session, variant, dataset, guesses, normalization, max_iterations):
    matrix_dtype, vector_dtype = mode_dtypes(variant["mode"])

    # System index as AMGX reads it, assembled on demand so only one matrix is in host memory at a time
    def system(index):
        matrix = amgx_csr(assemble_csr(dataset["face_x"][index], dataset["face_y"][index], normalization), matrix_dtype)
        return (matrix, np.ascontiguousarray(dataset["rhs"][index].ravel(), dtype=vector_dtype),
                np.ascontiguousarray(guesses[index].ravel(), dtype=vector_dtype))

    histories = solve_systems(session, variant, system, len(guesses), HISTORY_TOLERANCE, max_iterations)
    timings = solve_systems(session, variant, system, len(guesses), TIMING_TOLERANCE, max_iterations)
    curves = [result["history"] / result["history"][0] for result in histories]
    length = max(len(curve) for curve in curves)
    padded = np.stack([np.concatenate([curve, np.full(length - len(curve), curve[-1])]) for curve in curves])
    return padded, float(np.median([result["setup"] for result in timings])), float(np.median([result["solve"] for result in timings]))


# GPU AMG baseline (NVIDIA AMGX through pyamgx): classical and aggregation AMG-PCG per system, setup and solve timed per system (AMGX solves one system at a time)
def execute_amgx_run(run, config, log=print, pyamgx=None):
    if pyamgx is None:
        import pyamgx  # only in the amgx environment (slurm/build_amgx.slurm)
    name = amgx_name(run)
    path = os.path.join(config["results_dir"], f"{name}.json")
    if os.path.exists(path):
        log(f"skip {name}: result exists")
        return
    test, grid, normalization = config["test"], run["grid"], config["normalization"]
    dataset = load_test_set(config["paper_cache_dir"], config["cache_dir"], run["kind"], grid, test["n_test"], normalization)
    guesses = initial_guesses(test["n_test"], grid)
    session = AmgxSession(pyamgx)
    summaries, arrays = {}, {}
    try:
        for variant in config["variants"]:
            log(f"{name} {variant['label']}: start ({variant['mode']})")
            curves, setup, solve = amgx_variant(session, variant, dataset, guesses, normalization, test["max_iterations"])
            iterations = {f"{tol:g}": summarize_systems(curves, tol) for tol in test["tolerances"]}
            summaries[variant["label"]] = {"iterations": iterations, "setup_ms_per_system": 1e3 * setup, "solve_ms_per_system": 1e3 * solve}
            arrays[f"{variant['label']}__iterations"] = np.stack([per_system_iterations(curves, tol) for tol in test["tolerances"]])
            stats = iterations["1e-06"]
            log(f"{name} {variant['label']}: 1e-6 failures {stats['failure_rate']:.2f} median {stats['median']} p90 {stats['p90']} | "
                f"setup {1e3 * setup:.2f} ms, solve {1e3 * solve:.2f} ms per system")
    except Exception:
        # Print the Python error before finalize(), which can abort the process when AMGX is left in a bad state
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        session.close()
    os.makedirs(config["results_dir"], exist_ok=True)
    np.savez_compressed(path[:-len(".json")] + ".npz", tolerances=np.array(test["tolerances"]), **arrays)
    write_json(path, {**run, "name": name, "variants": summaries})
