import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from fcgno.data.assembly import apply_stencil
from fcgno.experiments.runner import write_json
from fcgno.experiments.settings import resolve_iterations
from fcgno.linalg.krylov import fcg
from nfcg.data.test_sets import initial_guesses, load_test_set
from nfcg.eval.per_system import per_system_iterations, summarize_systems
from nfcg.eval.timing import seconds_per_iteration
from nfcg.data.coefficient_features import log_coefficient
from nfcg.experiments.main_table import load_run_model, main_key, main_test_set, main_timing
from nfcg.linalg.multigrid import (build_hierarchy, cast_hierarchy, coarse_solver, hierarchy_interpolations, level_sizes, spectral_upper,
                                   vcycle_preconditioner)
from nfcg.linalg.multilevel import multilevel_preconditioner
from nfcg.linalg.smoothers import make_smoother, stencil_diagonal
from nfcg.linalg.two_level import interpolation_matrix, two_level_preconditioner

MULTIGRID_AXES = ("kind", "grid")
DTYPES = {None: None, "float32": jnp.float32}


# Name of one multigrid-baseline run, e.g. "multigrid_diffusion_256"
def multigrid_name(run):
    return f"multigrid_{run['kind']}_{run['grid']}"


# Hierarchy sizes and whether the coarsest level needs dense factors, for a classical variant on a fine grid
def variant_sizes(variant, grid):
    if variant["method"] == "gmg":
        return level_sizes(grid), True
    if variant["coarse"] == "vcycle":
        return level_sizes(grid, variant["coarse_grid"]), True
    return [grid, variant["coarse_grid"]], variant["coarse"] == "exact"


# Setup of a classical variant for a batch of systems: the Galerkin hierarchy, plus the fine Chebyshev bound for the V-cycle preconditioner
@eqx.filter_jit
def classical_setup(face_x, face_y, normalization, sizes, factor, interpolations):
    operator = lambda u: apply_stencil(face_x, face_y, u, normalization)
    levels = build_hierarchy(operator, face_x.shape[0], list(sizes), face_x.dtype, factor, interpolations)
    return levels, spectral_upper(operator, stencil_diagonal(face_x, face_y, normalization))


# Jitted FCG with a classical preconditioner on a prepared hierarchy: the GMG V-cycle, or the two-level scheme with a classical coarse solver
@eqx.filter_jit
def classical_solve(face_x, face_y, rhs, guess, setup, n_iter, m_max, normalization, options):
    variant = dict(options)
    levels, fine_upper = setup
    operator = lambda u: apply_stencil(face_x, face_y, u, normalization)
    diagonal = stencil_diagonal(face_x, face_y, normalization)
    if variant["method"] == "gmg":
        coarse_levels = levels if variant.get("coarse_dtype") is None else cast_hierarchy(levels, DTYPES[variant["coarse_dtype"]])
        precondition = vcycle_preconditioner(operator, diagonal, fine_upper, coarse_levels, variant["sweeps"])
    else:
        coarse = coarse_solver(variant["coarse"], levels, variant.get("coarse_sweeps", 2), DTYPES[variant.get("coarse_dtype")])
        smooth = make_smoother("chebyshev", operator, diagonal, variant.get("sweeps", 2))
        precondition = two_level_preconditioner(coarse, smooth, operator, interpolation_matrix(rhs.shape[-1], variant["coarse_grid"]))
    return fcg(operator, precondition, rhs, guess, n_iter, m_max)["residual_norm"]


# Jitted FCG with the setup-free classical multi-level preconditioner on rediscretized levels (its coarse faces and bottom factors are built inside the solve)
@eqx.filter_jit
def rediscretized_solve(face_x, face_y, rhs, guess, n_iter, m_max, normalization, options):
    operator = lambda u: apply_stencil(face_x, face_y, u, normalization)
    precondition = multilevel_preconditioner(face_x, face_y, normalization, dict(options))
    return fcg(operator, precondition, rhs, guess, n_iter, m_max)["residual_norm"]


# Curves and timings of a rediscretized variant; its setup is inside the timed solve, so the setup entries are zero
def rediscretized_variant(variant, dataset, guesses, test, n_iter, normalization):
    options, curves = classical_options(variant), []
    for start in range(0, len(guesses), test["chunk"]):
        piece = [jnp.asarray(dataset[name][start:start + test["chunk"]]) for name in ("face_x", "face_y", "rhs")]
        norms = np.asarray(rediscretized_solve(*piece, jnp.asarray(guesses[start:start + test["chunk"]]), n_iter, test["m_max"], normalization, options))
        curves.append(norms / norms[:, :1])
    timings = {}
    for batch in test["timing_batches"]:
        arguments = [jnp.asarray(dataset[name][:batch]) for name in ("face_x", "face_y", "rhs")] + [jnp.asarray(guesses[:batch])]
        solve = lambda *values: rediscretized_solve(*values, test["timing_iterations"], test["m_max"], normalization, options)
        timings[str(batch)] = seconds_per_iteration(solve, arguments, test["timing_iterations"])
    return np.concatenate(curves), timings, {str(batch): 0.0 for batch in test["timing_batches"]}


# Hashable options of a classical variant (the label is dropped so equal variants share a compilation)
def classical_options(variant):
    return tuple(sorted((name, value) for name, value in variant.items() if name != "label"))


# Median wall-clock seconds of the setup for one batch, compilation excluded
def setup_seconds(arguments, normalization, sizes, factor, interpolations, repeats=5):
    jax.block_until_ready(classical_setup(*arguments, normalization, sizes, factor, interpolations))
    durations = []
    for unused in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(classical_setup(*arguments, normalization, sizes, factor, interpolations))
        durations.append(time.perf_counter() - start)
    return float(np.median(durations))


# Curves and timings of a classical variant over the test set (setup per chunk; timing of setup and of solve iterations per batch size)
def classical_variant(variant, dataset, guesses, test, n_iter, normalization):
    sizes, factor = variant_sizes(variant, dataset["rhs"].shape[-1])
    sizes, options, curves = tuple(sizes), classical_options(variant), []
    interpolations = hierarchy_interpolations(sizes, dataset["face_x"].dtype)
    for start in range(0, len(guesses), test["chunk"]):
        piece = [jnp.asarray(dataset[name][start:start + test["chunk"]]) for name in ("face_x", "face_y", "rhs")]
        setup = classical_setup(piece[0], piece[1], normalization, sizes, factor, interpolations)
        norms = np.asarray(classical_solve(*piece, jnp.asarray(guesses[start:start + test["chunk"]]), setup, n_iter, test["m_max"], normalization, options))
        curves.append(norms / norms[:, :1])
    iteration_seconds, setups = {}, {}
    for batch in test["timing_batches"]:
        faces = [jnp.asarray(dataset[name][:batch]) for name in ("face_x", "face_y")]
        setups[str(batch)] = setup_seconds(faces, normalization, sizes, factor, interpolations)
        setup = classical_setup(*faces, normalization, sizes, factor, interpolations)
        arguments = faces + [jnp.asarray(dataset["rhs"][:batch]), jnp.asarray(guesses[:batch])]
        solve = lambda *values: classical_solve(*values, setup, test["timing_iterations"], test["m_max"], normalization, options)
        iteration_seconds[str(batch)] = seconds_per_iteration(solve, arguments, test["timing_iterations"])
    return np.concatenate(curves), iteration_seconds, setups


# Placement of a network variant: the two-level slot (network on coarse_grid, or on grid / coarsen; with fine_network also on the fine grid), or with
# levels "multi" the multi-level scheme down to coarse_grid (recursive when network_max is given)
def network_placement(variant, grid):
    if variant.get("levels") == "multi":
        options = {"coarse_grid": variant["coarse_grid"], "smoother": "chebyshev", "sweeps": variant.get("sweeps", 2)}
        return {**options, "network_max": variant["network_max"]} if "network_max" in variant else options
    coarsen = variant["coarsen"] if "coarsen" in variant else grid // variant["coarse_grid"]
    options = {"coarsen": coarsen, "smoother": "chebyshev", "sweeps": variant.get("sweeps", 2)}
    return {**options, "fine_network": True} if variant.get("fine_network") else options


# Curves and timings of a trained network in the two- or multi-level scheme (TF32, Chebyshev(2) smoothing), for same-job timing against the classical solvers;
# the model may come from another problem kind (zero-shot transfer)
def network_variant(variant, kind, dataset, guesses, test, n_iter, config):
    source = variant["model"]
    run = {"kind": source.get("kind", kind), "seed": source["seed"], "model": {**source, "source": "training"}}
    model, history, coefficient = load_run_model(run, config)
    fields = log_coefficient(dataset) if coefficient else None
    key = main_key({"two_level": network_placement(variant, dataset["rhs"].shape[-1]), "precision": {"dtype": "float32"}})
    curves, unused_shares = main_test_set(model, history, dataset, fields, guesses, n_iter, test["m_max"], config["normalization"], key, test["chunk"])
    timings = {str(batch): main_timing(model, history, dataset, fields, guesses, test["m_max"], config["normalization"], key, batch, test["timing_iterations"])
               for batch in test["timing_batches"]}
    return curves, timings, {str(batch): 0.0 for batch in test["timing_batches"]}


# Classical multigrid baselines (GMG V-cycle; the two-level scheme with zero, Chebyshev, V-cycle or exact coarse solves; setup-free rediscretized multi-level)
# next to the trained network in the same coarse slot, on the 100-system test set: per-system iterations, setup seconds and seconds per iteration
def execute_multigrid_run(run, config, log=print):
    name = multigrid_name(run)
    path = os.path.join(config["results_dir"], f"{name}.json")
    if os.path.exists(path):
        log(f"skip {name}: result exists")
        return
    test, grid, normalization = config["test"], run["grid"], config["normalization"]
    dataset = load_test_set(config["paper_cache_dir"], config["cache_dir"], run["kind"], grid, test["n_test"], normalization)
    guesses = initial_guesses(test["n_test"], grid)
    n_iter = resolve_iterations(test["cap"], grid)
    summaries, arrays = {}, {}
    for variant in config["variants"]:
        if variant.get("coarse_grid", 0) >= grid:
            continue
        if variant["method"] == "network":
            curves, iteration_seconds, setups = network_variant(variant, run["kind"], dataset, guesses, test, n_iter, config)
        elif variant["method"] == "rediscretized":
            curves, iteration_seconds, setups = rediscretized_variant(variant, dataset, guesses, test, n_iter, normalization)
        else:
            curves, iteration_seconds, setups = classical_variant(variant, dataset, guesses, test, n_iter, normalization)
        summary = {"iterations": {f"{tol:g}": summarize_systems(curves, tol) for tol in test["tolerances"]}, "seconds_per_iteration": iteration_seconds,
                   "setup_seconds": setups}
        summaries[variant["label"]] = summary
        arrays[f"{variant['label']}__iterations"] = np.stack([per_system_iterations(curves, tol) for tol in test["tolerances"]])
        stats = summary["iterations"]["1e-06"]
        log(f"{name} {variant['label']}: 1e-6 failures {stats['failure_rate']:.2f} median {stats['median']} p90 {stats['p90']} | "
            f"ms/it {', '.join(f'b{batch} {1e3 * value:.3f}' for batch, value in iteration_seconds.items())} | "
            f"setup ms {', '.join(f'b{batch} {1e3 * value:.2f}' for batch, value in setups.items())}")
    os.makedirs(config["results_dir"], exist_ok=True)
    np.savez_compressed(path[:-len(".json")] + ".npz", tolerances=np.array(test["tolerances"]), **arrays)
    write_json(path, {**run, "name": name, "variants": summaries})
