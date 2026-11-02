import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from fcgno.experiments.runner import train_seed, write_json
from fcgno.linalg.krylov import fcg
from fcgno.training.losses import LOSSES
from fcgno.training.trainer import train
from nfcg.eval.per_system import per_system_iterations, summarize_systems
from nfcg.eval.timing import seconds_per_iteration
from nfcg.linalg.history_fcg import history_fcg
from nfcg.training.records import merge, subsample
from nfcg.three_d.data import load_or_make_3d
from nfcg.three_d.model import SNO3D
from nfcg.three_d.operator import apply_stencil_3d
from nfcg.three_d.solver import FACE_NAMES, classical_preconditioner_3d, collect_records_3d, make_step_loss_3d, network_preconditioner_3d

THREE_D_TRAINING_AXES = ("kind", "grid", "seed", "variant")
THREE_D_AXES = ("kind", "grid")
TEST_SEED = 23
GUESS_SEED = 24


# Name of one 3D training run, e.g. "h1_step_c2_diffusion_32_seed0"
def three_d_training_name(run):
    return f"{run['variant']['label']}_{run['kind']}_{run['grid']}_seed{run['seed']}"


# Name of one 3D deployment run, e.g. "three_d_diffusion_64"
def three_d_name(run):
    return f"three_d_{run['kind']}_{run['grid']}"


# Grid sizes from the fine grid down to the network's bottom grid by halving (two entries when the bottom is one halving away)
def halving_sizes(fine, bottom):
    sizes = [fine]
    while sizes[-1] > bottom:
        sizes.append(sizes[-1] // 2)
    return tuple(sizes)


# Fresh 3D network for a variant
def build_model_3d(variant, seed):
    return SNO3D(variant["features"], jax.random.PRNGKey(seed), in_channels=1 + 2 * variant["history"], n_modes=variant.get("n_modes", 12))


# Load a trained 3D model and its spec, or None when missing
def load_model_3d(directory, name):
    stem = os.path.join(directory, name)
    if not os.path.exists(stem + ".eqx"):
        return None
    with open(stem + ".json") as handle:
        spec = json.load(handle)
    return eqx.tree_deserialise_leaves(stem + ".eqx", build_model_3d(spec["variant"], 0)), spec


# Iterations to 1e-6 of a model on a test set in the deployed scheme (median, failure rate), for the per-round log
def evaluate_model_3d(model, history, test_set, guesses, sizes, normalization, n_iter):
    faces = tuple(jnp.asarray(test_set[name]) for name in FACE_NAMES)
    norms = np.asarray(deployed_solve(model, history, faces, jnp.asarray(test_set["rhs"]), jnp.asarray(guesses), n_iter, 20, normalization, sizes, None))
    return summarize_systems(norms / norms[:, :1], 1e-6)


# Train one 3D variant: round 0 on CG records, then on-policy rounds through the deployed scheme with fine-tuning (step loss, Theorem 1)
def execute_three_d_training_run(run, config, log=print):
    name, variant, grid, normalization = three_d_training_name(run), run["variant"], run["grid"], config["normalization"]
    stem = os.path.join(config["checkpoint_dir"], f"three_d_{name}")
    if os.path.exists(stem + ".eqx"):
        log(f"skip {name}: checkpoint exists")
        return
    sizes, history, train_options = halving_sizes(grid, variant["bottom"]), variant["history"], config["train"]
    dataset = load_or_make_3d(config["cache_dir"], run["kind"], grid, train_options["problems"], train_seed(run), normalization, solve=True)
    test_set = load_or_make_3d(config["cache_dir"], run["kind"], grid, config["test"]["n_test"], TEST_SEED, normalization)
    rng = np.random.default_rng(train_seed(run))
    test_guesses = np.random.default_rng(GUESS_SEED).standard_normal(test_set["rhs"].shape)
    base = collect_records_3d(None, history, dataset, rng.standard_normal(dataset["rhs"].shape), train_options["steps"], 20, normalization, train_options["chunk"], sizes)
    problems = {face: dataset[face] for face in FACE_NAMES}
    loss_name = f"three_d_step_h{history}_{'_'.join(map(str, sizes))}"
    LOSSES[loss_name] = make_step_loss_3d(history, sizes, variant.get("scale_weight", 0.05))
    settings = {"learning_rate": train_options["learning_rate"], "decay_epochs": 50, "decay_factor": 0.5, "weight_decay": 1e-2, "batch_size": train_options["batch_size"]}
    model, records, start, rounds = build_model_3d(variant, run["seed"]), base, time.perf_counter(), []
    for index in range(variant["rounds"] + 1):
        if index > 0:
            new = collect_records_3d(model, history, dataset, rng.standard_normal(dataset["rhs"].shape), train_options["steps"], 20, normalization,
                                     train_options["chunk"], sizes)
            records = merge(records, subsample(new, int(train_options["round_fraction"] * len(base["owner"])), train_seed(run) + index))
        epochs = train_options["epochs"] if index == 0 else train_options["fine_tune_epochs"]
        model, curve = train(model, {"pairs": {key: records[key] for key in ("residual", "error", "updates", "changes")}, "problems": problems, "owner": records["owner"]},
                             loss_name, {**settings, "epochs": epochs}, normalization, seed=run["seed"] + index, log=log)
        stats = evaluate_model_3d(model, history, test_set, test_guesses, sizes, normalization, 4 * grid)
        rounds.append({"round": index, "records": int(len(records["owner"])), "final_loss": float(curve[-1]), "iterations_1e-06": stats})
        log(f"{name} round {index}: records {len(records['owner'])} | test 1e-6 median {stats['median']} failures {stats['failure_rate']:.2f}")
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    eqx.tree_serialise_leaves(stem + ".eqx", model)
    write_json(stem + ".json", {"variant": variant, "train_seconds": time.perf_counter() - start, "rounds": rounds})
    write_json(os.path.join(config["results_dir"], f"{name}.json"), {**run, "name": name, "rounds": rounds, "train_seconds": time.perf_counter() - start})


# Jitted FCG with a 3D network preconditioner (model given) or a classical one (variant options given); returns residual norms
@eqx.filter_jit
def deployed_solve(model, history, faces, rhs, guess, n_iter, m_max, normalization, sizes, classical):
    operator = lambda u: apply_stencil_3d(faces, u, normalization)
    if model is None:
        return fcg(operator, classical_preconditioner_3d(faces, normalization, dict(classical)), rhs, guess, n_iter, m_max)["residual_norm"]
    precondition = network_preconditioner_3d(model, history, faces, normalization, sizes, jnp.float32)
    return history_fcg(operator, precondition, rhs, guess, n_iter, m_max, max(history, 1))["residual_norm"]


# Curves and timings of one 3D variant over the test set (setup of the rediscretized levels is inside the timed solve)
def three_d_variant(variant, config, kind, test_set, guesses, n_iter):
    test, normalization, grid = config["test"], config["normalization"], test_set["rhs"].shape[-1]
    model, history, sizes, classical = None, 0, None, None
    if variant["method"] == "network":
        loaded = load_model_3d(config["checkpoint_dir"], f"three_d_{variant['model']}_{kind}_{variant['train_grid']}_seed{variant['seed']}")
        if loaded is None:
            raise FileNotFoundError(f"missing 3D checkpoint for {variant['model']} seed {variant['seed']}; run e19 training first")
        model, spec = loaded
        history, sizes = spec["variant"]["history"], (halving_sizes(grid, variant["bottom"]) if variant.get("levels") == "multi" else (grid, variant["bottom"]))
    else:
        options = {**variant, "sizes": halving_sizes(grid, variant["bottom"])} if variant["method"] == "multilevel" else variant
        classical = tuple(sorted((key, tuple(value) if isinstance(value, list) else value) for key, value in options.items() if key != "label"))
    solve = lambda *values: deployed_solve(model, history, tuple(values[:3]), values[3], values[4], values[5], test["m_max"], normalization, sizes, classical)
    curves = []
    for start in range(0, len(guesses), test["chunk"]):
        values = [jnp.asarray(test_set[name][start:start + test["chunk"]]) for name in (*FACE_NAMES, "rhs")] + [jnp.asarray(guesses[start:start + test["chunk"]])]
        norms = np.asarray(solve(*values, n_iter))
        curves.append(norms / norms[:, :1])
    timings = {}
    for batch in test["timing_batches"]:
        arguments = [jnp.asarray(test_set[name][:batch]) for name in (*FACE_NAMES, "rhs")] + [jnp.asarray(guesses[:batch]), test["timing_iterations"]]
        timings[str(batch)] = seconds_per_iteration(solve, arguments, test["timing_iterations"])
    return np.concatenate(curves), timings


# 3D deployment: Chebyshev-PCG, rediscretized multigrid (setup-free) and the trained 3D networks in the two-/multi-level scheme, in the multigrid report format
def execute_three_d_run(run, config, log=print):
    name = three_d_name(run)
    path = os.path.join(config["results_dir"], f"{name}.json")
    if os.path.exists(path):
        log(f"skip {name}: result exists")
        return
    test, grid = config["test"], run["grid"]
    test_set = load_or_make_3d(config["cache_dir"], run["kind"], grid, test["n_test"], TEST_SEED, config["normalization"])
    guesses = np.random.default_rng(GUESS_SEED).standard_normal(test_set["rhs"].shape)
    summaries, arrays = {}, {}
    for variant in config["variants"]:
        if variant.get("bottom", 0) >= grid:
            continue
        curves, timings = three_d_variant(variant, config, run["kind"], test_set, guesses, test["cap"] * grid)
        summaries[variant["label"]] = {"iterations": {f"{tol:g}": summarize_systems(curves, tol) for tol in test["tolerances"]}, "seconds_per_iteration": timings,
                                       "setup_seconds": {batch: 0.0 for batch in timings}}
        arrays[f"{variant['label']}__iterations"] = np.stack([per_system_iterations(curves, tol) for tol in test["tolerances"]])
        stats = summaries[variant["label"]]["iterations"]["1e-06"]
        log(f"{name} {variant['label']}: 1e-6 failures {stats['failure_rate']:.2f} median {stats['median']} p90 {stats['p90']} | "
            f"ms/it {', '.join(f'b{batch} {1e3 * value:.3f}' for batch, value in timings.items())}")
    os.makedirs(config["results_dir"], exist_ok=True)
    np.savez_compressed(path[:-len(".json")] + ".npz", tolerances=np.array(test["tolerances"]), **arrays)
    write_json(path, {**run, "name": name, "variants": summaries})
