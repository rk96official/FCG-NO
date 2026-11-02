import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from fcgno.data.assembly import apply_stencil
from fcgno.experiments.runner import train_seed, write_json
from fcgno.experiments.settings import resolve_iterations
from fcgno.linalg.krylov import fcg
from nfcg.data.test_sets import cached_dataset, initial_guesses, load_test_set
from nfcg.eval.per_system import per_system_iterations, summarize_systems
from nfcg.eval.timing import seconds_per_iteration
from nfcg.linalg.incomplete_cholesky import factor_solve, ic0_factor
from nfcg.models.neuralif import NeuralIF, frobenius_loss, neural_factor

NEURALIF_TRAINING_AXES = ("kind", "grid", "seed")
NEURALIF_AXES = ("kind", "grid")


# Name of one NeuralIF training run, e.g. "neuralif_diffusion_128_seed0"
def neuralif_training_name(run):
    return f"neuralif_{run['kind']}_{run['grid']}_seed{run['seed']}"


# Name of one NeuralIF / IC(0) deployment run, e.g. "factorization_diffusion_256"
def neuralif_name(run):
    return f"factorization_{run['kind']}_{run['grid']}"


# Checkpoint path stem of a trained NeuralIF model
def checkpoint_stem(directory, kind, grid, seed):
    return os.path.join(directory, f"neuralif_{kind}_{grid}_seed{seed}")


# Fresh NeuralIF model with the configured width and depth
def build_model(config, seed):
    return NeuralIF(config["model"]["features"], jax.random.PRNGKey(seed), config["model"]["layers"])


# Load a trained NeuralIF model, or None when its training has not run
def load_neuralif(config, kind, seed):
    stem = checkpoint_stem(config["checkpoint_dir"], kind, config["train_grid"], seed)
    if not os.path.exists(stem + ".eqx"):
        return None
    with open(stem + ".json") as handle:
        spec = json.load(handle)
    model = NeuralIF(spec["features"], jax.random.PRNGKey(0), spec["layers"])
    return eqx.tree_deserialise_leaves(stem + ".eqx", model)


# Train NeuralIF on our training problems (same set as the HC-NO models) with its stochastic Frobenius loss; Adam, fresh probes every step
def execute_neuralif_training_run(run, config, log=print):
    name, stem = neuralif_training_name(run), checkpoint_stem(config["checkpoint_dir"], run["kind"], run["grid"], run["seed"])
    if os.path.exists(stem + ".eqx"):
        log(f"skip {name}: checkpoint exists")
        return
    grid, train = run["grid"], config["train"]
    dataset = cached_dataset(config["paper_cache_dir"], config["cache_dir"], run["kind"], grid, grid, train_seed(run), config["normalization"])
    faces = jax.device_put((jnp.asarray(dataset["face_x"]), jnp.asarray(dataset["face_y"])))
    model = build_model(config, run["seed"])
    optimizer = optax.adam(optax.cosine_decay_schedule(train["learning_rate"], train["steps"]))
    params, static = eqx.partition(model, eqx.is_array)
    opt_state = optimizer.init(params)

    # One Adam step on a random batch of problems with fresh Gaussian probes
    @jax.jit
    def step(params, opt_state, key):
        choice_key, probe_key = jax.random.split(key)
        chosen = jax.random.choice(choice_key, faces[0].shape[0], (train["batch_size"],))
        objective = lambda trainable: frobenius_loss(eqx.combine(trainable, static), faces[0][chosen], faces[1][chosen], config["normalization"], probe_key)
        loss, grads = jax.value_and_grad(objective)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    key, losses, start = jax.random.PRNGKey(run["seed"]), [], time.perf_counter()
    for index in range(train["steps"]):
        key, step_key = jax.random.split(key)
        params, opt_state, loss = step(params, opt_state, step_key)
        losses.append(float(loss))
        if index % 500 == 0 or index == train["steps"] - 1:
            log(f"{name} step {index}: frobenius loss {np.mean(losses[-100:]):.4e}")
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    eqx.tree_serialise_leaves(stem + ".eqx", eqx.combine(params, static))
    write_json(stem + ".json", {"features": config["model"]["features"], "layers": config["model"]["layers"], "train_seconds": time.perf_counter() - start,
                                "loss_curve": losses[::10]})


# Factor of a variant for a batch of operators: classical IC(0) or the trained NeuralIF network (its setup)
@eqx.filter_jit
def factor_setup(model, face_x, face_y, normalization):
    return ic0_factor(face_x, face_y, normalization) if model is None else neural_factor(model, face_x, face_y, normalization)


# Jitted FCG with the incomplete-factorization preconditioner (L L^T)^-1 applied by wavefront triangular solves
@eqx.filter_jit
def factor_fcg(face_x, face_y, rhs, guess, factor, n_iter, m_max, normalization):
    operator = lambda u: apply_stencil(face_x, face_y, u, normalization)
    return fcg(operator, lambda residual: factor_solve(factor, residual), rhs, guess, n_iter, m_max)["residual_norm"]


# Median wall-clock seconds of a variant's setup for one batch, compilation excluded
def setup_seconds(model, faces, normalization, repeats=5):
    jax.block_until_ready(factor_setup(model, *faces, normalization))
    durations = []
    for unused in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(factor_setup(model, *faces, normalization))
        durations.append(time.perf_counter() - start)
    return float(np.median(durations))


# Curves, seconds per iteration and setup seconds of one variant over the test set
def factor_variant(model, dataset, guesses, test, n_iter, normalization):
    curves = []
    for start in range(0, len(guesses), test["chunk"]):
        piece = [jnp.asarray(dataset[name][start:start + test["chunk"]]) for name in ("face_x", "face_y", "rhs")]
        factor = factor_setup(model, piece[0], piece[1], normalization)
        norms = np.asarray(factor_fcg(*piece, jnp.asarray(guesses[start:start + test["chunk"]]), factor, n_iter, test["m_max"], normalization))
        curves.append(norms / norms[:, :1])
    iteration_seconds, setups = {}, {}
    for batch in test["timing_batches"]:
        faces = [jnp.asarray(dataset[name][:batch]) for name in ("face_x", "face_y")]
        setups[str(batch)] = setup_seconds(model, faces, normalization)
        factor = factor_setup(model, *faces, normalization)
        arguments = faces + [jnp.asarray(dataset["rhs"][:batch]), jnp.asarray(guesses[:batch])]
        solve = lambda *values: factor_fcg(*values[:4], factor, test["timing_iterations"], test["m_max"], normalization)
        iteration_seconds[str(batch)] = seconds_per_iteration(solve, arguments, test["timing_iterations"])
    return np.concatenate(curves), iteration_seconds, setups


# Learned (NeuralIF) and classical IC(0) incomplete factorizations on the 100-system test set, in the multigrid report's format
def execute_neuralif_run(run, config, log=print):
    name = neuralif_name(run)
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
        model = None if variant["method"] == "ic0" else load_neuralif(config, run["kind"], variant["seed"])
        if variant["method"] == "neuralif" and model is None:
            raise FileNotFoundError(f"missing NeuralIF checkpoint for {run['kind']} seed {variant['seed']}; run its training first")
        curves, iteration_seconds, setups = factor_variant(model, dataset, guesses, test, n_iter, normalization)
        summaries[variant["label"]] = {"iterations": {f"{tol:g}": summarize_systems(curves, tol) for tol in test["tolerances"]},
                                       "seconds_per_iteration": iteration_seconds, "setup_seconds": setups}
        arrays[f"{variant['label']}__iterations"] = np.stack([per_system_iterations(curves, tol) for tol in test["tolerances"]])
        stats = summaries[variant["label"]]["iterations"]["1e-06"]
        log(f"{name} {variant['label']}: 1e-6 failures {stats['failure_rate']:.2f} median {stats['median']} p90 {stats['p90']} | "
            f"ms/it {', '.join(f'b{batch} {1e3 * value:.3f}' for batch, value in iteration_seconds.items())} | "
            f"setup ms {', '.join(f'b{batch} {1e3 * value:.2f}' for batch, value in setups.items())}")
    os.makedirs(config["results_dir"], exist_ok=True)
    np.savez_compressed(path[:-len(".json")] + ".npz", tolerances=np.array(test["tolerances"]), **arrays)
    write_json(path, {**run, "name": name, "variants": summaries})
