import os
import time

import numpy as np
import jax
import jax.numpy as jnp

from fcgno.experiments.runner import train_seed, write_json
from fcgno.experiments.settings import resolve_iterations, training_settings
from fcgno.training.checkpoint import save_model
from fcgno.training.trainer import train
from nfcg.data.coefficient_features import log_coefficient
from nfcg.data.test_sets import cached_dataset, initial_guesses, load_test_set
from nfcg.eval.per_system import per_system_iterations, summarize_systems
from nfcg.eval.timing import seconds_per_iteration
from nfcg.experiments.checkpoints import load_checkpoint, sno_checkpoint_name
from nfcg.experiments.solve import history_relative_residuals, solve_history_batch
from nfcg.models.checkpoint import load_history_model
from nfcg.models.history import build_history_sno
from nfcg.training.records import collect_records, merge, problem_table, subsample, training_data, two_level_key
from nfcg.training.step_losses import register_loss

TRAINING_AXES = ("kind", "grid", "seed", "variant")
EXTRA_SEED_STRIDE = 100


# Name of one training run, e.g. "h1_on_step_poisson_128_seed0"
def training_name(run):
    return f"{run['variant']['label']}_{run['kind']}_{run['grid']}_seed{run['seed']}"


# Network width of a run: the variant's own features when given, else the per-kind setting for the training grid
def model_features(run, config):
    return run["variant"].get("features", config["settings"][run["kind"]]["features"][run["grid"]])


# Whether a variant feeds the centered log coefficient to the model
def uses_coefficient(variant):
    return bool(variant.get("coefficient", False))


# Table 8 epochs and batch sizes with the configured per-kind overrides (learning rate and batch sizes, optional epochs)
def run_settings(run, config):
    overrides = config["settings"][run["kind"]]
    batch_sizes = overrides.get("batch_size") or {}
    by_grid = {run["grid"]: {"batch_size": batch_sizes[run["grid"]]}} if run["grid"] in batch_sizes else None
    general = {key: overrides[key] for key in ("learning_rate", "epochs") if key in overrides}
    return training_settings(run["kind"], run["grid"], general, by_grid)


# Loss of a variant, registered in fcgno's loss table under a name that encodes its options
def variant_loss(variant):
    return register_loss(variant["objective"], variant["history"], variant.get("aggregate", "mean"), variant.get("q", 0.1),
                         variant.get("scale_weight", 0.0), uses_coefficient(variant), variant.get("two_level"))


# Train a model on records with a variant's loss and save it with the spec needed to reload it; returns the model and training seconds
def train_and_save(model, records, problems, run, config, settings, path, log):
    start = time.perf_counter()
    model, curve = train(model, training_data(records, problems), variant_loss(run["variant"]), settings, config["normalization"], seed=run["seed"], log=log)
    seconds = time.perf_counter() - start
    features = model_features(run, config)
    save_model(path, model, {"features": features, "history": run["variant"]["history"], "coefficient": uses_coefficient(run["variant"]),
                             "settings": settings, "train_seconds": seconds, "records": int(len(records["owner"])),
                             "loss_curve": curve.tolist(), "variant": run["variant"]})
    return model, seconds


# Round-0 model: the existing notebook checkpoint, or one trained on CG records with the variant's loss. Each run keeps its own checkpoint:
# concurrent array tasks sharing one would each train it anyway, and GPU nondeterminism makes their models differ
def round_zero(run, config, base, problems, log):
    variant = run["variant"]
    if variant["init"] == "notebook":
        model, unused_spec = load_checkpoint(config["notebook_checkpoint_dir"], sno_checkpoint_name(run["kind"], run["grid"], run["seed"], "notebook"))
        return model, 0.0
    path = os.path.join(config["checkpoint_dir"], f"{training_name(run)}_round0")
    loaded = load_history_model(path)
    if loaded is not None:
        log(f"loaded round-0 checkpoint {path}")
        return loaded[0], loaded[1]["train_seconds"]
    features = model_features(run, config)
    model = build_history_sno(features, jax.random.PRNGKey(run["seed"]), variant["history"], uses_coefficient(variant))
    return train_and_save(model, base, problems, run, config, run_settings(run, config), path, log)


# New records of one round: the current model's own FCG trajectories (on-policy), or CG trajectories on fresh systems (equal-size control)
def round_records(run, config, model, index, dataset, n_problems, limit):
    variant, collection, normalization = run["variant"], config["collection"], config["normalization"]
    seed = train_seed(run) + index
    coefficient = uses_coefficient(variant)
    if variant["sampling"] == "onpolicy":
        guesses = np.random.default_rng(seed).standard_normal(dataset["rhs"].shape)
        records = collect_records(model, variant["history"], dataset, guesses, collection["steps"], config["m_max"], normalization,
                                  collection["chunk"], collection["floor"], coefficient=coefficient, two_level=variant.get("two_level"))
        return subsample(records, limit, seed), None
    extra = cached_dataset(config["paper_cache_dir"], config["cache_dir"], run["kind"], run["grid"], run["grid"],
                           train_seed(run) + EXTRA_SEED_STRIDE * index, normalization)
    guesses = np.random.default_rng(seed + EXTRA_SEED_STRIDE * index).standard_normal(extra["rhs"].shape)
    records = collect_records(None, variant["history"], extra, guesses, config["krylov_iterations"], config["m_max"], normalization, collection["chunk"])
    records["owner"] = records["owner"] + n_problems
    return subsample(records, limit, seed), problem_table(extra, coefficient)


# Per-system relative residual curves of a model on the test set and their summaries per tolerance
def evaluate_model(model, variant, test_set, guesses, config, grid):
    test = config["test"]
    curves = history_relative_residuals(model, variant["history"], test_set, guesses, resolve_iterations(test["cap"], grid), config["m_max"],
                                        config["normalization"], test["chunk"], uses_coefficient(variant), variant.get("two_level"))
    return curves, {f"{tol:g}": summarize_systems(curves, tol) for tol in test["tolerances"]}


# Seconds per FCG iteration of a model at the configured batch sizes
def model_timing(model, variant, test_set, guesses, config):
    fields = log_coefficient(test_set) if uses_coefficient(variant) else None
    n_iter = config["test"]["timing_iterations"]
    timings = {}
    for batch in config["test"]["timing_batches"]:
        arguments = [jnp.asarray(test_set[name][:batch]) for name in ("face_x", "face_y", "rhs")] + [jnp.asarray(guesses[:batch])]
        field = None if fields is None else jnp.asarray(fields[:batch])
        solve = lambda *values: solve_history_batch(model, *values, field, n_iter, config["m_max"], variant["history"], config["normalization"],
                                                    two_level_key(variant.get("two_level")))
        timings[str(batch)] = seconds_per_iteration(solve, arguments, n_iter)
    return timings


# Model of one round: reload its checkpoint if present, otherwise fine-tune the previous model on the aggregated records
def fine_tune_round(model, records, problems, run, config, index, log):
    round_path = os.path.join(config["checkpoint_dir"], f"{training_name(run)}_round{index}")
    loaded = load_history_model(round_path)
    if loaded is not None:
        return loaded[0], loaded[1]["train_seconds"]
    settings = run_settings(run, config)
    settings.update(epochs=config["fine_tune"]["epochs"], learning_rate=settings["learning_rate"] * config["fine_tune"]["learning_rate_scale"])
    return train_and_save(model, records, problems, run, config, settings, round_path, log)


# Train one variant (round 0, then on-policy or control rounds with fine-tuning), evaluating on the 100-system test set after every round
def execute_training_run(run, config, log=print):
    name, variant, grid, normalization = training_name(run), run["variant"], run["grid"], config["normalization"]
    path = os.path.join(config["results_dir"], f"{name}.json")
    if os.path.exists(path):
        log(f"skip {name}: result exists")
        return
    if variant["init"] == "notebook" and (uses_coefficient(variant) or variant["history"] > 0):
        raise ValueError(f"{name}: notebook initialization exists only for memoryless models without coefficient input")
    coefficient = uses_coefficient(variant)
    dataset = cached_dataset(config["paper_cache_dir"], config["cache_dir"], run["kind"], grid, grid, train_seed(run), normalization)
    guesses = np.random.default_rng(train_seed(run)).standard_normal(dataset["rhs"].shape)
    base = collect_records(None, variant["history"], dataset, guesses, config["krylov_iterations"], config["m_max"], normalization,
                           config["collection"]["chunk"], coefficient=coefficient)
    problems = problem_table(dataset, coefficient)
    test_set = load_test_set(config["paper_cache_dir"], config["cache_dir"], run["kind"], grid, config["test"]["n_test"], normalization)
    test_guesses = initial_guesses(config["test"]["n_test"], grid)
    model, seconds = round_zero(run, config, base, problems, log)
    records, rounds, curves = base, [], None
    for index in range(variant["rounds"] + 1):
        if index > 0:
            limit = int(config["collection"]["round_fraction"] * len(base["owner"]))
            new_records, extra = round_records(run, config, model, index, dataset, len(problems["face_x"]), limit)
            if extra is not None:
                problems = {key: np.concatenate([problems[key], extra[key]]) for key in problems}
            records = merge(records, new_records)
            model, seconds = fine_tune_round(model, records, problems, run, config, index, log)
        curves, iterations = evaluate_model(model, variant, test_set, test_guesses, config, grid)
        rounds.append({"round": index, "records": int(len(records["owner"])), "train_seconds": seconds, "iterations": iterations})
        stats = iterations["1e-06"]
        log(f"{name} round {index}: records {len(records['owner'])} | 1e-6 failures {stats['failure_rate']:.2f} median {stats['median']} p90 {stats['p90']}")
    result = {**run, "name": name, "rounds": rounds, "seconds_per_iteration": model_timing(model, variant, test_set, test_guesses, config)}
    os.makedirs(config["results_dir"], exist_ok=True)
    tolerances = config["test"]["tolerances"]
    np.savez_compressed(path[:-len(".json")] + ".npz", tolerances=np.array(tolerances), relative_residual=curves.astype(np.float32),
                        iterations=np.stack([per_system_iterations(curves, tol) for tol in tolerances]))
    write_json(path, result)
