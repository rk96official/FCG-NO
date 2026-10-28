import json
import os
import time

import numpy as np
import jax

from fcgno.data.generate import load_or_make_dataset
from fcgno.data.coefficients import DEFAULT_MAX_MODE
from fcgno.models.registry import TABLE7_FEATURES, build_model, count_parameters
from fcgno.training.sampling import krylov_pairs, random_pairs
from fcgno.training.trainer import train
from fcgno.training.checkpoint import save_model, load_model
from fcgno.eval.evaluate import solve_test_set, iteration_table, mean_curve_iterations
from fcgno.eval.timing import time_to_tolerances
from fcgno.experiments.settings import TABLE_KIND, training_settings, resolve_test_grids, resolve_iterations

TRAIN_SEED_BASE = 2
TRAIN_SEED_STRIDE = 1000


# Write JSON atomically so readers never see a partial file
def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


# Training-set seed of a run; distinct from the test seed and from other run seeds
def train_seed(run):
    return TRAIN_SEED_BASE + TRAIN_SEED_STRIDE * run["seed"]


# Training pairs of a run: Krylov (CG history) or random residuals, their errors, and the problem owning each pair
def training_data(run, config):
    data_options = config.get("data") or {}
    dataset = load_or_make_dataset(config["cache_dir"], run["kind"], run["grid"], run["n_train"], train_seed(run), normalization=run["normalization"],
                                   max_mode=data_options.get("max_mode", DEFAULT_MAX_MODE), sample=data_options.get("train_sample", "rhs"))
    if run["sampling"] == "krylov":
        residuals, errors, owner = krylov_pairs(dataset, run["krylov_iterations"] - 1, train_seed(run), run["normalization"])
    else:
        residuals, errors, owner = random_pairs(dataset, run["krylov_iterations"], train_seed(run), run["normalization"])
    return {"pairs": {"residual": residuals, "error": errors}, "problems": {"face_x": dataset["face_x"], "face_y": dataset["face_y"]}, "owner": owner}


# Per-model config value: one value, or a mapping by training grid; None when the model or grid has no entry
def per_model_value(config, key, run):
    value = (config.get(key) or {}).get(run["model"])
    return value.get(run["grid"]) if isinstance(value, dict) else value


# Feature count from a config override or paper Table 7 (non-smooth kinds use the Diffusion column)
def model_features(run, config):
    override = per_model_value(config, "features", run)
    return int(override) if override is not None else TABLE7_FEATURES[TABLE_KIND[run["kind"]]][run["model"]][run["grid"]]


# Trained preconditioner of a run from the checkpoint directory, training and saving it first if missing
def train_or_load(run, config, log=print):
    path = os.path.join(config["checkpoint_dir"], run["name"])
    loaded = load_model(path)
    if loaded is not None:
        log(f"loaded checkpoint {path}")
        return loaded[0]
    settings = training_settings(run["kind"], run["grid"], config.get("training"), config.get("training_by_grid"))
    settings["precision"] = per_model_value(config, "precision", run) or "float64"
    features = model_features(run, config)
    model = build_model(run["model"], jax.random.PRNGKey(run["seed"]), features=features)
    log(f"training {run['name']}: {count_parameters(model)} parameters, {settings}")
    start = time.perf_counter()
    data = training_data(run, config)
    log(f"training pairs: {len(data['owner'])}, generated in {time.perf_counter() - start:.1f}s")
    model, history = train(model, data, run["loss"], settings, run["normalization"], seed=run["seed"], log=log)
    spec = {**run, "features": features, "settings": settings, "parameters": count_parameters(model),
            "train_seconds": time.perf_counter() - start, "history": history.tolist()}
    save_model(path, model, spec)
    return model


# Test a preconditioner on each test grid against CG, saving curves and returning result rows (optional GPU timing)
def evaluate(model, run, config, log=print):
    test = config["test"]
    rows = []
    for test_grid in resolve_test_grids(test, run["grid"]):
        dataset = load_or_make_dataset(config["cache_dir"], run["kind"], test_grid, test["n_test"], test["seed"], normalization=run["normalization"],
                                       max_mode=(config.get("data") or {}).get("max_mode", DEFAULT_MAX_MODE))
        guesses = np.random.default_rng(test["initial_guess_seed"]).standard_normal(dataset["rhs"].shape)
        n_iter = resolve_iterations(test["n_iter"], test_grid)
        cg_iter = resolve_iterations(test.get("cg_n_iter", "6grid"), test_grid)
        with_model = solve_test_set(model, dataset, guesses, n_iter, test["m_max"], run["normalization"])
        baseline = solve_test_set(None, dataset, guesses, cg_iter, test["m_max"], run["normalization"])
        row = {**run, "test_grid": test_grid, "n_iter": n_iter, "cg_n_iter": cg_iter,
               "fcg": iteration_table(with_model["relative_residual"], test["tolerances"]),
               "cg": iteration_table(baseline["relative_residual"], test["tolerances"])}
        if test.get("timing"):
            row["timing"] = time_to_tolerances(model, dataset, guesses, run["normalization"], test["m_max"],
                                               mean_curve_iterations(row["fcg"]), mean_curve_iterations(row["cg"]))
        np.savez(os.path.join(config["results_dir"], f"{run['name']}_test{test_grid}.npz"),
                 fcg_residual=with_model["relative_residual"], fcg_notay=with_model["notay"],
                 cg_residual=baseline["relative_residual"], cg_notay=baseline["notay"])
        log(f"{run['name']} on {test_grid}: NO+FCG {mean_curve_iterations(row['fcg'])}, CG {mean_curve_iterations(row['cg'])}")
        rows.append(row)
    return rows


# Train or load, evaluate, and save the result rows of one preconditioner run
def execute_preconditioner_run(run, config, log=print):
    os.makedirs(config["results_dir"], exist_ok=True)
    model = train_or_load(run, config, log)
    write_json(os.path.join(config["results_dir"], f"{run['name']}.json"), evaluate(model, run, config, log))
