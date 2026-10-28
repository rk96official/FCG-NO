import os
import time

import numpy as np
import jax
import yaml

from fcgno.data.generate import load_or_make_dataset
from fcgno.models.registry import build_model, count_parameters
from fcgno.training.trainer import train
from fcgno.training.checkpoint import save_model, load_model
from fcgno.eval.evaluate import solve_test_set
from fcgno.eval.surrogate import restrict_to_grid, relative_errors
from fcgno.experiments.settings import COMMON_TRAINING, expand_runs
from fcgno.experiments.runner import train_or_load, write_json


# Predictions of a single-sample model on a batch of fields, in chunks to bound GPU memory
def predict(model, fields, chunk=10):
    batched = jax.jit(jax.vmap(model))
    return np.concatenate([np.asarray(batched(fields[start:start + chunk, None]))[:, 0] for start in range(0, len(fields), chunk)])


# The SNO preconditioner for the NO+FCG curve at a grid, taken from (or trained by) another experiment config
def fcg_preconditioner(config, grid, log):
    with open(config["preconditioner"]["config"]) as handle:
        source = yaml.safe_load(handle)
    wanted = {**config["preconditioner"]["run"], **(config["preconditioner"].get("by_grid") or {}).get(grid, {})}
    matches = [run for run in expand_runs(source) if all(run[key] == value for key, value in wanted.items())]
    if len(matches) != 1:
        raise ValueError(f"preconditioner run {wanted} matches {len(matches)} runs of {config['preconditioner']['config']}")
    return train_or_load(matches[0], source, log), matches[0]["name"]


# Surrogate trained as a direct solver f -> u, loaded from its checkpoint when one exists
def train_or_load_surrogate(name, features, data, config, grid, log):
    settings = dict(COMMON_TRAINING, **config["training"])
    path = os.path.join(config["checkpoint_dir"], f"surrogate_{name}_{config['kind']}_{grid}_{config['normalization']}_n{config['n_train']}_e{settings['epochs']}_seed{config['model_seed']}")
    loaded = load_model(path)
    if loaded is not None:
        log(f"loaded checkpoint {path}")
        return loaded[0], loaded[1]["details"]
    model = build_model(name, jax.random.PRNGKey(config["model_seed"]), features=features)
    start = time.perf_counter()
    model, history = train(model, data, "surrogate", settings, config["normalization"], seed=config["model_seed"], log=log)
    details = {"parameters": count_parameters(model), "train_seconds": time.perf_counter() - start, "final_loss": float(history[-1])}
    save_model(path, model, {"model": name, "features": features, "details": details, "history": history.tolist()})
    return model, details


# Fig. 1 at one grid: SNO and U-Net trained as direct solvers f -> u versus NO+FCG, errors against a fine-grid reference
def execute_surrogate_run(run, config, log=print):
    grid, kind, normalization = run["grid"], config["kind"], config["normalization"]
    cache = config["cache_dir"]
    train_set = load_or_make_dataset(cache, kind, grid, config["n_train"], config["train_seed"], normalization=normalization)
    test_set = load_or_make_dataset(cache, kind, grid, config["n_test"], config["test_seed"], normalization=normalization)
    reference = load_or_make_dataset(cache, kind, config["reference_grid"], config["n_test"], config["test_seed"], normalization=normalization)
    reference_on_grid = restrict_to_grid(reference["solution"], grid)
    row = {**run, "discrete": relative_errors(test_set["solution"], reference_on_grid).tolist()}
    data = {"pairs": {"rhs": train_set["rhs"], "solution": train_set["solution"]}, "problems": {}, "owner": np.arange(len(train_set["rhs"]))}
    for name, features in config["surrogates"].items():
        model, row[f"{name}_details"] = train_or_load_surrogate(name, int(features), data, config, grid, log)
        row[name] = relative_errors(predict(model, test_set["rhs"]), reference_on_grid).tolist()
        log(f"{name} surrogate on {grid}: mean relative error {np.mean(row[name]):.3e}")
    preconditioner, source_name = fcg_preconditioner(config, grid, log)
    guesses = np.random.default_rng(config["initial_guess_seed"]).standard_normal(test_set["rhs"].shape)
    solved = solve_test_set(preconditioner, test_set, guesses, config["fcg"]["n_iter"], config["fcg"]["m_max"], normalization)
    row["no_fcg"] = relative_errors(solved["iterate"], reference_on_grid).tolist()
    row["no_fcg_details"] = {"preconditioner": source_name, "final_relative_residual": solved["relative_residual"][:, -1].tolist()}
    log(f"NO+FCG on {grid}: mean relative error {np.mean(row['no_fcg']):.3e}, discrete {np.mean(row['discrete']):.3e}")
    write_json(os.path.join(config["results_dir"], f"{run['name']}.json"), [row])
