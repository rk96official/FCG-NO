import json

import pytest
import yaml

from fcgno.experiments.settings import experiment_runs, resolve_iterations, resolve_test_grids, training_settings
from fcgno.experiments.runner import model_features, per_model_value
from scripts.make_tables import build_table, to_markdown

EXPECTED_RUNS = {"r2_losses": 12, "r2_physical": 6, "r3_sampling": 6, "r4_cross_grid": 6, "r5_architectures": 18, "r8_nonsmooth": 4, "r9_surrogates": 5, "r2_seeds": 12, "r4_seeds": 12, "r2_notebook_poisson": 9, "r2_notebook_diffusion": 9, "r2_l2mse": 6, "r2_modes5": 6, "r2_sampleu": 6, "r2_paperdata": 9}


# Every experiment config expands to the run count its SLURM array expects, with unique names
@pytest.mark.parametrize("experiment", sorted(EXPECTED_RUNS))
def test_config_run_counts(experiment):
    with open(f"configs/{experiment}.yaml") as handle:
        runs = experiment_runs(yaml.safe_load(handle))
    assert len(runs) == EXPECTED_RUNS[experiment]
    assert len({run["name"] for run in runs}) == len(runs)
    with open(f"slurm/{experiment}.slurm") as handle:
        assert f"--array=0-{len(runs) - 1}" in handle.read()


# Iteration budgets, test grids and Table 8 settings resolve as the paper specifies
def test_resolvers():
    assert resolve_iterations("2grid", 64) == 128 and resolve_iterations(300, 64) == 300 and resolve_iterations("2.5grid", 32) == 80
    assert resolve_test_grids({"grids": "same"}, 32) == [32]
    assert resolve_test_grids({"grids": {32: [64, 128]}}, 32) == [64, 128]
    assert training_settings("a1", 64)["batch_size"] == 16 and training_settings("poisson", 128)["epochs"] == 150


# Comparison tables match rows to paper entries and format "ours (paper)"
def test_build_table(tmp_path):
    row = {"model": "sno", "kind": "poisson", "grid": 32, "test_grid": 32, "loss": "notay", "sampling": "krylov",
           "fcg": {"0.001": {"iterations_mean_curve": 4}, "1e-06": {"iterations_mean_curve": 10}, "1e-12": {"iterations_mean_curve": -1}}}
    (tmp_path / "r2_losses").mkdir()
    (tmp_path / "r2_losses" / "run.json").write_text(json.dumps([row]))
    spec = {"title": "test", "sources": ["r2_losses"], "filter": {"model": "sno"}, "column": "loss", "metric": "iterations",
            "paper": {"notay": {"poisson": {32: [4, 9, 20]}}}}
    table = build_table(spec, ["0.001", "1e-06", "1e-12"], str(tmp_path))
    assert list(table["ours_mean"].fillna(-1)) == [4, 10, -1] and list(table["paper"]) == [4, 9, 20]
    markdown = to_markdown(spec, table)
    assert "4 (4)" in markdown and "10 (9)" in markdown and "— (20)" in markdown


# Feature overrides apply per model and optionally per training grid, otherwise Table 7
def test_model_features():
    run = {"model": "dilresnet", "kind": "diffusion", "grid": 128}
    assert model_features(run, {"features": {"dilresnet": {128: 16}}}) == 16
    assert model_features({**run, "grid": 64}, {"features": {"dilresnet": {128: 16}}}) == 24
    assert model_features(run, {"features": {"dilresnet": 8}}) == 8 and model_features(run, {}) == 32
    assert per_model_value({"precision": {"dilresnet": {128: "float32"}}}, "precision", run) == "float32"
    assert per_model_value({"precision": {"dilresnet": {128: "float32"}}}, "precision", {**run, "grid": 64}) is None


# Tagged variants get distinct run names and per-grid training overrides apply on top of Table 8
def test_tags_and_grid_overrides():
    with open("configs/r2_notebook_diffusion.yaml") as handle:
        runs = experiment_runs(yaml.safe_load(handle))
    assert all(run["name"].endswith("_notebook") for run in runs)
    assert training_settings("diffusion", 64, {"learning_rate": 8e-4}, {64: {"batch_size": 32}})["batch_size"] == 32
    assert training_settings("diffusion", 128, {"learning_rate": 8e-4}, {64: {"batch_size": 32}})["batch_size"] == 4
