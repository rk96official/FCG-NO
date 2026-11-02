import numpy as np
import jax
import jax.numpy as jnp
import pytest

from fcgno.data.generate import make_dataset
from nfcg.data.coefficient_features import coefficient_contrast, log_coefficient
from nfcg.experiments.solve import history_relative_residuals
from nfcg.models.history import build_history_sno, history_preconditioner
from nfcg.training.records import collect_records, problem_table
from nfcg.training.step_losses import record_inputs

GRID = 32
NORMALIZATION = "upstream"


# Small non-smooth a1 set (jump coefficient) with N(0, I) guesses
@pytest.fixture(scope="module")
def problem():
    dataset = make_dataset("a1", GRID, 2, seed=3, normalization=NORMALIZATION)
    return dataset, np.random.default_rng(3).standard_normal(dataset["rhs"].shape)


# Batch of records with the per-problem table attached through the owners
def record_batch(records, table):
    batch = {name: jnp.asarray(records[name]) for name in ("residual", "error", "updates", "changes")}
    batch.update({name: jnp.asarray(values[records["owner"]]) for name, values in table.items()})
    return batch


# The log coefficient is centered per problem and vanishes for Poisson; contrast is max/min of a
def test_coefficient_features(problem):
    dataset, unused_guesses = problem
    np.testing.assert_allclose(log_coefficient(dataset).mean(axis=(-2, -1)), 0.0, atol=1e-12)
    assert np.all(coefficient_contrast(dataset) > 1.0)
    poisson = make_dataset("poisson", GRID, 1, seed=0, normalization=NORMALIZATION)
    np.testing.assert_allclose(log_coefficient(poisson), 0.0, atol=1e-12)
    np.testing.assert_allclose(coefficient_contrast(poisson), 1.0)


# With a coefficient channel, training inputs still match the run-time preconditioner (HC-NO and memoryless)
@pytest.mark.parametrize("history", [0, 1])
def test_coefficient_inputs_match_preconditioner(problem, history):
    dataset, guesses = problem
    model = build_history_sno(8, jax.random.PRNGKey(0), history=history, coefficient=True)
    records = collect_records(model, history, dataset, guesses, 5, 20, NORMALIZATION, chunk=2, coefficient=True)
    batch = record_batch(records, problem_table(dataset, coefficient=True))
    from_records = jax.vmap(model)(record_inputs(batch, history, coefficient=True))[:, 0]
    at_run_time = history_preconditioner(model, history, batch["log_coefficient"])(
        batch["residual"], jnp.moveaxis(batch["updates"], 1, 0), jnp.moveaxis(batch["changes"], 1, 0))
    np.testing.assert_allclose(np.asarray(from_records), np.asarray(at_run_time), rtol=1e-9, atol=1e-12)
    curves = history_relative_residuals(model, history, dataset, guesses, 4, 20, NORMALIZATION, chunk=2, coefficient=True)
    assert curves.shape == (2, 5) and np.all(np.isfinite(curves))
