import numpy as np
import jax
import jax.numpy as jnp
import pytest

from fcgno.data.assembly import apply_stencil
from fcgno.data.generate import make_dataset
from fcgno.models.registry import as_preconditioner, build_model
from fcgno.training.checkpoint import save_model
from fcgno.training.sampling import krylov_pairs
from fcgno.training.trainer import train
from nfcg.linalg.traced_fcg import fcg_traced
from nfcg.models.checkpoint import load_history_model
from nfcg.models.history import build_history_sno, history_preconditioner
from nfcg.training.records import collect_records, merge, subsample, training_data
from nfcg.training.step_losses import aggregate, per_sample_losses, record_inputs, register_loss

GRID = 32
SYSTEMS = 3
NORMALIZATION = "upstream"


# Small Poisson training set at the SNO's smallest sensible grid, with N(0, I) guesses
@pytest.fixture(scope="module")
def problem():
    dataset = make_dataset("poisson", GRID, SYSTEMS, seed=5, normalization=NORMALIZATION)
    guesses = np.random.default_rng(5).standard_normal(dataset["rhs"].shape)
    return dataset, guesses


# Batch of records with each record's face coefficients attached
def record_batch(records, dataset):
    batch = {name: jnp.asarray(records[name]) for name in ("residual", "error", "updates", "changes")}
    batch.update(face_x=jnp.asarray(dataset["face_x"][records["owner"]]), face_y=jnp.asarray(dataset["face_y"][records["owner"]]))
    return batch


# CG records equal fcgno's Krylov pairs (same guesses), so the round-0 data matches the reproduction's training data
def test_cg_records_match_krylov_pairs(problem):
    dataset, unused_guesses = problem
    guesses = np.random.default_rng(7).standard_normal(dataset["rhs"].shape)
    records = collect_records(None, 0, dataset, guesses, 10, 20, NORMALIZATION, chunk=SYSTEMS)
    residuals, errors, unused_owner = krylov_pairs(dataset, 9, 7, NORMALIZATION)
    reorder = lambda field: field.reshape(10, SYSTEMS, GRID, GRID).transpose(1, 0, 2, 3).reshape(-1, GRID, GRID)
    np.testing.assert_allclose(reorder(records["residual"]), residuals, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(reorder(records["error"]), errors, rtol=1e-9, atol=1e-12)


# Training inputs follow the channel order of the run-time history preconditioner
def test_record_inputs_match_preconditioner(problem):
    dataset, guesses = problem
    model = build_history_sno(8, jax.random.PRNGKey(0), history=1)
    records = collect_records(model, 1, dataset, guesses, 6, 20, NORMALIZATION, chunk=SYSTEMS)
    from_records = jax.vmap(model)(record_inputs(record_batch(records, dataset), 1))[:, 0]
    at_run_time = history_preconditioner(model, 1)(jnp.asarray(records["residual"]), jnp.moveaxis(jnp.asarray(records["updates"]), 1, 0),
                                                jnp.moveaxis(jnp.asarray(records["changes"]), 1, 0))
    np.testing.assert_allclose(np.asarray(from_records), np.asarray(at_run_time), rtol=1e-9, atol=1e-12)


# Prop. 1 in training form: with m_max = 1 the step loss of a record equals the contraction FCG actually achieved at that step
def test_step_loss_is_measured_contraction(problem):
    dataset, guesses = problem
    model = build_model("sno", jax.random.PRNGKey(1), features=8)
    steps = 6
    records = collect_records(model, 0, dataset, guesses, steps, 1, NORMALIZATION, chunk=SYSTEMS)
    batch = record_batch(records, dataset)
    unused_notay, step_loss = per_sample_losses(as_preconditioner(model)(batch["residual"]), batch, NORMALIZATION)
    face_x, face_y = jnp.asarray(dataset["face_x"]), jnp.asarray(dataset["face_y"])
    operator = lambda u: apply_stencil(face_x, face_y, u, NORMALIZATION)
    traced = fcg_traced(operator, as_preconditioner(model), jnp.asarray(dataset["rhs"]), jnp.asarray(guesses), jnp.asarray(dataset["solution"]), steps, m_max=1)
    measured = np.asarray(traced["contraction"]).T.reshape(-1)
    np.testing.assert_allclose(np.asarray(step_loss), measured, rtol=1e-6, atol=1e-4)


# CVaR aggregate is the mean of the largest ceil(q B) values
def test_cvar_aggregate():
    values = jnp.arange(1.0, 11.0)
    assert float(aggregate(values, "cvar", 0.2)) == 9.5
    assert float(aggregate(values, "mean", 0.2)) == 5.5


# Subsampling caps the record count, merging concatenates, and one epoch of HC-NO training with the step loss runs and saves/reloads
def test_training_round_trip(problem, tmp_path):
    dataset, guesses = problem
    records = merge(*[collect_records(None, 1, dataset, guesses, 8, 20, NORMALIZATION, chunk=SYSTEMS)] * 2)
    records = subsample(records, 30, seed=0)
    assert len(records["owner"]) == 30
    data = training_data(records, {"face_x": dataset["face_x"], "face_y": dataset["face_y"]})
    settings = {"learning_rate": 1e-3, "decay_epochs": 50, "decay_factor": 0.5, "weight_decay": 1e-2, "epochs": 1, "batch_size": 8}
    model, curve = train(build_history_sno(8, jax.random.PRNGKey(2), 1), data, register_loss("step", 1, scale_weight=0.05), settings, NORMALIZATION, log=lambda message: None)
    assert np.all(np.isfinite(curve))
    path = str(tmp_path / "hcno")
    save_model(path, model, {"features": 8, "history": 1})
    reloaded, spec = load_history_model(path)
    probe = jnp.asarray(np.random.default_rng(0).standard_normal((1, 3, GRID, GRID)))
    np.testing.assert_allclose(np.asarray(jax.vmap(reloaded)(probe)), np.asarray(jax.vmap(model)(probe)), rtol=1e-12)
    assert spec["history"] == 1
