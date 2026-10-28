import numpy as np
import jax
import jax.numpy as jnp
import pytest

from fcgno.data.assembly import assemble_csr
from fcgno.data.generate import make_dataset
from fcgno.models.registry import build_model
from fcgno.training.losses import notay_loss, l2_loss, mse_loss
from fcgno.training.sampling import krylov_pairs, random_pairs
from fcgno.training.trainer import learning_rate_schedule, train
from fcgno.training.checkpoint import save_model, load_model

N = 8


# One Diffusion problem with a dense inverse and a batch of normalized residuals and exact errors
@pytest.fixture(scope="module")
def problem():
    dataset = make_dataset("diffusion", N, 1, seed=0)
    matrix = assemble_csr(dataset["face_x"][0], dataset["face_y"][0]).toarray()
    residual = np.random.default_rng(0).standard_normal((3, N * N))
    residual /= np.linalg.norm(residual, axis=1, keepdims=True)
    batch = {
        "residual": jnp.asarray(residual.reshape(3, N, N)),
        "error": jnp.asarray(np.linalg.solve(matrix, residual.T).T.reshape(3, N, N)),
        "face_x": jnp.repeat(jnp.asarray(dataset["face_x"]), 3, axis=0),
        "face_y": jnp.repeat(jnp.asarray(dataset["face_y"]), 3, axis=0),
    }
    inverse = jnp.asarray(np.linalg.inv(matrix))
    exact_model = lambda field: (inverse @ field.reshape(-1)).reshape(field.shape)
    return dataset, batch, exact_model


# Both losses vanish for the exact inverse and equal one for a zero preconditioner
@pytest.mark.parametrize("loss", [notay_loss, l2_loss])
def test_losses_exact_and_zero(problem, loss):
    _, batch, exact_model = problem
    assert float(loss(exact_model, batch, "physical")) < 1e-10
    assert float(loss(lambda field: jnp.zeros_like(field), batch, "physical")) == pytest.approx(1.0, rel=1e-12)


# Krylov pairs: one pair per CG iterate, unit residuals, and A e = r for every pair
def test_krylov_pairs():
    dataset = make_dataset("poisson", N, 2, seed=1)
    residuals, errors, owner = krylov_pairs(dataset, 5, seed=0, normalization="physical")
    assert residuals.shape == errors.shape == (12, N, N)
    np.testing.assert_array_equal(owner, np.repeat([0, 1], 6))
    np.testing.assert_allclose(np.linalg.norm(residuals.reshape(12, -1), axis=1), 1.0, rtol=1e-12)
    for residual, error, problem_index in zip(residuals, errors, owner):
        matrix = assemble_csr(dataset["face_x"][problem_index], dataset["face_y"][problem_index])
        assert np.linalg.norm(matrix @ error.ravel() - residual.ravel()) < 1e-8


# Random pairs: n_repeats per operator and A e = r exactly
def test_random_pairs():
    dataset = make_dataset("diffusion", N, 2, seed=2)
    residuals, errors, owner = random_pairs(dataset, 4, seed=0, normalization="upstream")
    assert residuals.shape == (8, N, N)
    np.testing.assert_array_equal(owner, np.repeat([0, 1], 4))
    for residual, error, problem_index in zip(residuals, errors, owner):
        matrix = assemble_csr(dataset["face_x"][problem_index], dataset["face_y"][problem_index], "upstream")
        assert np.linalg.norm(matrix @ error.ravel() - residual.ravel()) < 1e-10


# Learning rate halves every 50 epochs as in upstream train_model
def test_learning_rate_schedule():
    schedule = learning_rate_schedule(1e-3, steps_per_epoch=10)
    assert float(schedule(0)) == pytest.approx(1e-3)
    assert float(schedule(499)) == pytest.approx(1e-3)
    assert float(schedule(501)) == pytest.approx(5e-4)
    assert float(schedule(1001)) == pytest.approx(2.5e-4)


# A few epochs of Notay training reduce the loss of a small SNO
def test_training_reduces_loss():
    dataset = make_dataset("poisson", 16, 2, seed=3)
    residuals, errors, owner = krylov_pairs(dataset, 20, seed=0, normalization="upstream")
    data = {"pairs": {"residual": residuals, "error": errors}, "problems": {"face_x": dataset["face_x"], "face_y": dataset["face_y"]}, "owner": owner}
    settings = {"epochs": 10, "batch_size": 8, "learning_rate": 1e-3, "decay_epochs": 50, "decay_factor": 0.5, "weight_decay": 1e-2}
    model = build_model("sno", jax.random.PRNGKey(0), features=4)
    _, history = train(model, data, "notay", settings, "upstream", log=lambda message: None)
    assert len(history) == 10 and np.all(np.isfinite(history))
    assert history[-1] < history[0]


# Checkpoints round-trip weights and spec
def test_checkpoint_roundtrip(tmp_path):
    model = build_model("unet", jax.random.PRNGKey(1), features=4)
    path = str(tmp_path / "unet_test")
    save_model(path, model, {"model": "unet", "features": 4, "note": "test"})
    loaded, spec = load_model(path)
    field = jax.random.normal(jax.random.PRNGKey(2), (1, 16, 16))
    np.testing.assert_array_equal(model(field), loaded(field))
    assert spec["note"] == "test" and load_model(str(tmp_path / "missing")) is None


# Float32 training runs, lowers the loss and hands back a float64 model
def test_float32_training():
    dataset = make_dataset("poisson", 16, 2, seed=4)
    residuals, errors, owner = krylov_pairs(dataset, 20, seed=0, normalization="upstream")
    data = {"pairs": {"residual": residuals, "error": errors}, "problems": {"face_x": dataset["face_x"], "face_y": dataset["face_y"]}, "owner": owner}
    settings = {"epochs": 10, "batch_size": 8, "learning_rate": 1e-3, "decay_epochs": 50, "decay_factor": 0.5, "weight_decay": 1e-2, "precision": "float32"}
    model, history = train(build_model("dilresnet", jax.random.PRNGKey(0), features=4), data, "notay", settings, "upstream", log=lambda message: None)
    assert np.all(np.isfinite(history)) and history[-1] < history[0]
    assert model.encoder.weight.dtype == jnp.float64


# Upstream-form MSE loss vanishes for the exact inverse and equals the mean squared error norm for zero output
def test_mse_loss(problem):
    _, batch, exact_model = problem
    assert float(mse_loss(exact_model, batch, "physical")) < 1e-18
    expected = float(jnp.mean(jnp.sum(batch["error"].reshape(3, -1) ** 2, axis=1)))
    assert float(mse_loss(lambda field: jnp.zeros_like(field), batch, "physical")) == pytest.approx(expected, rel=1e-12)
