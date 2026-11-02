import numpy as np
import jax
import jax.numpy as jnp
import pytest

from fcgno.data.assembly import apply_stencil, face_coefficients
from fcgno.data.generate import make_dataset
from fcgno.linalg.krylov import fcg
import nfcg.training.records as records
from nfcg.data.coefficient_features import log_coefficient
from nfcg.experiments.main_table import main_key, solve_main
from nfcg.linalg.multilevel import coarse_faces, multilevel_preconditioner
from nfcg.models.history import build_history_sno
from nfcg.training.records import two_level_key

FINE = 64
NORMALIZATION = "upstream"


# Small diffusion batch: faces, operator, right-hand sides, N(0, I) guesses and the dataset
@pytest.fixture(scope="module")
def problem():
    dataset = make_dataset("diffusion", FINE, 2, seed=7, normalization=NORMALIZATION)
    face_x, face_y = jnp.asarray(dataset["face_x"]), jnp.asarray(dataset["face_y"])
    operator = lambda u: apply_stencil(face_x, face_y, u, NORMALIZATION)
    guesses = jnp.asarray(np.random.default_rng(7).standard_normal((2, FINE, FINE)))
    return face_x, face_y, operator, jnp.asarray(dataset["rhs"]), guesses, dataset


# Iterations of each system to reach tol (len when never reached)
def iterations(norms, tol):
    relative = np.asarray(norms) / np.asarray(norms)[:, :1]
    return [int(np.argmax(row <= tol)) if (row <= tol).any() else len(row) for row in relative]


# Rediscretized faces of a smooth coefficient match the coefficient evaluated on the coarse faces
def test_coarse_faces_sample_the_coefficient():
    coefficient = lambda x, y: jnp.exp(jnp.sin(2.0 * x) * jnp.cos(3.0 * y))
    fine_x, fine_y = face_coefficients(coefficient, FINE)
    expected_x, expected_y = face_coefficients(coefficient, 16)
    sampled_x, sampled_y = coarse_faces(fine_x[None], fine_y[None], 16)
    np.testing.assert_allclose(np.asarray(sampled_x[0]), np.asarray(expected_x), rtol=5e-3)
    np.testing.assert_allclose(np.asarray(sampled_y[0]), np.asarray(expected_y), rtol=5e-3)


# Setup-free multi-level PCG converges fast with a dense bottom, and a Chebyshev bottom on 32 is slower but converges
def test_rediscretized_multilevel_converges(problem):
    face_x, face_y, operator, rhs, guesses, unused_dataset = problem
    counts = {}
    for bottom, coarse_grid in (("dense", 8), ("chebyshev", 32)):
        options = {"coarse_grid": coarse_grid, "smoother": "chebyshev", "sweeps": 2, "bottom": bottom, "bottom_sweeps": 8}
        precondition = multilevel_preconditioner(face_x, face_y, NORMALIZATION, options)
        counts[bottom] = max(iterations(fcg(operator, precondition, rhs, guesses, 80, 20)["residual_norm"], 1e-6))
    assert counts["dense"] <= 12 and counts["dense"] < counts["chebyshev"] < 80


# A history network in the multi-level and recursive forms runs end to end in the main-table solver
def test_multilevel_network_variants(problem):
    face_x, face_y, unused_operator, rhs, guesses, unused_dataset = problem
    model = build_history_sno(8, jax.random.PRNGKey(2), history=1)
    for options in ({"coarse_grid": 16, "smoother": "chebyshev", "sweeps": 2}, {"coarse_grid": 16, "smoother": "chebyshev", "sweeps": 2, "network_max": 32}):
        result = solve_main(model, face_x, face_y, rhs, guesses, None, 6, 20, NORMALIZATION, 1, main_key({"two_level": options, "precision": {"dtype": "float32"}}))
        assert np.all(np.isfinite(np.asarray(result["residual_norm"])))


# On-policy records and one epoch of training through the recursive multi-level preconditioner run and give finite losses
def test_recursive_training_round(problem):
    unused_faces_x, unused_faces_y, unused_operator, unused_rhs, guesses, dataset = problem
    from fcgno.training.trainer import train
    from nfcg.training.records import collect_records, problem_table, training_data
    from nfcg.training.step_losses import register_loss
    options = {"coarse_grid": 16, "smoother": "chebyshev", "sweeps": 2, "network_max": 32}
    model = build_history_sno(8, jax.random.PRNGKey(3), history=1)
    records = collect_records(model, 1, dataset, np.asarray(guesses), 4, 20, NORMALIZATION, chunk=2, two_level=options)
    settings = {"learning_rate": 1e-3, "decay_epochs": 50, "decay_factor": 0.5, "weight_decay": 1e-2, "epochs": 1, "batch_size": 4}
    loss = register_loss("step", 1, scale_weight=0.05, two_level=options)
    unused_model, curve = train(model, training_data(records, problem_table(dataset)), loss, settings, NORMALIZATION, log=lambda message: None)
    assert "network_max32" in loss and np.all(np.isfinite(curve))



# fine_network places the network on the fine grid (line-searched) and as the coarse solve on grid / coarsen; the plain two-level slot only on the coarse grid
def test_fine_network_calls_both_grids(problem, monkeypatch):
    face_x, face_y, unused_operator, rhs, unused_guesses, unused_dataset = problem
    seen = []

    # Stand-in network that records the grid it is applied on
    def recording_preconditioner(model, history, field):
        def network(residual, updates, changes):
            seen.append(residual.shape[-1])
            return residual
        return network

    monkeypatch.setattr(records, "model_preconditioner", recording_preconditioner)
    empty = jnp.zeros((1, *rhs.shape))
    options = {"coarsen": 2, "smoother": "chebyshev", "sweeps": 2}
    fine = records.deployed_preconditioner(object(), 1, face_x, face_y, NORMALIZATION, FINE, None, two_level_key({**options, "fine_network": True}))
    fine_output = fine(rhs, empty, empty)
    assert sorted(set(seen)) == [FINE // 2, FINE]
    seen.clear()
    plain_output = records.deployed_preconditioner(object(), 1, face_x, face_y, NORMALIZATION, FINE, None, two_level_key(options))(rhs, empty, empty)
    assert seen == [FINE // 2]
    assert not np.allclose(np.asarray(fine_output), np.asarray(plain_output))


# Standardized log a: unchanged for mild coefficients (std < 1), unit standard deviation for binary jumps
def test_log_coefficient_standardization():
    mild = {"coefficient": np.exp(0.3 * np.random.default_rng(0).standard_normal((2, 8, 8)))}
    centered = np.log(mild["coefficient"]) - np.log(mild["coefficient"]).mean(axis=(-2, -1), keepdims=True)
    np.testing.assert_allclose(log_coefficient(mild), centered)
    jump = {"coefficient": np.where(np.random.default_rng(1).random((2, 8, 8)) > 0.5, 1e6, 1.0)}
    np.testing.assert_allclose(log_coefficient(jump).std(axis=(-2, -1)), 1.0, rtol=1e-6)
