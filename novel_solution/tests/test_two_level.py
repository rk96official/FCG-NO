import numpy as np
import jax
import jax.numpy as jnp
import pytest

from fcgno.data.assembly import apply_stencil, assemble_csr
from fcgno.data.generate import make_dataset
from nfcg.experiments.main_table import main_key, solve_main
from nfcg.linalg.history_fcg import history_fcg
from nfcg.linalg.smoothers import make_smoother, stencil_diagonal
from nfcg.linalg.two_level import interpolation_matrix, prolongate, restrict, two_level_history_preconditioner
from nfcg.models.history import build_history_sno

FINE = 32
COARSE = 16
NORMALIZATION = "upstream"


# Small diffusion batch on the fine grid: operator, diagonal, right-hand sides, N(0, I) guesses and the dataset
@pytest.fixture(scope="module")
def problem():
    dataset = make_dataset("diffusion", FINE, 2, seed=6, normalization=NORMALIZATION)
    face_x, face_y = jnp.asarray(dataset["face_x"]), jnp.asarray(dataset["face_y"])
    operator = lambda u: apply_stencil(face_x, face_y, u, NORMALIZATION)
    guesses = jnp.asarray(np.random.default_rng(6).standard_normal((2, FINE, FINE)))
    return operator, stencil_diagonal(face_x, face_y, NORMALIZATION), jnp.asarray(dataset["rhs"]), guesses, dataset


# Iterations of a relative-residual curve to reach tol (len when never reached)
def iterations(norms, tol):
    relative = np.asarray(norms) / np.asarray(norms)[:, :1]
    return [int(np.argmax(row <= tol)) if (row <= tol).any() else len(row) for row in relative]


# Interpolation reproduces a smooth field to second order, and restriction preserves constants away from the boundary
def test_grid_transfers():
    interpolation = interpolation_matrix(FINE, COARSE)
    mode = lambda n: np.sin(np.pi * np.arange(1, n + 1) / (n + 1))
    coarse, fine = np.outer(mode(COARSE), mode(COARSE)), np.outer(mode(FINE), mode(FINE))
    assert np.linalg.norm(np.asarray(prolongate(jnp.asarray(coarse), interpolation)) - fine) < 1e-2 * np.linalg.norm(fine)
    restricted = np.asarray(restrict(jnp.ones((FINE, FINE)), interpolation))
    np.testing.assert_allclose(restricted[2:-2, 2:-2], 1.0, atol=0.05)


# The energy line search makes the two-level output independent of the network's output scale
def test_scale_invariance(problem):
    operator, diagonal, rhs, unused_guesses, unused_dataset = problem
    interpolation = interpolation_matrix(FINE, COARSE)
    smooth = make_smoother("chebyshev", operator, diagonal, 2)
    network = lambda residual, updates, changes: jnp.roll(residual, 1, axis=-1) + residual
    empty = jnp.zeros((1, 2, COARSE, COARSE))
    unit = two_level_history_preconditioner(network, smooth, operator, interpolation)(rhs, jnp.zeros((1,) + rhs.shape), jnp.zeros((1,) + rhs.shape))
    scaled = two_level_history_preconditioner(lambda *fields: 1e3 * network(*fields), smooth, operator, interpolation)(
        rhs, jnp.zeros((1,) + rhs.shape), jnp.zeros((1,) + rhs.shape))
    np.testing.assert_allclose(np.asarray(scaled), np.asarray(unit), rtol=1e-9)
    assert empty.shape[-1] == COARSE


# With an exact Galerkin coarse solve in place of the network, the scheme is a two-grid method: few iterations, far fewer than smoothing alone
def test_two_grid_with_exact_coarse_solve(problem):
    operator, diagonal, rhs, guesses, dataset = problem
    interpolation = np.asarray(interpolation_matrix(FINE, COARSE))
    prolongation = np.kron(interpolation, interpolation)
    restriction = ((COARSE + 1) / (FINE + 1)) ** 2 * prolongation.T
    coarse_operators = [restriction @ assemble_csr(dataset["face_x"][index], dataset["face_y"][index], NORMALIZATION).toarray() @ prolongation for index in range(2)]
    inverses = jnp.asarray(np.stack([np.linalg.inv(matrix) for matrix in coarse_operators]))
    exact = lambda residual, updates, changes: jnp.einsum("bij,bj->bi", inverses, residual.reshape(2, -1)).reshape(residual.shape)
    smooth = make_smoother("chebyshev", operator, diagonal, 2)
    two_grid = two_level_history_preconditioner(exact, smooth, operator, jnp.asarray(interpolation))
    smoothing_only = two_level_history_preconditioner(lambda residual, updates, changes: jnp.zeros_like(residual), smooth, operator, jnp.asarray(interpolation))
    two_grid_iterations = iterations(history_fcg(operator, two_grid, rhs, guesses, 40)["residual_norm"], 1e-8)
    smoothing_iterations = iterations(history_fcg(operator, smoothing_only, rhs, guesses, 40)["residual_norm"], 1e-8)
    assert max(two_grid_iterations) <= 15
    assert min(smoothing_iterations) > 2 * max(two_grid_iterations)


# Two-level variants run end to end in the main-table solver for a history model, with and without the safeguard
def test_two_level_main_variants(problem):
    unused_operator, unused_diagonal, rhs, guesses, dataset = problem
    model = build_history_sno(8, jax.random.PRNGKey(2), history=1)
    face_x, face_y = jnp.asarray(dataset["face_x"]), jnp.asarray(dataset["face_y"])
    for variant in ({"two_level": {"coarsen": 2, "smoother": "chebyshev", "sweeps": 2}},
                    {"two_level": {"coarsen": 2, "smoother": "chebyshev", "sweeps": 2}, "safeguard": {"fallback": "chebyshev", "sweeps": 2, "every": 2},
                     "precision": {"dtype": "float32"}}):
        result = solve_main(model, face_x, face_y, rhs, guesses, None, 8, 20, NORMALIZATION, 1, main_key(variant))
        assert np.all(np.isfinite(np.asarray(result["residual_norm"])))


# On-policy records with the two-level preconditioner and one epoch of training through it (step loss) run and give finite losses
def test_two_level_training_round(problem):
    unused_operator, unused_diagonal, unused_rhs, guesses, dataset = problem
    from fcgno.training.trainer import train
    from nfcg.training.records import collect_records, problem_table, training_data
    from nfcg.training.step_losses import register_loss
    options = {"coarsen": 2, "smoother": "chebyshev", "sweeps": 2}
    model = build_history_sno(8, jax.random.PRNGKey(3), history=1)
    records = collect_records(model, 1, dataset, np.asarray(guesses), 6, 20, NORMALIZATION, chunk=2, two_level=options)
    assert len(records["owner"]) == 12
    settings = {"learning_rate": 1e-3, "decay_epochs": 50, "decay_factor": 0.5, "weight_decay": 1e-2, "epochs": 1, "batch_size": 4}
    loss = register_loss("step", 1, scale_weight=0.05, two_level=options)
    unused_model, curve = train(model, training_data(records, problem_table(dataset)), loss, settings, NORMALIZATION, log=lambda message: None)
    assert "_2l" in loss and np.all(np.isfinite(curve))
