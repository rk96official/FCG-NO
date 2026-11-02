import numpy as np
import jax
import jax.numpy as jnp
import pytest

from fcgno.data.assembly import apply_stencil, assemble_csr
from fcgno.data.generate import make_dataset
from fcgno.linalg.krylov import batch_dot, fcg
from fcgno.models.registry import as_preconditioner, build_model
from nfcg.linalg.history_fcg import history_fcg, history_init, history_step
from nfcg.linalg.precision import low_precision_preconditioner
from nfcg.linalg.safeguarded_fcg import safeguarded_fcg
from nfcg.linalg.smoothers import chebyshev_sweeps, jacobi_sweeps, make_smoother, smoothed_preconditioner, stencil_diagonal
from nfcg.linalg.traced_fcg import fcg_traced
from nfcg.models.history import build_history_sno, history_preconditioner

N = 16
MODEL_GRID = 32


# Small diffusion batch: operator, diagonal, right-hand sides, N(0, I) guesses, exact solutions and the raw dataset
@pytest.fixture(scope="module")
def problem():
    dataset = make_dataset("diffusion", N, 3, seed=2)
    face_x, face_y = jnp.asarray(dataset["face_x"]), jnp.asarray(dataset["face_y"])
    operator = lambda u: apply_stencil(face_x, face_y, u)
    guesses = jnp.asarray(np.random.default_rng(2).standard_normal((3, N, N)))
    return operator, stencil_diagonal(face_x, face_y, "physical"), jnp.asarray(dataset["rhs"]), guesses, jnp.asarray(dataset["solution"]), dataset


# A poor nonlinear preconditioner, so that the safeguard has something to fix
def poor(residual):
    return jnp.roll(residual, 3, axis=-1) + 0.1 * residual


# Squared A-norm of the error of an iterate, per sample
def error_energy(operator, solution, iterate):
    error = solution - iterate
    return np.asarray(batch_dot(error, operator(error)))


# The GPU diagonal equals the assembled matrix's diagonal
def test_stencil_diagonal(problem):
    dataset, diagonal = problem[5], problem[1]
    matrix = assemble_csr(dataset["face_x"][0], dataset["face_y"][0], "physical")
    np.testing.assert_allclose(np.asarray(diagonal[0]).ravel(), matrix.diagonal(), rtol=1e-12)


# Many Jacobi sweeps, and Chebyshev on the true eigenvalue interval of D^-1 A, both converge to A^-1 r
def test_smoothers_converge(problem):
    operator, diagonal, rhs, unused_guesses, solution, dataset = problem
    matrix = assemble_csr(dataset["face_x"][0], dataset["face_y"][0], "physical").toarray()
    eigenvalues = np.linalg.eigvals(matrix / matrix.diagonal()[:, None]).real
    scale = np.abs(np.asarray(solution[0])).max()
    chebyshev = chebyshev_sweeps(operator, diagonal, rhs, 80, lower=eigenvalues.min(), upper=eigenvalues.max())
    np.testing.assert_allclose(np.asarray(chebyshev[0]), np.asarray(solution[0]), atol=1e-6 * scale)
    jacobi = jacobi_sweeps(operator, diagonal, rhs, 3000)
    np.testing.assert_allclose(np.asarray(jacobi[0]), np.asarray(solution[0]), atol=1e-4 * scale)


# Proposition 2 at the first step: the safeguarded step reduces the error at least as much as either direction alone
def test_safeguard_dominates_single_directions(problem):
    operator, diagonal, rhs, guesses, solution, unused_dataset = problem
    fallback = make_smoother("jacobi", operator, diagonal, 1)
    combined = error_energy(operator, solution, safeguarded_fcg(operator, poor, fallback, rhs, guesses, 1)["iterate"])
    network_only = error_energy(operator, solution, fcg(operator, poor, rhs, guesses, 1)["iterate"])
    fallback_only = error_energy(operator, solution, fcg(operator, fallback, rhs, guesses, 1)["iterate"])
    assert np.all(combined <= np.minimum(network_only, fallback_only) * (1 + 1e-10))


# With a useless network (zero output) the safeguard still converges, driven by the fallback alone
def test_safeguard_survives_useless_network(problem):
    operator, diagonal, rhs, guesses, unused_solution, unused_dataset = problem
    fallback = make_smoother("chebyshev", operator, diagonal, 2)
    for every in (1, 2):
        result = safeguarded_fcg(operator, jnp.zeros_like, fallback, rhs, guesses, 60, every=every)
        relative = np.asarray(result["residual_norm"])
        assert np.all(relative[:, -1] < 1e-6 * relative[:, 0])
        np.testing.assert_allclose(np.asarray(result["fallback_share"])[:, :20], 1.0, atol=1e-8)


# The wrapper with a zero smoother reduces to the network; with a zero network it is a two-sided smoother
def test_smoothing_wrapper(problem):
    operator, diagonal, rhs, unused_guesses, unused_solution, unused_dataset = problem
    np.testing.assert_allclose(np.asarray(smoothed_preconditioner(poor, jnp.zeros_like, operator)(rhs)), np.asarray(poor(rhs)), rtol=1e-12)
    smooth = make_smoother("jacobi", operator, diagonal, 1)
    two_sided = smoothed_preconditioner(jnp.zeros_like, smooth, operator)(rhs)
    expected = smooth(rhs) + smooth(rhs - operator(smooth(rhs)))
    np.testing.assert_allclose(np.asarray(two_sided), np.asarray(expected), rtol=1e-12)


# Low-precision inference stays close to float64 inference, and the observer measures the gap delta
def test_low_precision_preconditioner():
    model = build_model("sno", jax.random.PRNGKey(0), features=8)
    residual = jnp.asarray(np.random.default_rng(3).standard_normal((2, MODEL_GRID, MODEL_GRID)))
    reference = np.asarray(as_preconditioner(model)(residual))
    single = np.asarray(low_precision_preconditioner(model, "float32", "highest")(residual))
    half = np.asarray(low_precision_preconditioner(model, "bfloat16")(residual))
    assert np.linalg.norm(single - reference) < 1e-5 * np.linalg.norm(reference)
    assert np.linalg.norm(half - reference) < 5e-2 * np.linalg.norm(reference)


# An observer equal to the preconditioner has zero deviation
def test_observer_deviation(problem):
    operator, unused_diagonal, rhs, guesses, solution, unused_dataset = problem
    traced = fcg_traced(operator, poor, rhs, guesses, solution, 5, observer=poor)
    np.testing.assert_allclose(np.asarray(traced["observer_deviation"]), 0.0, atol=1e-12)


# A history preconditioner that ignores the history reproduces fcgno's FCG
def test_history_fcg_matches_fcg(problem):
    operator, unused_diagonal, rhs, guesses, unused_solution, unused_dataset = problem
    ignore_history = lambda residual, updates, changes: poor(residual)
    history = history_fcg(operator, ignore_history, rhs, guesses, 20, history=2)
    reference = fcg(operator, poor, rhs, guesses, 20)
    np.testing.assert_allclose(np.asarray(history["residual_norm"]), np.asarray(reference["residual_norm"]), rtol=1e-12)


# The history stacks hold the latest updates (newest first) and their images under A
def test_history_stacks(problem):
    operator, unused_diagonal, rhs, guesses, unused_solution, unused_dataset = problem
    precondition = lambda residual, updates, changes: poor(residual)
    states = [history_init(operator, rhs, guesses, 20, 2)]
    for unused in range(2):
        states.append(history_step(states[-1], operator, precondition, 20)[0])
    iterates = [state.solver.iterate for state in states]
    np.testing.assert_allclose(np.asarray(states[2].updates[0]), np.asarray(iterates[2] - iterates[1]), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(states[2].updates[1]), np.asarray(iterates[1] - iterates[0]), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(states[2].changes), np.asarray(jax.vmap(operator)(states[2].updates)), rtol=1e-8, atol=1e-10)


# An HC-NO model runs inside the history FCG and gives finite residuals
def test_history_model_runs():
    dataset = make_dataset("poisson", MODEL_GRID, 2, seed=0, normalization="upstream")
    face_x, face_y = jnp.asarray(dataset["face_x"]), jnp.asarray(dataset["face_y"])
    operator = lambda u: apply_stencil(face_x, face_y, u, "upstream")
    model = build_history_sno(8, jax.random.PRNGKey(1), history=1)
    result = history_fcg(operator, history_preconditioner(model, 1), jnp.asarray(dataset["rhs"]), jnp.zeros((2, MODEL_GRID, MODEL_GRID)), 5)
    assert np.all(np.isfinite(np.asarray(result["residual_norm"])))
