import numpy as np
import jax.numpy as jnp
import pytest

from fcgno.data.assembly import apply_stencil
from fcgno.data.generate import make_dataset
from fcgno.linalg.krylov import fcg
from nfcg.eval.diagnostics import active_steps
from nfcg.linalg.traced_fcg import fcg_traced

N = 16
STEPS = 30


# Small diffusion batch with exact solutions and N(0, I) initial guesses
@pytest.fixture(scope="module")
def problem():
    dataset = make_dataset("diffusion", N, 3, seed=0)
    face_x, face_y = jnp.asarray(dataset["face_x"]), jnp.asarray(dataset["face_y"])
    operator = lambda u: apply_stencil(face_x, face_y, u)
    guesses = jnp.asarray(np.random.default_rng(0).standard_normal((3, N, N)))
    return operator, jnp.asarray(dataset["rhs"]), guesses, jnp.asarray(dataset["solution"])


# A nonlinear preconditioner, so the checks do not rely on linearity
def nonlinear(residual):
    return residual + 0.3 * jnp.tanh(residual / jnp.max(jnp.abs(residual)))


# Tracing does not change the iteration: residual norms equal fcgno's FCG
def test_matches_fcgno_fcg(problem):
    operator, rhs, guesses, solution = problem
    traced = fcg_traced(operator, nonlinear, rhs, guesses, solution, STEPS)
    reference = fcg(operator, nonlinear, rhs, guesses, STEPS, solution=solution)
    np.testing.assert_allclose(traced["residual_norm"], reference["residual_norm"], rtol=1e-12)
    np.testing.assert_allclose(traced["notay"], reference["notay"], rtol=1e-10)


# Proposition 1: the measured energy contraction equals sin^2 of the A-angle between the search direction and the error
def test_contraction_equals_direction_sine(problem):
    operator, rhs, guesses, solution = problem
    traced = fcg_traced(operator, nonlinear, rhs, guesses, solution, STEPS)
    mask = active_steps(np.asarray(traced["residual_norm"]) / np.asarray(traced["residual_norm"])[:, :1], 1e-8)
    np.testing.assert_allclose(np.asarray(traced["contraction"])[mask], np.asarray(traced["direction_sine_squared"])[mask], rtol=1e-6, atol=1e-12)


# The A-angle is the Notay ratio at the best scale, so sin^2 never exceeds the squared ratio
def test_angle_bounded_by_notay_ratio(problem):
    operator, rhs, guesses, solution = problem
    traced = fcg_traced(operator, nonlinear, rhs, guesses, solution, STEPS)
    assert np.all(np.asarray(traced["preconditioned_sine_squared"]) <= np.asarray(traced["notay"]) ** 2 + 1e-12)


# Full orthogonalization (m_max >= n_iter) runs and still decreases the error energy monotonically
def test_full_orthogonalization(problem):
    operator, rhs, guesses, solution = problem
    traced = fcg_traced(operator, nonlinear, rhs, guesses, solution, STEPS, m_max=STEPS)
    energies = np.asarray(traced["error_energy"])
    assert np.all(np.diff(energies, axis=1) <= 1e-12 * energies[:, :1])
