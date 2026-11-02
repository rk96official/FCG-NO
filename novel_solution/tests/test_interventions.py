import numpy as np
import jax.numpy as jnp
import pytest

from fcgno.data.assembly import apply_stencil
from fcgno.data.generate import make_dataset
from fcgno.linalg.krylov import batch_dot
from nfcg.eval.interventions import angle_cap, residual_swap
from nfcg.linalg.traced_fcg import angle_sine_squared, fcg_traced

N = 16
STEPS = 25


# Small diffusion batch with exact solutions, N(0, I) initial guesses and the matching initial errors
@pytest.fixture(scope="module")
def problem():
    dataset = make_dataset("diffusion", N, 3, seed=1)
    face_x, face_y = jnp.asarray(dataset["face_x"]), jnp.asarray(dataset["face_y"])
    operator = lambda u: apply_stencil(face_x, face_y, u)
    guesses = jnp.asarray(np.random.default_rng(1).standard_normal((3, N, N)))
    solution = jnp.asarray(dataset["solution"])
    return operator, jnp.asarray(dataset["rhs"]), guesses, solution


# A poor nonlinear preconditioner, so that some steps have a bad angle
def poor(residual):
    return jnp.roll(residual, 3, axis=-1) + 0.1 * residual


# Squared A-angle sine of a field against the error
def sine_squared(field, error, operator):
    error_energy = batch_dot(error, operator(error))
    return np.asarray(angle_sine_squared(field, operator(field), operator(error), error_energy))


# The angle cap leaves good steps alone and puts bad steps exactly at the threshold
def test_angle_cap(problem):
    operator, rhs, guesses, solution = problem
    error = solution - guesses
    residual = rhs - operator(guesses)
    proposed = poor(residual)
    before = sine_squared(proposed, error, operator)
    threshold = float(np.median(before))
    capped, mask = angle_cap(threshold)(proposed, residual, error, operator)
    after = sine_squared(capped, error, operator)
    np.testing.assert_allclose(after[np.asarray(mask)], threshold, rtol=1e-8)
    np.testing.assert_array_equal(np.asarray(capped)[~np.asarray(mask)], np.asarray(proposed)[~np.asarray(mask)])


# The residual swap replaces exactly the bad steps by the residual
def test_residual_swap(problem):
    operator, rhs, guesses, solution = problem
    residual = rhs - operator(guesses)
    swapped, mask = residual_swap(0.0)(poor(residual), residual, solution - guesses, operator)
    assert np.asarray(mask).all()
    np.testing.assert_array_equal(np.asarray(swapped), np.asarray(residual))


# An intervention that never fires and an observer equal to the preconditioner leave the traces unchanged
def test_traced_options_are_neutral(problem):
    operator, rhs, guesses, solution = problem
    base = fcg_traced(operator, poor, rhs, guesses, solution, STEPS)
    options = fcg_traced(operator, poor, rhs, guesses, solution, STEPS, intervention=angle_cap(1.0), observer=poor)
    np.testing.assert_allclose(options["residual_norm"], base["residual_norm"], rtol=1e-12)
    np.testing.assert_allclose(options["observer_sine_squared"], base["preconditioned_sine_squared"], rtol=1e-10)
    assert not np.asarray(options["intervened"]).any()


# Capping bad angles never slows the error decrease relative to the uncapped run at the first step
def test_cap_improves_first_step(problem):
    operator, rhs, guesses, solution = problem
    base = fcg_traced(operator, poor, rhs, guesses, solution, 1)
    capped = fcg_traced(operator, poor, rhs, guesses, solution, 1, intervention=angle_cap(0.5))
    assert np.all(np.asarray(capped["contraction"]) <= np.asarray(base["contraction"]) + 1e-12)
