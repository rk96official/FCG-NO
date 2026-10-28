import numpy as np
import jax.numpy as jnp
import pytest

from fcgno.data.assembly import apply_stencil, assemble_csr, interior_mesh
from fcgno.data.generate import make_dataset
from fcgno.linalg.krylov import fcg, cg
from fcgno.eval.evaluate import solve_test_set, iteration_table, mean_curve_iterations, exact_notay_curves
from fcgno.eval.timing import make_runner, make_eager_runner, time_to_tolerances
from fcgno.eval.surrogate import restrict_to_grid, relative_errors

N = 16


# Small test set with random initial guesses
@pytest.fixture(scope="module")
def test_set():
    dataset = make_dataset("diffusion", N, 3, seed=5, normalization="upstream")
    return dataset, np.random.default_rng(0).standard_normal((3, N, N))


# Timing runner without a model reproduces plain CG after k steps
def test_runner_matches_cg(test_set):
    dataset, guesses = test_set
    args = [jnp.asarray(dataset[name]) for name in ("face_x", "face_y", "rhs")] + [jnp.asarray(guesses)]
    operator = lambda u: apply_stencil(args[0], args[1], u, "upstream")
    expected = cg(operator, args[2], args[3], 12)["iterate"]
    np.testing.assert_allclose(make_runner(None, "upstream", 20)(*args, jnp.asarray(12)), expected, rtol=1e-10, atol=1e-10)


# Timing runner with an identity model reproduces FCG with B = I after k steps
def test_runner_matches_fcg(test_set):
    dataset, guesses = test_set
    args = [jnp.asarray(dataset[name]) for name in ("face_x", "face_y", "rhs")] + [jnp.asarray(guesses)]
    operator = lambda u: apply_stencil(args[0], args[1], u, "upstream")
    expected = fcg(operator, lambda r: r, args[2], args[3], 9, 20)["iterate"]
    np.testing.assert_allclose(make_runner(lambda field: field, "upstream", 20)(*args, jnp.asarray(9)), expected, rtol=1e-9, atol=1e-9)


# Test-set solve returns curves starting at one, Notay ratios per iteration, and iteration tables
def test_solve_test_set(test_set):
    dataset, guesses = test_set
    result = solve_test_set(None, dataset, guesses, 120, 20, "upstream")
    assert result["relative_residual"].shape == (3, 121) and result["notay"].shape == (3, 120)
    np.testing.assert_allclose(result["relative_residual"][:, 0], 1.0)
    table = iteration_table(result["relative_residual"], [1e-3, 1e-6])
    assert set(table) == {"0.001", "1e-06"}
    assert 0 < mean_curve_iterations(table)["0.001"] <= mean_curve_iterations(table)["1e-06"]


# Timing produces a finite saving for every reachable tolerance and skips unreachable ones
def test_time_to_tolerances(test_set):
    dataset, guesses = test_set
    rows = time_to_tolerances(lambda field: field, dataset, guesses, "upstream", 20, {"0.001": 10, "1e-12": -1}, {"0.001": 12, "1e-12": 30}, repeats=2)
    assert rows["1e-12"] is None and np.isfinite(rows["0.001"]["saving_percent"])
    assert np.isfinite(rows["0.001"]["saving_vs_identity_fcg_percent"]) and np.isfinite(rows["0.001"]["saving_eager_percent"])


# Eager paper-style runner gives the same iterate as the jitted FCG with B = I
def test_eager_runner_matches_fcg(test_set):
    dataset, guesses = test_set
    args = [jnp.asarray(dataset[name]) for name in ("face_x", "face_y", "rhs")] + [jnp.asarray(guesses)]
    expected = make_runner(None, "upstream", 20, identity_fcg=True)(*args, jnp.asarray(7))
    np.testing.assert_allclose(make_eager_runner(None, "upstream", 20)(*args, 7), expected, rtol=1e-10, atol=1e-10)


# Cubic restriction of a smooth fine-grid field matches the analytic values on the coarse grid
def test_restrict_to_grid():
    fine_x, fine_y = interior_mesh(127)
    fine = np.asarray(jnp.sin(jnp.pi * fine_x) * jnp.sin(jnp.pi * fine_y))[None]
    coarse_x, coarse_y = interior_mesh(15)
    expected = np.asarray(jnp.sin(jnp.pi * coarse_x) * jnp.sin(jnp.pi * coarse_y))[None]
    assert relative_errors(restrict_to_grid(fine, 15), expected)[0] < 1e-6


# Exact Notay curves: finite per-iteration ratios, zero for the exact inverse, and residual curves starting at one
def test_exact_notay_curves(test_set):
    dataset, guesses = test_set
    result = exact_notay_curves(None, dataset, guesses, 10, 20, "upstream")
    assert result["notay"].shape == (3, 10) and np.all(np.isfinite(result["notay"])) and np.all(result["notay"] > 0)
    np.testing.assert_allclose(result["relative_residual"][:, 0], 1.0)
    inverse = jnp.asarray(np.linalg.inv(assemble_csr(dataset["face_x"][0], dataset["face_y"][0], "upstream").toarray()))
    single = {name: values[:1] for name, values in dataset.items()}
    exact_model = lambda field: (inverse @ field.reshape(-1)).reshape(field.shape)
    assert exact_notay_curves(exact_model, single, guesses[:1], 1, 20, "upstream")["notay"][0, 0] < 1e-8
