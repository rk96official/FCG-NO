import numpy as np
import jax
import jax.numpy as jnp
import pytest

from fcgno.data.assembly import apply_stencil
from fcgno.data.generate import make_dataset
from fcgno.models.registry import build_model
from nfcg.experiments.composed import solve_composed, variant_key
from nfcg.experiments.main_table import main_key, solve_main
from nfcg.linalg.history_fcg import history_safeguarded_fcg
from nfcg.linalg.precision import low_precision_model
from nfcg.linalg.safeguarded_fcg import safeguarded_fcg
from nfcg.linalg.smoothers import make_smoother, smoothed_history_preconditioner, smoothed_preconditioner, stencil_diagonal
from nfcg.models.history import build_history_sno

GRID = 32
NORMALIZATION = "upstream"
VARIANTS = [
    {"label": "base"},
    {"label": "wrapper_safeguard_every2_fp32", "wrapper": {"smoother": "chebyshev", "sweeps": 2},
     "safeguard": {"fallback": "chebyshev", "sweeps": 2, "every": 2}, "precision": {"dtype": "float32", "matmul": "highest"}},
]


# Small diffusion batch at the SNO's grid: face coefficients, operator, diagonal, right-hand sides and N(0, I) guesses
@pytest.fixture(scope="module")
def problem():
    dataset = make_dataset("diffusion", GRID, 2, seed=4, normalization=NORMALIZATION)
    face_x, face_y = jnp.asarray(dataset["face_x"]), jnp.asarray(dataset["face_y"])
    operator = lambda u: apply_stencil(face_x, face_y, u, NORMALIZATION)
    guesses = jnp.asarray(np.random.default_rng(4).standard_normal((2, GRID, GRID)))
    return face_x, face_y, operator, stencil_diagonal(face_x, face_y, NORMALIZATION), jnp.asarray(dataset["rhs"]), guesses


# A poor nonlinear preconditioner
def poor(residual):
    return jnp.roll(residual, 3, axis=-1) + 0.1 * residual


# A history-aware safeguard whose network ignores the history reproduces the memoryless safeguard
@pytest.mark.parametrize("every", [1, 2])
def test_history_safeguard_matches_safeguard(problem, every):
    unused_x, unused_y, operator, diagonal, rhs, guesses = problem
    fallback = make_smoother("chebyshev", operator, diagonal, 2)
    history = history_safeguarded_fcg(operator, lambda residual, updates, changes: poor(residual), fallback, rhs, guesses, 15, every=every)
    plain = safeguarded_fcg(operator, poor, fallback, rhs, guesses, 15, every=every)
    np.testing.assert_allclose(np.asarray(history["residual_norm"]), np.asarray(plain["residual_norm"]), rtol=1e-10)


# The history-aware wrapper with a history-blind network equals the memoryless wrapper
def test_history_wrapper_matches_wrapper(problem):
    unused_x, unused_y, operator, diagonal, rhs, unused_guesses = problem
    smooth = make_smoother("jacobi", operator, diagonal, 2)
    empty = jnp.zeros((1,) + rhs.shape)
    history = smoothed_history_preconditioner(lambda residual, updates, changes: poor(residual), smooth, operator)(rhs, empty, empty)
    np.testing.assert_allclose(np.asarray(history), np.asarray(smoothed_preconditioner(poor, smooth, operator)(rhs)), rtol=1e-12)


# For a memoryless model, the main-table solver reproduces the E2b composed solver variant by variant
@pytest.mark.parametrize("variant", VARIANTS, ids=[variant["label"] for variant in VARIANTS])
def test_main_matches_composed(problem, variant):
    face_x, face_y, unused_operator, unused_diagonal, rhs, guesses = problem
    model = build_model("sno", jax.random.PRNGKey(0), features=8)
    main = solve_main(model, face_x, face_y, rhs, guesses, None, 12, 20, NORMALIZATION, 0, main_key(variant))
    composed = solve_composed(model, face_x, face_y, rhs, guesses, 12, 20, NORMALIZATION, variant_key(variant))
    np.testing.assert_allclose(np.asarray(main["residual_norm"]), np.asarray(composed["residual_norm"]), rtol=1e-9)


# A history model runs under every composition, and float32 inference stays close to float64
def test_history_model_compositions(problem):
    face_x, face_y, unused_operator, unused_diagonal, rhs, guesses = problem
    model = build_history_sno(8, jax.random.PRNGKey(1), history=1)
    probe = jnp.asarray(np.random.default_rng(5).standard_normal((3, GRID, GRID)))
    single = low_precision_model(model, "float32", "highest")(probe)
    assert single.dtype == probe.dtype
    np.testing.assert_allclose(np.asarray(single), np.asarray(model(probe)), rtol=1e-4, atol=1e-5 * float(jnp.abs(model(probe)).max()))
    for variant in VARIANTS:
        result = solve_main(model, face_x, face_y, rhs, guesses, None, 6, 20, NORMALIZATION, 1, main_key(variant))
        assert np.all(np.isfinite(np.asarray(result["residual_norm"])))
