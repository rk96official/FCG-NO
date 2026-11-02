import numpy as np
import jax.numpy as jnp
import pytest

from fcgno.data.assembly import apply_stencil
from fcgno.data.generate import make_dataset
from fcgno.linalg.krylov import fcg
from nfcg.linalg.multigrid import (apply_stencil_operator, build_hierarchy, coarse_solver, level_sizes, probe_stencil, spectral_upper,
                                   vcycle_preconditioner)
from nfcg.linalg.smoothers import make_smoother, stencil_diagonal
from nfcg.linalg.two_level import interpolation_matrix, prolongate, restrict, two_level_preconditioner

FINE = 48
NORMALIZATION = "upstream"


# Small diffusion batch: operator, diagonal, right-hand sides and N(0, I) guesses
@pytest.fixture(scope="module")
def problem():
    dataset = make_dataset("diffusion", FINE, 2, seed=4, normalization=NORMALIZATION)
    face_x, face_y = jnp.asarray(dataset["face_x"]), jnp.asarray(dataset["face_y"])
    operator = lambda u: apply_stencil(face_x, face_y, u, NORMALIZATION)
    guesses = jnp.asarray(np.random.default_rng(4).standard_normal((2, FINE, FINE)))
    return operator, stencil_diagonal(face_x, face_y, NORMALIZATION), jnp.asarray(dataset["rhs"]), guesses


# Iterations of each system to reach tol (len when never reached)
def iterations(norms, tol):
    relative = np.asarray(norms) / np.asarray(norms)[:, :1]
    return [int(np.argmax(row <= tol)) if (row <= tol).any() else len(row) for row in relative]


# Probing recovers the 5-point operator exactly, and every Galerkin level R A P exactly (bandwidth within the stencil radius)
def test_probed_stencils_are_exact(problem):
    operator = problem[0]
    vector = jnp.asarray(np.random.default_rng(0).standard_normal((2, FINE, FINE)))
    fine = probe_stencil(operator, 2, FINE, jnp.float64)
    np.testing.assert_allclose(np.asarray(apply_stencil_operator(fine, vector)), np.asarray(operator(vector)), atol=1e-10)
    apply_above = operator
    for level in build_hierarchy(operator, 2, level_sizes(FINE, 20), jnp.float64):
        interpolation, size = level["interpolation"], level["interpolation"].shape[1]
        coarse = jnp.asarray(np.random.default_rng(size).standard_normal((2, size, size)))
        expected = restrict(apply_above(prolongate(coarse, interpolation)), interpolation)
        np.testing.assert_allclose(np.asarray(apply_stencil_operator(level["stencil"], coarse)), np.asarray(expected), atol=1e-10)
        apply_above = lambda u, stencil=level["stencil"]: apply_stencil_operator(stencil, u)


# GMG-PCG converges in a handful of iterations, with fp64 or fp32 coarse levels
def test_vcycle_preconditioner_converges(problem):
    operator, diagonal, rhs, guesses = problem
    levels = build_hierarchy(operator, 2, level_sizes(FINE), jnp.float64)
    upper = spectral_upper(operator, diagonal)
    for coarse_levels in (levels, [{name: value.astype(jnp.float32) for name, value in level.items()} for level in levels]):
        precondition = vcycle_preconditioner(operator, diagonal, upper, coarse_levels, 2)
        assert max(iterations(fcg(operator, precondition, rhs, guesses, 30, 20)["residual_norm"], 1e-6)) <= 10


# Classical coarse solvers in the two-level slot: exact and V-cycle beat Chebyshev sweeps, which beat smoothing only
def test_coarse_solvers_order(problem):
    operator, diagonal, rhs, guesses = problem
    smooth = make_smoother("chebyshev", operator, diagonal, 2)
    interpolation = interpolation_matrix(FINE, 24)
    counts = {}
    for kind, sizes, factor in (("zero", [FINE, 24], False), ("chebyshev", [FINE, 24], False), ("vcycle", level_sizes(FINE, 24), True), ("exact", [FINE, 24], True)):
        levels = build_hierarchy(operator, 2, sizes, jnp.float64, factor)
        precondition = two_level_preconditioner(coarse_solver(kind, levels, 4), smooth, operator, interpolation)
        counts[kind] = max(iterations(fcg(operator, precondition, rhs, guesses, 80, 20)["residual_norm"], 1e-6))
    assert counts["exact"] <= counts["vcycle"] + 1 and counts["vcycle"] < counts["chebyshev"] < counts["zero"]
