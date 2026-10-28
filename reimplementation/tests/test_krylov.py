import numpy as np
import jax
import jax.numpy as jnp
import pytest

from fcgno.data.assembly import apply_stencil, assemble_csr
from fcgno.data.generate import make_dataset
from fcgno.linalg.krylov import fcg, cg, truncation_length
from fcgno.linalg.reference_krylov import fcg_numpy
from fcgno.linalg.classical import build_classical, jacobi, symmetric_gauss_seidel
from fcgno.eval.metrics import iterations_to_tolerance

N = 16


# Small diffusion batch with random N(0, 1) initial guesses as in Algorithm 1
@pytest.fixture(scope="module")
def problem():
    dataset = make_dataset("diffusion", N, 3, seed=0)
    guesses = np.random.default_rng(0).standard_normal((3, N, N))
    return dataset, guesses


# Batched stencil operator for a dataset as a closure
def operator_for(dataset):
    face_x, face_y = jnp.asarray(dataset["face_x"]), jnp.asarray(dataset["face_y"])
    return lambda u: apply_stencil(face_x, face_y, u)


# Number of directions kept by the upstream list-reset rule in their FCG loop
def upstream_truncation_lengths(n_steps, m_max):
    lengths, kept = [], 0
    for step in range(n_steps):
        lengths.append(kept)
        kept = 1 if step % (m_max + 1) in (0, m_max) else kept + 1
    return lengths


# Paper formula for m_i reproduces the truncation of the upstream implementation
@pytest.mark.parametrize("m_max", [3, 20])
def test_truncation_matches_upstream(m_max):
    ours = [int(truncation_length(step, m_max)) for step in range(100)]
    assert ours == upstream_truncation_lengths(100, m_max)


# FCG with B = I and standard CG produce the same residual history
def test_fcg_identity_equals_cg(problem):
    dataset, guesses = problem
    operator = operator_for(dataset)
    rhs, guess = jnp.asarray(dataset["rhs"]), jnp.asarray(guesses)
    flexible = jax.jit(lambda: fcg(operator, lambda r: r, rhs, guess, 25))()
    standard = jax.jit(lambda: cg(operator, rhs, guess, 25))()
    np.testing.assert_allclose(flexible["residual_norm"], standard["residual_norm"], rtol=1e-6)


# JAX FCG matches the NumPy reference with a diagonal (Jacobi(1)) preconditioner
def test_fcg_matches_reference(problem):
    dataset, guesses = problem
    operator = operator_for(dataset)
    diagonals = np.stack([assemble_csr(fx, fy).diagonal().reshape(N, N) for fx, fy in zip(dataset["face_x"], dataset["face_y"])])
    result = jax.jit(lambda: fcg(operator, lambda r: r / diagonals, jnp.asarray(dataset["rhs"]), jnp.asarray(guesses), 30, m_max=5))()
    for sample in range(3):
        matrix = assemble_csr(dataset["face_x"][sample], dataset["face_y"][sample])
        reference = jacobi(matrix, 1)
        _, norms = fcg_numpy(matrix, reference, dataset["rhs"][sample].ravel(), guesses[sample].ravel(), 30, m_max=5)
        np.testing.assert_allclose(result["residual_norm"][sample], norms, rtol=1e-6)


# CG converges to the exact discrete solution
def test_cg_converges(problem):
    dataset, guesses = problem
    result = jax.jit(lambda: cg(operator_for(dataset), jnp.asarray(dataset["rhs"]), jnp.asarray(guesses), N * N))()
    relative = result["residual_norm"][:, -1] / result["residual_norm"][:, 0]
    assert np.all(relative < 1e-10)
    np.testing.assert_allclose(result["iterate"], dataset["solution"], rtol=1e-7, atol=1e-7 * np.abs(dataset["solution"]).max())


# With B = A^{-1}, FCG converges in one step and the Notay ratio is zero
def test_exact_preconditioner(problem):
    dataset, guesses = problem
    inverses = jnp.asarray(np.stack([np.linalg.inv(assemble_csr(fx, fy).toarray()) for fx, fy in zip(dataset["face_x"], dataset["face_y"])]))
    exact = lambda r: jnp.einsum("bij,bj->bi", inverses, r.reshape(3, -1)).reshape(r.shape)
    result = jax.jit(lambda: fcg(operator_for(dataset), exact, jnp.asarray(dataset["rhs"]), jnp.asarray(guesses), 3, solution=jnp.asarray(dataset["solution"])))()
    assert np.all(result["residual_norm"][:, 1] / result["residual_norm"][:, 0] < 1e-10)
    assert np.all(result["notay"][:, 0] < 1e-8)


# History mode returns residuals consistent with the recorded norms
def test_history_shapes(problem):
    dataset, guesses = problem
    result = jax.jit(lambda: fcg(operator_for(dataset), lambda r: r, jnp.asarray(dataset["rhs"]), jnp.asarray(guesses), 7, keep_history=True))()
    assert result["residual_history"].shape == (3, 8, N, N)
    assert result["iterate_history"].shape == (3, 8, N, N)
    np.testing.assert_allclose(np.linalg.norm(result["residual_history"].reshape(3, 8, -1), axis=-1), result["residual_norm"], rtol=1e-12)


# Jacobi(k) and SGS(k) act as symmetric linear operators
@pytest.mark.parametrize("builder", [lambda m: jacobi(m, 4), lambda m: symmetric_gauss_seidel(m, 2)])
def test_classical_symmetric(problem, builder):
    dataset, _ = problem
    matrix = assemble_csr(dataset["face_x"][0], dataset["face_y"][0])
    apply = builder(matrix)
    first, second = np.random.default_rng(1).standard_normal((2, N * N))
    assert first @ apply(second) == pytest.approx(second @ apply(first), rel=1e-10)


# Every classical preconditioner needs fewer iterations than CG to reach 1e-6
def test_classical_beats_cg(problem):
    dataset, guesses = problem
    matrix = assemble_csr(dataset["face_x"][0], dataset["face_y"][0])
    rhs, guess = dataset["rhs"][0].ravel(), guesses[0].ravel()
    counts = {}
    for spec in ["identity", "jacobi:4", "sgs:1", "sgs:4", "ilu:8"]:
        apply, _ = build_classical(spec, matrix)
        _, norms = fcg_numpy(matrix, apply, rhs, guess, 4 * N)
        counts[spec] = int(iterations_to_tolerance(norms / norms[0], 1e-6))
    assert all(0 < counts[spec] < counts["identity"] for spec in counts if spec != "identity"), counts


# Iteration counting picks the first index at or below the tolerance
def test_iterations_to_tolerance():
    curves = np.array([[1.0, 1e-2, 1e-4, 1e-7], [1.0, 0.5, 0.2, 0.1]])
    np.testing.assert_array_equal(iterations_to_tolerance(curves, 1e-6), [3, -1])
    np.testing.assert_array_equal(iterations_to_tolerance(curves, 1e-2), [1, -1])


# ILU accepts a SuperLU column ordering in its spec and still reduces CG iterations
def test_ilu_ordering(problem):
    dataset, guesses = problem
    matrix = assemble_csr(dataset["face_x"][0], dataset["face_y"][0])
    for spec in ("ilu:8::NATURAL", "ilu:8:1e-5:MMD_AT_PLUS_A"):
        apply, ratio = build_classical(spec, matrix)
        _, norms = fcg_numpy(matrix, apply, dataset["rhs"][0].ravel(), guesses[0].ravel(), 4 * N)
        assert ratio > 1 and 0 < iterations_to_tolerance(norms / norms[0], 1e-6) < 40
