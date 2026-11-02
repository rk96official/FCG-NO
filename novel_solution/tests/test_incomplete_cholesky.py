import numpy as np
import jax
import jax.numpy as jnp

from fcgno.data.assembly import apply_stencil
from fcgno.data.generate import make_dataset
from fcgno.linalg.krylov import fcg
from nfcg.linalg.incomplete_cholesky import factor_product, factor_solve, ic0_factor
from nfcg.models.neuralif import NeuralIF, frobenius_loss, neural_factor

NORMALIZATION = "upstream"


# Small diffusion batch on an n-grid: faces, operator and the dataset
def diffusion(n, seed=5):
    dataset = make_dataset("diffusion", n, 2, seed=seed, normalization=NORMALIZATION)
    face_x, face_y = jnp.asarray(dataset["face_x"]), jnp.asarray(dataset["face_y"])
    return face_x, face_y, (lambda u: apply_stencil(face_x, face_y, u, NORMALIZATION)), jnp.asarray(dataset["rhs"])


# Dense matrices of a batched linear map on an n-grid, one per system
def dense(apply, n):
    basis = jnp.eye(n * n).reshape(n * n, 1, n, n)
    columns = jnp.stack([apply(jnp.broadcast_to(vector, (2, n, n))) for vector in basis[:, 0]], axis=-1)
    return np.asarray(columns.reshape(2, n * n, n * n))


# IC(0): L L^T equals A on A's sparsity pattern, and the wavefront solve inverts L L^T
def test_ic0_matches_pattern_and_solve_inverts():
    n = 7
    face_x, face_y, operator, unused_rhs = diffusion(n)
    factor = ic0_factor(face_x, face_y, NORMALIZATION)
    product, matrix = dense(lambda w: factor_product(factor, w), n), dense(operator, n)
    pattern = matrix != 0
    np.testing.assert_allclose(product[pattern], matrix[pattern], rtol=1e-10, atol=1e-12)
    vector = jnp.asarray(np.random.default_rng(1).standard_normal((2, n, n)))
    np.testing.assert_allclose(np.asarray(factor_product(factor, factor_solve(factor, vector))), np.asarray(vector), rtol=1e-9, atol=1e-9)


# IC(0)-PCG converges in clearly fewer iterations than CG
def test_ic0_preconditioning_helps():
    n = 32
    face_x, face_y, operator, rhs = diffusion(n)
    guess = jnp.asarray(np.random.default_rng(2).standard_normal((2, n, n)))
    factor = ic0_factor(face_x, face_y, NORMALIZATION)
    count = lambda precondition: max(int(np.argmax(row / row[0] <= 1e-6)) for row in np.asarray(fcg(operator, precondition, rhs, guess, 200, 20)["residual_norm"]))
    assert count(lambda residual: factor_solve(factor, residual)) < 0.6 * count(lambda residual: residual)


# NeuralIF: the Frobenius loss is finite, one gradient step lowers it, and its factor gives a convergent preconditioner
def test_neuralif_trains_and_preconditions():
    n = 16
    face_x, face_y, operator, rhs = diffusion(n)
    model = NeuralIF(8, jax.random.PRNGKey(0), 2)
    key = jax.random.PRNGKey(1)
    loss, grads = jax.value_and_grad(lambda m: frobenius_loss(m, face_x, face_y, NORMALIZATION, key))(model)
    stepped = jax.tree_util.tree_map(lambda value, grad: value - 1e-2 * grad if grad is not None else value, model, grads)
    assert np.isfinite(float(loss)) and float(frobenius_loss(stepped, face_x, face_y, NORMALIZATION, key)) < float(loss)
    factor = neural_factor(model, face_x, face_y, NORMALIZATION)
    guess = jnp.asarray(np.random.default_rng(3).standard_normal((2, n, n)))
    norms = np.asarray(fcg(operator, lambda residual: factor_solve(factor, residual), rhs, guess, 100, 20)["residual_norm"])
    assert np.all(norms[:, -1] / norms[:, 0] < 1e-6)
