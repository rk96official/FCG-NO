import numpy as np
import jax.numpy as jnp

from fcgno.data.assembly import apply_stencil
from fcgno.data.generate import make_dataset
from nfcg.linalg.safeguarded_fcg import safeguarded_fcg
from nfcg.linalg.smoothers import make_smoother, stencil_diagonal
from nfcg.linalg.two_level import interpolation_matrix, prolongate

NORMALIZATION = "upstream"


# Dense matrix of a linear map on (n_in x n_in) fields with (n_out x n_out) outputs
def dense(apply, n_in):
    basis = np.eye(n_in * n_in).reshape(n_in * n_in, n_in, n_in)
    return np.stack([np.asarray(apply(jnp.asarray(vector))).ravel() for vector in basis], axis=1)


# Squared A-cosine between two vectors
def cosine_squared(matrix, first, second):
    return (first @ matrix @ second) ** 2 / ((first @ matrix @ first) * (second @ matrix @ second))


# Small diffusion operator on an n-grid: faces, operator, dense matrix
def diffusion(n, seed=11):
    dataset = make_dataset("diffusion", n, 1, seed=seed, normalization=NORMALIZATION)
    face_x, face_y = jnp.asarray(dataset["face_x"][0]), jnp.asarray(dataset["face_y"][0])
    operator = lambda u: apply_stencil(face_x, face_y, u, NORMALIZATION)
    return face_x, face_y, operator, dense(operator, n)


# Theorem 3: cos^2_A(P w, e) = (1 - sin^2_{A_c}(w, x*)) cos^2_A(P x*, e) for the Galerkin A_c = P^T A P and any w
def test_learned_coarse_solve_identity():
    fine, coarse = 15, 7
    unused_x, unused_y, unused_operator, matrix = diffusion(fine)
    interpolation = interpolation_matrix(fine, coarse)
    transfer = dense(lambda field: prolongate(field, interpolation), coarse)
    coarse_matrix = transfer.T @ matrix @ transfer
    rng = np.random.default_rng(0)
    for unused in range(5):
        error, guess = rng.standard_normal(fine * fine), rng.standard_normal(coarse * coarse)
        exact = np.linalg.solve(coarse_matrix, transfer.T @ matrix @ error)
        delta = 1.0 - cosine_squared(coarse_matrix, guess, exact)
        np.testing.assert_allclose(cosine_squared(matrix, transfer @ guess, error), (1.0 - delta) * cosine_squared(matrix, transfer @ exact, error), rtol=1e-9)


# Theorem 2: with an arbitrary (here adversarial) network, every safeguarded step contracts the A-norm error at least by (kappa_f - 1) / (kappa_f + 1)
def test_safeguard_bound_holds_for_any_network():
    n = 12
    face_x, face_y, operator, matrix = diffusion(n, seed=12)
    fallback = make_smoother("chebyshev", operator, stencil_diagonal(face_x, face_y, NORMALIZATION), 2)
    eigenvalues = np.linalg.eigvals(dense(fallback, n) @ matrix).real
    kappa = eigenvalues.max() / eigenvalues.min()
    rng = np.random.default_rng(1)
    noise = jnp.asarray(rng.standard_normal((1, n, n)))
    network = lambda residual: jnp.sin(37.0 * residual) * noise - 3.0 * residual
    rhs, guess = jnp.asarray(rng.standard_normal((1, n, n))), jnp.asarray(rng.standard_normal((1, n, n)))
    solution = np.linalg.solve(matrix, np.asarray(rhs).ravel())
    batched = lambda u: apply_stencil(face_x, face_y, u, NORMALIZATION)
    batched_fallback = make_smoother("chebyshev", batched, stencil_diagonal(face_x, face_y, NORMALIZATION), 2)
    energies = []
    for steps in range(0, 10):
        iterate = np.asarray(safeguarded_fcg(batched, network, batched_fallback, rhs, guess, steps, 20)["iterate"]).ravel()
        error = solution - iterate
        energies.append(np.sqrt(error @ matrix @ error))
    ratios = np.array(energies[1:]) / np.array(energies[:-1])
    assert np.all(ratios <= (kappa - 1.0) / (kappa + 1.0) + 1e-8)
