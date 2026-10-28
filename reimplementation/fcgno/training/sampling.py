import numpy as np
import jax
import scipy.sparse.linalg as spla

from fcgno.data.assembly import apply_stencil, assemble_csr, interior_mesh
from fcgno.data.coefficients import DEFAULT_MAX_MODE, DEFAULT_ALPHA, sample_rhs
from fcgno.linalg.krylov import fcg


# Divide residual/error pairs by the residual norm, as upstream does before training
def normalize_pairs(residuals, errors):
    norms = np.linalg.norm(residuals.reshape(len(residuals), -1), axis=1)[:, None, None]
    return residuals / norms, errors / norms


# Krylov pairs r ~ p_Km (paper eq. 9): residuals and errors of FCG with B = I (CG) from u0 ~ N(0, I), n_steps + 1 per problem
def krylov_pairs(dataset, n_steps, seed, normalization, m_max=20, chunk=8):
    n = dataset["rhs"].shape[-1]
    guesses = np.random.default_rng(seed).standard_normal(dataset["rhs"].shape)

    # CG histories for a chunk of problems
    def run(face_x, face_y, rhs, guess):
        operator = lambda u: apply_stencil(face_x, face_y, u, normalization)
        return fcg(operator, lambda r: r, rhs, guess, n_steps, m_max, keep_history=True)

    run = jax.jit(run)
    residuals, errors = [], []
    for start in range(0, len(guesses), chunk):
        part = slice(start, start + chunk)
        result = run(dataset["face_x"][part], dataset["face_y"][part], dataset["rhs"][part], guesses[part])
        residuals.append(np.asarray(result["residual_history"]))
        errors.append(dataset["solution"][part][:, None] - np.asarray(result["iterate_history"]))
    residuals, errors = normalize_pairs(np.concatenate(residuals).reshape(-1, n, n), np.concatenate(errors).reshape(-1, n, n))
    return residuals, errors, np.repeat(np.arange(len(guesses)), n_steps + 1)


# Random pairs r ~ p_K0 (paper Section 3.3.3): r = f - A u with fresh f ~ P and u ~ N(0, I) per training operator, e = A^{-1} f - u
def random_pairs(dataset, n_repeats, seed, normalization, max_mode=DEFAULT_MAX_MODE, alpha=DEFAULT_ALPHA):
    n = dataset["rhs"].shape[-1]
    mesh_x, mesh_y = interior_mesh(n)
    rng = np.random.default_rng(seed)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(dataset["rhs"]) * n_repeats).reshape(len(dataset["rhs"]), n_repeats, -1)
    evaluate_rhs = jax.jit(jax.vmap(lambda key: sample_rhs(key, max_mode, alpha)(mesh_x, mesh_y)))
    residuals, errors = [], []
    for problem, (face_x, face_y) in enumerate(zip(dataset["face_x"], dataset["face_y"])):
        matrix = assemble_csr(face_x, face_y, normalization)
        right_hand_sides = np.asarray(evaluate_rhs(keys[problem])).reshape(n_repeats, -1)
        guesses = rng.standard_normal((n_repeats, n * n))
        exact = spla.splu(matrix.tocsc()).solve(right_hand_sides.T).T
        residuals.append(right_hand_sides - (matrix @ guesses.T).T)
        errors.append(exact - guesses)
    residuals, errors = normalize_pairs(np.concatenate(residuals).reshape(-1, n, n), np.concatenate(errors).reshape(-1, n, n))
    return residuals, errors, np.repeat(np.arange(len(dataset["rhs"])), n_repeats)
