import numpy as np
import jax.numpy as jnp
import equinox as eqx
import scipy.sparse.linalg as spla

from fcgno.data.assembly import apply_stencil, assemble_csr
from fcgno.linalg.krylov import fcg
from fcgno.models.registry import as_preconditioner
from fcgno.eval.metrics import summarize_iterations


# FCG on a whole test set with the model as preconditioner, or B = I (CG) when model is None
def solve_test_set(model, dataset, guesses, n_iter, m_max, normalization):
    # Jitted batched FCG with per-iteration Notay ratios against the exact solutions
    @eqx.filter_jit
    def solve(model, face_x, face_y, rhs, guess, solution):
        precondition = (lambda r: r) if model is None else as_preconditioner(model)
        operator = lambda u: apply_stencil(face_x, face_y, u, normalization)
        return fcg(operator, precondition, rhs, guess, n_iter, m_max, solution=solution)

    arrays = [jnp.asarray(dataset[name]) for name in ("face_x", "face_y", "rhs")]
    result = solve(model, *arrays, jnp.asarray(guesses), jnp.asarray(dataset["solution"]))
    norms = np.asarray(result["residual_norm"])
    return {"relative_residual": norms / norms[:, :1], "notay": np.asarray(result["notay"]), "iterate": np.asarray(result["iterate"])}


# Iteration counts per tolerance, keyed by the tolerance written as a string
def iteration_table(relative_residual, tolerances):
    return {f"{tol:g}": summarize_iterations(relative_residual, tol) for tol in tolerances}


# Iterations needed per tolerance from the mean curve, the quantity reported in the paper's tables
def mean_curve_iterations(table):
    return {tol: summary["iterations_mean_curve"] for tol, summary in table.items()}


# Paper-style Notay ratio per iteration, ||B(r_i) - A^{-1} r_i||_A / ||A^{-1} r_i||_A with A^{-1} r_i from a sparse LU (upstream res_func)
def exact_notay_curves(model, dataset, guesses, n_iter, m_max, normalization, chunk=64):
    # Jitted batched FCG keeping every residual
    @eqx.filter_jit
    def solve(model, face_x, face_y, rhs, guess):
        precondition = (lambda r: r) if model is None else as_preconditioner(model)
        return fcg(lambda u: apply_stencil(face_x, face_y, u, normalization), precondition, rhs, guess, n_iter, m_max, keep_history=True)

    arrays = [jnp.asarray(dataset[name]) for name in ("face_x", "face_y", "rhs")]
    result = solve(model, *arrays, jnp.asarray(guesses))
    residuals = np.asarray(result["residual_history"])[:, :-1]
    batch, steps, n = residuals.shape[0], residuals.shape[1], residuals.shape[-1]
    apply = eqx.filter_jit(lambda module, fields: fields if module is None else as_preconditioner(module)(fields))
    flat = residuals.reshape(-1, n, n)
    preconditioned = np.concatenate([np.asarray(apply(model, jnp.asarray(flat[start:start + chunk]))) for start in range(0, len(flat), chunk)])
    preconditioned = preconditioned.reshape(batch, steps, n * n)
    curves = np.zeros((batch, steps))
    for sample in range(batch):
        matrix = assemble_csr(dataset["face_x"][sample], dataset["face_y"][sample], normalization)
        errors = spla.splu(matrix.tocsc()).solve(residuals[sample].reshape(steps, -1).T).T
        difference = preconditioned[sample] - errors
        energy = lambda fields: np.einsum("ij,ij->i", fields, (matrix @ fields.T).T)
        curves[sample] = np.sqrt(energy(difference) / energy(errors))
    norms = np.asarray(result["residual_norm"])
    return {"notay": curves, "relative_residual": norms / norms[:, :1]}
