import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# Jacobi(k): k sweeps of x <- x + D^{-1} (z - A x) starting from x = 0 (paper eq. 15)
def jacobi(matrix, steps):
    inverse_diagonal = 1.0 / matrix.diagonal()

    # Apply k Jacobi sweeps to z
    def apply(rhs):
        solution = np.zeros_like(rhs)
        for _ in range(steps):
            solution = solution + inverse_diagonal * (rhs - matrix @ solution)
        return solution

    return apply


# SGS(k): k forward-then-backward Gauss-Seidel sweeps starting from x = 0 (paper eq. 16)
def symmetric_gauss_seidel(matrix, steps):
    lower = sp.tril(matrix, format="csr")
    upper = sp.triu(matrix, format="csr")

    # Apply k symmetric Gauss-Seidel sweeps to z
    def apply(rhs):
        solution = np.zeros_like(rhs)
        for _ in range(steps):
            solution = solution + spla.spsolve_triangular(lower, rhs - matrix @ solution, lower=True)
            solution = solution + spla.spsolve_triangular(upper, rhs - matrix @ solution, lower=False)
        return solution

    return apply


# ILU from SuperLU with the given fill_factor and column ordering (paper Table 1); also returns nnz(L + U) / nnz(A)
def incomplete_lu(matrix, fill_factor, drop_tol=None, ordering=None):
    factor = spla.spilu(matrix.tocsc(), fill_factor=fill_factor, drop_tol=drop_tol, permc_spec=ordering or "COLAMD")
    return factor.solve, (factor.L.nnz + factor.U.nnz) / matrix.nnz


# Build a preconditioner from a spec such as "identity", "jacobi:4", "sgs:1", "ilu:8", "ilu:8:1e-5" or "ilu:1::NATURAL" (fill, drop tol, ordering)
def build_classical(spec, matrix):
    name, _, parameter = spec.partition(":")
    parameter, _, options = parameter.partition(":")
    drop_tol, _, ordering = options.partition(":")
    if name == "identity":
        return (lambda rhs: rhs.copy()), float("nan")
    if name == "jacobi":
        return jacobi(matrix, int(parameter)), float("nan")
    if name == "sgs":
        return symmetric_gauss_seidel(matrix, int(parameter)), float("nan")
    if name == "ilu":
        return incomplete_lu(matrix, float(parameter), float(drop_tol) if drop_tol else None, ordering or None)
    raise ValueError(f"unknown classical preconditioner: {spec}")
