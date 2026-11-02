import jax.numpy as jnp

from fcgno.data.assembly import operator_scale

JACOBI_DAMPING = 2.0 / 3.0
CHEBYSHEV_INTERVAL = (0.2, 2.2)


# Diagonal of the 5-point operator from face coefficients, shape (..., n, n); same values as assemble_csr's diagonal
def stencil_diagonal(face_x, face_y, normalization):
    n = face_x.shape[-1]
    return (face_x[..., :-1, :] + face_x[..., 1:, :] + face_y[..., :, :-1] + face_y[..., :, 1:]) * operator_scale(n, normalization)


# Damped-Jacobi sweeps for A x = r starting from x = 0
def jacobi_sweeps(apply_operator, diagonal, residual, sweeps, damping=JACOBI_DAMPING):
    correction = damping * residual / diagonal
    for unused in range(sweeps - 1):
        correction = correction + damping * (residual - apply_operator(correction)) / diagonal
    return correction


# Chebyshev iteration for A x = r from x = 0, preconditioned by the diagonal, for eigenvalues of D^-1 A in [lower, upper] (Saad, Alg. 12.1)
def chebyshev_sweeps(apply_operator, diagonal, residual, sweeps, lower=CHEBYSHEV_INTERVAL[0], upper=CHEBYSHEV_INTERVAL[1]):
    theta, delta = (upper + lower) / 2.0, (upper - lower) / 2.0
    sigma = theta / delta
    rho = 1.0 / sigma
    update = residual / diagonal / theta
    correction = update
    for unused in range(sweeps - 1):
        rho_next = 1.0 / (2.0 * sigma - rho)
        update = rho_next * rho * update + 2.0 * rho_next / delta * (residual - apply_operator(correction)) / diagonal
        correction = correction + update
        rho = rho_next
    return correction


# Linear map r -> correction for a smoother kind: none (zero), identity, damped Jacobi or Chebyshev with a number of sweeps
def make_smoother(kind, apply_operator, diagonal, sweeps=1):
    if kind == "none":
        return lambda residual: jnp.zeros_like(residual)
    if kind == "identity":
        return lambda residual: residual
    if kind == "jacobi":
        return lambda residual: jacobi_sweeps(apply_operator, diagonal, residual, sweeps)
    if kind == "chebyshev":
        return lambda residual: chebyshev_sweeps(apply_operator, diagonal, residual, sweeps)
    raise ValueError(f"unknown smoother: {kind}")


# Smoothing wrapper around a history-aware network B(r, updates, changes); the history inputs pass through unchanged
def smoothed_history_preconditioner(network, smooth, apply_operator):
    # Pre-smooth, history-aware network on the remaining residual, post-smooth
    def apply(residual, updates, changes):
        correction = smooth(residual)
        correction = correction + network(residual - apply_operator(correction), updates, changes)
        return correction + smooth(residual - apply_operator(correction))

    return apply


# Smoothing wrapper B(r) = S_post o NO o S_pre: the network corrects what pre-smoothing leaves, post-smoothing cleans up (network as coarse correction)
def smoothed_preconditioner(network, smooth, apply_operator):
    # Pre-smooth, network on the remaining residual, post-smooth
    def apply(residual):
        correction = smooth(residual)
        correction = correction + network(residual - apply_operator(correction))
        return correction + smooth(residual - apply_operator(correction))

    return apply
