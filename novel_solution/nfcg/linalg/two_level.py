import numpy as np
import jax.numpy as jnp

from fcgno.linalg.krylov import DENOMINATOR_EPS, batch_dot, batch_scale


# Linear interpolation (fine_n, coarse_n) from coarse interior nodes j / (coarse_n + 1) to fine interior nodes i / (fine_n + 1), zero on the boundary
def interpolation_matrix(fine_n, coarse_n):
    position = np.arange(1, fine_n + 1) / (fine_n + 1) * (coarse_n + 1)
    left = np.floor(position).astype(int)
    weight = position - left
    matrix = np.zeros((fine_n, coarse_n))
    rows = np.arange(fine_n)
    inside_left = (left >= 1) & (left <= coarse_n)
    inside_right = (left + 1 >= 1) & (left + 1 <= coarse_n)
    matrix[rows[inside_left], left[inside_left] - 1] += 1.0 - weight[inside_left]
    matrix[rows[inside_right], left[inside_right]] += weight[inside_right]
    return jnp.asarray(matrix)


# Prolongate coarse fields (..., coarse_n, coarse_n) to the fine grid by bilinear interpolation
def prolongate(fields, interpolation):
    return jnp.einsum("ia,...ab,jb->...ij", interpolation, fields, interpolation)


# Restrict fine fields (..., fine_n, fine_n) to the coarse grid with the transpose of interpolation, scaled per axis so constants are preserved (full weighting)
def restrict(fields, interpolation):
    scale = ((interpolation.shape[1] + 1) / (interpolation.shape[0] + 1)) ** 2
    return scale * jnp.einsum("ia,...ij,jb->...ab", interpolation, fields, interpolation)


# One two-level application: pre-smooth on the fine grid; coarse direction from the restricted remainder; prolongation with an exact energy line
# search (so the coarse direction's scale is irrelevant); post-smooth
def two_level_apply(coarse, smooth, apply_operator, interpolation, residual):
    correction = smooth(residual)
    remainder = residual - apply_operator(correction)
    direction = prolongate(coarse(restrict(remainder, interpolation)), interpolation)
    image = apply_operator(direction)
    correction = correction + batch_scale(batch_dot(direction, remainder) / (batch_dot(direction, image) + DENOMINATOR_EPS), direction)
    return correction + smooth(residual - apply_operator(correction))


# Two-level preconditioner around a history-aware network (r, updates, changes) on the coarse grid; the history is restricted alongside the residual
def two_level_history_preconditioner(network, smooth, apply_operator, interpolation):
    # One two-level application to a batch of residuals with their history stacks
    def apply(residual, updates, changes):
        coarse = lambda coarse_residual: network(coarse_residual, restrict(updates, interpolation), restrict(changes, interpolation))
        return two_level_apply(coarse, smooth, apply_operator, interpolation, residual)

    return apply


# Two-level preconditioner r -> correction around a memoryless coarse solver (classical baselines in the network's slot)
def two_level_preconditioner(coarse_solve, smooth, apply_operator, interpolation):
    return lambda residual: two_level_apply(coarse_solve, smooth, apply_operator, interpolation, residual)
