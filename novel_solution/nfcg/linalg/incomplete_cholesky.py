import numpy as np
import jax.numpy as jnp
from jax import lax

from fcgno.data.assembly import operator_scale


# Gather indices of the anti-diagonal (wavefront) layout: row i of diagonal k holds node (i, k - i); invalid slots point at node (0, 0) and are masked
def wavefront_indices(n):
    diagonal, row = np.meshgrid(np.arange(2 * n - 1), np.arange(n), indexing="ij")
    column = diagonal - row
    valid = (column >= 0) & (column < n)
    return np.where(valid, row, 0), np.where(valid, column, 0), valid


# Fields (..., n, n) in wavefront layout (2n - 1, ..., n), invalid slots set to fill
def to_wavefront(field, fill=0.0):
    rows, columns, valid = wavefront_indices(field.shape[-1])
    gathered = jnp.where(valid, field[..., rows, columns], fill)
    return jnp.moveaxis(gathered, -2, 0)


# Wavefront layout (2n - 1, ..., n) back to fields (..., n, n)
def from_wavefront(layout):
    n = layout.shape[-1]
    rows, columns, valid = wavefront_indices(n)
    layout = jnp.moveaxis(layout, 0, -2)
    field = jnp.zeros(layout.shape[:-2] + (n, n), layout.dtype)
    return field.at[..., rows[valid], columns[valid]].set(layout[..., valid])


# Lower entries of the 5-point operator per node: diagonal, coupling to (i - 1, j) and to (i, j - 1) (zero on the boundary)
def operator_entries(face_x, face_y, normalization):
    n = face_x.shape[-1]
    scale = operator_scale(n, normalization)
    diagonal = (face_x[..., :-1, :] + face_x[..., 1:, :] + face_y[..., :, :-1] + face_y[..., :, 1:]) * scale
    west = (-face_x[..., :-1, :] * scale).at[..., 0, :].set(0.0)
    south = (-face_y[..., :, :-1] * scale).at[..., :, 0].set(0.0)
    return diagonal, west, south


# Field whose entry i along axis reads entry i + offset of values (zero past the edge)
def shifted(values, axis, offset):
    n = values.shape[axis]
    index = jnp.arange(n) + offset
    shape = [1] * values.ndim
    shape[axis] = n
    return jnp.roll(values, -offset, axis=axis) * ((index >= 0) & (index < n)).reshape(shape)


# IC(0) factor of the 5-point operator (sparsity of the lower triangle): diagonal d, west and south entries of L with A ~ L L^T, by one wavefront sweep
def ic0_factor(face_x, face_y, normalization):
    diagonal, west, south = operator_entries(face_x, face_y, normalization)
    entries = [to_wavefront(field, fill) for field, fill in ((diagonal, 1.0), (west, 0.0), (south, 0.0))]

    # Diagonal k from diagonal k - 1: l_w = a_w / d(i - 1, j), l_s = a_s / d(i, j - 1), d^2 = a_d - l_w^2 - l_s^2
    def step(previous, values):
        a_diagonal, a_west, a_south = values
        above = shifted(previous, -1, -1)
        lower_west = a_west / jnp.where(above > 0, above, 1.0)
        lower_south = a_south / previous
        current = jnp.sqrt(jnp.maximum(a_diagonal - lower_west ** 2 - lower_south ** 2, 1e-12 * a_diagonal))
        return current, (current, lower_west, lower_south)

    unused, factors = lax.scan(step, jnp.ones_like(entries[0][0]), tuple(entries))
    return tuple(from_wavefront(layout) for layout in factors)


# Apply (L L^T)^-1 to residuals by a forward and a backward wavefront triangular solve; factor is (diagonal, west, south) fields of L
def factor_solve(factor, residual):
    diagonal, west, south = factor
    west_above, south_right = shifted(west, -2, 1), shifted(south, -1, 1)
    layouts = [to_wavefront(field, fill) for field, fill in ((diagonal, 1.0), (west, 0.0), (south, 0.0), (west_above, 0.0), (south_right, 0.0))]
    rhs = to_wavefront(residual)

    # Forward: y(i, j) = (r - l_w y(i - 1, j) - l_s y(i, j - 1)) / d
    def forward(previous, values):
        right, d, lower_west, lower_south = values
        current = (right - lower_west * shifted(previous, -1, -1) - lower_south * previous) / d
        return current, current

    # Backward: x(i, j) = (y - l_w(i + 1, j) x(i + 1, j) - l_s(i, j + 1) x(i, j + 1)) / d
    def backward(following, values):
        right, d, upper_west, upper_south = values
        current = (right - upper_west * shifted(following, -1, 1) - upper_south * following) / d
        return current, current

    zero = jnp.zeros_like(rhs[0])
    unused, solved = lax.scan(forward, zero, (rhs, layouts[0], layouts[1], layouts[2]))
    unused, solved = lax.scan(backward, zero, (solved, layouts[0], layouts[3], layouts[4]), reverse=True)
    return from_wavefront(solved)


# Product L L^T w for a factor in (diagonal, west, south) form, without solves (NeuralIF training loss)
def factor_product(factor, field):
    diagonal, west, south = factor
    transposed = diagonal * field + shifted(west * field, -2, 1) + shifted(south * field, -1, 1)
    return diagonal * transposed + west * shifted(transposed, -2, -1) + south * shifted(transposed, -1, -1)
