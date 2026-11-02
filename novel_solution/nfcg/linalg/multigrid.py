import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

from nfcg.linalg.smoothers import chebyshev_sweeps
from nfcg.linalg.two_level import interpolation_matrix, prolongate, restrict

STENCIL_RADIUS = 3
COARSEST_N = 16
POWER_STEPS = 15
UPPER_MARGIN = 1.1
INTERVAL_RATIO = 11.0


# Grid sizes of a hierarchy: the fine grid, then first_coarse (when given), then halving down to at most COARSEST_N
def level_sizes(fine_n, first_coarse=None, coarsest=COARSEST_N):
    sizes = [fine_n] if first_coarse is None else [fine_n, first_coarse]
    while sizes[-1] > coarsest:
        sizes.append(sizes[-1] // 2)
    return sizes


# Apply a variable stencil operator: (A u)[i, j] = sum_pq stencil[..., p, q, i, j] u[i + p - r, j + q - r], zero outside the grid
def apply_stencil_operator(stencil, u):
    width, n = stencil.shape[-3], u.shape[-1]
    radius = width // 2
    padded = jnp.pad(u, [(0, 0)] * (u.ndim - 2) + [(radius, radius), (radius, radius)])
    return sum(stencil[..., p, q, :, :] * padded[..., p:p + n, q:q + n] for p in range(width) for q in range(width))


# Stencil of a banded operator (bandwidth <= radius per axis) from (2r + 1)^2 probing products with coloured unit vectors
def probe_stencil(apply_operator, batch, n, dtype, radius=STENCIL_RADIUS):
    width = 2 * radius + 1
    rows = np.arange(n)
    stencil = jnp.zeros((batch, width, width, n, n), dtype)
    row_index, column_index = np.meshgrid(rows, rows, indexing="ij")
    for a in range(width):
        for b in range(width):
            colour = jnp.asarray(np.outer(rows % width == a, rows % width == b).astype(np.float64), dtype)
            image = apply_operator(jnp.broadcast_to(colour, (batch, n, n)))
            offset_p, offset_q = (a - row_index + radius) % width, (b - column_index + radius) % width
            stencil = stencil.at[:, offset_p, offset_q, row_index, column_index].set(image)
    return stencil


# Galerkin coarse stencil R A P of a fine operator, with the bilinear transfers between non-nested grids
def galerkin_stencil(apply_fine, batch, coarse_n, interpolation, dtype):
    apply_coarse = lambda v: restrict(apply_fine(prolongate(v, interpolation)), interpolation)
    return probe_stencil(apply_coarse, batch, coarse_n, dtype)


# Per-system upper bound on the spectrum of D^-1 A by power iteration, with a safety margin; shape (batch, 1, 1)
def spectral_upper(apply_operator, diagonal, key=0):
    vector = jax.random.normal(jax.random.PRNGKey(key), diagonal.shape, diagonal.dtype)
    norm = lambda v: jnp.sqrt(jnp.sum(v * v, axis=(-2, -1), keepdims=True))
    for unused in range(POWER_STEPS):
        vector = apply_operator(vector) / diagonal
        vector = vector / norm(vector)
    return UPPER_MARGIN * norm(apply_operator(vector) / diagonal)


# Dense Cholesky factors of a stencil operator on a small grid, one per system
def dense_factor(stencil):
    n = stencil.shape[-1]
    basis = jnp.eye(n * n, dtype=stencil.dtype).reshape(n * n, n, n)
    columns = jax.vmap(lambda single: apply_stencil_operator(single[None], basis).reshape(n * n, n * n).T)(stencil)
    return jax.vmap(lambda matrix: cho_factor(matrix, lower=True)[0])(columns)


# Exact solve with the dense Cholesky factors of each system
def dense_solve(factor, residual):
    n = residual.shape[-1]
    solve = lambda lower, rhs: cho_solve((lower, True), rhs.reshape(-1)).reshape(n, n)
    return jax.vmap(solve)(factor, residual)


# Bilinear interpolation matrices between consecutive levels; passed into jitted setups as arguments so XLA does not constant-fold the probes
def hierarchy_interpolations(sizes, dtype):
    return [jnp.asarray(interpolation_matrix(above, size), dtype) for above, size in zip(sizes[:-1], sizes[1:])]


# Galerkin hierarchy below a fine operator: per coarse level its stencil, diagonal, Chebyshev upper bound and interpolation from the level above;
# the coarsest level also holds dense Cholesky factors when factor is set. Built once per batch of systems (setup)
def build_hierarchy(apply_fine, batch, sizes, dtype, factor=True, interpolations=None):
    levels, apply_above = [], apply_fine
    interpolations = interpolations or hierarchy_interpolations(sizes, dtype)
    for size, interpolation in zip(sizes[1:], interpolations):
        stencil = galerkin_stencil(apply_above, batch, size, interpolation, dtype)
        diagonal = stencil[:, STENCIL_RADIUS, STENCIL_RADIUS]
        apply_level = lambda v, stencil=stencil: apply_stencil_operator(stencil, v)
        levels.append({"stencil": stencil, "diagonal": diagonal, "upper": spectral_upper(apply_level, diagonal), "interpolation": interpolation})
        apply_above = apply_level
    if factor:
        levels[-1]["factor"] = dense_factor(levels[-1]["stencil"])
    return levels


# Same hierarchy with every array cast to dtype (low-precision coarse solves)
def cast_hierarchy(levels, dtype):
    return [{name: value.astype(dtype) for name, value in level.items()} for level in levels]


# Chebyshev smoothing on one level with its estimated interval [upper / 11, upper] (the fine interval 0.2-2.2 has the same ratio)
def level_smooth(apply_operator, diagonal, upper, residual, sweeps):
    return chebyshev_sweeps(apply_operator, diagonal, residual, sweeps, upper / INTERVAL_RATIO, upper)


# One V-cycle for A x = r on level index of a hierarchy (Chebyshev pre/post smoothing, Galerkin coarse correction, exact solve at the bottom)
def vcycle(levels, index, residual, sweeps):
    level = levels[index]
    if index == len(levels) - 1:
        return dense_solve(level["factor"], residual)
    apply_level = lambda v: apply_stencil_operator(level["stencil"], v)
    return smoothed_correction(apply_level, level["diagonal"], level["upper"], levels, index, residual, sweeps)


# Pre-smooth, coarse correction by a V-cycle on the next level (in that level's precision), post-smooth, for any operator above levels[index + 1]
def smoothed_correction(apply_operator, diagonal, upper, levels, index, residual, sweeps):
    below = levels[index + 1]
    correction = level_smooth(apply_operator, diagonal, upper, residual, sweeps)
    remainder = restrict(residual - apply_operator(correction), below["interpolation"]).astype(below["stencil"].dtype)
    correction = correction + prolongate(vcycle(levels, index + 1, remainder, sweeps), below["interpolation"])
    return correction + level_smooth(apply_operator, diagonal, upper, residual - apply_operator(correction), sweeps)


# Geometric multigrid V-cycle preconditioner on the fine grid: the hierarchy holds the coarse levels (levels[0] is the first coarse level)
def vcycle_preconditioner(apply_fine, fine_diagonal, fine_upper, levels, sweeps):
    stacked = [None] + levels
    return lambda residual: smoothed_correction(apply_fine, fine_diagonal, fine_upper, stacked, 0, residual, sweeps)


# Classical coarse solver for the two-level slot of the network, on levels[0]: zero (smoothing only), Chebyshev sweeps, one V-cycle, or the exact solve.
# dtype casts the coarse work (and the residual) to lower precision, as TF32 does for the network
def coarse_solver(kind, levels, sweeps=2, dtype=None):
    levels = levels if dtype is None else cast_hierarchy(levels, dtype)
    first = levels[0]
    apply_first = lambda v: apply_stencil_operator(first["stencil"], v)
    solvers = {
        "zero": lambda residual: jnp.zeros_like(residual),
        "chebyshev": lambda residual: level_smooth(apply_first, first["diagonal"], first["upper"], residual, sweeps),
        "vcycle": lambda residual: vcycle(levels, 0, residual, sweeps),
        "exact": lambda residual: dense_solve(first["factor"], residual),
    }
    solve = solvers[kind]
    return solve if dtype is None else lambda residual: solve(residual.astype(dtype)).astype(residual.dtype)
