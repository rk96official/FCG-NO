import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

from fcgno.data.assembly import apply_stencil
from fcgno.linalg.krylov import DENOMINATOR_EPS, batch_dot, batch_scale
from nfcg.linalg.multigrid import level_sizes
from nfcg.linalg.smoothers import chebyshev_sweeps, make_smoother, stencil_diagonal
from nfcg.linalg.two_level import interpolation_matrix, prolongate, restrict


# Linear interpolation matrix (targets, sources) between increasing 1D positions, clamped at both ends
def sampling_matrix(sources, targets):
    index = np.clip(np.searchsorted(sources, targets) - 1, 0, len(sources) - 2)
    weight = np.clip((targets - sources[index]) / (sources[index + 1] - sources[index]), 0.0, 1.0)
    matrix = np.zeros((len(targets), len(sources)))
    rows = np.arange(len(targets))
    matrix[rows, index] = 1.0 - weight
    matrix[rows, index + 1] += weight
    return matrix


# Face and node positions of the interior grid with n unknowns per axis
def grid_positions(n):
    return (np.arange(n + 1) + 0.5) / (n + 1), np.arange(1, n + 1) / (n + 1)


# Rediscretized face coefficients on a coarser grid: log a sampled bilinearly at the coarse faces (geometric, keeps positivity and contrast); no setup cost
def coarse_faces(face_x, face_y, coarse_n):
    fine_faces, fine_nodes = grid_positions(face_x.shape[-1])
    coarse_face_positions, coarse_node_positions = grid_positions(coarse_n)
    along_faces = jnp.asarray(sampling_matrix(fine_faces, coarse_face_positions), face_x.dtype)
    along_nodes = jnp.asarray(sampling_matrix(fine_nodes, coarse_node_positions), face_x.dtype)
    sample = lambda field, rows, columns: jnp.exp(jnp.einsum("ai,...ij,bj->...ab", rows, jnp.log(field), columns))
    return sample(face_x, along_faces, along_nodes), sample(face_y, along_nodes, along_faces)


# Levels from the fine grid down to the one above the bottom grid: per level its operator, smoother and interpolation to the next level
def rediscretized_levels(face_x, face_y, normalization, sizes, smoother, sweeps):
    levels = []
    for size, below in zip(sizes[:-1], sizes[1:]):
        level_x, level_y = (face_x, face_y) if size == sizes[0] else coarse_faces(face_x, face_y, size)
        operator = lambda u, level_x=level_x, level_y=level_y: apply_stencil(level_x, level_y, u, normalization)
        smooth = make_smoother(smoother, operator, stencil_diagonal(level_x, level_y, normalization), sweeps)
        levels.append({"operator": operator, "smooth": smooth, "interpolation": interpolation_matrix(size, below).astype(face_x.dtype)})
    return levels


# Add the energy-optimal multiple of direction to correction for A x = residual (exact line search, so the direction's scale is irrelevant)
def line_searched(operator, correction, direction, residual):
    remainder = residual - operator(correction)
    step = batch_dot(direction, remainder) / (batch_dot(direction, operator(direction)) + DENOMINATOR_EPS)
    return correction + batch_scale(step, direction)


# One application from level index down: pre-smooth, optional learned correction on this grid, line-searched coarse direction from the levels below, post-smooth
def multilevel_apply(levels, index, bottom, learned, residual):
    if index == len(levels):
        return bottom(residual)
    level = levels[index]
    operator = level["operator"]
    correction = level["smooth"](residual)
    if learned[index] is not None:
        correction = line_searched(operator, correction, learned[index](residual - operator(correction)), residual)
    coarse = multilevel_apply(levels, index + 1, bottom, learned, restrict(residual - operator(correction), level["interpolation"]))
    correction = line_searched(operator, correction, prolongate(coarse, level["interpolation"]), residual)
    return correction + level["smooth"](residual - operator(correction))


# Fine fields restricted to every level and the bottom grid (index 0 is the fine grid itself); None stays None
def restricted_per_level(levels, fields):
    restricted = [fields]
    for level in levels:
        restricted.append(None if fields is None else restrict(restricted[-1], level["interpolation"]))
    return restricted


# Multi-level preconditioner (residual, updates, changes) -> correction around one history-aware network: rediscretized levels down to options["coarse_grid"],
# the network as the bottom solve and, on every level no larger than options["network_max"], also as a line-searched correction (recursive form, shared weights)
def multilevel_history_preconditioner(network_factory, face_x, face_y, normalization, log_coefficient_field, options):
    sizes = level_sizes(face_x.shape[-1], None, options["coarse_grid"])
    levels = rediscretized_levels(face_x, face_y, normalization, sizes, options["smoother"], options.get("sweeps", 2))
    networks = [network_factory(field) for field in restricted_per_level(levels, log_coefficient_field)]
    network_max = options.get("network_max", 0)

    # Network at each level reads that level's restriction of the history stacks
    def apply(residual, updates, changes):
        histories = list(zip(restricted_per_level(levels, updates), restricted_per_level(levels, changes)))
        at_level = lambda index: (lambda level_residual: networks[index](level_residual, *histories[index]))
        learned = [at_level(index) if sizes[index] <= network_max else None for index in range(len(levels))]
        return multilevel_apply(levels, 0, at_level(len(levels)), learned, residual)

    return apply


# Dense Cholesky solve on the bottom grid, factorized once per system
def dense_bottom(face_x, face_y, normalization):
    n = face_x.shape[-1]
    basis = jnp.eye(n * n, dtype=face_x.dtype).reshape(n * n, n, n)
    matrices = jax.vmap(lambda single_x, single_y: apply_stencil(single_x, single_y, basis, normalization).reshape(n * n, n * n))(face_x, face_y)
    factors = jax.vmap(lambda matrix: cho_factor(matrix, lower=True)[0])(matrices)
    solve = lambda lower, rhs: cho_solve((lower, True), rhs.reshape(-1)).reshape(n, n)
    return lambda residual: jax.vmap(solve)(factors, residual)


# Classical multi-level preconditioner r -> correction on the same rediscretized levels; bottom solve dense (Cholesky) or Chebyshev sweeps
def multilevel_preconditioner(face_x, face_y, normalization, options):
    sizes = level_sizes(face_x.shape[-1], None, options["coarse_grid"])
    levels = rediscretized_levels(face_x, face_y, normalization, sizes, options["smoother"], options.get("sweeps", 2))
    bottom_x, bottom_y = coarse_faces(face_x, face_y, sizes[-1])
    if options["bottom"] == "dense":
        bottom = dense_bottom(bottom_x, bottom_y, normalization)
    else:
        operator = lambda u: apply_stencil(bottom_x, bottom_y, u, normalization)
        bottom = lambda residual: chebyshev_sweeps(operator, stencil_diagonal(bottom_x, bottom_y, normalization), residual, options["bottom_sweeps"])
    return lambda residual: multilevel_apply(levels, 0, bottom, [None] * len(levels), residual)
