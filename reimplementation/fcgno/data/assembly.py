import numpy as np
import scipy.sparse as sp
import jax.numpy as jnp

DEFAULT_NORMALIZATION = "physical"


# Interior nodes x_i = i h, h = 1/(n+1), as an (n, n) mesh with x along axis 0
def interior_mesh(n):
    nodes = jnp.linspace(0.0, 1.0, n + 2)[1:-1]
    return jnp.meshgrid(nodes, nodes, indexing="ij")


# Stencil factor: 1/h^2 for the physical operator, or 1/(h^2 n^2) for the A / N^2 of upstream FD_2D
def operator_scale(n, normalization=DEFAULT_NORMALIZATION):
    h = 1.0 / (n + 1)
    if normalization == "physical":
        return 1.0 / h ** 2
    if normalization == "upstream":
        return 1.0 / (h ** 2 * n ** 2)
    raise ValueError(f"unknown normalization: {normalization}")


# Coefficient on cell faces: face_x[k, j] = a((k + 1/2) h, y_j), face_y[i, k] = a(x_i, (k + 1/2) h)
def face_coefficients(coefficient, n):
    h = 1.0 / (n + 1)
    faces = (jnp.arange(n + 1) + 0.5) * h
    nodes = jnp.arange(1, n + 1) * h
    face_grid_x, node_grid_y = jnp.meshgrid(faces, nodes, indexing="ij")
    node_grid_x, face_grid_y = jnp.meshgrid(nodes, faces, indexing="ij")
    return coefficient(face_grid_x, node_grid_y), coefficient(node_grid_x, face_grid_y)


# Matrix-free 5-point operator A u with zero Dirichlet boundary; u has shape (..., n, n)
def apply_stencil(face_x, face_y, u, normalization=DEFAULT_NORMALIZATION):
    n = u.shape[-1]
    padded = jnp.pad(u, [(0, 0)] * (u.ndim - 2) + [(1, 1), (1, 1)])
    flux_x = face_x * (padded[..., 1:, 1:-1] - padded[..., :-1, 1:-1])
    flux_y = face_y * (padded[..., 1:-1, 1:] - padded[..., 1:-1, :-1])
    divergence = (flux_x[..., :-1, :] - flux_x[..., 1:, :]) + (flux_y[..., :, :-1] - flux_y[..., :, 1:])
    return divergence * operator_scale(n, normalization)


# Same operator as a SciPy CSR matrix, unknowns ordered row-major so index = i n + j
def assemble_csr(face_x, face_y, normalization=DEFAULT_NORMALIZATION):
    face_x, face_y = np.asarray(face_x), np.asarray(face_y)
    n = face_x.shape[1]
    index = np.arange(n * n).reshape(n, n)
    west, east = face_x[:-1, :], face_x[1:, :]
    south, north = face_y[:, :-1], face_y[:, 1:]
    rows = [index, index[:-1, :], index[1:, :], index[:, :-1], index[:, 1:]]
    cols = [index, index[1:, :], index[:-1, :], index[:, 1:], index[:, :-1]]
    values = [west + east + south + north, -east[:-1, :], -west[1:, :], -north[:, :-1], -south[:, 1:]]
    flatten = lambda blocks: np.concatenate([block.ravel() for block in blocks])
    matrix = sp.coo_matrix((flatten(values), (flatten(rows), flatten(cols))), shape=(n * n, n * n))
    return (matrix * operator_scale(n, normalization)).tocsr()
