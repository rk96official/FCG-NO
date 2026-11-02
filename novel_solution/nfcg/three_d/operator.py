import numpy as np
import jax.numpy as jnp
import scipy.sparse as sp

from fcgno.data.assembly import operator_scale
from nfcg.linalg.multilevel import grid_positions, sampling_matrix


# 7-point operator A u with zero Dirichlet boundary on (..., n, n, n) fields; faces = (face_x (..., n+1, n, n), face_y (..., n, n+1, n), face_z (..., n, n, n+1))
def apply_stencil_3d(faces, u, normalization):
    n = u.shape[-1]
    padded = jnp.pad(u, [(0, 0)] * (u.ndim - 3) + [(1, 1)] * 3)
    face_x, face_y, face_z = faces
    flux_x = face_x * (padded[..., 1:, 1:-1, 1:-1] - padded[..., :-1, 1:-1, 1:-1])
    flux_y = face_y * (padded[..., 1:-1, 1:, 1:-1] - padded[..., 1:-1, :-1, 1:-1])
    flux_z = face_z * (padded[..., 1:-1, 1:-1, 1:] - padded[..., 1:-1, 1:-1, :-1])
    divergence = (flux_x[..., :-1, :, :] - flux_x[..., 1:, :, :]) + (flux_y[..., :, :-1, :] - flux_y[..., :, 1:, :]) + (flux_z[..., :-1] - flux_z[..., 1:])
    return divergence * operator_scale(n, normalization)


# Sum of the six face coefficients around each node, shape (..., n, n, n)
def face_sum(faces):
    face_x, face_y, face_z = faces
    return face_x[..., :-1, :, :] + face_x[..., 1:, :, :] + face_y[..., :, :-1, :] + face_y[..., :, 1:, :] + face_z[..., :-1] + face_z[..., 1:]


# Diagonal of the 7-point operator, shape (..., n, n, n)
def diagonal_3d(faces, normalization):
    return face_sum(faces) * operator_scale(faces[0].shape[-1], normalization)


# Same operator as a SciPy CSR matrix (row-major unknowns), for the CPU reference solves
def assemble_csr_3d(faces, normalization):
    faces = [np.asarray(face) for face in faces]
    n = faces[0].shape[-1]
    index = np.arange(n ** 3).reshape(n, n, n)
    rows, columns, values = [index.ravel()], [index.ravel()], [face_sum(faces).ravel()]
    for axis, face in enumerate(faces):
        inner = np.take(face, np.arange(1, n), axis=axis).ravel()
        lower, upper = np.take(index, np.arange(n - 1), axis=axis).ravel(), np.take(index, np.arange(1, n), axis=axis).ravel()
        rows += [lower, upper]
        columns += [upper, lower]
        values += [-inner, -inner]
    matrix = sp.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=(n ** 3, n ** 3))
    return (matrix * operator_scale(n, normalization)).tocsr()


# Prolongate coarse fields (..., m, m, m) to the fine grid by trilinear interpolation
def prolongate_3d(fields, interpolation):
    return jnp.einsum("ia,jb,kc,...abc->...ijk", interpolation, interpolation, interpolation, fields)


# Restrict fine fields with the transpose of interpolation, scaled so constants are preserved
def restrict_3d(fields, interpolation):
    scale = ((interpolation.shape[1] + 1) / (interpolation.shape[0] + 1)) ** 3
    return scale * jnp.einsum("ia,jb,kc,...ijk->...abc", interpolation, interpolation, interpolation, fields)


# Rediscretized faces on a coarser grid: log a sampled trilinearly at the coarse faces (no setup cost)
def coarse_faces_3d(faces, coarse_n):
    fine_faces, fine_nodes = grid_positions(faces[0].shape[-1])
    coarse_face_positions, coarse_node_positions = grid_positions(coarse_n)
    along_faces = jnp.asarray(sampling_matrix(fine_faces, coarse_face_positions), faces[0].dtype)
    along_nodes = jnp.asarray(sampling_matrix(fine_nodes, coarse_node_positions), faces[0].dtype)
    sampled = []
    for axis, face in enumerate(faces):
        matrices = [along_faces if index == axis else along_nodes for index in range(3)]
        sampled.append(jnp.exp(jnp.einsum("ai,bj,ck,...ijk->...abc", *matrices, jnp.log(face))))
    return tuple(sampled)
