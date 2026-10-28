import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import scipy.sparse.linalg as spla
import pytest

from fcgno.data.assembly import interior_mesh, operator_scale, face_coefficients, apply_stencil, assemble_csr
from fcgno.data.coefficients import COEFFICIENT_SAMPLERS, sample_rhs

UPSTREAM_ROOT = Path(__file__).resolve().parents[2]
KINDS = sorted(COEFFICIENT_SAMPLERS)


# Face coefficients for one random sample of the given kind
def sample_faces(kind, n, seed=0):
    coefficient = COEFFICIENT_SAMPLERS[kind](jax.random.PRNGKey(seed))
    return face_coefficients(coefficient, n), coefficient


# Import upstream FD_2D read-only, skipping if it no longer imports with the installed JAX
def load_upstream_fd():
    sys.path.insert(0, str(UPSTREAM_ROOT))
    try:
        from solvers import FD_2D
    except Exception as error:
        pytest.skip(f"upstream solvers.py cannot be imported: {error}")
    finally:
        sys.path.remove(str(UPSTREAM_ROOT))
    return FD_2D


# Matrix-free stencil and assembled CSR give the same product
@pytest.mark.parametrize("kind", KINDS)
def test_stencil_matches_csr(kind):
    n = 12
    (face_x, face_y), _ = sample_faces(kind, n)
    u = jax.random.normal(jax.random.PRNGKey(1), (n, n))
    from_stencil = np.asarray(apply_stencil(face_x, face_y, u)).ravel()
    from_csr = assemble_csr(face_x, face_y) @ np.asarray(u).ravel()
    np.testing.assert_allclose(from_stencil, from_csr, rtol=1e-12, atol=1e-12)


# Stencil works on a batch of fields with per-sample coefficients
def test_stencil_batched():
    n = 8
    faces = [sample_faces("diffusion", n, seed)[0] for seed in range(3)]
    face_x = jnp.stack([pair[0] for pair in faces])
    face_y = jnp.stack([pair[1] for pair in faces])
    u = jax.random.normal(jax.random.PRNGKey(2), (3, n, n))
    batched = apply_stencil(face_x, face_y, u)
    for sample in range(3):
        np.testing.assert_allclose(batched[sample], apply_stencil(face_x[sample], face_y[sample], u[sample]), rtol=1e-13)


# Assembled matrix is symmetric positive definite for every coefficient kind
@pytest.mark.parametrize("kind", KINDS)
def test_operator_is_spd(kind):
    (face_x, face_y), _ = sample_faces(kind, 8)
    dense = assemble_csr(face_x, face_y).toarray()
    np.testing.assert_allclose(dense, dense.T, rtol=0, atol=1e-13)
    assert np.linalg.eigvalsh(dense).min() > 0


# Matrix and right-hand side match upstream FD_2D entry by entry
@pytest.mark.parametrize("kind", ["poisson", "diffusion"])
def test_matches_upstream_fd(kind):
    fd_2d = load_upstream_fd()
    n = 8
    (face_x, face_y), coefficient = sample_faces(kind, n)
    rhs = sample_rhs(jax.random.PRNGKey(3))
    upstream_rhs, upstream_matrix = fd_2d(n, [coefficient, rhs])
    mesh_x, mesh_y = interior_mesh(n)
    ours = assemble_csr(face_x, face_y, normalization="upstream").toarray()
    np.testing.assert_allclose(ours, np.asarray(upstream_matrix.todense()), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(rhs(mesh_x, mesh_y)).ravel(), np.asarray(upstream_rhs), rtol=1e-12, atol=1e-12)


# Poisson discretization error decreases at second order for u = sin(pi x) sin(pi y)
def test_poisson_second_order():
    errors = []
    for n in (16, 32):
        (face_x, face_y), _ = sample_faces("poisson", n)
        mesh_x, mesh_y = interior_mesh(n)
        exact = np.asarray(jnp.sin(jnp.pi * mesh_x) * jnp.sin(jnp.pi * mesh_y))
        approximate = spla.spsolve(assemble_csr(face_x, face_y).tocsc(), 2 * np.pi ** 2 * exact.ravel())
        errors.append(np.abs(approximate - exact.ravel()).max())
    ratio = errors[0] / errors[1]
    assert 3.5 < ratio < 4.5, (errors, ratio)


# Operator scale is 1/h^2 for the physical operator and 1/(h^2 N^2) for upstream FD_2D
def test_operator_scale():
    n = 32
    h = 1.0 / (n + 1)
    assert operator_scale(n) == pytest.approx(1.0 / h ** 2, rel=1e-14)
    assert operator_scale(n, "upstream") == pytest.approx(1.0 / (h ** 2 * n ** 2), rel=1e-14)
    with pytest.raises(ValueError):
        operator_scale(n, "unknown")


# Upstream normalization only rescales the operator by 1/N^2
def test_normalizations_differ_by_constant():
    n = 8
    (face_x, face_y), _ = sample_faces("diffusion", n)
    physical = assemble_csr(face_x, face_y).toarray()
    upstream = assemble_csr(face_x, face_y, normalization="upstream").toarray()
    np.testing.assert_allclose(upstream * n ** 2, physical, rtol=1e-13)
