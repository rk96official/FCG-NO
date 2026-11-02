import numpy as np
import jax.numpy as jnp

from fcgno.models.spectral import real_fourier_synthesis


# Orthonormal basis (n, n_modes) of the real Fourier modes an SNO with n_modes keeps on an n-point grid
def retained_basis(n, n_modes=20):
    orthonormal, unused = np.linalg.qr(np.asarray(real_fourier_synthesis(n, n_modes)))
    return jnp.asarray(orthonormal)


# Fraction of each field's squared Euclidean norm outside the separable span of the retained modes; fields have shape (batch, n, n)
def energy_outside_modes(fields, basis):
    coefficients = jnp.einsum("ia,bij,jc->bac", basis, fields, basis)
    inside = jnp.sum(coefficients ** 2, axis=(1, 2))
    total = jnp.sum(fields ** 2, axis=(1, 2))
    return 1.0 - inside / total


# Mask of steps taken before the relative residual first reaches floor; later steps are dominated by round-off
def active_steps(relative_residual, floor=1e-10):
    reached = np.asarray(relative_residual)[:, :-1] <= floor
    return ~np.logical_or.accumulate(reached, axis=1)
