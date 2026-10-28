import numpy as np
import jax.numpy as jnp


# Basis order of upstream Real_Fourier: sorted by |frequency| with cos before sin, i.e. cos0, cos1, sin1, cos2, sin2, ...
def real_fourier_modes(n_modes):
    n_cos = n_modes // 2
    modes = [("cos", k) for k in range(n_cos)] + [("sin", k) for k in range(1, n_modes - n_cos + 1)]
    return sorted(modes, key=lambda mode: mode[1])


# Synthesis matrix S of shape (n, n_modes): truncated real Fourier basis on the periodic grid x_j = j / n
def real_fourier_synthesis(n, n_modes):
    points = np.arange(n) / n
    columns = [np.cos(2 * np.pi * k * points) if kind == "cos" else np.sin(2 * np.pi * k * points) for kind, k in real_fourier_modes(n_modes)]
    return jnp.asarray(np.stack(columns, axis=1))


# Analysis matrix A = D S^T of shape (n_modes, n): rectangle-rule Fourier coefficients (paper A.1)
def real_fourier_analysis(n, n_modes):
    weights = np.full(n_modes, 2.0 / n)
    weights[0] = 1.0 / n
    return jnp.asarray(weights)[:, None] * real_fourier_synthesis(n, n_modes).T


# Apply one matrix along each spatial axis of a (features, n1, n2) field
def apply_separable(field, first, second):
    return jnp.einsum("ai,bj,fij->fab", first, second, field)
