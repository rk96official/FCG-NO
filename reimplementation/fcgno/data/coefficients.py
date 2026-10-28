import jax
import jax.numpy as jnp

DEFAULT_MAX_MODE = 4
DEFAULT_ALPHA = 2.0
DIFFUSION_SHIFT = 10.0
JUMP_LOCATION = 0.5
JUMP_SCALE = 0.1
A2_AMPLITUDE = 0.1


# Draw complex Gaussian coefficients c_mn, m, n = 0..max_mode, of a random trigonometric polynomial
def sample_trig_coefficients(key, max_mode=DEFAULT_MAX_MODE):
    return jax.random.normal(key, (max_mode + 1, max_mode + 1), dtype=jnp.complex128)


# Evaluate Re(sum_mn c_mn exp(2 pi i (m x + n y)) / (1 + m + n)^alpha) pointwise
def trig_polynomial(coefficients, x, y, alpha=DEFAULT_ALPHA):
    x, y = jnp.asarray(x), jnp.asarray(y)
    modes_x = jnp.arange(coefficients.shape[0])
    modes_y = jnp.arange(coefficients.shape[1])
    weights = coefficients / (1.0 + modes_x[:, None] + modes_y[None, :]) ** alpha
    phase_x = jnp.exp(2j * jnp.pi * x[..., None] * modes_x)
    phase_y = jnp.exp(2j * jnp.pi * y[..., None] * modes_y)
    return jnp.real(jnp.einsum("mn,...m,...n->...", weights, phase_x, phase_y))


# Random right-hand side f ~ P(max_mode, max_mode, alpha)
def sample_rhs(key, max_mode=DEFAULT_MAX_MODE, alpha=DEFAULT_ALPHA):
    coefficients = sample_trig_coefficients(key, max_mode)
    return lambda x, y: trig_polynomial(coefficients, x, y, alpha)


# Poisson dataset coefficient a(x) = 1 (paper eq. 13)
def poisson_coefficient(key, max_mode=DEFAULT_MAX_MODE, alpha=DEFAULT_ALPHA):
    return lambda x, y: jnp.ones(jnp.broadcast_shapes(jnp.shape(x), jnp.shape(y)))


# Diffusion dataset coefficient a(x) = P(max_mode, max_mode, alpha) + 10 (paper eq. 14)
def diffusion_coefficient(key, max_mode=DEFAULT_MAX_MODE, alpha=DEFAULT_ALPHA):
    coefficients = sample_trig_coefficients(key, max_mode)
    return lambda x, y: trig_polynomial(coefficients, x, y, alpha) + DIFFUSION_SHIFT


# Non-smooth a1(x) = 1 for x1 >= 0.5 and eps ~ 0.1 U(0, 1) otherwise (paper eq. 17)
def jump_coefficient_a1(key, max_mode=DEFAULT_MAX_MODE, alpha=DEFAULT_ALPHA):
    epsilon = JUMP_SCALE * jax.random.uniform(key, dtype=jnp.float64)
    return lambda x, y: jnp.where(jnp.asarray(x) >= JUMP_LOCATION, 1.0, epsilon) * jnp.ones_like(y)


# Non-smooth a2(x) = 0.1 P(max_mode, max_mode, alpha) + 1 for x1 >= 0.5 and eps otherwise (paper eq. 18)
def jump_coefficient_a2(key, max_mode=DEFAULT_MAX_MODE, alpha=DEFAULT_ALPHA):
    epsilon_key, polynomial_key = jax.random.split(key)
    epsilon = JUMP_SCALE * jax.random.uniform(epsilon_key, dtype=jnp.float64)
    coefficients = sample_trig_coefficients(polynomial_key, max_mode)
    smooth_part = lambda x, y: A2_AMPLITUDE * trig_polynomial(coefficients, x, y, alpha) + 1.0
    return lambda x, y: jnp.where(jnp.asarray(x) >= JUMP_LOCATION, smooth_part(x, y), epsilon)


COEFFICIENT_SAMPLERS = {
    "poisson": poisson_coefficient,
    "diffusion": diffusion_coefficient,
    "a1": jump_coefficient_a1,
    "a2": jump_coefficient_a2,
}
