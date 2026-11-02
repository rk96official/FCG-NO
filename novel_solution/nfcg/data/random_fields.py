import jax.numpy as jnp

from fcgno.data.coefficients import COEFFICIENT_SAMPLERS, sample_trig_coefficients, trig_polynomial
from fcgno.experiments.settings import TABLE_KIND

FIELD_MODES = 8
FIELD_DECAY = 1.5
REFERENCE_POINTS = 65
CONTRAST_DECADES = {"grf2": 2.0, "grf4": 4.0, "grf6": 6.0}
JUMP_DECADES = {"jump2": 2.0, "jump4": 4.0, "jump6": 6.0}


# Random smooth field (trigonometric polynomial with decaying spectrum) shifted and scaled to the range [-1/2, 1/2] on a reference grid
def normalized_field(key):
    coefficients = sample_trig_coefficients(key, FIELD_MODES)
    field = lambda x, y: trig_polynomial(coefficients, x, y, FIELD_DECAY)
    reference_x, reference_y = jnp.meshgrid(jnp.linspace(0.0, 1.0, REFERENCE_POINTS), jnp.linspace(0.0, 1.0, REFERENCE_POINTS), indexing="ij")
    values = field(reference_x, reference_y)
    low, high = values.min(), values.max()
    return lambda x, y: (field(x, y) - (low + high) / 2.0) / (high - low)


# Sampler of the smooth high-contrast coefficient a = 10^(decades g), so max a / min a = 10^decades on the reference grid; fcgno's sampler signature
def contrast_sampler(decades):
    # Coefficient function for one key; max_mode and alpha of fcgno's call are not used
    def sample(key, max_mode=None, alpha=None):
        field = normalized_field(key)
        return lambda x, y: 10.0 ** (decades * field(x, y))

    return sample


# Sampler of the binary jump coefficient a = 10^decades where the smooth field is positive and 1 elsewhere: curved interfaces not aligned with any grid
def jump_sampler(decades):
    # Coefficient function for one key; max_mode and alpha of fcgno's call are not used
    def sample(key, max_mode=None, alpha=None):
        field = normalized_field(key)
        return lambda x, y: jnp.where(field(x, y) > 0.0, 10.0 ** decades, 1.0)

    return sample


# Register the high-contrast and jump kinds in fcgno's coefficient table (data generation) and kind table (Diffusion training settings)
def register_random_fields():
    for kind, decades in CONTRAST_DECADES.items():
        COEFFICIENT_SAMPLERS.setdefault(kind, contrast_sampler(decades))
        TABLE_KIND.setdefault(kind, "diffusion")
    for kind, decades in JUMP_DECADES.items():
        COEFFICIENT_SAMPLERS.setdefault(kind, jump_sampler(decades))
        TABLE_KIND.setdefault(kind, "diffusion")
