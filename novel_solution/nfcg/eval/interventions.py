import jax.numpy as jnp

from fcgno.linalg.krylov import DENOMINATOR_EPS, batch_dot, batch_scale
from nfcg.linalg.traced_fcg import angle_sine_squared


# Mask of steps whose proposed w has a squared A-angle sine to the error above threshold
def bad_angle(preconditioned, residual, error, apply_operator, threshold):
    error_energy = batch_dot(error, apply_operator(error))
    return angle_sine_squared(preconditioned, apply_operator(preconditioned), residual, error_energy) > threshold


# Causal test with a cheap fallback: at bad-angle steps the residual replaces the network output
def residual_swap(threshold):
    # Replace w by r where the angle is bad
    def intervene(preconditioned, residual, error, apply_operator):
        bad = bad_angle(preconditioned, residual, error, apply_operator, threshold)
        return jnp.where(bad[:, None, None], residual, preconditioned), bad

    return intervene


# Causal test with an oracle: at bad-angle steps the part of w that is A-orthogonal to e is shrunk until sin^2 equals threshold
def angle_cap(threshold):
    # Keep w's off-error direction but cap its weight
    def intervene(preconditioned, residual, error, apply_operator):
        operator_error = apply_operator(error)
        error_energy = batch_dot(error, operator_error)
        along = batch_dot(preconditioned, operator_error) / (error_energy + DENOMINATOR_EPS)
        orthogonal = preconditioned - batch_scale(along, error)
        orthogonal_energy = batch_dot(orthogonal, apply_operator(orthogonal))
        weight = jnp.sqrt(threshold * error_energy / ((1.0 - threshold) * orthogonal_energy + DENOMINATOR_EPS))
        capped = error + batch_scale(weight, orthogonal)
        bad = bad_angle(preconditioned, residual, error, apply_operator, threshold)
        return jnp.where(bad[:, None, None], capped, preconditioned), bad

    return intervene


INTERVENTIONS = {"residual_swap": residual_swap, "angle_cap": angle_cap}
