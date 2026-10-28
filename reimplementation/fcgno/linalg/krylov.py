from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

DENOMINATOR_EPS = 1e-30


class FCGState(NamedTuple):
    iterate: jnp.ndarray
    residual: jnp.ndarray
    step: jnp.ndarray
    directions: jnp.ndarray
    operator_directions: jnp.ndarray
    direction_energies: jnp.ndarray


class CGState(NamedTuple):
    iterate: jnp.ndarray
    residual: jnp.ndarray
    direction: jnp.ndarray
    residual_energy: jnp.ndarray


# Inner product over all non-batch axes, one value per sample; fields have shape (batch, ...)
def batch_dot(a, b):
    return jnp.sum((a * b).reshape(a.shape[0], -1), axis=1)


# Multiply each sample of a batched field by its own scalar
def batch_scale(scalars, field):
    return scalars.reshape((-1,) + (1,) * (field.ndim - 1)) * field


# Number m_i of previous directions FCG orthogonalizes against at step i (Algorithm 1)
def truncation_length(step, m_max):
    return jnp.minimum(step, jnp.maximum(1, step % (m_max + 1)))


# Initial FCG state with empty ring buffers holding the last m_max directions p_k and s_k = A p_k
def fcg_init(apply_operator, rhs, initial_guess, m_max):
    buffer = jnp.zeros((m_max,) + rhs.shape, rhs.dtype)
    energies = jnp.zeros((m_max, rhs.shape[0]), rhs.dtype)
    residual = rhs - apply_operator(initial_guess)
    return FCGState(initial_guess, residual, jnp.array(0), buffer, buffer, energies)


# One FCG iteration of Algorithm 1; returns the new state and the preconditioned residual w_i
def fcg_step(state, apply_operator, preconditioner, m_max):
    preconditioned = preconditioner(state.residual)
    slots = jnp.arange(m_max)
    active = ((state.step - 1 - slots) % m_max) < truncation_length(state.step, m_max)
    projections = jnp.sum((state.operator_directions * preconditioned[None]).reshape(m_max, preconditioned.shape[0], -1), axis=2)
    coefficients = jnp.where(active[:, None], projections / (state.direction_energies + DENOMINATOR_EPS), 0.0)
    direction = preconditioned - jnp.einsum("kb,kb...->b...", coefficients, state.directions)
    operator_direction = apply_operator(direction)
    energy = batch_dot(direction, operator_direction)
    step_size = batch_dot(direction, state.residual) / (energy + DENOMINATOR_EPS)
    slot = state.step % m_max
    new_state = FCGState(
        state.iterate + batch_scale(step_size, direction),
        state.residual - batch_scale(step_size, operator_direction),
        state.step + 1,
        state.directions.at[slot].set(direction),
        state.operator_directions.at[slot].set(operator_direction),
        state.direction_energies.at[slot].set(energy),
    )
    return new_state, preconditioned


# Notay ratio ||w - e||_A / ||e||_A of a preconditioned residual w against the exact error e
def notay_ratio(apply_operator, preconditioned, error):
    difference = preconditioned - error
    return jnp.sqrt(batch_dot(difference, apply_operator(difference)) / batch_dot(error, apply_operator(error)))


# Run n_iter FCG steps; returns final iterate, residual norms and optional histories and Notay ratios
def fcg(apply_operator, preconditioner, rhs, initial_guess, n_iter, m_max=20, solution=None, keep_history=False):
    # Record per-step outputs alongside each FCG update
    def body(state, unused):
        outputs = {"residual_norm": jnp.sqrt(batch_dot(state.residual, state.residual))}
        if keep_history:
            outputs["residual"], outputs["iterate"] = state.residual, state.iterate
        new_state, preconditioned = fcg_step(state, apply_operator, preconditioner, m_max)
        if solution is not None:
            outputs["notay"] = notay_ratio(apply_operator, preconditioned, solution - state.iterate)
        return new_state, outputs

    state = fcg_init(apply_operator, rhs, initial_guess, m_max)
    state, outputs = lax.scan(body, state, None, length=n_iter)
    return finish_history(state.iterate, state.residual, outputs, keep_history)


# Initial state of standard (unpreconditioned) CG
def cg_init(apply_operator, rhs, initial_guess):
    residual = rhs - apply_operator(initial_guess)
    return CGState(initial_guess, residual, residual, batch_dot(residual, residual))


# One standard CG iteration
def cg_step(state, apply_operator):
    operator_direction = apply_operator(state.direction)
    step_size = state.residual_energy / (batch_dot(state.direction, operator_direction) + DENOMINATOR_EPS)
    iterate = state.iterate + batch_scale(step_size, state.direction)
    residual = state.residual - batch_scale(step_size, operator_direction)
    residual_energy = batch_dot(residual, residual)
    beta = residual_energy / (state.residual_energy + DENOMINATOR_EPS)
    return CGState(iterate, residual, residual + batch_scale(beta, state.direction), residual_energy)


# Run n_iter standard CG steps; same outputs as fcg
def cg(apply_operator, rhs, initial_guess, n_iter, keep_history=False):
    # Record per-step outputs alongside each CG update
    def body(state, unused):
        outputs = {"residual_norm": jnp.sqrt(state.residual_energy)}
        if keep_history:
            outputs["residual"], outputs["iterate"] = state.residual, state.iterate
        return cg_step(state, apply_operator), outputs

    state, outputs = lax.scan(body, cg_init(apply_operator, rhs, initial_guess), None, length=n_iter)
    return finish_history(state.iterate, state.residual, outputs, keep_history)


# Append the final state to the scanned histories and move the iteration axis after the batch axis
def finish_history(iterate, residual, outputs, keep_history):
    result = {"iterate": iterate}
    final_norm = jnp.sqrt(batch_dot(residual, residual))
    result["residual_norm"] = jnp.concatenate([outputs["residual_norm"], final_norm[None]]).T
    if keep_history:
        result["residual_history"] = jnp.moveaxis(jnp.concatenate([outputs["residual"], residual[None]]), 0, 1)
        result["iterate_history"] = jnp.moveaxis(jnp.concatenate([outputs["iterate"], iterate[None]]), 0, 1)
    if "notay" in outputs:
        result["notay"] = outputs["notay"].T
    return result
