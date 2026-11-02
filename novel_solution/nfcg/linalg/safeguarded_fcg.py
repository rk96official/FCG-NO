import jax.numpy as jnp
from jax import lax

from fcgno.linalg.krylov import DENOMINATOR_EPS, FCGState, batch_dot, batch_scale, fcg_init, truncation_length


# A-orthogonalize a field against the active stored directions (Algorithm 1, line 5)
def orthogonalize(state, field, m_max):
    slots = jnp.arange(m_max)
    active = ((state.step - 1 - slots) % m_max) < truncation_length(state.step, m_max)
    projections = jnp.sum((state.operator_directions * field[None]).reshape(m_max, field.shape[0], -1), axis=2)
    coefficients = jnp.where(active[:, None], projections / (state.direction_energies + DENOMINATOR_EPS), 0.0)
    return field - jnp.einsum("kb,kb...->b...", coefficients, state.directions)


# One safeguarded FCG step (midterm eq. 15): network and fallback directions, A-orthogonalized, combined by exact energy minimization.
# The network runs only when step % every == 0; the combined direction is stored, so memory equals plain FCG
def safeguarded_step(state, apply_operator, network, fallback, m_max, every):
    proposed = lax.cond(state.step % every == 0, network, jnp.zeros_like, state.residual)
    return safeguarded_update(state, apply_operator, proposed, fallback, m_max)


# Safeguarded update from a given network output w_i (zero when the network was skipped); returns the new state and the fallback's share of the decrease
def safeguarded_update(state, apply_operator, proposed, fallback, m_max):
    first = orthogonalize(state, proposed, m_max)
    first_image = apply_operator(first)
    first_energy = batch_dot(first, first_image)
    second = orthogonalize(state, fallback(state.residual), m_max)
    second = second - batch_scale(batch_dot(second, first_image) / (first_energy + DENOMINATOR_EPS), first)
    second_image = apply_operator(second)
    second_energy = batch_dot(second, second_image)
    first_gain = batch_dot(first, state.residual) ** 2 / (first_energy + DENOMINATOR_EPS)
    second_gain = batch_dot(second, state.residual) ** 2 / (second_energy + DENOMINATOR_EPS)
    first_step = batch_dot(first, state.residual) / (first_energy + DENOMINATOR_EPS)
    second_step = batch_dot(second, state.residual) / (second_energy + DENOMINATOR_EPS)
    direction = batch_scale(first_step, first) + batch_scale(second_step, second)
    image = batch_scale(first_step, first_image) + batch_scale(second_step, second_image)
    slot = state.step % m_max
    new_state = FCGState(state.iterate + direction, state.residual - image, state.step + 1, state.directions.at[slot].set(direction),
                         state.operator_directions.at[slot].set(image), state.direction_energies.at[slot].set(batch_dot(direction, image)))
    return new_state, second_gain / (first_gain + second_gain + DENOMINATOR_EPS)


# Safeguarded FCG for n_iter steps; returns the final iterate, residual norms and, per step, the share of the energy decrease from the fallback
def safeguarded_fcg(apply_operator, network, fallback, rhs, initial_guess, n_iter, m_max=20, every=1):
    # One safeguarded step plus its outputs
    def body(state, unused):
        new_state, fallback_share = safeguarded_step(state, apply_operator, network, fallback, m_max, every)
        return new_state, {"residual_norm": jnp.sqrt(batch_dot(state.residual, state.residual)), "fallback_share": fallback_share}

    state, outputs = lax.scan(body, fcg_init(apply_operator, rhs, initial_guess, m_max), None, length=n_iter)
    final_norm = jnp.sqrt(batch_dot(state.residual, state.residual))
    return {"iterate": state.iterate, "residual_norm": jnp.concatenate([outputs["residual_norm"], final_norm[None]]).T,
            "fallback_share": outputs["fallback_share"].T}
