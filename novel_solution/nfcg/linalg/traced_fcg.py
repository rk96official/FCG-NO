import jax.numpy as jnp
from jax import lax

from fcgno.linalg.krylov import DENOMINATOR_EPS, batch_dot, fcg_init, fcg_step


# Squared sine of the A-angle between each direction d and the error e, using (d, A e) = (d, r); invariant to the scale of d
def angle_sine_squared(direction, operator_direction, residual, error_energy):
    cosine_squared = batch_dot(direction, residual) ** 2 / (batch_dot(direction, operator_direction) * error_energy + DENOMINATOR_EPS)
    return jnp.clip(1.0 - cosine_squared, 0.0, 1.0)


# Notay ratio ||w - e||_A / ||e||_A and squared A-angle sine of a preconditioned residual w against the error e
def preconditioner_quality(preconditioned, residual, error, error_energy, apply_operator):
    difference = preconditioned - error
    notay = jnp.sqrt(batch_dot(difference, apply_operator(difference)) / (error_energy + DENOMINATOR_EPS))
    return notay, angle_sine_squared(preconditioned, apply_operator(preconditioned), residual, error_energy)


# FCG recording per step the residual norm, error energy, quality of the proposed w_i, the A-angle of the search direction p_i and the measured contraction.
# intervention(w, r, e, apply_operator) -> (w', mask) may replace w_i before the step; observer(r) is a second preconditioner evaluated on the same residuals but not used
def fcg_traced(apply_operator, preconditioner, rhs, initial_guess, solution, n_iter, m_max=20, error_probe=None, intervention=None, observer=None):
    # One FCG step of Algorithm 1 plus its diagnostics against the exact error
    def body(state, unused):
        error = solution - state.iterate
        error_energy = batch_dot(error, apply_operator(error))
        proposed = preconditioner(state.residual)
        applied, intervened = (proposed, jnp.zeros(error_energy.shape, bool)) if intervention is None else intervention(proposed, state.residual, error, apply_operator)
        new_state, unused_applied = fcg_step(state, apply_operator, lambda unused_residual: applied, m_max)
        slot = state.step % m_max
        new_error = solution - new_state.iterate
        notay, sine_squared = preconditioner_quality(proposed, state.residual, error, error_energy, apply_operator)
        outputs = {
            "residual_norm": jnp.sqrt(batch_dot(state.residual, state.residual)),
            "error_energy": error_energy,
            "notay": notay,
            "preconditioned_sine_squared": sine_squared,
            "direction_sine_squared": angle_sine_squared(new_state.directions[slot], new_state.operator_directions[slot], state.residual, error_energy),
            "contraction": batch_dot(new_error, apply_operator(new_error)) / (error_energy + DENOMINATOR_EPS),
            "intervened": intervened,
        }
        if error_probe is not None:
            outputs["error_probe"] = error_probe(error)
        if observer is not None:
            observed = observer(state.residual)
            outputs["observer_notay"], outputs["observer_sine_squared"] = preconditioner_quality(observed, state.residual, error, error_energy, apply_operator)
            gap = proposed - observed
            outputs["observer_deviation"] = jnp.sqrt(batch_dot(gap, apply_operator(gap)) / (batch_dot(observed, apply_operator(observed)) + DENOMINATOR_EPS))
        return new_state, outputs

    state, outputs = lax.scan(body, fcg_init(apply_operator, rhs, initial_guess, m_max), None, length=n_iter)
    traces = {name: values.T for name, values in outputs.items()}
    final_norm = jnp.sqrt(batch_dot(state.residual, state.residual))
    traces["residual_norm"] = jnp.concatenate([outputs["residual_norm"], final_norm[None]]).T
    return traces
