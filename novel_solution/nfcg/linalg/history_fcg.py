from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from fcgno.linalg.krylov import FCGState, batch_dot, fcg_init, fcg_step
from nfcg.linalg.safeguarded_fcg import safeguarded_update


class HistoryState(NamedTuple):
    solver: FCGState
    updates: jnp.ndarray
    changes: jnp.ndarray


# Initial state: plain FCG state plus empty stacks (history, batch, n, n) of past updates u_i - u_{i-1} and their images r_{i-1} - r_i
def history_init(apply_operator, rhs, initial_guess, m_max, history):
    empty = jnp.zeros((history,) + rhs.shape, rhs.dtype)
    return HistoryState(fcg_init(apply_operator, rhs, initial_guess, m_max), empty, empty)


# Push the newest field onto a history stack, dropping the oldest
def push(stack, newest):
    return jnp.concatenate([newest[None], stack[:-1]])


# One FCG step whose preconditioner also sees the latest updates and their images (newest first); returns the new state and w_i
def history_step(state, apply_operator, preconditioner, m_max):
    proposed = preconditioner(state.solver.residual, state.updates, state.changes)
    solver, unused_proposed = fcg_step(state.solver, apply_operator, lambda unused_residual: proposed, m_max)
    update = solver.iterate - state.solver.iterate
    change = state.solver.residual - solver.residual
    return HistoryState(solver, push(state.updates, update), push(state.changes, change)), proposed


# Safeguarded FCG step with a history-aware network (run when step % every == 0, else zero) plus a fallback; the history records the combined update
def history_safeguarded_step(state, apply_operator, preconditioner, fallback, m_max, every):
    proposed = lax.cond(state.solver.step % every == 0, lambda operands: preconditioner(*operands), lambda operands: jnp.zeros_like(operands[0]),
                        (state.solver.residual, state.updates, state.changes))
    solver, fallback_share = safeguarded_update(state.solver, apply_operator, proposed, fallback, m_max)
    update = solver.iterate - state.solver.iterate
    change = state.solver.residual - solver.residual
    return HistoryState(solver, push(state.updates, update), push(state.changes, change)), fallback_share


# Safeguarded FCG with a history-aware network; returns the final iterate, residual norms and the fallback share per step
def history_safeguarded_fcg(apply_operator, preconditioner, fallback, rhs, initial_guess, n_iter, m_max=20, history=1, every=1):
    # One safeguarded history step, recording the residual norm before it
    def body(state, unused):
        new_state, fallback_share = history_safeguarded_step(state, apply_operator, preconditioner, fallback, m_max, every)
        return new_state, {"residual_norm": jnp.sqrt(batch_dot(state.solver.residual, state.solver.residual)), "fallback_share": fallback_share}

    state, outputs = lax.scan(body, history_init(apply_operator, rhs, initial_guess, m_max, history), None, length=n_iter)
    final_norm = jnp.sqrt(batch_dot(state.solver.residual, state.solver.residual))
    return {"iterate": state.solver.iterate, "residual_norm": jnp.concatenate([outputs["residual_norm"], final_norm[None]]).T,
            "fallback_share": outputs["fallback_share"].T}


# FCG with a history-conditioned preconditioner B(r, updates, changes); returns the final iterate and residual norms per step
def history_fcg(apply_operator, preconditioner, rhs, initial_guess, n_iter, m_max=20, history=1):
    # One history step, recording the residual norm before it
    def body(state, unused):
        new_state, unused_proposed = history_step(state, apply_operator, preconditioner, m_max)
        return new_state, jnp.sqrt(batch_dot(state.solver.residual, state.solver.residual))

    state, norms = lax.scan(body, history_init(apply_operator, rhs, initial_guess, m_max, history), None, length=n_iter)
    final_norm = jnp.sqrt(batch_dot(state.solver.residual, state.solver.residual))
    return {"iterate": state.solver.iterate, "residual_norm": jnp.concatenate([norms, final_norm[None]]).T}
