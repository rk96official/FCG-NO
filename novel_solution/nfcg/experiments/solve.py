import numpy as np
import jax.numpy as jnp
import equinox as eqx

from fcgno.data.assembly import apply_stencil
from fcgno.linalg.krylov import fcg
from fcgno.models.registry import as_preconditioner
from nfcg.data.coefficient_features import log_coefficient
from nfcg.linalg.history_fcg import history_fcg
from nfcg.training.records import deployed_preconditioner, two_level_key


# Jitted batched FCG returning residual norms; model None gives B = I, i.e. CG
@eqx.filter_jit
def solve_batch(model, face_x, face_y, rhs, guess, n_iter, m_max, normalization):
    precondition = (lambda r: r) if model is None else as_preconditioner(model)
    operator = lambda u: apply_stencil(face_x, face_y, u, normalization)
    return fcg(operator, precondition, rhs, guess, n_iter, m_max)["residual_norm"]


# Jitted batched FCG with a history-aware model (memoryless when history = 0; log a channel when given; two-level when given), returning residual norms
@eqx.filter_jit
def solve_history_batch(model, face_x, face_y, rhs, guess, log_coefficient_field, n_iter, m_max, history, normalization, two_level=None):
    operator = lambda u: apply_stencil(face_x, face_y, u, normalization)
    precondition = deployed_preconditioner(model, history, face_x, face_y, normalization, rhs.shape[-1], log_coefficient_field, two_level)
    return history_fcg(operator, precondition, rhs, guess, n_iter, m_max, max(history, 1))["residual_norm"]


# Relative residual curves (systems, n_iter + 1) of FCG on a test set, solved in chunks of fixed size
def relative_residuals(model, dataset, guesses, n_iter, m_max, normalization, chunk):
    curves = []
    for start in range(0, len(guesses), chunk):
        part = [jnp.asarray(dataset[name][start:start + chunk]) for name in ("face_x", "face_y", "rhs")]
        norms = np.asarray(solve_batch(model, *part, jnp.asarray(guesses[start:start + chunk]), n_iter, m_max, normalization))
        curves.append(norms / norms[:, :1])
    return np.concatenate(curves)


# Relative residual curves of FCG with a history-aware (or memoryless, history = 0) model, solved in chunks; coefficient feeds log a to the model
def history_relative_residuals(model, history, dataset, guesses, n_iter, m_max, normalization, chunk, coefficient=False, two_level=None):
    fields = log_coefficient(dataset) if coefficient else None
    curves = []
    for start in range(0, len(guesses), chunk):
        part = [jnp.asarray(dataset[name][start:start + chunk]) for name in ("face_x", "face_y", "rhs")]
        field = None if fields is None else jnp.asarray(fields[start:start + chunk])
        norms = np.asarray(solve_history_batch(model, *part, jnp.asarray(guesses[start:start + chunk]), field, n_iter, m_max, history, normalization,
                                               two_level_key(two_level)))
        curves.append(norms / norms[:, :1])
    return np.concatenate(curves)
