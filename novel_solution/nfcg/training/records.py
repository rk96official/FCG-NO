import numpy as np
import jax.numpy as jnp
import equinox as eqx
from jax import lax

from fcgno.data.assembly import apply_stencil
from fcgno.linalg.krylov import batch_dot
from fcgno.models.registry import as_preconditioner
from nfcg.data.coefficient_features import log_coefficient
from nfcg.linalg.history_fcg import history_init, history_step
from nfcg.linalg.multilevel import multilevel_history_preconditioner
from nfcg.linalg.smoothers import make_smoother, stencil_diagonal
from nfcg.linalg.two_level import interpolation_matrix, restrict, two_level_history_preconditioner
from nfcg.models.history import history_preconditioner

RECORD_FIELDS = ("residual", "error", "updates", "changes", "owner")


# History-aware preconditioner of a model: HC-NO models read the history (and log a when given), memoryless ones ignore it; model None gives B = I (CG)
def model_preconditioner(model, history, log_coefficient_field=None):
    if model is None:
        return lambda residual, updates, changes: residual
    if history == 0 and log_coefficient_field is None:
        network = as_preconditioner(model)
        return lambda residual, updates, changes: network(residual)
    return history_preconditioner(model, history, log_coefficient_field)


# Preconditioner of a model as deployed: the history-aware network, the two-level scheme around it (network on a grid coarsened by
# two_level["coarsen"], Chebyshev or Jacobi smoothing on the fine grid), or with two_level["coarse_grid"] the multi-level scheme (network at the bottom
# grid, and on levels up to two_level["network_max"] in the recursive form; with two_level["fine_network"] the two-level slot with the network also as a
# line-searched correction on the fine grid); two_level is None or a sorted tuple of option pairs
def deployed_preconditioner(model, history, face_x, face_y, normalization, grid, log_coefficient_field=None, two_level=None):
    if model is None or two_level is None:
        return model_preconditioner(model, history, log_coefficient_field)
    options = dict(two_level)
    if options.get("fine_network"):
        options = {**options, "coarse_grid": grid // options["coarsen"], "network_max": grid}
    if "coarse_grid" in options:
        factory = lambda field: model_preconditioner(model, history, field)
        return multilevel_history_preconditioner(factory, face_x, face_y, normalization, log_coefficient_field, options)
    operator = lambda u: apply_stencil(face_x, face_y, u, normalization)
    interpolation = interpolation_matrix(grid, grid // options["coarsen"])
    coarse_field = None if log_coefficient_field is None else restrict(log_coefficient_field, interpolation)
    smooth = make_smoother(options["smoother"], operator, stencil_diagonal(face_x, face_y, normalization), options.get("sweeps", 1))
    return two_level_history_preconditioner(model_preconditioner(model, history, coarse_field), smooth, operator, interpolation)


# Sorted tuple form of a two-level option dict (hashable for jit), or None
def two_level_key(options):
    return None if options is None else tuple(sorted(options.items()))


# FCG trajectories (CG when model is None) on one chunk, recording before each step r_i, e_i = u - u_i, the history stacks and ||r_i||
@eqx.filter_jit
def trajectory_chunk(model, face_x, face_y, rhs, guess, solution, log_coefficient_field, n_steps, m_max, history, normalization, two_level=None):
    operator = lambda u: apply_stencil(face_x, face_y, u, normalization)
    precondition = deployed_preconditioner(model, history, face_x, face_y, normalization, rhs.shape[-1], log_coefficient_field, two_level)

    # Record the state, then take one history-aware FCG step
    def body(state, unused):
        record = {"residual": state.solver.residual, "error": solution - state.solver.iterate, "updates": state.updates,
                  "changes": state.changes, "norm": jnp.sqrt(batch_dot(state.solver.residual, state.solver.residual))}
        new_state, unused_proposed = history_step(state, operator, precondition, m_max)
        return new_state, record

    unused_state, records = lax.scan(body, history_init(operator, rhs, guess, m_max, max(history, 1)), None, length=n_steps)
    return records


# Records of a preconditioner on a training set: steps before ||r_i|| / ||r_0|| first reaches floor, every stride-th, with r_i and e_i divided by ||r_i||
def collect_records(model, history, dataset, guesses, n_steps, m_max, normalization, chunk, floor=0.0, stride=1, coefficient=False, two_level=None):
    pieces = {name: [] for name in RECORD_FIELDS}
    fields = log_coefficient(dataset) if coefficient else None
    for start in range(0, len(guesses), chunk):
        arrays = [jnp.asarray(dataset[name][start:start + chunk]) for name in ("face_x", "face_y", "rhs", "solution")]
        field = None if fields is None else jnp.asarray(fields[start:start + chunk])
        records = trajectory_chunk(model, *arrays[:3], jnp.asarray(guesses[start:start + chunk]), arrays[3], field, n_steps, m_max, history, normalization,
                                   two_level_key(two_level))
        norm = np.asarray(records["norm"])
        reached = np.logical_or.accumulate(norm / norm[:1] <= floor, axis=0)
        steps, systems = np.nonzero(~reached & (np.arange(n_steps)[:, None] % stride == 0))
        scale = norm[steps, systems][:, None, None]
        pieces["residual"].append(np.asarray(records["residual"])[steps, systems] / scale)
        pieces["error"].append(np.asarray(records["error"])[steps, systems] / scale)
        pieces["updates"].append(np.moveaxis(np.asarray(records["updates"]), 1, 2)[steps, systems])
        pieces["changes"].append(np.moveaxis(np.asarray(records["changes"]), 1, 2)[steps, systems])
        pieces["owner"].append(start + systems)
    return {name: np.concatenate(values) for name, values in pieces.items()}


# Random subset of at most limit records (all of them when there are fewer)
def subsample(records, limit, seed):
    count = len(records["owner"])
    if count <= limit:
        return records
    chosen = np.sort(np.random.default_rng(seed).choice(count, limit, replace=False))
    return {name: values[chosen] for name, values in records.items()}


# Records of several sets in one; owners must already index the same problem table
def merge(*record_sets):
    return {name: np.concatenate([records[name] for records in record_sets]) for name in RECORD_FIELDS}


# Per-problem table for training: face coefficients, plus the centered log coefficient when the model reads it
def problem_table(dataset, coefficient=False):
    table = {"face_x": dataset["face_x"], "face_y": dataset["face_y"]}
    if coefficient:
        table["log_coefficient"] = log_coefficient(dataset)
    return table


# Training data in the layout of fcgno's trainer: per-record pairs, per-problem arrays, and the owning problem of each record
def training_data(records, problems):
    pairs = {name: records[name] for name in ("residual", "error", "updates", "changes")}
    return {"pairs": pairs, "problems": dict(problems), "owner": records["owner"]}
