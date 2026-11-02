import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import lax
from jax.scipy.linalg import cho_factor, cho_solve

from fcgno.linalg.krylov import DENOMINATOR_EPS, batch_dot
from nfcg.linalg.history_fcg import history_init, history_step
from nfcg.linalg.multilevel import line_searched
from nfcg.linalg.smoothers import chebyshev_sweeps, make_smoother
from nfcg.linalg.two_level import interpolation_matrix
from nfcg.three_d.model import history_preconditioner_3d
from nfcg.three_d.operator import apply_stencil_3d, coarse_faces_3d, diagonal_3d, prolongate_3d, restrict_3d
from nfcg.training.step_losses import project_out_history

FACE_NAMES = ("face_x", "face_y", "face_z")


# Levels for grid sizes [fine, ..., bottom] (two-level when sizes has two entries): per level above the bottom its operator, Chebyshev(2) smoother and
# interpolation to the next level; coarse levels rediscretized from the fine faces
def levels_3d(faces, normalization, sizes, sweeps=2):
    levels = []
    for size, below in zip(sizes[:-1], sizes[1:]):
        level_faces = faces if size == sizes[0] else coarse_faces_3d(faces, size)
        operator = lambda u, level_faces=level_faces: apply_stencil_3d(level_faces, u, normalization)
        smooth = make_smoother("chebyshev", operator, diagonal_3d(level_faces, normalization), sweeps)
        levels.append({"operator": operator, "smooth": smooth, "interpolation": interpolation_matrix(size, below).astype(faces[0].dtype)})
    return levels


# One application from level index down: pre-smooth, line-searched coarse direction from the levels below (bottom solve at the end), post-smooth
def multilevel_apply_3d(levels, index, bottom, residual):
    if index == len(levels):
        return bottom(residual)
    level = levels[index]
    operator = level["operator"]
    correction = level["smooth"](residual)
    coarse = multilevel_apply_3d(levels, index + 1, bottom, restrict_3d(residual - operator(correction), level["interpolation"]))
    correction = line_searched(operator, correction, prolongate_3d(coarse, level["interpolation"]), residual)
    return correction + level["smooth"](residual - operator(correction))


# Deployed 3D preconditioner (residual, updates, changes) -> correction: the network as the bottom solve of the levels for sizes, history restricted to the bottom
def network_preconditioner_3d(model, history, faces, normalization, sizes, dtype=None):
    levels = levels_3d(faces, normalization, sizes)
    network = history_preconditioner_3d(model, history, dtype)

    # Restrict the history stacks through every level, then run the scheme with the network at the bottom
    def apply(residual, updates, changes):
        for level in levels:
            updates, changes = restrict_3d(updates, level["interpolation"]), restrict_3d(changes, level["interpolation"])
        return multilevel_apply_3d(levels, 0, lambda bottom_residual: network(bottom_residual, updates, changes), residual)

    return apply


# Classical 3D preconditioner r -> correction: Chebyshev(sweeps)-Jacobi alone, or the rediscretized levels down to variant["sizes"] with a dense Cholesky
# or Chebyshev(8) bottom_solver
def classical_preconditioner_3d(faces, normalization, variant):
    operator = lambda u: apply_stencil_3d(faces, u, normalization)
    if variant["method"] == "chebyshev":
        return make_smoother("chebyshev", operator, diagonal_3d(faces, normalization), variant["sweeps"])
    levels = levels_3d(faces, normalization, variant["sizes"])
    bottom_faces = coarse_faces_3d(faces, variant["sizes"][-1])
    bottom_operator = lambda u: apply_stencil_3d(bottom_faces, u, normalization)
    if variant["bottom_solver"] == "dense":
        n = variant["sizes"][-1]
        basis = jnp.eye(n ** 3, dtype=faces[0].dtype).reshape(n ** 3, n, n, n)
        matrices = jax.vmap(lambda single: apply_stencil_3d(single, basis, normalization).reshape(n ** 3, n ** 3))(bottom_faces)
        factors = jax.vmap(lambda matrix: cho_factor(matrix, lower=True)[0])(matrices)
        bottom = lambda residual: jax.vmap(lambda lower, rhs: cho_solve((lower, True), rhs.reshape(-1)).reshape(n, n, n))(factors, residual)
    else:
        bottom = lambda residual: chebyshev_sweeps(bottom_operator, diagonal_3d(bottom_faces, normalization), residual, 8)
    return lambda residual: multilevel_apply_3d(levels, 0, bottom, residual)


# FCG trajectories on one chunk (CG when model is None), recording r_i, e_i, the history stacks and ||r_i|| before each step
@eqx.filter_jit
def trajectory_chunk_3d(model, history, faces, rhs, guess, solution, n_steps, m_max, normalization, sizes):
    operator = lambda u: apply_stencil_3d(faces, u, normalization)
    precondition = (lambda residual, updates, changes: residual) if model is None else network_preconditioner_3d(model, history, faces, normalization, sizes)

    # Record the state, then take one history-aware FCG step
    def body(state, unused):
        record = {"residual": state.solver.residual, "error": solution - state.solver.iterate, "updates": state.updates, "changes": state.changes,
                  "norm": jnp.sqrt(batch_dot(state.solver.residual, state.solver.residual))}
        new_state, unused_proposed = history_step(state, operator, precondition, m_max)
        return new_state, record

    unused_state, records = lax.scan(body, history_init(operator, rhs, guess, m_max, max(history, 1)), None, length=n_steps)
    return records


# Training records of a model on a solved dataset: steps before ||r_i|| / ||r_0|| reaches floor, r_i and e_i divided by ||r_i||
def collect_records_3d(model, history, dataset, guesses, n_steps, m_max, normalization, chunk, sizes, floor=1e-10):
    pieces = {name: [] for name in ("residual", "error", "updates", "changes", "owner")}
    for start in range(0, len(guesses), chunk):
        faces = tuple(jnp.asarray(dataset[name][start:start + chunk]) for name in FACE_NAMES)
        arrays = [jnp.asarray(dataset[name][start:start + chunk]) for name in ("rhs", "solution")]
        records = trajectory_chunk_3d(model, history, faces, arrays[0], jnp.asarray(guesses[start:start + chunk]), arrays[1], n_steps, m_max, normalization,
                                      None if model is None else tuple(sizes))
        norm = np.asarray(records["norm"])
        reached = np.logical_or.accumulate(norm / norm[:1] <= floor, axis=0)
        steps, systems = np.nonzero(~reached)
        scale = norm[steps, systems].reshape((-1, 1, 1, 1))
        pieces["residual"].append(np.asarray(records["residual"])[steps, systems] / scale)
        pieces["error"].append(np.asarray(records["error"])[steps, systems] / scale)
        pieces["updates"].append(np.moveaxis(np.asarray(records["updates"]), 1, 2)[steps, systems])
        pieces["changes"].append(np.moveaxis(np.asarray(records["changes"]), 1, 2)[steps, systems])
        pieces["owner"].append(start + systems)
    return {name: np.concatenate(values) for name, values in pieces.items()}


# Step loss of the deployed 3D preconditioner (exact one-step contraction sin^2_A(Pi w, e), Theorem 1) plus a weak Notay term fixing the scale
def make_step_loss_3d(history, sizes, scale_weight):
    # Scalar loss for fcgno's trainer: (model, batch, normalization) -> loss
    def loss(model, batch, normalization):
        faces = tuple(batch[name] for name in FACE_NAMES)
        operator = lambda u: apply_stencil_3d(faces, u, normalization)
        precondition = network_preconditioner_3d(model, history, faces, normalization, sizes)
        output = precondition(batch["residual"], jnp.moveaxis(batch["updates"], 1, 0), jnp.moveaxis(batch["changes"], 1, 0))
        error = batch["error"]
        operator_error = operator(error)
        energy = batch_dot(error, operator_error)
        difference = output - error
        notay = jnp.sqrt(batch_dot(difference, operator(difference)) / (energy + DENOMINATOR_EPS))
        projected = project_out_history(output, batch["updates"], batch["changes"])
        cosine = batch_dot(projected, operator_error) ** 2 / (batch_dot(projected, operator(projected)) * energy + DENOMINATOR_EPS)
        return jnp.mean(jnp.clip(1.0 - cosine, 0.0, 1.0)) + scale_weight * jnp.mean(notay)

    return loss
