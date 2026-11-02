import math

import jax
import jax.numpy as jnp

from fcgno.data.assembly import apply_stencil
from fcgno.linalg.krylov import DENOMINATOR_EPS, batch_dot, batch_scale
from fcgno.training.losses import LOSSES
from nfcg.models.history import normalize_fields
from nfcg.training.records import deployed_preconditioner, two_level_key

LOG_FLOOR = 1e-8


# Model inputs for a batch of records: residual, `history` normalized updates and changes, optionally log a; same order as history_preconditioner
def record_inputs(batch, history, coefficient=False):
    channels = [batch["residual"]]
    channels += [normalize_fields(batch["updates"][:, index]) for index in range(history)]
    channels += [normalize_fields(batch["changes"][:, index]) for index in range(history)]
    if coefficient:
        channels.append(batch["log_coefficient"])
    return jnp.stack(channels, axis=1)


# A-orthogonalize w against the recorded updates (the latest FCG directions, mutually A-orthogonal) as FCG does before stepping; empty history is skipped
def project_out_history(field, updates, changes):
    for index in range(updates.shape[1]):
        update, change = updates[:, index], changes[:, index]
        field = field - batch_scale(batch_dot(field, change) / (batch_dot(update, change) + DENOMINATOR_EPS), update)
    return field


# Per-sample Notay ratio of w and exact one-step contraction sin^2_A(Pi w, e) (Prop. 1) for a batch of records
def per_sample_losses(output, batch, normalization):
    operator = lambda u: apply_stencil(batch["face_x"], batch["face_y"], u, normalization)
    error = batch["error"]
    operator_error = operator(error)
    error_energy = batch_dot(error, operator_error)
    difference = output - error
    notay = jnp.sqrt(batch_dot(difference, operator(difference)) / (error_energy + DENOMINATOR_EPS))
    projected = project_out_history(output, batch["updates"], batch["changes"])
    cosine_squared = batch_dot(projected, operator_error) ** 2 / (batch_dot(projected, operator(projected)) * error_energy + DENOMINATOR_EPS)
    return notay, jnp.clip(1.0 - cosine_squared, 0.0, 1.0)


# Batch aggregate: the mean, or CVaR_q estimated as the mean of the largest ceil(q B) values
def aggregate(values, mode, q):
    if mode == "mean":
        return jnp.mean(values)
    return jnp.mean(jax.lax.top_k(values, max(1, math.ceil(q * values.shape[0])))[0])


# Training loss of a variant: objective notay (paper), step (exact contraction) or log_step, aggregated by mean or CVaR, plus a weak Notay term fixing the output scale
def make_loss(objective, history, mode="mean", q=0.1, scale_weight=0.0, coefficient=False, two_level=None):
    # Scalar loss for fcgno's trainer: (model, batch, normalization) -> loss; with two_level the loss sees the whole two-level preconditioner
    def loss(model, batch, normalization):
        if two_level is None:
            output = jax.vmap(model)(record_inputs(batch, history, coefficient))[:, 0]
        else:
            precondition = deployed_preconditioner(model, history, batch["face_x"], batch["face_y"], normalization, batch["residual"].shape[-1],
                                                   batch["log_coefficient"] if coefficient else None, two_level_key(two_level))
            output = precondition(batch["residual"], jnp.moveaxis(batch["updates"], 1, 0), jnp.moveaxis(batch["changes"], 1, 0))
        notay, step = per_sample_losses(output, batch, normalization)
        values = {"notay": notay, "step": step, "log_step": jnp.log(step + LOG_FLOOR)}[objective]
        return aggregate(values, mode, q) + scale_weight * jnp.mean(notay)

    return loss


# Register a loss in fcgno's loss table, which fcgno.training.trainer.train reads by name; returns the name
def register_loss(objective, history, mode="mean", q=0.1, scale_weight=0.0, coefficient=False, two_level=None):
    level = "" if two_level is None else "_2l" + "_".join(f"{key}{value}" for key, value in sorted(two_level.items()))
    name = f"nfcg_{objective}_{mode}_q{q:g}_h{history}_scale{scale_weight:g}_c{int(coefficient)}{level}"
    LOSSES[name] = make_loss(objective, history, mode, q, scale_weight, coefficient, two_level)
    return name
