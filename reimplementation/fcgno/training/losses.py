import jax
import jax.numpy as jnp

from fcgno.data.assembly import apply_stencil
from fcgno.linalg.krylov import batch_dot


# Apply a single-sample model to a batch of single-channel fields of shape (batch, n, n)
def apply_batched(model, fields):
    return jax.vmap(model)(fields[:, None])[:, 0]


# Squared energy norm v^T A v per sample for operators given by face coefficients
def energy_norm_squared(face_x, face_y, field, normalization):
    return batch_dot(field, apply_stencil(face_x, face_y, field, normalization))


# Mean relative L2 error ||prediction - target||_2 / ||target||_2 over the batch
def relative_l2(prediction, target):
    flatten = lambda field: field.reshape(field.shape[0], -1)
    return jnp.mean(jnp.linalg.norm(flatten(prediction - target), axis=1) / jnp.linalg.norm(flatten(target), axis=1))


# Notay loss (paper eq. 11): mean ||B(r) - e||_A / ||e||_A with e = A^{-1} r
def notay_loss(model, batch, normalization):
    difference = apply_batched(model, batch["residual"]) - batch["error"]
    numerator = energy_norm_squared(batch["face_x"], batch["face_y"], difference, normalization)
    denominator = energy_norm_squared(batch["face_x"], batch["face_y"], batch["error"], normalization)
    return jnp.mean(jnp.sqrt(numerator / denominator))


# L2 loss (paper eq. 12): mean ||B(r) - e||_2 / ||e||_2
def l2_loss(model, batch, normalization):
    return relative_l2(apply_batched(model, batch["residual"]), batch["error"])


# L2 loss as written (commented out) in the upstream pipelines: mean squared error ||B(r) - e||_2^2, not normalized
def mse_loss(model, batch, normalization):
    difference = apply_batched(model, batch["residual"]) - batch["error"]
    return jnp.mean(jnp.sum(difference.reshape(difference.shape[0], -1) ** 2, axis=1))


# Supervised surrogate loss for Fig. 1: relative L2 between the predicted and exact solution of A u = f
def surrogate_loss(model, batch, normalization):
    return relative_l2(apply_batched(model, batch["rhs"]), batch["solution"])


LOSSES = {"notay": notay_loss, "l2": l2_loss, "mse": mse_loss, "surrogate": surrogate_loss}
