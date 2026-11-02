import jax
import jax.numpy as jnp

from fcgno.models.sno import SNO


# History-conditioned SNO (HC-NO): input r_i, the update and its image for each of `history` past steps, and optionally the log coefficient; output w_i
def build_history_sno(features, key, history=1, coefficient=False, n_layers=4, n_modes=20):
    return SNO(features, key, in_channels=1 + 2 * history + int(coefficient), n_layers=n_layers, n_modes=n_modes)


# Divide each sample of a batched field by its norm; zero fields (an empty history) stay zero
def normalize_fields(fields):
    norms = jnp.linalg.norm(fields.reshape(fields.shape[0], -1), axis=1)
    return fields / jnp.where(norms > 0, norms, 1.0)[:, None, None]


# Batched preconditioner B(r, updates, changes) = ||r|| model([r, updates, changes, log a]); stacks are (stored, batch, n, n) and the first `history` are used
def history_preconditioner(model, history, log_coefficient=None):
    # Assemble the normalized channels, run the model per sample, restore the residual's scale
    def apply(residual, updates, changes):
        norms = jnp.linalg.norm(residual.reshape(residual.shape[0], -1), axis=1)
        safe_norms = jnp.where(norms > 0, norms, 1.0)[:, None, None]
        channels = [residual / safe_norms] + [normalize_fields(field) for field in updates[:history]] + [normalize_fields(field) for field in changes[:history]]
        if log_coefficient is not None:
            channels.append(log_coefficient)
        return jax.vmap(model)(jnp.stack(channels, axis=1))[:, 0] * safe_norms

    return apply
