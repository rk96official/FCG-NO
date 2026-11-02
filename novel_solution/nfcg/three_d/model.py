import jax
import jax.numpy as jnp
import equinox as eqx

from fcgno.models.spectral import real_fourier_analysis, real_fourier_synthesis


# Apply one matrix along each spatial axis of a (features, n1, n2, n3) field
def apply_separable_3d(field, first, second, third):
    return jnp.einsum("ai,bj,ck,fijk->fabc", first, second, third, field)


class SpectralCell3D(eqx.Module):
    convolutions: list

    # 3x3x3 convolutions on the (features, modes, modes, modes) coefficient grid, GELU between them
    def __init__(self, features, key, depth=3):
        self.convolutions = [eqx.nn.Conv3d(features, features, 3, padding=1, key=k) for k in jax.random.split(key, depth)]

    # Convolutions with GELU between, none after the last
    def __call__(self, coefficients):
        for convolution in self.convolutions[:-1]:
            coefficients = jax.nn.gelu(convolution(coefficients))
        return self.convolutions[-1](coefficients)


class SNO3D(eqx.Module):
    encoder: eqx.nn.Conv3d
    decoder: eqx.nn.Conv3d
    cells: list
    skips: list
    n_modes: int = eqx.field(static=True)

    # The 2D SNO (encoder-processor-decoder in the real Fourier basis) with 3D convolutions and separable 3D transforms
    def __init__(self, features, key, in_channels=3, n_layers=4, n_modes=12):
        keys = jax.random.split(key, 4)
        self.encoder = eqx.nn.Conv3d(in_channels, features, 1, key=keys[0])
        self.decoder = eqx.nn.Conv3d(features, 1, 1, key=keys[1])
        self.cells = [SpectralCell3D(features, k) for k in jax.random.split(keys[2], n_layers)]
        self.skips = [eqx.nn.Conv3d(features, features, 1, key=k) for k in jax.random.split(keys[3], n_layers)]
        self.n_modes = n_modes

    # Map a (channels, n, n, n) field to (1, n, n, n); the Fourier matrices take the field's dtype, so float32 inputs run fully in float32
    def __call__(self, field):
        analysis = [real_fourier_analysis(n, self.n_modes).astype(field.dtype) for n in field.shape[1:]]
        synthesis = [real_fourier_synthesis(n, self.n_modes).astype(field.dtype) for n in field.shape[1:]]
        hidden = self.encoder(field)
        for index, (cell, skip) in enumerate(zip(self.cells, self.skips)):
            hidden = apply_separable_3d(cell(apply_separable_3d(hidden, *analysis)), *synthesis) + skip(hidden)
            if index < len(self.cells) - 1:
                hidden = jax.nn.gelu(hidden)
        return self.decoder(hidden)


# Divide each sample of a batched field by its norm; zero fields stay zero
def normalize_batch(fields):
    norms = jnp.linalg.norm(fields.reshape(fields.shape[0], -1), axis=1)
    return fields / jnp.where(norms > 0, norms, 1.0).reshape((-1,) + (1,) * (fields.ndim - 1))


# Batched history preconditioner B(r, updates, changes) = ||r|| model([r, update, change]) for 3D fields (history 0 or 1), optionally in float32
def history_preconditioner_3d(model, history, dtype=None):
    network = model if dtype is None else jax.tree_util.tree_map(lambda leaf: leaf.astype(dtype) if eqx.is_inexact_array(leaf) else leaf, model)

    # Normalized channels, per-sample network, the residual's scale restored
    def apply(residual, updates, changes):
        norms = jnp.linalg.norm(residual.reshape(residual.shape[0], -1), axis=1).reshape((-1, 1, 1, 1))
        safe = jnp.where(norms > 0, norms, 1.0)
        channels = [residual / safe] + [normalize_batch(stack) for stack in (*updates[:history], *changes[:history])]
        inputs = jnp.stack(channels, axis=1)
        outputs = jax.vmap(network)(inputs if dtype is None else inputs.astype(dtype))[:, 0]
        return outputs.astype(residual.dtype) * safe

    return apply
