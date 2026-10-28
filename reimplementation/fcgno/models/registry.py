import jax
import jax.numpy as jnp
import equinox as eqx

from fcgno.models.sno import SNO
from fcgno.models.fno import FNO
from fcgno.models.convnets import DilResNet, UNet

MODEL_CLASSES = {"sno": SNO, "fno": FNO, "dilresnet": DilResNet, "unet": UNet}

TABLE7_FEATURES = {
    "poisson": {
        "sno": {32: 32, 64: 32, 128: 80},
        "fno": {32: 32, 64: 32, 128: 32},
        "dilresnet": {32: 24, 64: 24, 128: 32},
        "unet": {32: 10, 64: 10, 128: 16},
    },
    "diffusion": {
        "sno": {32: 48, 64: 64, 128: 80},
        "fno": {32: 48, 64: 64, 128: 80},
        "dilresnet": {32: 24, 64: 24, 128: 32},
        "unet": {32: 10, 64: 10, 128: 16},
    },
}


# Build a model by name; features default to paper Table 7 for the dataset kind and training grid
def build_model(name, key, kind=None, grid=None, features=None, **options):
    if features is None:
        features = TABLE7_FEATURES[kind][name][grid]
    return MODEL_CLASSES[name](features, key, **options)


# Batched preconditioner B(r) = ||r|| model(r / ||r||) on residuals of shape (batch, n, n), as in upstream FCG
def as_preconditioner(model):
    # Normalize each residual, apply the model per sample, restore the scale
    def apply(residual):
        norms = jnp.linalg.norm(residual.reshape(residual.shape[0], -1), axis=1)
        safe_norms = jnp.where(norms > 0, norms, 1.0)[:, None, None]
        output = jax.vmap(model)((residual / safe_norms)[:, None])[:, 0]
        return output * safe_norms

    return apply


# Number of trainable scalar parameters
def count_parameters(model):
    return sum(leaf.size for leaf in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
