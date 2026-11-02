import jax
import jax.numpy as jnp
import equinox as eqx

from fcgno.data.assembly import apply_stencil
from nfcg.linalg.incomplete_cholesky import factor_product, operator_entries, shifted


class NeuralIF(eqx.Module):
    layers: list
    output: eqx.nn.Conv2d

    # Message passing on the 5-point grid graph (3x3 convolutions over per-node edge features), after NeuralIF (Hausner et al., TMLR 2024)
    def __init__(self, features, key, n_layers=4):
        keys = jax.random.split(key, n_layers + 1)
        channels = [5] + [features] * n_layers
        self.layers = [eqx.nn.Conv2d(channels[index], channels[index + 1], 3, padding=1, key=keys[index]) for index in range(n_layers)]
        output = eqx.nn.Conv2d(features, 3, 3, padding=1, key=keys[-1])
        self.output = eqx.tree_at(lambda layer: (layer.weight, layer.bias), output, (jnp.zeros_like(output.weight), jnp.zeros_like(output.bias)))

    # Corrections (log diagonal scale, west and south factors) from (5, n, n) node features
    def __call__(self, features):
        hidden = features
        for layer in self.layers:
            hidden = jax.nn.gelu(layer(hidden))
        return self.output(hidden)


# Diagonal-normalized node features of a batch of operators: log of the diagonal over its mean, and the four couplings over the diagonal
def node_features(diagonal, west, south):
    east, north = shifted(west, -2, 1), shifted(south, -1, 1)
    log_diagonal = jnp.log(diagonal / jnp.mean(diagonal, axis=(-2, -1), keepdims=True))
    return jnp.stack([log_diagonal, west / diagonal, south / diagonal, east / diagonal, north / diagonal], axis=1)


# Sparse lower factor L (IC(0) pattern) predicted by the network: scaled-Jacobi factor entries times learned corrections; positive diagonal
def neural_factor(model, face_x, face_y, normalization):
    diagonal, west, south = operator_entries(face_x, face_y, normalization)
    corrections = jax.vmap(model)(node_features(diagonal, west, south))
    root = jnp.sqrt(diagonal)
    lower_diagonal = root * jnp.exp(corrections[:, 0])
    lower_west = west / shifted(root, -2, -1).at[..., 0, :].set(1.0) * (1.0 + corrections[:, 1])
    lower_south = south / shifted(root, -1, -1).at[..., :, 0].set(1.0) * (1.0 + corrections[:, 2])
    return lower_diagonal, lower_west, lower_south


# NeuralIF training loss: stochastic Frobenius norm || (L L^T - A) w ||^2 / || A w ||^2 with w ~ N(0, I), averaged over the batch
def frobenius_loss(model, face_x, face_y, normalization, key):
    probe = jax.random.normal(key, face_x.shape[:-2] + (face_x.shape[-1], face_x.shape[-1]), face_x.dtype)
    image = apply_stencil(face_x, face_y, probe, normalization)
    difference = factor_product(neural_factor(model, face_x, face_y, normalization), probe) - image
    return jnp.mean(jnp.sum(difference ** 2, axis=(-2, -1)) / jnp.sum(image ** 2, axis=(-2, -1)))
