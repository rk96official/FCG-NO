import jax
import jax.numpy as jnp
import equinox as eqx


# Contract (out, in, kx, ky) weights with an (in, kx, ky) spectrum per mode
def mix_modes(weights, spectrum):
    return jnp.einsum("oixy,ixy->oxy", weights, spectrum)


class FourierLayer(eqx.Module):
    skip: eqx.nn.Conv2d
    positive: jnp.ndarray
    negative: jnp.ndarray
    zero_positive: jnp.ndarray
    zero_negative: jnp.ndarray
    n_modes: int = eqx.field(static=True)

    # Spectral weights as in upstream FNO2D; complex weights stored as stacked (real, imag) parts
    def __init__(self, features, n_modes, key):
        keys = jax.random.split(key, 5)
        complex_scale = jnp.sqrt(0.5) / features
        self.skip = eqx.nn.Conv2d(features, features, 1, key=keys[0])
        self.positive = jax.random.normal(keys[1], (2, features, features, n_modes + 1, n_modes)) * complex_scale
        self.negative = jax.random.normal(keys[2], (2, features, features, n_modes, n_modes)) * complex_scale
        self.zero_positive = jax.random.normal(keys[3], (features, features, n_modes + 1, 1)) / features
        self.zero_negative = jax.random.normal(keys[4], (features, features, n_modes, 1)) / features
        self.n_modes = n_modes

    # Multiply kept modes (same update order as upstream), zero the rest, add the 1x1 skip convolution
    def __call__(self, hidden):
        m = self.n_modes
        spectrum = jnp.fft.rfftn(hidden, axes=(1, 2))
        spectrum = spectrum.at[:, :m + 1, :1].set(mix_modes(self.zero_positive, spectrum[:, :m + 1, :1]))
        spectrum = spectrum.at[:, -m:, :1].set(mix_modes(self.zero_negative, spectrum[:, -m:, :1]))
        spectrum = spectrum.at[:, :m + 1, 1:m + 1].set(mix_modes(self.positive[0] + 1j * self.positive[1], spectrum[:, :m + 1, 1:m + 1]))
        spectrum = spectrum.at[:, -m:, 1:m + 1].set(mix_modes(self.negative[0] + 1j * self.negative[1], spectrum[:, -m:, 1:m + 1]))
        spectrum = spectrum.at[:, :, m + 1:].set(0)
        spectrum = spectrum.at[:, m + 1:-m, :].set(0)
        return jnp.fft.irfftn(spectrum, axes=(1, 2), s=hidden.shape[1:]) + self.skip(hidden)


class FNO(eqx.Module):
    encoder: eqx.nn.Conv2d
    decoder: eqx.nn.Conv2d
    layers: list

    # FNO with coordinate channels, n_layers Fourier layers and ReLU (paper A.2, upstream FNO2D)
    def __init__(self, features, key, in_channels=1, out_channels=1, n_layers=4, n_modes=16):
        keys = jax.random.split(key, 3)
        self.encoder = eqx.nn.Conv2d(in_channels + 2, features, 1, key=keys[0])
        self.decoder = eqx.nn.Conv2d(features, out_channels, 1, key=keys[1])
        self.layers = [FourierLayer(features, n_modes, k) for k in jax.random.split(keys[2], n_layers)]

    # Map a square (channels, n, n) field to (out_channels, n, n)
    def __call__(self, field):
        line = jnp.linspace(0.0, 1.0, field.shape[-1])
        hidden = self.encoder(jnp.concatenate([jnp.stack(jnp.meshgrid(line, line), 0), field], 0))
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index < len(self.layers) - 1:
                hidden = jax.nn.relu(hidden)
        return self.decoder(hidden)
