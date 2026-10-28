import jax
import equinox as eqx

from fcgno.models.spectral import real_fourier_analysis, real_fourier_synthesis, apply_separable


class SpectralCell(eqx.Module):
    convolutions: list

    # 3x3 convolutions on the (features, modes, modes) coefficient grid, as built by upstream get_SNO
    def __init__(self, features, key, depth=3, kernel_size=3):
        keys = jax.random.split(key, depth)
        self.convolutions = [eqx.nn.Conv2d(features, features, kernel_size, padding=kernel_size // 2, key=k) for k in keys]

    # GELU between convolutions, none after the last
    def __call__(self, coefficients):
        for convolution in self.convolutions[:-1]:
            coefficients = jax.nn.gelu(convolution(coefficients))
        return self.convolutions[-1](coefficients)


class SNO(eqx.Module):
    encoder: eqx.nn.Conv2d
    decoder: eqx.nn.Conv2d
    cells: list
    skips: list
    n_modes: int = eqx.field(static=True)

    # Encoder-processor-decoder SNO in the real Fourier basis (paper A.1, upstream fSNO)
    def __init__(self, features, key, in_channels=1, out_channels=1, n_layers=4, n_modes=20):
        keys = jax.random.split(key, 4)
        self.encoder = eqx.nn.Conv2d(in_channels, features, 1, key=keys[0])
        self.decoder = eqx.nn.Conv2d(features, out_channels, 1, key=keys[1])
        self.cells = [SpectralCell(features, k) for k in jax.random.split(keys[2], n_layers)]
        self.skips = [eqx.nn.Conv2d(features, features, 1, key=k) for k in jax.random.split(keys[3], n_layers)]
        self.n_modes = n_modes

    # Map a (channels, n1, n2) field to (out_channels, n1, n2) on any grid size
    def __call__(self, field):
        analysis = [real_fourier_analysis(n, self.n_modes) for n in field.shape[1:]]
        synthesis = [real_fourier_synthesis(n, self.n_modes) for n in field.shape[1:]]
        hidden = self.encoder(field)
        for index, (cell, skip) in enumerate(zip(self.cells, self.skips)):
            hidden = apply_separable(cell(apply_separable(hidden, *analysis)), *synthesis) + skip(hidden)
            if index < len(self.cells) - 1:
                hidden = jax.nn.gelu(hidden)
        return self.decoder(hidden)
