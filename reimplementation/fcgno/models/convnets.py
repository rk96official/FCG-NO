import jax
import jax.numpy as jnp
import equinox as eqx

DILATIONS = (1, 2, 4, 8, 4, 2, 1)


class DilatedBlock(eqx.Module):
    convolutions: list

    # 3x3 convolutions with dilations 1, 2, 4, 8, 4, 2, 1 (paper A.3, upstream DilatedConvBlock)
    def __init__(self, features, key, dilations=DILATIONS, kernel_size=3):
        keys = jax.random.split(key, len(dilations))
        self.convolutions = [
            eqx.nn.Conv2d(features, features, kernel_size, dilation=d, padding=d * (kernel_size // 2), key=k)
            for d, k in zip(dilations, keys)
        ]

    # GELU between convolutions, none after the last
    def __call__(self, hidden):
        for convolution in self.convolutions[:-1]:
            hidden = jax.nn.gelu(convolution(hidden))
        return self.convolutions[-1](hidden)


class DilResNet(eqx.Module):
    encoder: eqx.nn.Conv2d
    decoder: eqx.nn.Conv2d
    blocks: list

    # Dilated residual network with n_blocks residual blocks (paper A.3, upstream DilatedResNet)
    def __init__(self, features, key, in_channels=1, out_channels=1, n_blocks=4, kernel_size=3):
        keys = jax.random.split(key, 3)
        self.encoder = eqx.nn.Conv2d(in_channels, features, kernel_size, padding=kernel_size // 2, key=keys[0])
        self.decoder = eqx.nn.Conv2d(features, out_channels, kernel_size, padding=kernel_size // 2, key=keys[1])
        self.blocks = [DilatedBlock(features, k) for k in jax.random.split(keys[2], n_blocks)]

    # Residual updates with GELU on all but the last block
    def __call__(self, field):
        hidden = self.encoder(field)
        for index, block in enumerate(self.blocks):
            update = block(hidden)
            hidden = (jax.nn.gelu(update) if index < len(self.blocks) - 1 else update) + hidden
        return self.decoder(hidden)


class ConvStack(eqx.Module):
    convolutions: list

    # Same-width 3x3 convolutions each followed by GELU (upstream ConvNet with N_convs = 4 builds 3)
    def __init__(self, features, key, n_convs=3, kernel_size=3):
        keys = jax.random.split(key, n_convs)
        self.convolutions = [eqx.nn.Conv2d(features, features, kernel_size, padding=kernel_size // 2, key=k) for k in keys]

    # Apply every convolution with GELU
    def __call__(self, hidden):
        for convolution in self.convolutions:
            hidden = jax.nn.gelu(convolution(hidden))
        return hidden


# 2x2 max pooling with stride 2, equal to upstream AdaptivePool(n / 2, max) for even sizes
def max_pool_half(hidden):
    features, n1, n2 = hidden.shape
    return hidden.reshape(features, n1 // 2, 2, n2 // 2, 2).max(axis=(2, 4))


class UNet(eqx.Module):
    lift: eqx.nn.Conv2d
    project: eqx.nn.Conv2d
    down_stacks: list
    widen: list
    up_stacks: list
    narrow: list
    upsample: list

    # U-Net whose features double per level (paper A.4, upstream UNet with depth = 3)
    def __init__(self, features, key, in_channels=1, out_channels=1, depth=3, n_convs=3, kernel_size=3):
        widths = [features * 2 ** level for level in range(depth)]
        keys = jax.random.split(key, 7)
        padding = kernel_size // 2
        self.lift = eqx.nn.Conv2d(in_channels, features, kernel_size, padding=padding, key=keys[0])
        self.project = eqx.nn.Conv2d(features, out_channels, kernel_size, padding=padding, key=keys[1])
        self.down_stacks = [ConvStack(w, k, n_convs) for w, k in zip(widths, jax.random.split(keys[2], depth))]
        self.widen = [eqx.nn.Conv2d(w, 2 * w, kernel_size, padding=padding, key=k) for w, k in zip(widths[:-1], jax.random.split(keys[3], depth - 1))]
        self.up_stacks = [ConvStack(w, k, n_convs) for w, k in zip(widths[::-1], jax.random.split(keys[4], depth))]
        self.narrow = [eqx.nn.Conv2d(2 * w, w, kernel_size, padding=padding, key=k) for w, k in zip(widths[-2::-1], jax.random.split(keys[5], depth - 1))]
        self.upsample = [eqx.nn.ConvTranspose2d(2 * w, w, 2, stride=2, key=k) for w, k in zip(widths[-2::-1], jax.random.split(keys[6], depth - 1))]

    # Encoder path with max pooling, decoder path with transposed convolutions and skip concatenation
    def __call__(self, field):
        skips = [self.down_stacks[0](self.lift(field))]
        for level in range(1, len(self.down_stacks)):
            skips.append(self.down_stacks[level](self.widen[level - 1](max_pool_half(skips[-1]))))
        hidden = self.up_stacks[0](skips[-1])
        for level, (upsample, narrow) in enumerate(zip(self.upsample, self.narrow)):
            hidden = upsample(hidden)
            hidden = self.up_stacks[level + 1](narrow(jnp.concatenate([skips[-2 - level], hidden], 0)))
        return self.project(hidden)
