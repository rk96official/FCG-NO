import sys
import importlib
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from fcgno.models.spectral import real_fourier_analysis, real_fourier_synthesis
from fcgno.models.sno import SNO
from fcgno.models.fno import FNO
from fcgno.models.convnets import DilResNet, UNet
from fcgno.models.registry import MODEL_CLASSES, TABLE7_FEATURES, build_model, as_preconditioner, count_parameters

UPSTREAM_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_KEY = jax.random.PRNGKey(7)
N, FEATURES = 32, 8


# Import an upstream module read-only, with a jax.tree_map shim for old code; skip if it cannot load
def import_upstream(name):
    if not hasattr(jax, "tree_map"):
        jax.tree_map = jax.tree_util.tree_map
    sys.path.insert(0, str(UPSTREAM_ROOT))
    try:
        return importlib.import_module(name)
    except Exception as error:
        pytest.skip(f"upstream {name} cannot be imported: {error}")
    finally:
        sys.path.remove(str(UPSTREAM_ROOT))


# Replace parts of our model with upstream submodules or arrays
def transplant(model, replacements):
    for where, value in replacements:
        model = eqx.tree_at(where, model, value)
    return model


# A random (1, N, N) input field
def random_field(n=N, seed=3):
    return jax.random.normal(jax.random.PRNGKey(seed), (1, n, n))


# Our real Fourier analysis and synthesis matrices equal the upstream transforms
@pytest.mark.parametrize("n", [32, 64])
def test_spectral_matrices_match_upstream(n):
    utilities = import_upstream("transforms.utilities")
    spec = dict(polynomials=["Real_Fourier"], parameters=[[0.1, 0.1]], grids=[n], M_keep=[20], N_points=[n])
    np.testing.assert_allclose(real_fourier_analysis(n, 20), utilities.get_operators("analysis", **spec)[0], atol=1e-14)
    np.testing.assert_allclose(real_fourier_synthesis(n, 20), utilities.get_operators("synthesis", **spec)[0], atol=1e-14)


# SNO with upstream weights reproduces upstream fSNO with the get_SNO spectral cell
def test_sno_matches_upstream():
    fsno = import_upstream("architectures.fSNO")
    dilated = import_upstream("architectures.DilResNet")
    utilities = import_upstream("transforms.utilities")
    cell = lambda features, key: dilated.DilatedConvBlock([features] * 4, [[1, 1]] * 3, [[3, 3]] * 3, key, activation=lambda x: x)
    upstream = fsno.fSNO((1, N, N), 1, 4, FEATURES, cell, UPSTREAM_KEY)
    replacements = [(lambda m: m.encoder, upstream.encoder.convolutions[0]), (lambda m: m.decoder, upstream.decoder.convolutions[0])]
    for i in range(4):
        replacements.append((lambda m, i=i: m.skips[i], upstream.conv[i]))
        replacements += [(lambda m, i=i, j=j: m.cells[i].convolutions[j], upstream.processor[i].convolutions[j]) for j in range(3)]
    ours = transplant(SNO(FEATURES, jax.random.PRNGKey(0)), replacements)
    spec = dict(polynomials=["Real_Fourier"] * 2, parameters=[[0.1, 0.1]] * 2, grids=[N, N], M_keep=[20, 20], N_points=[N, N])
    expected = upstream(random_field(), utilities.get_operators("analysis", **spec), utilities.get_operators("synthesis", **spec))
    np.testing.assert_allclose(ours(random_field()), expected, rtol=1e-10, atol=1e-10)


# FNO with upstream weights reproduces upstream FNO2D, including its mode-update order
def test_fno_matches_upstream():
    fno = import_upstream("architectures.FNO")
    upstream = fno.FNO2D(4, [3, FEATURES, 1], 16, UPSTREAM_KEY)
    split = lambda weights: jnp.stack([weights.real, weights.imag])
    replacements = [(lambda m: m.encoder, upstream.encoder_decoder[0]), (lambda m: m.decoder, upstream.encoder_decoder[1])]
    for i in range(4):
        replacements += [
            (lambda m, i=i: m.layers[i].skip, upstream.convs[i]),
            (lambda m, i=i: m.layers[i].positive, split(upstream.Ap[i])),
            (lambda m, i=i: m.layers[i].negative, split(upstream.An[i])),
            (lambda m, i=i: m.layers[i].zero_positive, upstream.Dp[i]),
            (lambda m, i=i: m.layers[i].zero_negative, upstream.Dn[i]),
        ]
    ours = transplant(FNO(FEATURES, jax.random.PRNGKey(0)), replacements)
    np.testing.assert_allclose(ours(random_field()), upstream(random_field()), rtol=1e-10, atol=1e-10)


# DilResNet with upstream weights reproduces upstream DilatedResNet
def test_dilresnet_matches_upstream():
    dilated = import_upstream("architectures.DilResNet")
    upstream = dilated.DilatedResNet(UPSTREAM_KEY, [1, FEATURES, 1], 4, kernel_size=3, D=2)
    replacements = [(lambda m: m.encoder, upstream.encoder.convolutions[0]), (lambda m: m.decoder, upstream.decoder.convolutions[0])]
    for i in range(4):
        replacements += [(lambda m, i=i, j=j: m.blocks[i].convolutions[j], upstream.processor[i].convolutions[j]) for j in range(7)]
    ours = transplant(DilResNet(FEATURES, jax.random.PRNGKey(0)), replacements)
    np.testing.assert_allclose(ours(random_field()), upstream(random_field()), rtol=1e-10, atol=1e-10)


# U-Net with upstream weights reproduces upstream UNet with depth 3
def test_unet_matches_upstream():
    unet = import_upstream("architectures.UNet")
    upstream = unet.UNet(2, N, [1, FEATURES, 1], 3, 4, UPSTREAM_KEY, depth=3)
    replacements = [(lambda m: m.lift, upstream.transofrm_convs[0]), (lambda m: m.project, upstream.transofrm_convs[1])]
    for i in range(3):
        replacements += [(lambda m, i=i, j=j: m.down_stacks[i].convolutions[j], upstream.before_convs[i].convs[j]) for j in range(3)]
        replacements += [(lambda m, i=i, j=j: m.up_stacks[i].convolutions[j], upstream.after_convs[i].convs[j]) for j in range(3)]
    for i in range(2):
        replacements += [
            (lambda m, i=i: m.widen[i], upstream.after_d_convs[i]),
            (lambda m, i=i: m.narrow[i], upstream.right_after_convs[i]),
            (lambda m, i=i: m.upsample[i], upstream.conv_t[i]),
        ]
    ours = transplant(UNet(FEATURES, jax.random.PRNGKey(0)), replacements)
    np.testing.assert_allclose(ours(random_field()), upstream(random_field()), rtol=1e-10, atol=1e-10)


# Every model builds from Table 7 and maps (1, n, n) to a finite float64 (1, n, n) field
@pytest.mark.parametrize("name", sorted(MODEL_CLASSES))
@pytest.mark.parametrize("kind", sorted(TABLE7_FEATURES))
def test_table7_models_run(name, kind):
    model = build_model(name, jax.random.PRNGKey(0), kind=kind, grid=32)
    output = eqx.filter_jit(lambda module, field: module(field))(model, random_field())
    assert output.shape == (1, N, N) and output.dtype == jnp.float64
    assert np.all(np.isfinite(output)) and count_parameters(model) > 0


# Neural operators trained on one grid apply unchanged to finer grids
@pytest.mark.parametrize("name", ["sno", "fno"])
def test_operators_change_resolution(name):
    model = build_model(name, jax.random.PRNGKey(0), features=FEATURES)
    for n in (32, 64, 128):
        assert model(random_field(n)).shape == (1, n, n)


# Preconditioner wrapper is positively homogeneous and safe on zero residuals
def test_preconditioner_wrapper():
    apply = as_preconditioner(build_model("sno", jax.random.PRNGKey(0), features=FEATURES))
    residual = jax.random.normal(jax.random.PRNGKey(4), (2, N, N))
    np.testing.assert_allclose(apply(3.0 * residual), 3.0 * apply(residual), rtol=1e-10, atol=1e-12)
    assert np.all(np.isfinite(apply(jnp.zeros((1, N, N)))))
