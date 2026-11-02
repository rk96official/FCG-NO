import contextlib

import jax
import jax.numpy as jnp

from fcgno.models.registry import as_preconditioner
from fcgno.models.fno import FNO
from fcgno.models.sno import SNO
from fcgno.models.spectral import apply_separable, real_fourier_analysis, real_fourier_synthesis
from fcgno.training.trainer import cast_floats

DTYPES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}


# SNO forward pass (same as fcgno SNO.__call__) with the Fourier matrices in dtype; fcgno builds them in float64, which would promote the whole pass
def sno_forward(model, field, dtype):
    analysis = [real_fourier_analysis(n, model.n_modes).astype(dtype) for n in field.shape[1:]]
    synthesis = [real_fourier_synthesis(n, model.n_modes).astype(dtype) for n in field.shape[1:]]
    hidden = model.encoder(field)
    for index, (cell, skip) in enumerate(zip(model.cells, model.skips)):
        hidden = apply_separable(cell(apply_separable(hidden, *analysis)), *synthesis) + skip(hidden)
        if index < len(model.cells) - 1:
            hidden = jax.nn.gelu(hidden)
    return model.decoder(hidden)


# FNO forward pass (same as fcgno FNO.__call__) with the coordinate channels in dtype; fcgno builds them in float64, which would promote the input
def fno_forward(model, field, dtype):
    line = jnp.linspace(0.0, 1.0, field.shape[-1], dtype=dtype)
    hidden = model.encoder(jnp.concatenate([jnp.stack(jnp.meshgrid(line, line), 0), field], 0))
    for index, layer in enumerate(model.layers):
        hidden = layer(hidden)
        if index < len(model.layers) - 1:
            hidden = jax.nn.relu(hidden)
    return model.decoder(hidden)


# Per-sample model callable (channels, n, n) -> output that runs the network in a lower precision and returns the input's precision.
# matmul_precision "highest" forces true float32 matmuls, None keeps the XLA default (TF32 on H100)
def low_precision_model(model, dtype, matmul_precision=None):
    cast_model = cast_floats(model, DTYPES[dtype])
    if isinstance(model, SNO):
        forward = lambda field: sno_forward(cast_model, field, DTYPES[dtype])
    elif isinstance(model, FNO):
        forward = lambda field: fno_forward(cast_model, field, DTYPES[dtype])
    else:
        forward = cast_model

    # Cast in, run under the requested matmul precision, cast back
    def apply(field):
        context = jax.default_matmul_precision(matmul_precision) if matmul_precision else contextlib.nullcontext()
        with context:
            return forward(field.astype(DTYPES[dtype])).astype(field.dtype)

    return apply


# Preconditioner B(r) = ||r|| NO(r / ||r||) with only the network in a lower precision; norms and output stay in the residual's precision
def low_precision_preconditioner(model, dtype, matmul_precision=None):
    return as_preconditioner(low_precision_model(model, dtype, matmul_precision))
