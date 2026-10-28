import sys
import site
import importlib.metadata as metadata

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import scipy.sparse as sp
import scipy.sparse.linalg as spla

jax.config.update("jax_enable_x64", True)

PACKAGES = ["jax", "jaxlib", "equinox", "optax", "numpy", "scipy", "PyYAML", "pandas", "matplotlib"]


# Print installed versions of the core packages
def report_versions():
    print(f"python {sys.version.split()[0]}")
    for name in PACKAGES:
        print(f"{name} {metadata.version(name)}")


# Fail if packages from ~/.local can leak into the env
def check_isolation():
    if site.ENABLE_USER_SITE:
        raise RuntimeError(f"user site-packages {site.getusersitepackages()} is enabled, set PYTHONNOUSERSITE=1")
    print("user site-packages disabled")


# Fail unless JAX runs on a GPU
def check_gpu():
    devices = jax.devices()
    print(f"devices: {devices}")
    if devices[0].platform != "gpu":
        raise RuntimeError("JAX is not using a GPU, check the jax[cuda12] install and the driver")


# Solve a float64 system and round-trip a 2D FFT on the GPU
def check_float64_linalg():
    key = jax.random.PRNGKey(0)
    matrix = jax.random.normal(key, (256, 256)) + 256 * jnp.eye(256)
    rhs = jnp.ones(256)
    solution = jax.jit(jnp.linalg.solve)(matrix, rhs)
    residual = jnp.linalg.norm(matrix @ solution - rhs) / jnp.linalg.norm(rhs)
    assert solution.dtype == jnp.float64, solution.dtype
    assert residual < 1e-12, residual
    field = jax.random.normal(key, (64, 64))
    fft_error = jnp.max(jnp.abs(jnp.fft.irfft2(jnp.fft.rfft2(field), s=field.shape) - field))
    assert fft_error < 1e-12, fft_error
    print(f"float64 solve residual: {residual:.2e}, fft round-trip error: {fft_error:.2e}")


# Take one optimizer step on a small convolution to exercise equinox and optax
def check_training_step():
    key = jax.random.PRNGKey(1)
    model = eqx.nn.Conv2d(1, 1, kernel_size=3, padding=1, key=key)
    inputs = jnp.ones((1, 32, 32))
    optimizer = optax.adam(1e-3)
    state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Compute the loss and apply one Adam update
    @eqx.filter_jit
    def step(model, state):
        loss, grads = eqx.filter_value_and_grad(lambda m: jnp.mean(m(inputs) ** 2))(model)
        updates, state = optimizer.update(grads, state)
        return eqx.apply_updates(model, updates), state, loss

    model, state, loss = step(model, state)
    print(f"training step loss: {float(loss):.4e}")


# Build a small 2D Poisson matrix and check SciPy's ILU (SuperLU) runs
def check_scipy_ilu():
    size = 32
    lap1d = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(size, size))
    identity = sp.identity(size)
    matrix = (sp.kron(lap1d, identity) + sp.kron(identity, lap1d)).tocsc()
    ilu = spla.spilu(matrix, fill_factor=1)
    print(f"scipy ilu nnz(L+U): {ilu.L.nnz + ilu.U.nnz}")


if __name__ == "__main__":
    check_isolation()
    report_versions()
    check_gpu()
    check_float64_linalg()
    check_training_step()
    check_scipy_ilu()
    print("ENVIRONMENT OK")
