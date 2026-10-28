import time

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import lax

from fcgno.data.assembly import apply_stencil
from fcgno.linalg.krylov import fcg_init, fcg_step, cg_init, cg_step
from fcgno.models.registry import as_preconditioner


# Jitted solver running a traced number of steps: FCG with the model; with model None, plain CG or (paper-style) FCG with B = I
def make_runner(model, normalization, m_max, identity_fcg=False):
    # Run n_steps iterations without recording history and return the final iterate
    @eqx.filter_jit
    def run(model, face_x, face_y, rhs, guess, n_steps):
        operator = lambda u: apply_stencil(face_x, face_y, u, normalization)
        if model is None and not identity_fcg:
            state = lax.fori_loop(0, n_steps, lambda _, current: cg_step(current, operator), cg_init(operator, rhs, guess))
        else:
            precondition = (lambda r: r) if model is None else as_preconditioner(model)
            step = lambda _, current: fcg_step(current, operator, precondition, m_max)[0]
            state = lax.fori_loop(0, n_steps, step, fcg_init(operator, rhs, guess, m_max))
        return state.iterate

    return lambda *args: run(model, *args)


# Paper-style eager solver: a Python loop calling one jitted FCG step per iteration (B = I when model is None)
def make_eager_runner(model, normalization, m_max):
    # One jitted FCG step for given operator data
    @eqx.filter_jit
    def step(model, state, face_x, face_y):
        operator = lambda u: apply_stencil(face_x, face_y, u, normalization)
        precondition = (lambda r: r) if model is None else as_preconditioner(model)
        return fcg_step(state, operator, precondition, m_max)[0]

    # Initialize, then loop in Python over n_steps jitted steps
    def run(face_x, face_y, rhs, guess, n_steps):
        state = fcg_init(lambda u: apply_stencil(face_x, face_y, u, normalization), rhs, guess, m_max)
        for _ in range(int(n_steps)):
            state = step(model, state, face_x, face_y)
        return state.iterate

    return run


# Median wall-clock seconds over repeats after one warm-up call, synchronizing with block_until_ready
def median_seconds(function, args, repeats=5):
    jax.block_until_ready(function(*args))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(function(*args))
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


# Paper Tables 2 and 9: GPU time for FCG-NO, plain CG and FCG with B = I (the upstream CG) to reach each tolerance, with savings
def time_to_tolerances(model, dataset, guesses, normalization, m_max, fcg_iterations, cg_iterations, repeats=5):
    args = [jnp.asarray(dataset[name]) for name in ("face_x", "face_y", "rhs")] + [jnp.asarray(guesses)]
    fcg_runner = make_runner(model, normalization, m_max)
    cg_runner = make_runner(None, normalization, m_max)
    identity_runner = make_runner(None, normalization, m_max, identity_fcg=True)
    eager_fcg_runner = make_eager_runner(model, normalization, m_max)
    eager_identity_runner = make_eager_runner(None, normalization, m_max)
    rows = {}
    for tol, fcg_steps in fcg_iterations.items():
        cg_steps = cg_iterations[tol]
        if fcg_steps < 0 or cg_steps < 0:
            rows[tol] = None
            continue
        fcg_seconds = median_seconds(fcg_runner, args + [jnp.asarray(fcg_steps)], repeats)
        cg_seconds = median_seconds(cg_runner, args + [jnp.asarray(cg_steps)], repeats)
        identity_seconds = median_seconds(identity_runner, args + [jnp.asarray(cg_steps)], repeats)
        eager_fcg_seconds = median_seconds(eager_fcg_runner, args + [fcg_steps], repeats)
        eager_identity_seconds = median_seconds(eager_identity_runner, args + [cg_steps], repeats)
        rows[tol] = {
            "fcg_seconds": fcg_seconds, "cg_seconds": cg_seconds, "identity_fcg_seconds": identity_seconds,
            "eager_fcg_seconds": eager_fcg_seconds, "eager_identity_fcg_seconds": eager_identity_seconds,
            "fcg_iterations": fcg_steps, "cg_iterations": cg_steps,
            "saving_percent": 100.0 * (1.0 - fcg_seconds / cg_seconds),
            "saving_vs_identity_fcg_percent": 100.0 * (1.0 - fcg_seconds / identity_seconds),
            "saving_eager_percent": 100.0 * (1.0 - eager_fcg_seconds / eager_identity_seconds),
        }
    return rows
