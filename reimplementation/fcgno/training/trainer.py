import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jax import lax

from fcgno.training.losses import LOSSES


# Learning rate multiplied by decay_factor every decay_epochs epochs (paper Table 8, upstream train_model)
def learning_rate_schedule(learning_rate, steps_per_epoch, decay_epochs=50, decay_factor=0.5, max_epochs=1000):
    boundaries = {epoch * steps_per_epoch: decay_factor for epoch in range(decay_epochs, max_epochs, decay_epochs)}
    return optax.piecewise_constant_schedule(learning_rate, boundaries)


# Cast every floating-point array of a pytree to dtype, leaving integer arrays and static fields unchanged
def cast_floats(tree, dtype):
    return jax.tree_util.tree_map(lambda leaf: leaf.astype(dtype) if eqx.is_inexact_array(leaf) else leaf, tree)


# One training batch: per-pair arrays by index, per-problem arrays through the problem that owns each pair
def gather_batch(data, indices):
    owners = data["owner"][indices]
    batch = {name: values[indices] for name, values in data["pairs"].items()}
    batch.update({name: values[owners] for name, values in data["problems"].items()})
    return batch


# Train on data = {"pairs", "problems", "owner"} with AdamW in settings["precision"] (default float64); returns a float64 model
def train(model, data, loss_name, settings, normalization, seed=0, log=print):
    if settings.get("precision", "float64") == "float32":
        model, data = cast_floats(model, jnp.float32), cast_floats(data, jnp.float32)
    loss_function = LOSSES[loss_name]
    n_pairs = len(data["owner"])
    steps_per_epoch = max(1, n_pairs // settings["batch_size"])
    schedule = learning_rate_schedule(settings["learning_rate"], steps_per_epoch, settings["decay_epochs"], settings["decay_factor"])
    optimizer = optax.adamw(schedule, weight_decay=settings["weight_decay"])
    params, static = eqx.partition(model, eqx.is_array)
    opt_state = optimizer.init(params)

    # Scan AdamW steps over the batches of one epoch on the device
    @jax.jit
    def run_epoch(params, opt_state, data, batches):
        # One AdamW step on one batch of indices
        def step(carry, indices):
            params, opt_state = carry
            objective = lambda trainable: loss_function(eqx.combine(trainable, static), gather_batch(data, indices), normalization)
            loss, grads = jax.value_and_grad(objective)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), loss

        (params, opt_state), losses = lax.scan(step, (params, opt_state), batches)
        return params, opt_state, losses

    device_data = jax.device_put(data)
    key = jax.random.PRNGKey(seed)
    history = []
    for epoch in range(settings["epochs"]):
        key, batch_key = jax.random.split(key)
        batches = jax.random.choice(batch_key, n_pairs, (steps_per_epoch, settings["batch_size"]))
        params, opt_state, losses = run_epoch(params, opt_state, device_data, batches)
        history.append(float(losses.mean()))
        if epoch % 10 == 0 or epoch == settings["epochs"] - 1 or not np.isfinite(history[-1]):
            log(f"epoch {epoch}: {loss_name} loss {history[-1]:.4e}")
        if not np.isfinite(history[-1]):
            break
    return cast_floats(eqx.combine(params, static), jnp.float64), np.array(history)
