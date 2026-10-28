import json
import os

import jax
import equinox as eqx

from fcgno.models.registry import build_model


# Save model weights plus the spec needed to rebuild it, writing atomically so parallel jobs never see partial files
def save_model(path, model, spec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.{os.getpid()}.tmp"
    eqx.tree_serialise_leaves(temporary + ".eqx", model)
    with open(temporary + ".json", "w") as handle:
        json.dump(spec, handle, indent=2)
    os.replace(temporary + ".json", path + ".json")
    os.replace(temporary + ".eqx", path + ".eqx")


# Rebuild a saved model from its spec and weights; returns (model, spec) or None if no checkpoint exists
def load_model(path):
    if not (os.path.exists(path + ".eqx") and os.path.exists(path + ".json")):
        return None
    with open(path + ".json") as handle:
        spec = json.load(handle)
    skeleton = build_model(spec["model"], jax.random.PRNGKey(0), features=spec["features"])
    return eqx.tree_deserialise_leaves(path + ".eqx", skeleton), spec
