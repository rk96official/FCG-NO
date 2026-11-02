import json
import os

import jax
import equinox as eqx

from nfcg.models.history import build_history_sno


# History-model checkpoint (saved with fcgno's save_model; the spec holds features and history) as (model, spec), or None if missing
def load_history_model(path):
    if not (os.path.exists(path + ".eqx") and os.path.exists(path + ".json")):
        return None
    with open(path + ".json") as handle:
        spec = json.load(handle)
    skeleton = build_history_sno(spec["features"], jax.random.PRNGKey(0), spec["history"], spec.get("coefficient", False))
    return eqx.tree_deserialise_leaves(path + ".eqx", skeleton), spec
