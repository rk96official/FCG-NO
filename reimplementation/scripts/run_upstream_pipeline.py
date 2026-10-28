import argparse
import os
import runpy
import sys

import numpy as np
import jax
import jax.experimental.sparse.linalg

UPSTREAM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_FOLDERS = ("Poisson/SNO", "Elliptic/SNO")


# jnp.ravel_multi_index with int64 only when int32 cannot hold the flattened index (BCOO reshape of 256^2 x 256^2 matrices in current JAX)
def ravel_multi_index_64(original):
    # Same call, switching to a 64-bit dtype only for index spaces beyond int32
    def wrapped(multi_index, dims, *args, **kwargs):
        if "dtype" not in kwargs and np.prod(np.array(dims, dtype=np.float64)) > np.iinfo(np.int32).max:
            kwargs["dtype"] = jax.numpy.int64
        return original(multi_index, dims, *args, **kwargs)

    return wrapped


# jax.experimental.sparse.linalg.spsolve with int32 CSR indices, which its cuSOLVER kernel requires
def spsolve_int32(original):
    # Same solve with indices and indptr cast to int32 (values stay far below 2^31)
    def wrapped(data, indices, indptr, b, *args, **kwargs):
        return original(data, indices.astype(jax.numpy.int32), indptr.astype(jax.numpy.int32), b, *args, **kwargs)

    return wrapped


# Run one unmodified upstream pipeline script from a work directory, restoring jax.tree_map/tree_leaves removed in recent JAX
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True, help="file name in the upstream pipelines/ folder")
    parser.add_argument("--workdir", required=True)
    args, pipeline_args = parser.parse_known_args()
    jax.tree_map = jax.tree_util.tree_map
    jax.tree_leaves = jax.tree_util.tree_leaves
    jax.numpy.ravel_multi_index = ravel_multi_index_64(jax.numpy.ravel_multi_index)
    jax.experimental.sparse.linalg.spsolve = spsolve_int32(jax.experimental.sparse.linalg.spsolve)
    for folder in OUTPUT_FOLDERS:
        os.makedirs(os.path.join(args.workdir, folder), exist_ok=True)
    os.chdir(args.workdir)
    sys.path.insert(0, UPSTREAM_ROOT)
    script = os.path.join(UPSTREAM_ROOT, "pipelines", args.pipeline)
    sys.argv = [script] + pipeline_args
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
