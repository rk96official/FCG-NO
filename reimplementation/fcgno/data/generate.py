import os

import numpy as np
import jax
import scipy.sparse.linalg as spla

from fcgno.data.assembly import DEFAULT_NORMALIZATION, interior_mesh, face_coefficients, assemble_csr
from fcgno.data.coefficients import COEFFICIENT_SAMPLERS, DEFAULT_MAX_MODE, DEFAULT_ALPHA, sample_rhs


# One problem: face coefficients, nodal coefficient and right-hand side on the interior mesh
def sample_problem(key, kind, n, max_mode=DEFAULT_MAX_MODE, alpha=DEFAULT_ALPHA):
    coefficient_key, rhs_key = jax.random.split(key)
    coefficient = COEFFICIENT_SAMPLERS[kind](coefficient_key, max_mode, alpha)
    rhs = sample_rhs(rhs_key, max_mode, alpha)
    mesh_x, mesh_y = interior_mesh(n)
    face_x, face_y = face_coefficients(coefficient, n)
    return {"face_x": face_x, "face_y": face_y, "coefficient": coefficient(mesh_x, mesh_y), "rhs": rhs(mesh_x, mesh_y)}


# Exact discrete solution of A u = f with a sparse direct solver on the CPU
def exact_solution(face_x, face_y, rhs, normalization=DEFAULT_NORMALIZATION):
    matrix = assemble_csr(face_x, face_y, normalization).tocsc()
    return spla.spsolve(matrix, np.asarray(rhs).ravel()).reshape(np.shape(rhs))


# Generate n_samples problems with exact solutions; sample="rhs" draws f (upstream code), "solution" draws u and sets f = A u (paper text)
def make_dataset(kind, n, n_samples, seed, max_mode=DEFAULT_MAX_MODE, alpha=DEFAULT_ALPHA, normalization=DEFAULT_NORMALIZATION, sample="rhs"):
    keys = jax.random.split(jax.random.PRNGKey(seed), n_samples)
    batch = jax.vmap(lambda key: sample_problem(key, kind, n, max_mode, alpha))(keys)
    dataset = {name: np.asarray(values) for name, values in batch.items()}
    if min(dataset["face_x"].min(), dataset["face_y"].min()) <= 0:
        raise ValueError(f"non-positive coefficient sampled for kind={kind}, seed={seed}")
    samples = zip(dataset["face_x"], dataset["face_y"], dataset["rhs"])
    if sample == "solution":
        dataset["solution"] = dataset["rhs"]
        dataset["rhs"] = np.stack([(assemble_csr(face_x, face_y, normalization) @ field.ravel()).reshape(field.shape) for face_x, face_y, field in samples])
    elif sample == "rhs":
        dataset["solution"] = np.stack([exact_solution(face_x, face_y, rhs, normalization) for face_x, face_y, rhs in samples])
    else:
        raise ValueError(f"unknown sample mode: {sample}")
    return dataset


# Cache file name that encodes every generation parameter
def dataset_path(cache_dir, kind, n, n_samples, seed, max_mode=DEFAULT_MAX_MODE, alpha=DEFAULT_ALPHA, normalization=DEFAULT_NORMALIZATION, sample="rhs"):
    suffix = "" if sample == "rhs" else f"_sample{sample}"
    return os.path.join(cache_dir, f"{kind}_n{n}_samples{n_samples}_seed{seed}_mode{max_mode}_alpha{alpha:g}_{normalization}{suffix}.npz")


# Load a cached dataset, or generate it and save it to the cache
def load_or_make_dataset(cache_dir, kind, n, n_samples, seed, max_mode=DEFAULT_MAX_MODE, alpha=DEFAULT_ALPHA, normalization=DEFAULT_NORMALIZATION, sample="rhs"):
    path = dataset_path(cache_dir, kind, n, n_samples, seed, max_mode, alpha, normalization, sample)
    if os.path.exists(path):
        with np.load(path) as stored:
            return {name: stored[name] for name in stored.files}
    dataset = make_dataset(kind, n, n_samples, seed, max_mode, alpha, normalization, sample)
    os.makedirs(cache_dir, exist_ok=True)
    temporary = f"{path[:-len('.npz')]}.{os.getpid()}.tmp.npz"
    np.savez(temporary, **dataset)
    os.replace(temporary, path)
    return dataset
