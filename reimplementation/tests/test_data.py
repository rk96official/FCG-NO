import itertools

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from fcgno.data.assembly import assemble_csr, interior_mesh
from fcgno.data.coefficients import (
    COEFFICIENT_SAMPLERS, JUMP_LOCATION, JUMP_SCALE, sample_trig_coefficients, trig_polynomial,
    jump_coefficient_a1, jump_coefficient_a2,
)
from fcgno.data.generate import make_dataset, load_or_make_dataset


# Vectorized polynomial equals the double loop used in upstream random_polynomial_2D
def test_trig_polynomial_matches_loop():
    coefficients = sample_trig_coefficients(jax.random.PRNGKey(0))
    mesh_x, mesh_y = interior_mesh(10)
    reference = 0
    for m, n in itertools.product(range(coefficients.shape[0]), repeat=2):
        reference += coefficients[m, n] * jnp.exp(2j * jnp.pi * (m * mesh_x + n * mesh_y)) / (1 + m + n) ** 2
    np.testing.assert_allclose(trig_polynomial(coefficients, mesh_x, mesh_y), jnp.real(reference), rtol=1e-12, atol=1e-12)


# a1 is 1 right of the jump and a constant eps in (0, 0.1) left of it
def test_jump_coefficient_a1():
    mesh_x, mesh_y = interior_mesh(32)
    values = np.asarray(jump_coefficient_a1(jax.random.PRNGKey(4))(mesh_x, mesh_y))
    right = np.asarray(mesh_x) >= JUMP_LOCATION
    assert np.all(values[right] == 1.0)
    assert np.unique(values[~right]).size == 1
    assert 0 < values[~right][0] < JUMP_SCALE


# a2 is constant eps left of the jump and smooth around 1 right of it
def test_jump_coefficient_a2():
    mesh_x, mesh_y = interior_mesh(32)
    values = np.asarray(jump_coefficient_a2(jax.random.PRNGKey(5))(mesh_x, mesh_y))
    right = np.asarray(mesh_x) >= JUMP_LOCATION
    assert np.unique(values[~right]).size == 1
    assert 0 < values[~right][0] < JUMP_SCALE
    assert np.unique(values[right]).size > 1 and values[right].min() > 0


# Datasets have consistent shapes and the stored solution solves A u = f
@pytest.mark.parametrize("kind", sorted(COEFFICIENT_SAMPLERS))
def test_dataset_solution(kind):
    n, n_samples = 16, 3
    dataset = make_dataset(kind, n, n_samples, seed=0)
    assert dataset["face_x"].shape == (n_samples, n + 1, n)
    assert dataset["face_y"].shape == (n_samples, n, n + 1)
    for name in ("coefficient", "rhs", "solution"):
        assert dataset[name].shape == (n_samples, n, n) and dataset[name].dtype == np.float64
    for face_x, face_y, rhs, solution in zip(dataset["face_x"], dataset["face_y"], dataset["rhs"], dataset["solution"]):
        residual = assemble_csr(face_x, face_y) @ solution.ravel() - rhs.ravel()
        assert np.linalg.norm(residual) / np.linalg.norm(rhs) < 1e-10


# Different seeds give different samples and the same seed is reproducible
def test_dataset_seeds():
    first = make_dataset("diffusion", 8, 2, seed=0)
    again = make_dataset("diffusion", 8, 2, seed=0)
    other = make_dataset("diffusion", 8, 2, seed=1)
    np.testing.assert_array_equal(first["rhs"], again["rhs"])
    assert not np.allclose(first["rhs"], other["rhs"])
    assert not np.allclose(first["rhs"][0], first["rhs"][1])


# Cached dataset round-trips through disk unchanged
def test_dataset_cache(tmp_path):
    created = load_or_make_dataset(str(tmp_path), "a2", 8, 2, seed=7)
    loaded = load_or_make_dataset(str(tmp_path), "a2", 8, 2, seed=7)
    assert len(list(tmp_path.iterdir())) == 1
    for name in created:
        np.testing.assert_array_equal(created[name], loaded[name])


# Paper-text data option: u is sampled, f = A u holds exactly, and the cache keeps both variants apart
def test_sample_solution(tmp_path):
    from fcgno.data.generate import dataset_path
    dataset = make_dataset("diffusion", 12, 2, seed=3, max_mode=5, normalization="upstream", sample="solution")
    for face_x, face_y, rhs, solution in zip(dataset["face_x"], dataset["face_y"], dataset["rhs"], dataset["solution"]):
        residual = assemble_csr(face_x, face_y, "upstream") @ solution.ravel() - rhs.ravel()
        assert np.linalg.norm(residual) <= 1e-12 * np.linalg.norm(rhs)
    assert "_samplesolution" in dataset_path("c", "poisson", 32, 4, 0, sample="solution")
    assert "_samplesolution" not in dataset_path("c", "poisson", 32, 4, 0)
    with pytest.raises(ValueError):
        make_dataset("poisson", 8, 1, seed=0, sample="unknown")
