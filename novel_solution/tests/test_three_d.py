import numpy as np
import jax
import jax.numpy as jnp

from fcgno.linalg.krylov import fcg
from fcgno.training.losses import LOSSES
from fcgno.training.trainer import train
from nfcg.three_d.data import grid_coordinates, load_or_make_3d
from nfcg.three_d.model import SNO3D
from nfcg.three_d.operator import apply_stencil_3d, assemble_csr_3d, coarse_faces_3d
from nfcg.three_d.solver import FACE_NAMES, classical_preconditioner_3d, collect_records_3d, make_step_loss_3d

NORMALIZATION = "upstream"


# Faces of a smooth positive coefficient on an n-grid
def smooth_faces(n):
    coefficient = lambda x, y, z: np.exp(np.sin(2.0 * x) * np.cos(3.0 * y) + 0.5 * z)
    unused_nodes, face_axes = grid_coordinates(n)
    return tuple(jnp.asarray(coefficient(*np.meshgrid(*axes, indexing="ij"))) for axes in face_axes), coefficient


# The matrix-free 7-point operator equals the assembled CSR matrix
def test_operator_matches_csr():
    n = 5
    faces, unused = smooth_faces(n)
    field = np.random.default_rng(0).standard_normal((n, n, n))
    expected = assemble_csr_3d(faces, NORMALIZATION) @ field.ravel()
    np.testing.assert_allclose(np.asarray(apply_stencil_3d(faces, jnp.asarray(field), NORMALIZATION)).ravel(), expected, rtol=1e-12)


# Rediscretized faces of a smooth coefficient match the coefficient on the coarse faces
def test_coarse_faces_sample_the_coefficient():
    faces, coefficient = smooth_faces(32)
    unused_nodes, coarse_axes = grid_coordinates(8)
    for sampled, axes in zip(coarse_faces_3d(faces, 8), coarse_axes):
        np.testing.assert_allclose(np.asarray(sampled), coefficient(*np.meshgrid(*axes, indexing="ij")), rtol=1e-2)


# Rediscretized 3D multigrid with a dense bottom converges in a handful of iterations; Chebyshev-PCG needs many more
def test_classical_multilevel_3d_converges(tmp_path):
    dataset = load_or_make_3d(str(tmp_path), "diffusion", 16, 2, seed=0, normalization=NORMALIZATION)
    faces = tuple(jnp.asarray(dataset[name]) for name in FACE_NAMES)
    operator = lambda u: apply_stencil_3d(faces, u, NORMALIZATION)
    rhs, guess = jnp.asarray(dataset["rhs"]), jnp.asarray(np.random.default_rng(1).standard_normal(dataset["rhs"].shape))
    count = lambda variant: max(int(np.argmax(row / row[0] <= 1e-6)) for row in np.asarray(
        fcg(operator, classical_preconditioner_3d(faces, NORMALIZATION, variant), rhs, guess, 100, 20)["residual_norm"]))
    multilevel = count({"method": "multilevel", "sizes": (16, 8, 4), "bottom_solver": "dense"})
    chebyshev = count({"method": "chebyshev", "sweeps": 8})
    assert multilevel <= 10 and multilevel < chebyshev


# On-policy 3D records through the multi-level scheme and one epoch of step-loss training run with finite losses
def test_three_d_training_round(tmp_path):
    dataset = load_or_make_3d(str(tmp_path), "diffusion", 8, 2, seed=1, normalization=NORMALIZATION, solve=True)
    model = SNO3D(4, jax.random.PRNGKey(0), in_channels=3, n_layers=2, n_modes=4)
    guesses = np.random.default_rng(2).standard_normal(dataset["rhs"].shape)
    records = collect_records_3d(model, 1, dataset, guesses, 4, 20, NORMALIZATION, 2, (8, 4))
    LOSSES["three_d_test"] = make_step_loss_3d(1, (8, 4), 0.05)
    data = {"pairs": {key: records[key] for key in ("residual", "error", "updates", "changes")}, "problems": {name: dataset[name] for name in FACE_NAMES},
            "owner": records["owner"]}
    settings = {"learning_rate": 1e-3, "decay_epochs": 50, "decay_factor": 0.5, "weight_decay": 1e-2, "epochs": 1, "batch_size": 4}
    unused_model, curve = train(model, data, "three_d_test", settings, NORMALIZATION, log=lambda message: None)
    assert len(records["owner"]) == 8 and np.all(np.isfinite(curve))
