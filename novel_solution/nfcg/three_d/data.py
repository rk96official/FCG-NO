import os

import numpy as np
import pyamg
from scipy.sparse.linalg import cg

from nfcg.three_d.operator import assemble_csr_3d

MAX_MODE = 4
ALPHA = 2.0
DIFFUSION_SHIFT = 10.0
SOLVE_TOLERANCE = 1e-12


# Random smooth 3D trigonometric polynomial Re(sum c_lmn exp(2 pi i (l x + m y + n z)) / (1 + l + m + n)^alpha), the 3D analogue of the paper's P(4, 4, 2)
def trig_field_3d(rng, max_mode=MAX_MODE, alpha=ALPHA):
    shape = (max_mode + 1,) * 3
    modes = np.arange(max_mode + 1)
    weights = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / (1.0 + modes[:, None, None] + modes[None, :, None] + modes[None, None, :]) ** alpha
    phase = lambda points: np.exp(2j * np.pi * points[:, None] * modes)
    return lambda x, y, z: np.real(np.einsum("lmn,il,jm,kn->ijk", weights, phase(x), phase(y), phase(z)))


# 1D coordinates of the nodes and of the three face sets (x, y, z axes each) of the interior grid with n unknowns per axis; fields are separable in them
def grid_coordinates(n):
    h = 1.0 / (n + 1)
    nodes, faces = np.arange(1, n + 1) * h, (np.arange(n + 1) + 0.5) * h
    return (nodes, nodes, nodes), [(faces, nodes, nodes), (nodes, faces, nodes), (nodes, nodes, faces)]


# One 3D problem: Poisson (a = 1) or Diffusion (a = P + 10) face coefficients and a smooth random right-hand side
def sample_problem_3d(rng, kind, n):
    node_axes, face_axes = grid_coordinates(n)
    field = trig_field_3d(rng)
    coefficient = (lambda x, y, z: np.ones((len(x), len(y), len(z)))) if kind == "poisson" else (lambda x, y, z: field(x, y, z) + DIFFUSION_SHIFT)
    return {"face_x": coefficient(*face_axes[0]), "face_y": coefficient(*face_axes[1]), "face_z": coefficient(*face_axes[2]), "rhs": trig_field_3d(rng)(*node_axes)}


# Reference solution by smoothed-aggregation AMG-preconditioned CG to 1e-12 (CPU)
def reference_solution(problem, normalization):
    matrix = assemble_csr_3d((problem["face_x"], problem["face_y"], problem["face_z"]), normalization)
    preconditioner = pyamg.smoothed_aggregation_solver(matrix).aspreconditioner(cycle="V")
    solution, info = cg(matrix, problem["rhs"].ravel(), rtol=SOLVE_TOLERANCE, M=preconditioner, maxiter=500)
    if info != 0:
        raise RuntimeError(f"reference solve did not converge (info {info})")
    return solution.reshape(problem["rhs"].shape)


# Cached 3D dataset of count problems (with reference solutions when solve is set, as training needs the errors)
def load_or_make_3d(cache_dir, kind, n, count, seed, normalization, solve=False):
    path = os.path.join(cache_dir, f"three_d_{kind}_n{n}_samples{count}_seed{seed}_{normalization}{'_solved' if solve else ''}.npz")
    if os.path.exists(path):
        with np.load(path) as stored:
            return {name: stored[name] for name in stored.files}
    rng = np.random.default_rng(seed)
    problems = [sample_problem_3d(rng, kind, n) for unused in range(count)]
    if solve:
        for problem in problems:
            problem["solution"] = reference_solution(problem, normalization)
    dataset = {name: np.stack([problem[name] for problem in problems]) for name in problems[0]}
    os.makedirs(cache_dir, exist_ok=True)
    temporary = f"{path[:-len('.npz')]}.{os.getpid()}.tmp.npz"
    np.savez(temporary, **dataset)
    os.replace(temporary, path)
    return dataset
