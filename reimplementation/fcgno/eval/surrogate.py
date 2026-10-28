import numpy as np
from scipy.interpolate import RegularGridInterpolator


# Interpolate fine-grid solutions (zero Dirichlet boundary) to the interior nodes of an n x n grid with cubic splines
def restrict_to_grid(fine_solutions, n):
    fine_nodes = np.linspace(0.0, 1.0, fine_solutions.shape[-1] + 2)
    coarse_nodes = np.linspace(0.0, 1.0, n + 2)[1:-1]
    points = np.stack(np.meshgrid(coarse_nodes, coarse_nodes, indexing="ij"), axis=-1)
    restricted = [RegularGridInterpolator((fine_nodes, fine_nodes), np.pad(solution, 1), method="cubic")(points) for solution in fine_solutions]
    return np.stack(restricted)


# Relative L2 error of each prediction against its reference, as an array over samples
def relative_errors(predictions, references):
    flatten = lambda fields: np.asarray(fields).reshape(len(fields), -1)
    return np.linalg.norm(flatten(predictions) - flatten(references), axis=1) / np.linalg.norm(flatten(references), axis=1)
