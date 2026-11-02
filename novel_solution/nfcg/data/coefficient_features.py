import numpy as np


# Standardized log coefficient per problem, (log a - mean) / max(std, 1), on the interior nodes: unchanged for mild coefficients (std < 1, e.g. Diffusion),
# unit scale for high contrast and jumps; zero for Poisson
def log_coefficient(dataset):
    logarithm = np.log(dataset["coefficient"])
    centered = logarithm - logarithm.mean(axis=(-2, -1), keepdims=True)
    return centered / np.maximum(centered.std(axis=(-2, -1), keepdims=True), 1.0)


# Contrast max(a) / min(a) of each problem's coefficient on the cell faces
def coefficient_contrast(dataset):
    faces = [dataset["face_x"].reshape(len(dataset["face_x"]), -1), dataset["face_y"].reshape(len(dataset["face_y"]), -1)]
    values = np.concatenate(faces, axis=1)
    return values.max(axis=1) / values.min(axis=1)
