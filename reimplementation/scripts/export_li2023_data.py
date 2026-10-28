import argparse
import os

import numpy as np
import yaml

from fcgno.data.assembly import assemble_csr
from fcgno.data.generate import load_or_make_dataset
from fcgno.experiments.runner import write_json


# Export one grid's Poisson train/test problems, test initial guesses and CSR matrix for the Li et al. 2023 baseline
def export_grid(config, grid):
    normalization, test = config["normalization"], config["test"]
    train = load_or_make_dataset(config["cache_dir"], config["kind"], grid, config["n_train"], config["train_seed"], normalization=normalization)
    tested = load_or_make_dataset(config["cache_dir"], config["kind"], grid, test["n_test"], test["seed"], normalization=normalization)
    guesses = np.random.default_rng(test["initial_guess_seed"]).standard_normal(tested["rhs"].shape)
    matrix = assemble_csr(tested["face_x"][0], tested["face_y"][0], normalization)
    for faces in (train["face_x"], train["face_y"], tested["face_x"], tested["face_y"]):
        if not np.allclose(faces, faces[0]):
            raise ValueError("the Li et al. export assumes one operator shared by all samples (Poisson)")
    os.makedirs(config["export_dir"], exist_ok=True)
    path = os.path.join(config["export_dir"], f"{config['kind']}_{grid}.npz")
    flatten = lambda fields: fields.reshape(len(fields), -1)
    np.savez(path, data=matrix.data, indices=matrix.indices, indptr=matrix.indptr,
             train_rhs=flatten(train["rhs"]), train_solution=flatten(train["solution"]),
             test_rhs=flatten(tested["rhs"]), test_solution=flatten(tested["solution"]), test_guesses=flatten(guesses))
    write_json(path + ".json", {"grid": grid, "n_train": len(train["rhs"]), "n_test": len(tested["rhs"]), "normalization": normalization})
    print(f"wrote {path}", flush=True)


# Export the grid selected by --grid-index (or SLURM_ARRAY_TASK_ID), or all grids
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/r7_li2023.yaml")
    parser.add_argument("--grid-index", type=int)
    args = parser.parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    index = args.grid_index if args.grid_index is not None else os.environ.get("SLURM_ARRAY_TASK_ID")
    grids = config["grids"] if index is None else [config["grids"][int(index)]]
    for grid in grids:
        export_grid(config, grid)


if __name__ == "__main__":
    main()
