import argparse
import importlib
import os
import sys

import yaml

from nfcg.experiments.config import expand_axes

EXPERIMENTS = {
    "reference": ("nfcg.experiments.reference", "REFERENCE_AXES", "reference_name", "execute_reference_run"),
    "diagnosis": ("nfcg.experiments.diagnosis", "DIAGNOSIS_AXES", "diagnosis_name", "execute_diagnosis_run"),
    "supporting": ("nfcg.experiments.supporting", "SUPPORTING_AXES", "supporting_name", "execute_supporting_run"),
    "training": ("nfcg.experiments.training", "TRAINING_AXES", "training_name", "execute_training_run"),
    "classical": ("nfcg.experiments.classical", "CLASSICAL_AXES", "classical_name", "execute_classical_run"),
    "amg": ("nfcg.experiments.amg", "AMG_AXES", "amg_name", "execute_amg_run"),
    "composed": ("nfcg.experiments.composed", "COMPOSED_AXES", "composed_name", "execute_composed_run"),
    "main": ("nfcg.experiments.main_table", "MAIN_AXES", "main_name", "execute_main_run"),
    "multigrid": ("nfcg.experiments.multigrid_baselines", "MULTIGRID_AXES", "multigrid_name", "execute_multigrid_run"),
    "neuralif_training": ("nfcg.experiments.neuralif", "NEURALIF_TRAINING_AXES", "neuralif_training_name", "execute_neuralif_training_run"),
    "neuralif": ("nfcg.experiments.neuralif", "NEURALIF_AXES", "neuralif_name", "execute_neuralif_run"),
    "amgx": ("nfcg.experiments.amgx", "AMGX_AXES", "amgx_name", "execute_amgx_run"),
    "three_d_training": ("nfcg.experiments.three_d", "THREE_D_TRAINING_AXES", "three_d_training_name", "execute_three_d_training_run"),
    "three_d": ("nfcg.experiments.three_d", "THREE_D_AXES", "three_d_name", "execute_three_d_run"),
}


# Print with flush so SLURM logs update live
def log(message):
    print(message, flush=True)


# Axes, run-name function and executor of an experiment type, importing only that experiment's module
def experiment_parts(experiment_type):
    module_name, *attributes = EXPERIMENTS[experiment_type]
    module = importlib.import_module(module_name)
    return tuple(getattr(module, attribute) for attribute in attributes)


# Run one run of an nfcg experiment config (by --run-index or SLURM_ARRAY_TASK_ID), all runs, or list them
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-index", type=int)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    axes, name_of, execute = experiment_parts(config["type"])
    runs = expand_axes(config, axes)
    if args.list:
        for index, run in enumerate(runs):
            print(index, name_of(run))
        print(f"{len(runs)} runs")
        return
    index = args.run_index if args.run_index is not None else os.environ.get("SLURM_ARRAY_TASK_ID")
    if index is not None and int(index) >= len(runs):
        log(f"run index {index} out of range for {len(runs)} runs")
        sys.exit(1)
    for run in runs if index is None else [runs[int(index)]]:
        log(f"=== {name_of(run)}")
        execute(run, config, log)


if __name__ == "__main__":
    main()
