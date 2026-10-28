import argparse
import os
import sys

import yaml

from fcgno.experiments.settings import experiment_runs


# Print with flush so SLURM logs update live
def log(message):
    print(message, flush=True)


# Run one run of an experiment config (by --run-index or SLURM_ARRAY_TASK_ID), all runs, or list them
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-index", type=int)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    runs = experiment_runs(config)
    if args.list:
        for index, run in enumerate(runs):
            print(index, run["name"])
        print(f"{len(runs)} runs")
        return
    index = args.run_index if args.run_index is not None else os.environ.get("SLURM_ARRAY_TASK_ID")
    if index is not None and int(index) >= len(runs):
        log(f"run index {index} out of range for {len(runs)} runs")
        sys.exit(1)
    selected = runs if index is None else [runs[int(index)]]
    if config.get("type") == "surrogate":
        from fcgno.experiments.surrogate_run import execute_surrogate_run as execute
    else:
        from fcgno.experiments.runner import execute_preconditioner_run as execute
    for run in selected:
        log(f"=== {run['name']}")
        execute(run, config, log)


if __name__ == "__main__":
    main()
