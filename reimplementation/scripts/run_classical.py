import argparse
import os
import time

import numpy as np
import pandas as pd
import yaml

from fcgno.data.assembly import assemble_csr
from fcgno.data.generate import load_or_make_dataset
from fcgno.linalg.classical import build_classical
from fcgno.linalg.reference_krylov import fcg_numpy
from fcgno.eval.metrics import summarize_iterations


# Relative residual curves of FCG with one classical preconditioner over all test samples
def run_preconditioner(spec, dataset, initial_guesses, n_iter, m_max, normalization):
    curves, nnz_ratios = [], []
    for face_x, face_y, rhs, guess in zip(dataset["face_x"], dataset["face_y"], dataset["rhs"], initial_guesses):
        matrix = assemble_csr(face_x, face_y, normalization)
        preconditioner, nnz_ratio = build_classical(spec, matrix)
        _, norms = fcg_numpy(matrix, preconditioner, rhs.ravel(), guess.ravel(), n_iter, m_max)
        curves.append(norms / norms[0])
        nnz_ratios.append(nnz_ratio)
    return np.array(curves), float(np.mean(nnz_ratios))


# Paper column key and readable label of a spec; "ilu:8:1e-5" maps to the paper's ILU(8) with its drop tolerance noted
def paper_key_and_label(config, spec):
    base = ":".join(spec.split(":")[:2])
    extra = spec[len(base) + 1:]
    label = config["paper_labels"][base] + (f" ({extra})" if extra else "")
    return base, label


# Run every preconditioner on one (kind, grid) test set and return summary rows and curves
def run_case(config, kind, grid):
    dataset = load_or_make_dataset(config["cache_dir"], kind, grid, config["n_test"], config["test_seed"], normalization=config["normalization"])
    initial_guesses = np.random.default_rng(config["initial_guess_seed"]).standard_normal((config["n_test"], grid, grid))
    n_iter = config["max_iter_factor"] * grid
    paper = config["paper_iterations_1e-6"][kind][grid]
    rows, curves = [], {}
    for spec in config["preconditioners"]:
        base, label = paper_key_and_label(config, spec)
        start = time.perf_counter()
        curves[spec], nnz_ratio = run_preconditioner(spec, dataset, initial_guesses, n_iter, config["m_max"], config["normalization"])
        seconds = time.perf_counter() - start
        for tol in config["tolerances"]:
            summary = summarize_iterations(curves[spec], tol)
            rows.append({
                "kind": kind, "grid": grid, "preconditioner": label, "tolerance": tol,
                **summary, "paper_1e-6": paper[base] if tol == 1e-6 else np.nan,
                "nnz_ratio": nnz_ratio, "cpu_seconds_all_samples": seconds,
            })
        iterations = summarize_iterations(curves[spec], 1e-6)["iterations_mean_curve"]
        print(f"{kind} {grid} {spec}: {iterations} its to 1e-6 (paper {paper[base]}), nnz ratio {nnz_ratio:.2f}, {seconds:.1f}s", flush=True)
    return rows, curves


# Run the classical-preconditioner part of Table 1 and save CSV, curves and a comparison table
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/r1_classical.yaml")
    with open(parser.parse_args().config) as handle:
        config = yaml.safe_load(handle)
    os.makedirs(config["output_dir"], exist_ok=True)
    all_rows = []
    for kind in config["kinds"]:
        for grid in config["grids"]:
            rows, curves = run_case(config, kind, grid)
            all_rows.extend(rows)
            np.savez(os.path.join(config["output_dir"], f"curves_{kind}_{grid}.npz"), **{spec.replace(":", "_"): curve for spec, curve in curves.items()})
    results = pd.DataFrame(all_rows)
    results.to_csv(os.path.join(config["output_dir"], "iterations.csv"), index=False)
    at_paper_tol = results[results["tolerance"] == 1e-6]
    comparison = at_paper_tol.pivot_table(index=["kind", "grid"], columns="preconditioner", values=["iterations_mean_curve", "paper_1e-6"])
    comparison.to_csv(os.path.join(config["output_dir"], "table1_comparison.csv"))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(comparison)


if __name__ == "__main__":
    main()
