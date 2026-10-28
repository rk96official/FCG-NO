import numpy as np


# First iteration i with relative residual <= tol along the last axis; -1 where it is never reached
def iterations_to_tolerance(relative_curves, tol):
    reached = np.asarray(relative_curves) <= tol
    return np.where(reached.any(axis=-1), np.argmax(reached, axis=-1), -1)


# Iteration counts as in the paper (mean curve over samples) plus per-sample statistics
def summarize_iterations(relative_curves, tol):
    per_sample = iterations_to_tolerance(relative_curves, tol)
    all_reached = bool(np.all(per_sample >= 0))
    return {
        "iterations_mean_curve": int(iterations_to_tolerance(np.mean(relative_curves, axis=0), tol)),
        "iterations_sample_mean": float(per_sample.mean()) if all_reached else float("nan"),
        "iterations_sample_std": float(per_sample.std()) if all_reached else float("nan"),
        "samples_not_converged": int(np.sum(per_sample < 0)),
    }
