import json
import os

import numpy as np

KINDS = ("poisson", "diffusion")
KIND_TITLES = {"poisson": "Poisson", "diffusion": "Diffusion"}
SEEDS = (0, 1, 2, 3, 4)
REPRODUCTION_SYSTEMS = 20


# Per-system iteration counts of one variant at a tolerance from a saved .npz (rows ordered like its tolerances), optionally the first systems only; None when missing
def per_system(path, key, tolerance=1e-6, first=None):
    if not os.path.exists(path):
        return None
    stored = np.load(path)
    if f"{key}__iterations" not in stored.files:
        return None
    row = int(np.argmin(np.abs(np.log(stored["tolerances"]) - np.log(tolerance))))
    counts = stored[f"{key}__iterations"][row]
    return counts if first is None else counts[:first]


# Counts pooled over several files (seeds), or None when none exists
def pooled(paths, key, tolerance=1e-6, first=None):
    found = [counts for counts in (per_system(path, key, tolerance, first) for path in paths) if counts is not None]
    return None if not found else np.concatenate(found)


# Median that counts failures as infinite (the reports' convention): inf when at least half the systems fail
def median_iterations(counts):
    return np.inf if counts is None else float(np.quantile(counts, 0.5, method="inverted_cdf"))


# Upper quantile with failures as infinite
def upper_iterations(counts, level=0.9):
    return np.inf if counts is None else float(np.quantile(counts, level, method="inverted_cdf"))


# Variant entry of a result JSON next to an .npz path, or None
def variant_entry(path, key):
    json_path = path[:-len(".npz")] + ".json" if path.endswith(".npz") else path
    if not os.path.exists(json_path):
        return None
    return json.load(open(json_path))["variants"].get(key)


# Seconds per iteration of a variant at a batch size, or nan
def seconds_per_iteration(path, key, batch="1"):
    entry = variant_entry(path, key)
    return np.nan if entry is None else float(entry["seconds_per_iteration"].get(batch, np.nan))


# Setup seconds per system of a multigrid variant at a batch size (0 for setup-free variants), or nan
def setup_seconds_per_system(path, key, batch="1"):
    entry = variant_entry(path, key)
    if entry is None or "setup_seconds" not in entry:
        return np.nan
    return float(entry["setup_seconds"].get(batch, np.nan)) / int(batch)


# Median iterations stored in a result JSON (over all its systems), inf when the median system fails, nan when missing
def json_median(path, key, tolerance="1e-06"):
    entry = variant_entry(path, key)
    if entry is None:
        return np.nan
    median = entry["iterations"][tolerance]["median"]
    return np.inf if median is None else float(median)


# Milliseconds per system of a batched solver: setup per system plus median iterations times seconds per iteration divided by the batch size
def solve_milliseconds(counts, per_iteration, batch="1", setup=0.0):
    median = median_iterations(counts)
    if not np.isfinite(median) or not np.isfinite(per_iteration):
        return np.nan
    return 1e3 * (setup + median * per_iteration / int(batch))


# AMGX entry for one problem: per-system counts and the measured setup and solve milliseconds per system (one system at a time)
def amgx_entry(results, kind, grid, label="amgx_classical", first=None):
    path = os.path.join(results, "e0_amgx", f"amgx_{kind}_{grid}.npz")
    entry = variant_entry(path, label)
    if entry is None:
        return None, np.nan, np.nan
    return per_system(path, label, first=first), float(entry["setup_ms_per_system"]), float(entry["solve_ms_per_system"])
