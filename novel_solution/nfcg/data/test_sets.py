import os

import numpy as np

from fcgno.data.generate import dataset_path, load_or_make_dataset

PAPER_TEST_SEED = 12
EXTENSION_TEST_SEED = 13
PAPER_TEST_SIZE = 20
PAPER_GUESS_SEED = 1
EXTENSION_GUESS_SEED = 2


# Cached dataset, read from the fcgno cache when it is already there and otherwise generated into our own cache
def cached_dataset(paper_cache_dir, cache_dir, kind, grid, n_samples, seed, normalization):
    shared = dataset_path(paper_cache_dir, kind, grid, n_samples, seed, normalization=normalization)
    directory = paper_cache_dir if os.path.exists(shared) else cache_dir
    return load_or_make_dataset(directory, kind, grid, n_samples, seed, normalization=normalization)


# Test set of n_test problems: the 20 of the reproduction (seed 12) first, then fresh problems (seed 13)
def load_test_set(paper_cache_dir, cache_dir, kind, grid, n_test, normalization="upstream"):
    paper = cached_dataset(paper_cache_dir, cache_dir, kind, grid, PAPER_TEST_SIZE, PAPER_TEST_SEED, normalization)
    if n_test <= PAPER_TEST_SIZE:
        return {name: values[:n_test] for name, values in paper.items()}
    extension = cached_dataset(paper_cache_dir, cache_dir, kind, grid, n_test - PAPER_TEST_SIZE, EXTENSION_TEST_SEED, normalization)
    return {name: np.concatenate([paper[name], extension[name]]) for name in paper}


# Initial guesses u0 ~ N(0, I); the first 20 equal those of the fcgno evaluation (seed 1)
def initial_guesses(n_test, grid):
    paper = np.random.default_rng(PAPER_GUESS_SEED).standard_normal((min(n_test, PAPER_TEST_SIZE), grid, grid))
    if n_test <= PAPER_TEST_SIZE:
        return paper
    extension = np.random.default_rng(EXTENSION_GUESS_SEED).standard_normal((n_test - PAPER_TEST_SIZE, grid, grid))
    return np.concatenate([paper, extension])
