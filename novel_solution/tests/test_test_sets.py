import os

import numpy as np

from fcgno.data.generate import load_or_make_dataset, make_dataset
from nfcg.data.test_sets import PAPER_TEST_SEED, PAPER_TEST_SIZE, initial_guesses, load_test_set

N = 8


# The extended test set starts with the reproduction's 20 problems and is generated into our cache when the fcgno cache lacks it
def test_extended_set_keeps_paper_problems(tmp_path):
    paper_cache, own_cache = tmp_path / "paper", tmp_path / "own"
    dataset = load_test_set(str(paper_cache), str(own_cache), "diffusion", N, 25, "upstream")
    reference = make_dataset("diffusion", N, PAPER_TEST_SIZE, PAPER_TEST_SEED, normalization="upstream")
    assert dataset["rhs"].shape == (25, N, N)
    np.testing.assert_array_equal(dataset["rhs"][:PAPER_TEST_SIZE], reference["rhs"])
    np.testing.assert_array_equal(dataset["face_x"][:PAPER_TEST_SIZE], reference["face_x"])
    assert not paper_cache.exists() and len(os.listdir(own_cache)) == 2


# A dataset already in the fcgno cache is read from there, not regenerated elsewhere
def test_reads_existing_fcgno_cache(tmp_path):
    paper_cache, own_cache = tmp_path / "paper", tmp_path / "own"
    load_or_make_dataset(str(paper_cache), "poisson", N, PAPER_TEST_SIZE, PAPER_TEST_SEED, normalization="upstream")
    load_test_set(str(paper_cache), str(own_cache), "poisson", N, PAPER_TEST_SIZE, "upstream")
    assert not own_cache.exists()


# The first 20 initial guesses equal those of fcgno's evaluate (default_rng(1) over a (20, n, n) array)
def test_initial_guesses_match_fcgno():
    reference = np.random.default_rng(1).standard_normal((PAPER_TEST_SIZE, N, N))
    np.testing.assert_array_equal(initial_guesses(100, N)[:PAPER_TEST_SIZE], reference)
    np.testing.assert_array_equal(initial_guesses(5, N), reference[:5])
    assert initial_guesses(100, N).shape == (100, N, N)
