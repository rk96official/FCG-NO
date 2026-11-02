import numpy as np
import jax.numpy as jnp

from fcgno.models.spectral import real_fourier_synthesis
from nfcg.eval.diagnostics import active_steps, energy_outside_modes, retained_basis
from nfcg.eval.per_system import bootstrap_interval, paired_wilcoxon, per_system_iterations, summarize_systems
from nfcg.experiments.checkpoints import sno_checkpoint_name
from nfcg.experiments.reference import reproduction_check


# Three systems: converge at iterations 2 and 3, and one never reaches 1e-6
def curves():
    return np.array([[1, 1e-2, 1e-7, 1e-8, 1e-9], [1, 1e-1, 1e-3, 1e-7, 1e-8], [1, 1e-1, 1e-2, 1e-3, 1e-4]])


# Failures count as inf and make p90 and max undefined, while the median stays finite
def test_summarize_systems():
    summary = summarize_systems(curves(), 1e-6)
    assert summary["failure_rate"] == 1 / 3 and summary["median"] == 3.0
    assert summary["p90"] is None and summary["max"] is None and summary["systems"] == 3
    assert list(per_system_iterations(curves(), 1e-6)) == [2.0, 3.0, np.inf]


# Bootstrap interval is ordered, and identical counts give a Wilcoxon p-value of 1
def test_statistics():
    counts = np.array([5.0, 6.0, 7.0, 8.0, 30.0])
    lower, upper = bootstrap_interval(counts, np.median, n_resamples=200)
    assert lower <= np.median(counts) <= upper
    assert paired_wilcoxon(counts, counts, cap=100) == 1.0
    assert paired_wilcoxon(counts, counts + np.array([1, 2, 3, 4, np.inf]), cap=100) < 1.0


# A field built from retained modes has no energy outside them; the grid-scale checkerboard has all of it outside
def test_energy_outside_modes():
    n = 32
    synthesis = np.asarray(real_fourier_synthesis(n, 20))
    smooth = synthesis[:, 3][:, None] * synthesis[:, 7][None, :]
    checkerboard = np.cos(np.pi * np.arange(n))[:, None] * np.cos(np.pi * np.arange(n))[None, :]
    fractions = np.asarray(energy_outside_modes(jnp.asarray(np.stack([smooth, checkerboard])), retained_basis(n)))
    np.testing.assert_allclose(fractions, [0.0, 1.0], atol=1e-10)


# Steps stay active until the relative residual first reaches the floor
def test_active_steps():
    mask = active_steps(curves(), 1e-6)
    assert mask.tolist() == [[True, True, False, False], [True, True, True, False], [True, True, True, True]]


# Reproduction check compares the mean curve of the first 20 systems within the stored budget
def test_reproduction_check():
    stored = {"n_iter": 4, "fcg": {"1e-06": {"iterations_mean_curve": 4}}}
    rows = reproduction_check(np.tile(curves()[:1], (20, 1)), stored, [1e-6])
    assert rows["1e-06"] == {"ours": 2, "stored": 4, "match": False}
    assert sno_checkpoint_name("poisson", 128, 1, "notebook") == "sno_poisson_128_notay_krylov_n128_upstream_seed1_notebook"
