import glob
import json
import os
import re

import numpy as np
import pytest
import yaml

import fake_pyamgx
from nfcg.experiments.amgx import AmgxSession, amgx_csr, execute_amgx_run, mode_dtypes, solve_systems, solver_config
from fcgno.data.assembly import assemble_csr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMGX_SOURCE = os.path.expanduser("~/src/AMGX")
AMGX_CONFIGS = ["configs/smoke/e18_amgx_smoke.yaml", "configs/e18_amgx.yaml", "configs/e18_amgx_contrast.yaml", "configs/e20_amgx.yaml"]
FACTORY_KEYS = ("solver", "selector", "interpolator", "convergence", "coarse_solver", "cycle")


# Variants of every AMGX experiment config
def all_variants():
    return [variant for path in AMGX_CONFIGS for variant in yaml.safe_load(open(os.path.join(ROOT, path)))["variants"]]


# Fresh fake library state for every test
@pytest.fixture(autouse=True)
def fake_state():
    fake_pyamgx.reset()
    yield
    fake_pyamgx.reset()


# Runner config for a small Poisson test set in a temporary directory, with the smoke config's variants (classical, classical fp32, aggregation)
def small_config(tmp_path):
    smoke = yaml.safe_load(open(os.path.join(ROOT, "configs/smoke/e18_amgx_smoke.yaml")))
    return {**smoke, "results_dir": str(tmp_path / "results"), "cache_dir": str(tmp_path / "datasets"), "paper_cache_dir": str(tmp_path / "missing"),
            "test": {**smoke["test"], "n_test": 3}}


# The fake reproduces the original crash: a second live Resources object is rejected, as real AMGX aborts when one of them is destroyed
def test_fake_rejects_second_resources():
    fake_pyamgx.initialize()
    config = fake_pyamgx.Config().create("")
    fake_pyamgx.Resources().create_simple(config)
    with pytest.raises(fake_pyamgx.AMGXError, match="second live Resources"):
        fake_pyamgx.Resources().create_simple(config)


# The fake rejects destroying Resources while objects allocated from it are alive (AMGX's "non-empty temporary device pool" abort)
def test_fake_rejects_resources_destroyed_before_children():
    fake_pyamgx.initialize()
    config = fake_pyamgx.Config().create("")
    resources = fake_pyamgx.Resources().create_simple(config)
    fake_pyamgx.Vector().create(resources, "dDDI")
    with pytest.raises(fake_pyamgx.AMGXError, match="non-empty temporary device pool"):
        resources.destroy()


# Full runner on three variants (double, single precision, aggregation): results written, one Resources at a time, nothing left alive, finalized once
def test_runner_end_to_end(tmp_path):
    config = small_config(tmp_path)
    run = {"kind": "poisson", "grid": 32}
    execute_amgx_run(run, config, log=lambda message: None, pyamgx=fake_pyamgx)
    result = json.load(open(tmp_path / "results" / "amgx_poisson_32.json"))
    assert set(result["variants"]) == {variant["label"] for variant in config["variants"]}
    for summary in result["variants"].values():
        assert summary["iterations"]["1e-06"]["failure_rate"] == 0.0
        assert summary["setup_ms_per_system"] >= 0.0 and summary["solve_ms_per_system"] > 0.0
    arrays = np.load(tmp_path / "results" / "amgx_poisson_32.npz")
    assert arrays["amgx_classical__iterations"].shape == (3, 3)
    assert fake_pyamgx.STATE["live"] == [] and fake_pyamgx.STATE["max_resources"] == 1 and fake_pyamgx.STATE["finalized"] == 1


# Uploads follow the mode: double for dDDI, single for dFFI, int32 indices always
def test_uploads_follow_mode(tmp_path):
    config = small_config(tmp_path)
    execute_amgx_run({"kind": "poisson", "grid": 32}, config, log=lambda message: None, pyamgx=fake_pyamgx)
    dtypes = {(what, dtype) for what, dtype in fake_pyamgx.STATE["uploads"]}
    assert ("matrix data", np.dtype(np.float32)) in dtypes and ("matrix data", np.dtype(np.float64)) in dtypes
    assert ("vector data", np.dtype(np.float32)) in dtypes and ("vector data", np.dtype(np.float64)) in dtypes
    assert {dtype for what, dtype in dtypes if what in ("row_ptrs", "col_indices")} == {np.dtype(np.int32)}


# An AMGX error mid-run surfaces as a Python exception after every object is destroyed and AMGX finalized, instead of aborting the process
def test_error_releases_everything(tmp_path):
    fake_pyamgx.reset(fail_solve_at=2)
    with pytest.raises(fake_pyamgx.AMGXError, match="injected"):
        execute_amgx_run({"kind": "poisson", "grid": 32}, small_config(tmp_path), log=lambda message: None, pyamgx=fake_pyamgx)
    assert fake_pyamgx.STATE["live"] == [] and fake_pyamgx.STATE["finalized"] == 1
    assert not os.path.exists(tmp_path / "results" / "amgx_poisson_32.json")


# Residual histories start at the initial residual and match the iteration count; per-system timings are returned
def test_solve_systems_histories():
    session = AmgxSession(fake_pyamgx)
    matrix = amgx_csr(assemble_csr(np.ones((17, 16)), np.ones((16, 17)), "upstream"), np.float64)
    rhs = np.ascontiguousarray(np.random.default_rng(0).standard_normal(16 * 16))
    variant = {"mode": "dDDI", "preconditioner": all_variants()[0]["preconditioner"]}
    results = solve_systems(session, variant, lambda index: (matrix, rhs, np.zeros_like(rhs)), 2, 1e-8, 500)
    session.close()
    assert len(results) == 2
    for result in results:
        assert result["history"][-1] <= 1e-8 * result["history"][0] and len(result["history"]) > 2
    assert fake_pyamgx.STATE["live"] == []


# Mode strings map to the precisions AMGX reads
def test_mode_dtypes():
    assert mode_dtypes("dDDI") == (np.float64, np.float64)
    assert mode_dtypes("dFFI") == (np.float32, np.float32)


# Registered AMGX parameter and factory names, parsed from the AMGX sources the library was built from
def amgx_registry():
    sources = glob.glob(os.path.join(AMGX_SOURCE, "src", "**", "*.cu"), recursive=True) + glob.glob(os.path.join(AMGX_SOURCE, "include", "**", "*.h"), recursive=True)
    text = "\n".join(open(path, errors="ignore").read() for path in sources)
    parameters = set(re.findall(r'registerParameter<[^>]*>\(\s*"([^"]+)"', text))
    factories = set(re.findall(r'registerFactory\(\s*"([^"]+)"', text))
    return parameters, factories


# Every key of every experiment's AMGX config is a registered AMGX parameter and every solver/selector/interpolator/... value a registered factory
@pytest.mark.skipif(not os.path.isdir(os.path.join(AMGX_SOURCE, "src")), reason="AMGX sources not present")
def test_configs_use_registered_amgx_names():
    parameters, factories = amgx_registry()
    assert "store_res_history" in parameters and "PCG" in factories

    # Check one config scope recursively
    def check(scope):
        for key, value in scope.items():
            if key in ("config_version", "scope"):
                continue
            assert key in parameters, f"unknown AMGX parameter {key}"
            if isinstance(value, dict):
                check(value)
            elif key in FACTORY_KEYS:
                assert value in factories, f"unknown AMGX {key} {value}"
            elif key == "algorithm":
                assert value in ("CLASSICAL", "AGGREGATION")

    for variant in all_variants():
        check(solver_config(variant["preconditioner"], 1e-6, 500)["solver"])
