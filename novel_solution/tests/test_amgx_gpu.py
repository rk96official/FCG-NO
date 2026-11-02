import os

import numpy as np
import pytest
import scipy.sparse.linalg as spla
import yaml

from fcgno.data.assembly import assemble_csr
from nfcg.data.test_sets import initial_guesses, load_test_set
from nfcg.experiments.amgx import AmgxSession, amgx_csr, amgx_variant, solver_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = 64
pytestmark = pytest.mark.skipif(os.environ.get("AMGX_GPU_TESTS") != "1", reason="real AMGX on a GPU: run with AMGX_GPU_TESTS=1 in the amgx env")


# Variants of the AMGX smoke config (classical double, classical single precision, aggregation)
def smoke_variants():
    return yaml.safe_load(open(os.path.join(ROOT, "configs/smoke/e18_amgx_smoke.yaml")))["variants"]


# One real AMGX session for the module (one Resources per process, as in the experiments); closing it must not raise
@pytest.fixture(scope="module")
def session():
    import pyamgx
    opened = AmgxSession(pyamgx)
    yield opened
    opened.close()


# Small test set of a kind at GRID in a temporary cache
@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    cache = str(tmp_path_factory.mktemp("datasets"))
    return {kind: load_test_set(cache + "/missing", cache, kind, GRID, 3, "upstream") for kind in ("poisson", "diffusion", "jump6")}


# Real AMGX solves a Poisson system to 1e-10 and its solution matches a sparse direct solve
def test_solution_matches_direct_solve(session, datasets):
    pyamgx, dataset = session.pyamgx, datasets["poisson"]
    matrix = amgx_csr(assemble_csr(dataset["face_x"][0], dataset["face_y"][0], "upstream"), np.float64)
    rhs = np.ascontiguousarray(dataset["rhs"][0].ravel(), dtype=np.float64)
    created = []
    try:
        config = pyamgx.Config().create_from_dict(solver_config(smoke_variants()[0]["preconditioner"], 1e-10, 500))
        created.append(config)
        for item in (pyamgx.Matrix(), pyamgx.Vector(), pyamgx.Vector()):
            created.append(item.create(session.resources, "dDDI"))
        operator, right, solution = created[1:4]
        solver = pyamgx.Solver().create(session.resources, config, "dDDI")
        created.append(solver)
        operator.upload(matrix.indptr, matrix.indices, matrix.data, shape=list(matrix.shape))
        right.upload(rhs)
        solution.upload(np.zeros_like(rhs))
        solver.setup(operator)
        solver.solve(right, solution)
        values = np.zeros_like(rhs)
        solution.download(values)
        assert solver.iterations_number < 60
    finally:
        for item in reversed(created):
            item.destroy()
    expected = spla.spsolve(matrix.tocsc(), rhs)
    assert np.linalg.norm(values - expected) <= 1e-7 * np.linalg.norm(expected)


# Every smoke variant runs on Poisson, Diffusion and jump6 through the experiment code path and reduces the residual 1e3-fold (single precision or aggregation
# may legitimately stall lower at contrast 1e6, which is a measurement, not a wrapper failure); classical double-precision AMG reaches 1e-6 everywhere
@pytest.mark.parametrize("kind", ["poisson", "diffusion", "jump6"])
@pytest.mark.parametrize("index", range(3))
def test_variants_run(session, datasets, kind, index):
    variant, dataset = smoke_variants()[index], datasets[kind]
    curves, setup, solve = amgx_variant(session, variant, dataset, initial_guesses(3, GRID), "upstream", 500)
    reached = 1e-6 if variant["label"] == "amgx_classical" else 1e-3
    assert curves.shape[0] == 3 and np.all(curves.min(axis=1) <= reached), f"{variant['label']} on {kind}: best {curves.min(axis=1)}"
    assert setup > 0.0 and solve > 0.0
