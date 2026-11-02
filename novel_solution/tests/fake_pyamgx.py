import json

import numpy as np
import scipy.sparse as sp

# Strict CPU stand-in for pyamgx enforcing the AMGX object rules the real library aborts on; solves with plain CG so residual histories behave like AMGX's

STATE = {"initialized": False, "live": [], "max_resources": 0, "uploads": [], "fail_solve_at": None, "solves": 0, "finalized": 0}


# Reset the module state between tests
def reset(fail_solve_at=None):
    STATE.update(initialized=False, live=[], max_resources=0, uploads=[], fail_solve_at=fail_solve_at, solves=0, finalized=0)


# Error raised where real AMGX returns an error code or aborts
class AMGXError(RuntimeError):
    pass


# Raise like AMGX when a rule is broken
def require(condition, message):
    if not condition:
        raise AMGXError(message)


# AMGX_initialize, once
def initialize():
    require(not STATE["initialized"], "initialize called twice")
    STATE["initialized"] = True


# AMGX_finalize, only with no live objects
def finalize():
    require(STATE["initialized"], "finalize without initialize")
    require(not STATE["live"], f"finalize with live objects {[type(item).__name__ for item in STATE['live']]} (real AMGX aborts here)")
    STATE["initialized"] = False
    STATE["finalized"] += 1


# Base of every AMGX object: tracks liveness
class Handle:
    # Register a live object
    def register(self):
        require(STATE["initialized"], f"{type(self).__name__} created before initialize")
        self.alive = True
        STATE["live"].append(self)
        return self

    # Fail on use after destroy
    def check(self):
        require(getattr(self, "alive", False), f"{type(self).__name__} used after destroy or before create")

    # Destroy the object once
    def destroy(self):
        self.check()
        self.alive = False
        STATE["live"].remove(self)


# AMGX config
class Config(Handle):
    # Config from an AMGX option string (pyamgx tests use "")
    def create(self, options):
        self.params = {"options": options}
        return self.register()

    # Config from a dict, which pyamgx writes as JSON
    def create_from_dict(self, params):
        self.params = json.loads(json.dumps(params))
        require(self.params.get("config_version") == 2, "config_version 2 expected")
        return self.register()

    # A config may only be destroyed after every solver or resources created from it
    def destroy(self):
        users = [item for item in STATE["live"] if getattr(item, "config", None) is self]
        require(not users, f"Config destroyed while {[type(item).__name__ for item in users]} still use it")
        super().destroy()


# AMGX resources (device memory pool owner)
class Resources(Handle):
    # Single-threaded resources; a second live Resources breaks AMGX's shared device pool
    def create_simple(self, config):
        config.check()
        require(not any(isinstance(item, Resources) for item in STATE["live"]), "second live Resources (shared device pool; real AMGX aborts on destroy)")
        self.config = config
        self.register()
        STATE["max_resources"] = max(STATE["max_resources"], sum(isinstance(item, Resources) for item in STATE["live"]))
        return self

    # Destroying resources with live children frees a non-empty device pool
    def destroy(self):
        children = [item for item in STATE["live"] if getattr(item, "resources", None) is self]
        require(not children, f"Resources destroyed with live {[type(item).__name__ for item in children]} (non-empty temporary device pool)")
        super().destroy()


# Matrix and vector value dtypes of a mode string like "dDDI"
def mode_dtypes(mode):
    require(len(mode) == 4 and mode[0] == "d" and mode[3] == "I" and mode[1] in "DF" and mode[2] in "DF", f"unsupported mode {mode}")
    return tuple(np.dtype(np.float32 if mode[index] == "F" else np.float64) for index in (1, 2))


# Check an uploaded array is contiguous and of the expected dtype
def check_array(array, dtype, what):
    require(isinstance(array, np.ndarray) and array.dtype == dtype, f"{what} has dtype {getattr(array, 'dtype', type(array))}, expected {dtype}")
    require(array.flags.c_contiguous, f"{what} is not contiguous")
    STATE["uploads"].append((what, array.dtype))


# AMGX matrix
class Matrix(Handle):
    # Matrix in one mode on a live resources
    def create(self, resources, mode="dDDI"):
        resources.check()
        self.resources, self.mode, self.csr = resources, mode, None
        return self.register()

    # CSR upload: int32 indices, values in the mode's matrix precision
    def upload(self, row_ptrs, col_indices, data, block_dims=[1, 1], shape=None):
        self.check()
        check_array(row_ptrs, np.dtype(np.int32), "row_ptrs")
        check_array(col_indices, np.dtype(np.int32), "col_indices")
        check_array(data, mode_dtypes(self.mode)[0], "matrix data")
        rows = len(row_ptrs) - 1 if shape is None else shape[0]
        self.csr = sp.csr_matrix((data, col_indices, row_ptrs), shape=(rows, rows if shape is None else shape[1]))
        return self


# AMGX vector
class Vector(Handle):
    # Vector in one mode on a live resources
    def create(self, resources, mode="dDDI"):
        resources.check()
        self.resources, self.mode, self.values = resources, mode, None
        return self.register()

    # Upload values in the mode's vector precision
    def upload(self, data, block_dim=1):
        self.check()
        check_array(data, mode_dtypes(self.mode)[1], "vector data")
        require(data.ndim == 1, "vector data must be 1D")
        self.values = data.copy()
        return self


# AMGX solver
class Solver(Handle):
    # Solver from a config on a live resources, in one mode
    def create(self, resources, config, mode="dDDI"):
        resources.check()
        config.check()
        self.resources, self.config, self.mode, self.matrix, self.history = resources, config, mode, None, None
        return self.register()

    # Setup on a matrix of the same mode with uploaded values
    def setup(self, matrix):
        self.check()
        matrix.check()
        require(matrix.mode == self.mode, f"solver mode {self.mode} with matrix mode {matrix.mode}")
        require(matrix.csr is not None, "setup before matrix upload")
        self.matrix = matrix

    # CG solve to the config's relative tolerance, storing the residual history (index 0 the initial residual)
    def solve(self, right, solution, zero_initial_guess=False):
        self.check()
        right.check()
        solution.check()
        require(self.matrix is not None, "solve before setup")
        require(right.mode == self.mode and solution.mode == self.mode, "vector mode differs from solver mode")
        STATE["solves"] += 1
        if STATE["fail_solve_at"] is not None and STATE["solves"] == STATE["fail_solve_at"]:
            raise AMGXError("injected solve failure")
        options = self.config.params["solver"]
        matrix = self.matrix.csr.astype(np.float64)
        x = np.zeros_like(right.values, dtype=np.float64) if zero_initial_guess else solution.values.astype(np.float64)
        residual = right.values.astype(np.float64) - matrix @ x
        direction, history = residual.copy(), [np.linalg.norm(residual)]
        while len(history) <= options["max_iters"] and history[-1] > options["tolerance"] * history[0]:
            product = matrix @ direction
            step = (residual @ residual) / (direction @ product)
            x, new_residual = x + step * direction, residual - step * product
            direction = new_residual + (new_residual @ new_residual) / (residual @ residual) * direction
            residual = new_residual
            history.append(np.linalg.norm(residual))
        solution.values = x.astype(solution.values.dtype)
        self.history = history

    # Iterations of the last solve
    @property
    def iterations_number(self):
        self.check()
        return len(self.history) - 1

    # Residual of an iteration in 0..iterations_number, as AMGX's history
    def get_residual(self, iteration=-1, block_idx=0):
        self.check()
        iteration = self.iterations_number if iteration == -1 else iteration
        require(0 <= iteration <= self.iterations_number, f"residual history index {iteration} outside 0..{self.iterations_number}")
        return float(self.history[iteration])
