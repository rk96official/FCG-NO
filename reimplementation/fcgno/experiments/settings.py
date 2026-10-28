import itertools

AXES = ("kind", "grid", "model", "loss", "sampling", "normalization", "seed")

TABLE8 = {
    "poisson": {"epochs": 150, "batch_size": {32: 32, 64: 32, 128: 8}},
    "diffusion": {"epochs": 200, "batch_size": {32: 16, 64: 16, 128: 4}},
}
COMMON_TRAINING = {"learning_rate": 5e-4, "decay_epochs": 50, "decay_factor": 0.5, "weight_decay": 1e-2}
TABLE_KIND = {"poisson": "poisson", "diffusion": "diffusion", "a1": "diffusion", "a2": "diffusion"}


# A config entry as a list of axis values
def as_list(value):
    return value if isinstance(value, list) else [value]


# Table 8 training settings for a dataset kind and training grid, with config overrides (all grids, then per grid) applied
def training_settings(kind, grid, overrides=None, overrides_by_grid=None):
    table = TABLE8[TABLE_KIND[kind]]
    settings = dict(COMMON_TRAINING, epochs=table["epochs"], batch_size=table["batch_size"].get(grid))
    settings.update(overrides or {})
    settings.update((overrides_by_grid or {}).get(grid, {}))
    return settings


# Number of training problems: an integer, or "grid" for N_train = grid as in the paper
def resolve_count(value, grid):
    return grid if value == "grid" else int(value)


# Test grids of a run: "same" as training, an explicit list, or a mapping from training grid to test grids
def resolve_test_grids(test, grid):
    grids = test.get("grids", "same")
    if grids == "same":
        return [grid]
    if isinstance(grids, dict):
        return [int(value) for value in grids[grid]]
    return [int(value) for value in as_list(grids)]


# Iteration budget on a grid: an integer, or "<k>grid" for k * grid, k possibly fractional (paper: 2 * grid, upstream code: 2.5 * grid)
def resolve_iterations(value, grid):
    if isinstance(value, str) and value.endswith("grid"):
        return int(float(value[:-len("grid")] or 1) * grid)
    return int(value)


# Identifier of a trained preconditioner, shared by every experiment that trains the same configuration (tag marks setting variants)
def run_name(run):
    name = f"{run['model']}_{run['kind']}_{run['grid']}_{run['loss']}_{run['sampling']}_n{run['n_train']}_{run['normalization']}_seed{run['seed']}"
    return f"{name}_{run['tag']}" if run.get("tag") else name


# Expand the list-valued axes of a preconditioner config into one run per combination
def expand_runs(config):
    runs = []
    for values in itertools.product(*[as_list(config[axis]) for axis in AXES]):
        run = dict(zip(AXES, values))
        run["n_train"] = resolve_count(config.get("n_train", "grid"), run["grid"])
        run["krylov_iterations"] = int(config.get("krylov_iterations", 100))
        if config.get("tag"):
            run["tag"] = config["tag"]
        run["name"] = run_name(run)
        runs.append(run)
    return runs


# One Fig. 1 run per resolution
def surrogate_runs(config):
    return [{"grid": int(grid), "name": f"surrogate_{config['kind']}_{grid}_{config['normalization']}"} for grid in config["grids"]]


# Runs of any experiment config
def experiment_runs(config):
    return surrogate_runs(config) if config.get("type") == "surrogate" else expand_runs(config)
