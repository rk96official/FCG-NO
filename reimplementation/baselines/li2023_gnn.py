import argparse
import importlib.util
import json
import os
import sys
import time
import types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import yaml
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPOSITORY = os.path.join(ROOT, "third_party", "li2023_preconditioner")


# Import the unmodified upstream GNN (models/model_v1.py); stub the unused CPython test module xxlimited if this build lacks it
def import_li2023_net():
    if importlib.util.find_spec("xxlimited") is None:
        sys.modules["xxlimited"] = types.SimpleNamespace(new=None)
    sys.path.insert(0, REPOSITORY)
    from models.model_v1 import Net
    return Net


# Iteration count at which each tolerance is first reached (paper metric: mean relative-residual curve) plus per-sample stats
def iteration_table(relative_curves, tolerances):
    table = {}
    for tol in tolerances:
        first = lambda curve: int(np.argmax(curve <= tol)) if np.any(curve <= tol) else -1
        per_sample = np.array([first(curve) for curve in relative_curves])
        reached = bool(np.all(per_sample >= 0))
        table[f"{tol:g}"] = {"iterations_mean_curve": first(np.mean(relative_curves, axis=0)),
                             "iterations_sample_mean": float(per_sample.mean()) if reached else float("nan"),
                             "iterations_sample_std": float(per_sample.std()) if reached else float("nan"),
                             "samples_not_converged": int(np.sum(per_sample < 0))}
    return table


# Preconditioned CG in float64; precondition maps a residual to M^{-1} r; returns relative true-residual norms ||b - A x_i|| / ||r_0||
def pcg(matrix, precondition, rhs, guess, n_iter):
    solution = guess.copy()
    residual = rhs - matrix @ solution
    preconditioned = precondition(residual)
    direction = preconditioned.copy()
    energy = residual @ preconditioned
    norms = [np.linalg.norm(residual)]
    for _ in range(n_iter):
        operator_direction = matrix @ direction
        step = energy / (direction @ operator_direction)
        solution += step * direction
        residual -= step * operator_direction
        norms.append(np.linalg.norm(rhs - matrix @ solution))
        if norms[-1] == 0.0:
            break
        preconditioned = precondition(residual)
        new_energy = residual @ preconditioned
        if energy == 0.0:
            break
        direction = preconditioned + (new_energy / energy) * direction
        energy = new_energy
    norms = np.array(norms + [norms[-1]] * (n_iter + 1 - len(norms)))
    return norms / norms[0]


# Graph sample in the upstream format: nodes [x, y, f, dirichlet = 0], one edge per nonzero of A (both directions and self-loops), A_ij as the last edge feature
def build_graph(coo, positions, rhs, solution):
    scale = np.sqrt(np.mean(rhs ** 2))
    as_column = lambda values: torch.tensor(np.asarray(values)[:, None], dtype=torch.float32)
    nodes = np.column_stack([positions, rhs / scale, np.zeros(len(rhs))])
    return Data(x=torch.tensor(nodes, dtype=torch.float32), edge_attr=as_column(coo.data),
                edge_index=torch.tensor(np.stack([coo.row, coo.col]), dtype=torch.long),
                rhs=as_column(rhs / scale), u_next=as_column(solution / scale), diag=as_column(coo.tocsr().diagonal()),
                r=torch.zeros(len(coo.data), 1))


# Upstream Net with the train.py constructor arguments and the train.sh architecture settings
def build_net(net_class, model, n_nodes, n_edges):
    hidden = model["hidden_dim"]
    return net_class(types.SimpleNamespace(dataset="syn", use_global=False), 4, 1, 1, n_nodes, num_edges=n_edges,
                     out_dim_node=hidden, out_dim_edge=hidden, hidden_dim_node=hidden, hidden_dim_edge=hidden,
                     hidden_layers_node=model["hidden_layers_encoder"], hidden_layers_edge=model["hidden_layers_encoder"],
                     num_iterations=model["num_iterations"], hidden_dim_processor_node=hidden, hidden_dim_processor_edge=hidden,
                     hidden_layers_processor_node=model["hidden_layers_processor"], hidden_layers_processor_edge=model["hidden_layers_processor"],
                     hidden_dim_decoder=hidden, hidden_layers_decoder=model["hidden_layers_decoder"], dirichlet_idx=3, norm_type=model["norm"])


# Forward pass as in upstream train_epoch: input_x = exact solution, predicted x used, no global pooling
def forward(net, batch, input_x):
    return net(batch.x, batch.edge_attr, batch.edge_index, diag=batch.diag, input_r=batch.r, input_x=input_x, batch_idx=batch.batch,
               include_r=False, use_global=False, diagonalize=True, use_pred_x=True)


# Train with the upstream loss ||x_hat - u||^2 + ||L L^T x_hat - b||^2 (sum MSE), Adam and upstream ExpLR stepped per batch
def train(net, graphs, training, device, log):
    loader = DataLoader(graphs, batch_size=training["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=training["learning_rate"])
    base, minimum = training["learning_rate"], training["min_lr"]
    decay = lambda step: (minimum + (base - minimum) * training["decay_gamma"] ** (step / training["decay_steps"])) / base
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, decay)
    mse = torch.nn.MSELoss(reduction="sum")
    history = []
    for epoch in range(training["epochs"]):
        net.train()
        total = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            predicted_rhs, _, predicted_x = forward(net, batch, batch.u_next)
            loss = mse(predicted_x, batch.u_next) + mse(predicted_rhs, batch.rhs)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()
        history.append(total / len(graphs))
        if epoch % 10 == 0 or epoch == training["epochs"] - 1:
            log(f"epoch {epoch}: loss per sample {history[-1]:.4e}, lr {scheduler.get_last_lr()[0]:.2e}")
    return history


# Learned lower-triangular factor L (CSR, float64) for one graph
def learned_factor(net, graph, device):
    net.eval()
    with torch.no_grad():
        batch = graph.to(device)
        batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long, device=device)
        _, ((values, _, _), index), _ = forward(net, batch, torch.zeros_like(batch.u_next))
    index = index.cpu().numpy()
    return sp.csr_matrix((values.cpu().double().numpy().ravel(), (index[0], index[1])), shape=(graph.num_nodes, graph.num_nodes))


# Apply (L L^T)^{-1} with two sparse triangular solves
def factor_preconditioner(factor):
    upper = factor.T.tocsr()
    return lambda residual: spla.spsolve_triangular(upper, spla.spsolve_triangular(factor, residual, lower=True), lower=False)


# Node coordinates of the interior n x n grid in row-major order
def grid_positions(n):
    nodes = np.arange(1, n + 1) / (n + 1)
    mesh_x, mesh_y = np.meshgrid(nodes, nodes, indexing="ij")
    return np.column_stack([mesh_x.ravel(), mesh_y.ravel()])


# Train on one grid, evaluate PCG with the learned preconditioner and plain CG on the shared test set, save results
def run_grid(config, grid, device, log):
    data = np.load(os.path.join(config["export_dir"], f"{config['kind']}_{grid}.npz"))
    matrix = sp.csr_matrix((data["data"], data["indices"], data["indptr"]), shape=(grid * grid, grid * grid))
    coo, positions = matrix.tocoo(), grid_positions(grid)
    torch.manual_seed(config["model_seed"])
    train_graphs = [build_graph(coo, positions, f, u) for f, u in zip(data["train_rhs"], data["train_solution"])]
    net = build_net(import_li2023_net(), config["model"], grid * grid, len(coo.data)).to(device)
    log(f"grid {grid}: {sum(p.numel() for p in net.parameters())} parameters, {len(train_graphs)} training graphs")
    start = time.perf_counter()
    history = train(net, train_graphs, config["training"], device, log)
    train_seconds = time.perf_counter() - start
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    torch.save(net.state_dict(), os.path.join(config["checkpoint_dir"], f"li2023_{config['kind']}_{grid}_seed{config['model_seed']}.pt"))
    n_iter = int(config["test"]["n_iter"].replace("grid", "") or 1) * grid if isinstance(config["test"]["n_iter"], str) else int(config["test"]["n_iter"])
    learned, plain = [], []
    for f, u, guess in zip(data["test_rhs"], data["test_solution"], data["test_guesses"]):
        factor = learned_factor(net, build_graph(coo, positions, f, u), device)
        learned.append(pcg(matrix, factor_preconditioner(factor), f, guess, n_iter))
        plain.append(pcg(matrix, lambda residual: residual.copy(), f, guess, n_iter))
    learned, plain = np.array(learned), np.array(plain)
    tolerances = config["test"]["tolerances"]
    row = {"name": f"pcg_gnn_{config['kind']}_{grid}", "model": "pcg_gnn", "kind": config["kind"], "grid": grid, "test_grid": grid,
           "loss": "li2023", "sampling": "none", "normalization": config["normalization"], "seed": config["model_seed"], "n_iter": n_iter,
           "fcg": iteration_table(learned, tolerances), "cg": iteration_table(plain, tolerances),
           "train_seconds": train_seconds, "history": history, "factor_nnz_ratio": factor.nnz / matrix.nnz}
    os.makedirs(config["results_dir"], exist_ok=True)
    np.savez(os.path.join(config["results_dir"], f"{row['name']}_test{grid}.npz"), pcg_residual=learned, cg_residual=plain)
    with open(os.path.join(config["results_dir"], f"{row['name']}.json"), "w") as handle:
        json.dump([row], handle, indent=2)
    log(f"grid {grid}: PCG-GNN {({tol: v['iterations_mean_curve'] for tol, v in row['fcg'].items()})}, CG {({tol: v['iterations_mean_curve'] for tol, v in row['cg'].items()})}")


# Tiny end-to-end check of the environment: upstream Net forward/backward and PCG on an 8 x 8 Poisson problem
def self_test(device):
    n = 8
    lap = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))
    matrix = (sp.kron(lap, sp.identity(n)) + sp.kron(sp.identity(n), lap)).tocsr()
    rhs = np.random.default_rng(0).standard_normal(n * n)
    graphs = [build_graph(matrix.tocoo(), grid_positions(n), rhs, spla.spsolve(matrix.tocsc(), rhs)) for _ in range(4)]
    model = {"hidden_dim": 8, "num_iterations": 2, "hidden_layers_encoder": 1, "hidden_layers_processor": 1, "hidden_layers_decoder": 1, "norm": "LayerNorm"}
    net = build_net(import_li2023_net(), model, n * n, matrix.nnz).to(device)
    training = {"epochs": 2, "batch_size": 2, "learning_rate": 1e-3, "decay_steps": 40000, "decay_gamma": 0.1, "min_lr": 1e-8}
    history = train(net, graphs, training, device, print)
    factor = learned_factor(net, graphs[0], device)
    curve = pcg(matrix, factor_preconditioner(factor), rhs, np.zeros(n * n), 200)
    assert np.all(np.isfinite(history)) and sp.triu(factor, k=1).count_nonzero() == 0 and curve[-1] < 1e-8, (history, curve[-1])
    print(f"torch {torch.__version__}, cuda {torch.cuda.is_available()}, PCG-GNN self-test residual {curve[-1]:.1e}\nSELF TEST OK", flush=True)


# Run the Table 10 baseline for one grid (--grid-index or SLURM_ARRAY_TASK_ID), all grids, or the environment self-test
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/r7_li2023.yaml")
    parser.add_argument("--grid-index", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.self_test:
        return self_test(device)
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    index = args.grid_index if args.grid_index is not None else os.environ.get("SLURM_ARRAY_TASK_ID")
    for grid in (config["grids"] if index is None else [config["grids"][int(index)]]):
        run_grid(config, grid, device, lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
