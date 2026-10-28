import numpy as np

DENOMINATOR_EPS = 1e-30


# Single-system FCG (Algorithm 1) in NumPy for CPU preconditioners; returns final iterate and residual norms
def fcg_numpy(matrix, preconditioner, rhs, initial_guess, n_iter, m_max=20):
    iterate = np.array(initial_guess, dtype=np.float64)
    residual = rhs - matrix @ iterate
    history = []
    norms = [np.linalg.norm(residual)]
    for step in range(n_iter):
        preconditioned = preconditioner(residual)
        count = min(step, max(1, step % (m_max + 1)))
        direction = preconditioned.copy()
        for previous, operator_previous, energy in history[len(history) - count:]:
            direction -= (operator_previous @ preconditioned) / (energy + DENOMINATOR_EPS) * previous
        operator_direction = matrix @ direction
        energy = direction @ operator_direction
        step_size = (direction @ residual) / (energy + DENOMINATOR_EPS)
        iterate += step_size * direction
        residual -= step_size * operator_direction
        history = (history + [(direction, operator_direction, energy)])[-m_max:]
        norms.append(np.linalg.norm(residual))
    return iterate, np.array(norms)
