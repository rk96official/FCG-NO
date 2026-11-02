import time

import numpy as np
import jax


# Median wall-clock seconds per iteration of a jitted solver call over repeats, compilation and warm-up excluded
def seconds_per_iteration(solve, arguments, n_iter, repeats=5):
    jax.block_until_ready(solve(*arguments))
    durations = []
    for unused in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(solve(*arguments))
        durations.append(time.perf_counter() - start)
    return float(np.median(durations)) / n_iter
