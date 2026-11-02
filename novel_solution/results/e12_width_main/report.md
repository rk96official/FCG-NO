# E12 width

## diffusion 128² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 300 | 324 |  | 1.37 |  |
| **jacobi4** (GPU) |  | 0.0 | 126 | 137 |  | 0.66 |  |
| **chebyshev8** (GPU) |  | 0.0 | 59 | 64 |  | 0.41 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 6 | 6 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| h1_on_step | tf32 | 0.0 | 19 | 21 | 3.133 | 3.72 | — |
| h1_on_step | safeguard_every4_tf32 | 0.0 | 23 | 25 | 0.949 | 1.36 | — |
| h1_on_step | two_level_c2_cheb2_tf32 | 0.0 | 56 | 111 | 1.386 | 4.85 | — |
| h1_on_step | two_level_c2_cheb4_tf32 | 0.0 | 39 | 70 | 1.398 | 3.41 | — |
| h1_on_step | two_level_c2_cheb2_safeguard_tf32 | 0.0 | 53 | 107 | 1.424 | 4.72 | — |
| h1_on_step_w32 | tf32 | 0.0 | 18 | 21 | 0.806 | 0.91 | — |
| h1_on_step_w32 | safeguard_every4_tf32 | 0.0 | 22 | 23 | 0.343 | 0.47 | — |
| h1_on_step_w32 | two_level_c2_cheb2_tf32 | 0.0 | 28 | 34 | 0.502 | 0.88 | — |
| h1_on_step_w32 | two_level_c2_cheb4_tf32 | 0.0 | 25 | 30 | 0.515 | 0.81 | — |
| h1_on_step_w32 | two_level_c2_cheb2_safeguard_tf32 | 0.0 | 26 | 33 | 0.537 | 0.87 | — |
| h1_on_step_w48 | tf32 | 0.0 | 18 | 22 | 1.113 | 1.25 | — |
| h1_on_step_w48 | safeguard_every4_tf32 | 0.0 | 22 | 26 | 0.428 | 0.59 | — |
| h1_on_step_w48 | two_level_c2_cheb2_tf32 | 0.0 | 30 | 91 | 0.600 | 1.13 | — |
| h1_on_step_w48 | two_level_c2_cheb4_tf32 | 0.0 | 36 | 62 | 0.631 | 1.42 | — |
| h1_on_step_w48 | two_level_c2_cheb2_safeguard_tf32 | 0.0 | 32 | 88 | 0.638 | 1.28 | — |

## diffusion 256² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 606 | 661 |  | 6.29 |  |
| **jacobi4** (GPU) |  | 0.0 | 255 | 275 |  | 3.64 |  |
| **chebyshev8** (GPU) |  | 0.0 | 119 | 129 |  | 2.53 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 7 | 7 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| h1_on_step | tf32 | 0.0 | 45 | 49 | 10.566 | 29.72 | — |
| h1_on_step | safeguard_every4_tf32 | 0.0 | 35 | 36 | 2.905 | 6.35 | — |
| h1_on_step | two_level_c2_cheb2_tf32 | 0.0 | 11 | 15 | 3.435 | 2.36 | — |
| h1_on_step | two_level_c2_cheb4_tf32 | 0.0 | 12 | 26 | 3.530 | 2.65 | — |
| h1_on_step | two_level_c2_cheb2_safeguard_tf32 | 0.0 | 11 | 14 | 3.514 | 2.42 | — |
| h1_on_step_w32 | tf32 | 0.0 | 41 | 44 | 2.356 | 6.04 | — |
| h1_on_step_w32 | safeguard_every4_tf32 | 0.0 | 31 | 38 | 0.872 | 1.69 | — |
| h1_on_step_w32 | two_level_c2_cheb2_tf32 | 0.0 | 12 | 13 | 1.116 | 0.84 | — |
| h1_on_step_w32 | two_level_c2_cheb4_tf32 | 0.0 | 10 | 12 | 1.203 | 0.75 | — |
| h1_on_step_w32 | two_level_c2_cheb2_safeguard_tf32 | 0.0 | 11 | 12 | 1.202 | 0.83 | — |
| h1_on_step_w48 | tf32 | 0.0 | 42 | 46 | 3.531 | 9.27 | — |
| h1_on_step_w48 | safeguard_every4_tf32 | 0.0 | 31 | 34 | 1.193 | 2.31 | — |
| h1_on_step_w48 | two_level_c2_cheb2_tf32 | 0.0 | 12 | 27 | 1.430 | 1.07 | — |
| h1_on_step_w48 | two_level_c2_cheb4_tf32 | 0.0 | 12 | 22 | 1.543 | 1.16 | — |
| h1_on_step_w48 | two_level_c2_cheb2_safeguard_tf32 | 0.0 | 11 | 25 | 1.518 | 1.04 | — |
