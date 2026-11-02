# E6 main table

## diffusion 128² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook base |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 300 | 324 |  | 1.37 |  |
| **jacobi4** (GPU) |  | 0.0 | 126 | 137 |  | 0.66 |  |
| **chebyshev8** (GPU) |  | 0.0 | 59 | 64 |  | 0.41 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 6 | 6 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| h1_on_step | base | 0.0 | 19 | 21 | 6.774 | 8.04 | 0.0000 |
| h1_on_step | tf32 | 0.0 | 19 | 21 | 3.139 | 3.73 | 0.0000 |
| h1_on_step | wrapper_cheb2 | 33.7 | 10 | — | 6.809 | 4.26 | 0.0000 |
| h1_on_step | wrapper_cheb2_tf32 | 33.7 | 10 | — | 3.175 | 1.98 | 0.0000 |
| h1_on_step | wrapper_cheb4_tf32 | 44.7 | 297 | — | 3.182 | 59.07 | 0.0930 |
| h1_on_step | safeguard_tf32 | 0.0 | 13 | 14 | 3.188 | 2.59 | 0.0000 |
| h1_on_step | safeguard_every4_tf32 | 0.0 | 23 | 25 | 0.968 | 1.39 | 0.0000 |
| h1_on_step | wrapper_cheb2_safeguard_tf32 | 0.0 | 10 | 43 | 3.209 | 2.01 | 0.0000 |
| h1_on_step | wrapper_cheb2_safeguard_every4_tf32 | 0.0 | 35 | 51 | 0.973 | 2.13 | 0.0993 |
| m0_on_step | base | 33.3 | 20 | — | 6.743 | 8.43 | 0.0000 |
| m0_on_step | tf32 | 33.3 | 20 | — | 3.146 | 3.93 | 0.0000 |
| m0_on_step | wrapper_cheb2 | 33.3 | 10 | — | 6.768 | 4.23 | 0.0000 |
| m0_on_step | wrapper_cheb2_tf32 | 33.3 | 10 | — | 3.177 | 1.99 | 0.0000 |
| m0_on_step | wrapper_cheb4_tf32 | 56.7 | — | — | 3.186 | — | 0.0081 |
| m0_on_step | safeguard_tf32 | 0.0 | 14 | 143 | 3.191 | 2.79 | 0.0000 |
| m0_on_step | safeguard_every4_tf32 | 0.0 | 22 | 149 | 0.964 | 1.33 | 0.0000 |
| m0_on_step | wrapper_cheb2_safeguard_tf32 | 0.0 | 11 | 131 | 3.215 | 2.21 | 0.0000 |
| m0_on_step | wrapper_cheb2_safeguard_every4_tf32 | 0.0 | 27 | 141 | 0.970 | 1.64 | 0.0000 |
| notebook | base | 33.3 | 25 | — | 6.731 | 10.52 | 1.0000 |
| notebook | tf32 | 33.3 | 25 | — | 3.130 | 4.89 | 0.0000 |
| notebook | wrapper_cheb2 | 33.3 | 7 | — | 6.756 | 2.96 | 0.0000 |
| notebook | wrapper_cheb2_tf32 | 33.3 | 7 | — | 3.156 | 1.38 | 0.0000 |
| notebook | wrapper_cheb4_tf32 | 15.3 | 22 | — | 3.169 | 4.36 | 0.0000 |
| notebook | safeguard_tf32 | 0.0 | 15 | 145 | 3.172 | 2.97 | 0.0000 |
| notebook | safeguard_every4_tf32 | 0.0 | 23 | 149 | 0.961 | 1.38 | 0.0000 |
| notebook | wrapper_cheb2_safeguard_tf32 | 0.0 | 7 | 146 | 3.193 | 1.40 | 0.0000 |
| notebook | wrapper_cheb2_safeguard_every4_tf32 | 0.0 | 24 | 145 | 0.966 | 1.45 | 0.0000 |

## diffusion 256² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook base |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 606 | 661 |  | 6.29 |  |
| **jacobi4** (GPU) |  | 0.0 | 255 | 275 |  | 3.64 |  |
| **chebyshev8** (GPU) |  | 0.0 | 119 | 129 |  | 2.53 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 7 | 7 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| h1_on_step | base | 0.0 | 45 | 49 | 20.746 | 58.35 | 0.0000 |
| h1_on_step | tf32 | 0.0 | 45 | 49 | 10.740 | 30.21 | 0.0000 |
| h1_on_step | wrapper_cheb2 | 33.3 | 17 | — | 20.938 | 22.25 | 0.0000 |
| h1_on_step | wrapper_cheb2_tf32 | 33.3 | 17 | — | 10.760 | 11.43 | 0.0000 |
| h1_on_step | wrapper_cheb4_tf32 | 33.3 | 15 | — | 10.873 | 10.19 | 0.0000 |
| h1_on_step | safeguard_tf32 | 0.0 | 30 | 32 | 10.798 | 20.25 | 0.0000 |
| h1_on_step | safeguard_every4_tf32 | 0.0 | 35 | 36 | 2.952 | 6.46 | 0.0000 |
| h1_on_step | wrapper_cheb2_safeguard_tf32 | 0.0 | 12 | 73 | 10.928 | 8.20 | 0.0000 |
| h1_on_step | wrapper_cheb2_safeguard_every4_tf32 | 0.0 | 28 | 72 | 2.970 | 5.20 | 0.0000 |
| m0_on_step | base | 33.3 | 43 | — | 20.579 | 55.31 | 0.0000 |
| m0_on_step | tf32 | 33.3 | 43 | — | 10.600 | 28.49 | 0.0000 |
| m0_on_step | wrapper_cheb2 | 33.3 | 17 | — | 20.660 | 21.95 | 0.0000 |
| m0_on_step | wrapper_cheb2_tf32 | 33.3 | 17 | — | 10.684 | 11.35 | 0.0000 |
| m0_on_step | wrapper_cheb4_tf32 | 33.3 | 14 | — | 10.836 | 9.48 | 0.0000 |
| m0_on_step | safeguard_tf32 | 0.0 | 33 | 291 | 10.706 | 22.08 | 0.0000 |
| m0_on_step | safeguard_every4_tf32 | 0.0 | 35 | 309 | 2.933 | 6.42 | 0.0000 |
| m0_on_step | wrapper_cheb2_safeguard_tf32 | 0.0 | 13 | 276 | 10.786 | 8.76 | 0.0000 |
| m0_on_step | wrapper_cheb2_safeguard_every4_tf32 | 0.0 | 27 | 277 | 2.952 | 4.98 | 0.0000 |
| notebook | base | 33.3 | 50 | — | 20.522 | 64.13 | 1.0000 |
| notebook | tf32 | 33.3 | 50 | — | 10.584 | 33.07 | 0.0000 |
| notebook | wrapper_cheb2 | 33.3 | 15 | — | 20.644 | 19.35 | 0.0000 |
| notebook | wrapper_cheb2_tf32 | 33.3 | 15 | — | 10.689 | 10.02 | 0.0000 |
| notebook | wrapper_cheb4_tf32 | 33.3 | 13 | — | 10.791 | 8.77 | 0.0000 |
| notebook | safeguard_tf32 | 0.0 | 31 | 291 | 10.710 | 20.75 | 0.0000 |
| notebook | safeguard_every4_tf32 | 0.0 | 31 | 307 | 2.937 | 5.69 | 0.0000 |
| notebook | wrapper_cheb2_safeguard_tf32 | 0.0 | 13 | 265 | 10.788 | 8.77 | 0.0000 |
| notebook | wrapper_cheb2_safeguard_every4_tf32 | 0.0 | 23 | 276 | 2.961 | 4.26 | 0.0000 |

## poisson 128² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook base |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 316 | 319 |  | 1.77 |  |
| **jacobi4** (GPU) |  | 0.0 | 136 | 137 |  | 0.85 |  |
| **chebyshev8** (GPU) |  | 0.0 | 64 | 65 |  | 0.52 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 6 | 7 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| h1_on_step | base | 0.0 | 20 | 21 | 3.053 | 3.82 | 0.0000 |
| h1_on_step | tf32 | 0.0 | 20 | 21 | 0.828 | 1.03 | 0.0000 |
| h1_on_step | wrapper_cheb2 | 33.3 | 9 | — | 3.094 | 1.74 | 0.0993 |
| h1_on_step | wrapper_cheb2_tf32 | 33.3 | 9 | — | 0.862 | 0.49 | 0.0993 |
| h1_on_step | wrapper_cheb4_tf32 | 33.3 | 9 | — | 0.880 | 0.50 | 0.0994 |
| h1_on_step | safeguard_tf32 | 0.0 | 15 | 15 | 0.885 | 0.83 | 0.0000 |
| h1_on_step | safeguard_every4_tf32 | 0.0 | 24 | 26 | 0.368 | 0.55 | 0.0000 |
| h1_on_step | wrapper_cheb2_safeguard_tf32 | 0.0 | 9 | 44 | 0.903 | 0.51 | 0.0000 |
| h1_on_step | wrapper_cheb2_safeguard_every4_tf32 | 0.0 | 25 | 40 | 0.368 | 0.58 | 0.0000 |
| m0_on_step | base | 0.0 | 21 | 24 | 3.058 | 4.01 | 0.0000 |
| m0_on_step | tf32 | 0.0 | 21 | 24 | 0.818 | 1.07 | 0.0000 |
| m0_on_step | wrapper_cheb2 | 33.3 | 10 | — | 3.084 | 1.93 | 0.0989 |
| m0_on_step | wrapper_cheb2_tf32 | 33.3 | 10 | — | 0.847 | 0.53 | 0.0989 |
| m0_on_step | wrapper_cheb4_tf32 | 33.3 | 9 | — | 0.865 | 0.49 | 0.0989 |
| m0_on_step | safeguard_tf32 | 0.0 | 15 | 16 | 0.870 | 0.82 | 0.0000 |
| m0_on_step | safeguard_every4_tf32 | 0.0 | 23 | 25 | 0.369 | 0.53 | 0.0000 |
| m0_on_step | wrapper_cheb2_safeguard_tf32 | 0.0 | 10 | 118 | 0.886 | 0.55 | 0.8280 |
| m0_on_step | wrapper_cheb2_safeguard_every4_tf32 | 0.0 | 25 | 123 | 0.376 | 0.59 | 0.3746 |
| notebook | base | 0.0 | 35 | 80 | 3.048 | 6.67 | 1.0000 |
| notebook | tf32 | 4.3 | 35 | 160 | 0.819 | 1.79 | 0.0000 |
| notebook | wrapper_cheb2 | 0.0 | 7 | 10 | 3.075 | 1.35 | 0.0000 |
| notebook | wrapper_cheb2_tf32 | 0.0 | 7 | 10 | 0.843 | 0.37 | 0.0000 |
| notebook | wrapper_cheb4_tf32 | 0.0 | 6 | 7 | 0.860 | 0.32 | 0.0000 |
| notebook | safeguard_tf32 | 0.0 | 15 | 20 | 0.866 | 0.81 | 0.0000 |
| notebook | safeguard_every4_tf32 | 0.0 | 19 | 22 | 0.368 | 0.44 | 0.0000 |
| notebook | wrapper_cheb2_safeguard_tf32 | 0.0 | 7 | 10 | 0.883 | 0.39 | 0.0000 |
| notebook | wrapper_cheb2_safeguard_every4_tf32 | 0.0 | 17 | 21 | 0.371 | 0.39 | 0.0000 |

## poisson 256² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook base |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 632 | 639 |  | 7.20 |  |
| **jacobi4** (GPU) |  | 0.0 | 271 | 275 |  | 3.95 |  |
| **chebyshev8** (GPU) |  | 0.0 | 128 | 130 |  | 2.73 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 8 | 8 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| h1_on_step | base | 0.0 | 54 | 56 | 9.618 | 32.46 | 0.0000 |
| h1_on_step | tf32 | 0.0 | 54 | 57 | 2.399 | 8.10 | 0.0000 |
| h1_on_step | wrapper_cheb2 | 33.3 | 19 | — | 9.685 | 11.50 | 0.0996 |
| h1_on_step | wrapper_cheb2_tf32 | 33.3 | 19 | — | 2.482 | 2.95 | 0.0996 |
| h1_on_step | wrapper_cheb4_tf32 | 33.3 | 16 | — | 2.579 | 2.58 | 0.0996 |
| h1_on_step | safeguard_tf32 | 0.0 | 35 | 39 | 2.490 | 5.45 | 0.0000 |
| h1_on_step | safeguard_every4_tf32 | 0.0 | 38 | 39 | 0.891 | 2.12 | 0.0000 |
| h1_on_step | wrapper_cheb2_safeguard_tf32 | 0.0 | 13 | 114 | 2.560 | 2.08 | 0.0000 |
| h1_on_step | wrapper_cheb2_safeguard_every4_tf32 | 0.0 | 30 | 103 | 0.913 | 1.71 | 0.0000 |
| m0_on_step | base | 0.0 | 57 | 59 | 9.563 | 34.07 | 0.0000 |
| m0_on_step | tf32 | 0.0 | 57 | 59 | 2.384 | 8.49 | 0.0000 |
| m0_on_step | wrapper_cheb2 | 33.3 | 19 | — | 9.643 | 11.45 | 0.0996 |
| m0_on_step | wrapper_cheb2_tf32 | 33.3 | 19 | — | 2.473 | 2.94 | 0.0996 |
| m0_on_step | wrapper_cheb4_tf32 | 33.3 | 16 | — | 2.560 | 2.56 | 0.0996 |
| m0_on_step | safeguard_tf32 | 0.0 | 37 | 39 | 2.484 | 5.75 | 0.0000 |
| m0_on_step | safeguard_every4_tf32 | 0.0 | 38 | 39 | 0.876 | 2.08 | 0.0000 |
| m0_on_step | wrapper_cheb2_safeguard_tf32 | 0.0 | 13 | 293 | 2.547 | 2.07 | 0.0997 |
| m0_on_step | wrapper_cheb2_safeguard_every4_tf32 | 0.0 | 29 | 333 | 0.897 | 1.63 | 0.0997 |
| notebook | base | 0.0 | 57 | 86 | 9.544 | 34.00 | 1.0000 |
| notebook | tf32 | 0.0 | 62 | 84 | 2.384 | 9.24 | 0.0000 |
| notebook | wrapper_cheb2 | 0.0 | 16 | 21 | 9.627 | 9.63 | 0.0000 |
| notebook | wrapper_cheb2_tf32 | 0.0 | 16 | 21 | 2.466 | 2.47 | 0.0000 |
| notebook | wrapper_cheb4_tf32 | 0.0 | 14 | 15 | 2.563 | 2.24 | 0.0000 |
| notebook | safeguard_tf32 | 0.0 | 33 | 44 | 2.481 | 5.12 | 0.0000 |
| notebook | safeguard_every4_tf32 | 0.0 | 35 | 43 | 0.883 | 1.93 | 0.0000 |
| notebook | wrapper_cheb2_safeguard_tf32 | 0.0 | 14 | 17 | 2.546 | 2.23 | 0.0000 |
| notebook | wrapper_cheb2_safeguard_every4_tf32 | 0.0 | 25 | 31 | 0.899 | 1.41 | 0.0000 |
