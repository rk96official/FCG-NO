# E10m two-level-trained models deployed

## diffusion 128² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 300 | 324 |  | 1.37 |  |
| **jacobi4** (GPU) |  | 0.0 | 126 | 137 |  | 0.66 |  |
| **chebyshev8** (GPU) |  | 0.0 | 59 | 64 |  | 0.41 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 6 | 6 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| h1_on_step_2l2 | two_level_c4_cheb2 | 0.0 | 10 | 10 | 2.258 | 1.41 | — |
| h1_on_step_2l2 | two_level_c2_cheb2_tf32 | 0.0 | 6 | 6 | 1.356 | 0.51 | — |
| h1_on_step_2l2 | two_level_c4_cheb2_tf32 | 0.0 | 10 | 10 | 0.992 | 0.62 | — |
| h1_on_step_2l2 | two_level_c8_cheb2_tf32 | 0.0 | 75 | 116 | 0.860 | 4.03 | — |
| h1_on_step_2l2 | two_level_c16_cheb2_tf32 | 0.0 | 115 | 125 | 0.884 | 6.35 | — |
| h1_on_step_2l2 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 9 | 10 | 1.020 | 0.57 | — |
| h1_on_step_2l4 | two_level_c4_cheb2 | 0.0 | 7 | 7 | 2.262 | 0.99 | — |
| h1_on_step_2l4 | two_level_c2_cheb2_tf32 | 0.0 | 10 | 11 | 1.385 | 0.87 | — |
| h1_on_step_2l4 | two_level_c4_cheb2_tf32 | 0.0 | 7 | 7 | 1.016 | 0.44 | — |
| h1_on_step_2l4 | two_level_c8_cheb2_tf32 | 0.0 | 101 | 116 | 0.883 | 5.57 | — |
| h1_on_step_2l4 | two_level_c16_cheb2_tf32 | 0.0 | 127 | 147 | 0.913 | 7.25 | — |
| h1_on_step_2l4 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 7 | 7 | 1.048 | 0.46 | — |
| m0_on_step_2l4 | two_level_c4_cheb2 | 0.0 | 7 | 7 | 2.243 | 0.98 | — |
| m0_on_step_2l4 | two_level_c2_cheb2_tf32 | 0.0 | 10 | 10 | 1.359 | 0.85 | — |
| m0_on_step_2l4 | two_level_c4_cheb2_tf32 | 0.0 | 7 | 7 | 1.188 | 0.52 | — |
| m0_on_step_2l4 | two_level_c8_cheb2_tf32 | 0.0 | 82 | 117 | 1.204 | 6.17 | — |
| m0_on_step_2l4 | two_level_c16_cheb2_tf32 | 0.0 | 118 | 127 | 1.130 | 8.34 | — |
| m0_on_step_2l4 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 7 | 7 | 1.191 | 0.52 | — |

## diffusion 256² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 606 | 661 |  | 6.29 |  |
| **jacobi4** (GPU) |  | 0.0 | 255 | 275 |  | 3.64 |  |
| **chebyshev8** (GPU) |  | 0.0 | 119 | 129 |  | 2.53 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 7 | 7 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| h1_on_step_2l2 | two_level_c4_cheb2 | 0.0 | 8 | 9 | 3.432 | 1.72 | — |
| h1_on_step_2l2 | two_level_c2_cheb2_tf32 | 0.0 | 12 | 13 | 3.441 | 2.58 | — |
| h1_on_step_2l2 | two_level_c4_cheb2_tf32 | 0.0 | 8 | 9 | 1.576 | 0.79 | — |
| h1_on_step_2l2 | two_level_c8_cheb2_tf32 | 0.0 | 15 | 16 | 1.185 | 1.11 | — |
| h1_on_step_2l2 | two_level_c16_cheb2_tf32 | 0.0 | 161 | 258 | 1.058 | 10.65 | — |
| h1_on_step_2l2 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 8 | 9 | 1.654 | 0.83 | — |
| h1_on_step_2l4 | two_level_c4_cheb2 | 0.0 | 13 | 14 | 3.416 | 2.78 | — |
| h1_on_step_2l4 | two_level_c2_cheb2_tf32 | 0.0 | 22 | 25 | 3.435 | 4.72 | — |
| h1_on_step_2l4 | two_level_c4_cheb2_tf32 | 0.0 | 13 | 14 | 1.579 | 1.28 | — |
| h1_on_step_2l4 | two_level_c8_cheb2_tf32 | 0.0 | 13 | 13 | 1.186 | 0.96 | — |
| h1_on_step_2l4 | two_level_c16_cheb2_tf32 | 0.0 | 224 | 245 | 1.049 | 14.69 | — |
| h1_on_step_2l4 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 13 | 14 | 1.652 | 1.34 | — |
| m0_on_step_2l4 | two_level_c4_cheb2 | 0.0 | 13 | 13 | 3.384 | 2.75 | — |
| m0_on_step_2l4 | two_level_c2_cheb2_tf32 | 0.0 | 21 | 23 | 3.368 | 4.42 | — |
| m0_on_step_2l4 | two_level_c4_cheb2_tf32 | 0.0 | 13 | 13 | 1.514 | 1.23 | — |
| m0_on_step_2l4 | two_level_c8_cheb2_tf32 | 0.0 | 13 | 13 | 1.188 | 0.97 | — |
| m0_on_step_2l4 | two_level_c16_cheb2_tf32 | 0.0 | 169 | 237 | 1.205 | 12.73 | — |
| m0_on_step_2l4 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 13 | 13 | 1.598 | 1.30 | — |

## diffusion 512² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 1234 | 1351 |  | 53.21 |  |
| **jacobi4** (GPU) |  | 0.0 | 517 | 558 |  | 31.35 |  |
| **chebyshev8** (GPU) |  | 0.0 | 241 | 262 |  | 20.13 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 6 | 7 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| h1_on_step_2l2 | two_level_c4_cheb2 | 0.0 | 20 | 21 | 8.066 | 10.08 | — |
| h1_on_step_2l2 | two_level_c2_cheb2_tf32 | 0.0 | 36 | 39 | 12.171 | 27.38 | — |
| h1_on_step_2l2 | two_level_c4_cheb2_tf32 | 0.0 | 20 | 21 | 4.544 | 5.68 | — |
| h1_on_step_2l2 | two_level_c8_cheb2_tf32 | 0.0 | 14 | 14 | 2.516 | 2.20 | — |
| h1_on_step_2l2 | two_level_c16_cheb2_tf32 | 0.0 | 27 | 29 | 2.082 | 3.51 | — |
| h1_on_step_2l2 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 19 | 20 | 5.247 | 6.23 | — |
| h1_on_step_2l4 | two_level_c4_cheb2 | 0.0 | 29 | 31 | 8.067 | 14.62 | — |
| h1_on_step_2l4 | two_level_c2_cheb2_tf32 | 0.0 | 59 | 68 | 12.184 | 44.93 | — |
| h1_on_step_2l4 | two_level_c4_cheb2_tf32 | 0.0 | 29 | 31 | 4.571 | 8.29 | — |
| h1_on_step_2l4 | two_level_c8_cheb2_tf32 | 0.0 | 16 | 17 | 2.528 | 2.53 | — |
| h1_on_step_2l4 | two_level_c16_cheb2_tf32 | 0.0 | 24 | 25 | 2.101 | 3.15 | — |
| h1_on_step_2l4 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 29 | 31 | 5.255 | 9.52 | — |
| m0_on_step_2l4 | two_level_c4_cheb2 | 0.0 | 29 | 31 | 7.844 | 14.22 | — |
| m0_on_step_2l4 | two_level_c2_cheb2_tf32 | 0.0 | 60 | 64 | 11.849 | 44.44 | — |
| m0_on_step_2l4 | two_level_c4_cheb2_tf32 | 0.0 | 29 | 31 | 4.259 | 7.72 | — |
| m0_on_step_2l4 | two_level_c8_cheb2_tf32 | 0.0 | 16 | 17 | 2.356 | 2.36 | — |
| m0_on_step_2l4 | two_level_c16_cheb2_tf32 | 0.0 | 24 | 25 | 1.965 | 2.95 | — |
| m0_on_step_2l4 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 29 | 31 | 5.042 | 9.14 | — |

## poisson 128² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 316 | 319 |  | 1.77 |  |
| **jacobi4** (GPU) |  | 0.0 | 136 | 137 |  | 0.85 |  |
| **chebyshev8** (GPU) |  | 0.0 | 64 | 65 |  | 0.52 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 6 | 7 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| h1_on_step_2l2 | two_level_c4_cheb2 | 0.0 | 11 | 12 | 0.985 | 0.68 | — |
| h1_on_step_2l2 | two_level_c2_cheb2_tf32 | 0.0 | 6 | 7 | 0.509 | 0.19 | — |
| h1_on_step_2l2 | two_level_c4_cheb2_tf32 | 0.0 | 11 | 12 | 0.418 | 0.29 | — |
| h1_on_step_2l2 | two_level_c8_cheb2_tf32 | 0.0 | 68 | 77 | 0.387 | 1.64 | — |
| h1_on_step_2l2 | two_level_c16_cheb2_tf32 | 0.0 | 151 | 172 | 0.383 | 3.61 | — |
| h1_on_step_2l2 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 11 | 12 | 0.459 | 0.32 | — |
| h1_on_step_2l4 | two_level_c4_cheb2 | 0.0 | 7 | 7 | 0.969 | 0.42 | — |
| h1_on_step_2l4 | two_level_c2_cheb2_tf32 | 0.0 | 12 | 13 | 0.504 | 0.38 | — |
| h1_on_step_2l4 | two_level_c4_cheb2_tf32 | 0.0 | 7 | 7 | 0.414 | 0.18 | — |
| h1_on_step_2l4 | two_level_c8_cheb2_tf32 | 0.0 | 41 | 77 | 0.381 | 0.98 | — |
| h1_on_step_2l4 | two_level_c16_cheb2_tf32 | 0.0 | 135 | 160 | 0.378 | 3.19 | — |
| h1_on_step_2l4 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 7 | 7 | 0.448 | 0.20 | — |
| m0_on_step_2l4 | two_level_c4_cheb2 | 0.0 | 7 | 7 | 0.950 | 0.42 | — |
| m0_on_step_2l4 | two_level_c2_cheb2_tf32 | 0.0 | 11 | 12 | 0.489 | 0.34 | — |
| m0_on_step_2l4 | two_level_c4_cheb2_tf32 | 0.0 | 7 | 7 | 0.387 | 0.17 | — |
| m0_on_step_2l4 | two_level_c8_cheb2_tf32 | 0.0 | 47 | 54 | 0.367 | 1.08 | — |
| m0_on_step_2l4 | two_level_c16_cheb2_tf32 | 0.0 | 131 | 148 | 0.357 | 2.93 | — |
| m0_on_step_2l4 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 7 | 7 | 0.422 | 0.18 | — |

## poisson 256² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 632 | 639 |  | 7.20 |  |
| **jacobi4** (GPU) |  | 0.0 | 271 | 275 |  | 3.95 |  |
| **chebyshev8** (GPU) |  | 0.0 | 128 | 130 |  | 2.73 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 8 | 8 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| h1_on_step_2l2 | two_level_c4_cheb2 | 0.0 | 10 | 10 | 1.574 | 0.98 | — |
| h1_on_step_2l2 | two_level_c2_cheb2_tf32 | 0.0 | 15 | 15 | 1.135 | 1.06 | — |
| h1_on_step_2l2 | two_level_c4_cheb2_tf32 | 0.0 | 10 | 10 | 0.709 | 0.44 | — |
| h1_on_step_2l2 | two_level_c8_cheb2_tf32 | 0.0 | 17 | 18 | 0.605 | 0.64 | — |
| h1_on_step_2l2 | two_level_c16_cheb2_tf32 | 0.0 | 167 | 183 | 0.586 | 6.12 | — |
| h1_on_step_2l2 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 10 | 10 | 0.786 | 0.49 | — |
| h1_on_step_2l4 | two_level_c4_cheb2 | 0.0 | 16 | 17 | 1.555 | 1.56 | — |
| h1_on_step_2l4 | two_level_c2_cheb2_tf32 | 0.0 | 27 | 29 | 1.122 | 1.89 | — |
| h1_on_step_2l4 | two_level_c4_cheb2_tf32 | 0.0 | 16 | 17 | 0.697 | 0.70 | — |
| h1_on_step_2l4 | two_level_c8_cheb2_tf32 | 0.0 | 14 | 15 | 0.601 | 0.53 | — |
| h1_on_step_2l4 | two_level_c16_cheb2_tf32 | 0.0 | 96 | 190 | 0.559 | 3.36 | — |
| h1_on_step_2l4 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 15 | 17 | 0.772 | 0.72 | — |
| m0_on_step_2l4 | two_level_c4_cheb2 | 0.0 | 15 | 15 | 1.507 | 1.41 | — |
| m0_on_step_2l4 | two_level_c2_cheb2_tf32 | 0.0 | 25 | 27 | 1.036 | 1.62 | — |
| m0_on_step_2l4 | two_level_c4_cheb2_tf32 | 0.0 | 15 | 15 | 0.648 | 0.61 | — |
| m0_on_step_2l4 | two_level_c8_cheb2_tf32 | 0.0 | 14 | 15 | 0.551 | 0.48 | — |
| m0_on_step_2l4 | two_level_c16_cheb2_tf32 | 0.0 | 100 | 133 | 0.521 | 3.25 | — |
| m0_on_step_2l4 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 15 | 15 | 0.723 | 0.68 | — |

## poisson 512² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 1270 | 1291 |  | 55.48 |  |
| **jacobi4** (GPU) |  | 0.0 | 546 | 555 |  | 32.32 |  |
| **chebyshev8** (GPU) |  | 0.0 | 258 | 261 |  | 21.10 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 7 | 7 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| h1_on_step_2l2 | two_level_c4_cheb2 | 0.0 | 22 | 23 | 4.369 | 6.01 | — |
| h1_on_step_2l2 | two_level_c2_cheb2_tf32 | 0.0 | 40 | 46 | 3.882 | 9.71 | — |
| h1_on_step_2l2 | two_level_c4_cheb2_tf32 | 0.0 | 22 | 23 | 2.162 | 2.97 | — |
| h1_on_step_2l2 | two_level_c8_cheb2_tf32 | 0.0 | 16 | 17 | 1.662 | 1.66 | — |
| h1_on_step_2l2 | two_level_c16_cheb2_tf32 | 0.0 | 33 | 35 | 1.517 | 3.13 | — |
| h1_on_step_2l2 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 21 | 22 | 2.911 | 3.82 | — |
| h1_on_step_2l4 | two_level_c4_cheb2 | 0.0 | 37 | 40 | 4.354 | 10.07 | — |
| h1_on_step_2l4 | two_level_c2_cheb2_tf32 | 0.0 | 93 | 104 | 3.911 | 22.73 | — |
| h1_on_step_2l4 | two_level_c4_cheb2_tf32 | 0.0 | 38 | 40 | 2.178 | 5.17 | — |
| h1_on_step_2l4 | two_level_c8_cheb2_tf32 | 0.0 | 19 | 20 | 1.649 | 1.96 | — |
| h1_on_step_2l4 | two_level_c16_cheb2_tf32 | 0.0 | 28 | 29 | 1.520 | 2.66 | — |
| h1_on_step_2l4 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 36 | 39 | 2.934 | 6.60 | — |
| m0_on_step_2l4 | two_level_c4_cheb2 | 0.0 | 36 | 38 | 4.107 | 9.24 | — |
| m0_on_step_2l4 | two_level_c2_cheb2_tf32 | 0.0 | 84 | 101 | 3.487 | 18.31 | — |
| m0_on_step_2l4 | two_level_c4_cheb2_tf32 | 0.0 | 36 | 38 | 1.909 | 4.30 | — |
| m0_on_step_2l4 | two_level_c8_cheb2_tf32 | 0.0 | 19 | 20 | 1.487 | 1.77 | — |
| m0_on_step_2l4 | two_level_c16_cheb2_tf32 | 0.0 | 29 | 30 | 1.375 | 2.49 | — |
| m0_on_step_2l4 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 35 | 37 | 2.631 | 5.75 | — |
