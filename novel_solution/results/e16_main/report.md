# E16 multi-level

## diffusion 128² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 300 | 324 |  | 1.37 |  |
| **jacobi4** (GPU) |  | 0.0 | 126 | 137 |  | 0.66 |  |
| **chebyshev8** (GPU) |  | 0.0 | 59 | 64 |  | 0.41 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 6 | 6 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| final_2l2 | multi_b64_tf32 | 0.0 | 6 | 6 | 0.506 | 0.19 | — |
| final_2l2 | multi_b32_tf32 | 0.0 | 8 | 9 | 0.458 | 0.23 | — |
| final_2l2 | recursive_b32_max64_tf32 | 0.0 | 6 | 6 | 0.815 | 0.31 | — |
| final_2l2 | recursive_b32_max128_tf32 | 0.0 | 5 | 6 | 1.533 | 0.48 | — |
| final_2l4 | multi_b64_tf32 | 0.0 | 11 | 11 | 0.504 | 0.35 | — |
| final_2l4 | multi_b32_tf32 | 0.0 | 6 | 7 | 0.458 | 0.17 | — |
| final_2l4 | recursive_b32_max64_tf32 | 0.0 | 6 | 6 | 0.810 | 0.30 | — |
| final_2l4 | recursive_b32_max128_tf32 | 0.0 | 5 | 6 | 1.529 | 0.48 | — |
| ml_b32 | multi_b64_tf32 | 0.0 | 17 | 18 | 0.505 | 0.54 | — |
| ml_b32 | multi_b32_tf32 | 0.0 | 6 | 6 | 0.458 | 0.17 | — |
| ml_b32 | recursive_b32_max64_tf32 | 0.0 | 6 | 7 | 0.812 | 0.30 | — |
| ml_b32 | recursive_b32_max128_tf32 | 0.0 | 6 | 6 | 1.535 | 0.58 | — |
| rec_b32 | multi_b64_tf32 | 0.0 | 12 | 13 | 0.506 | 0.38 | — |
| rec_b32 | multi_b32_tf32 | 0.0 | 7 | 12 | 0.457 | 0.20 | — |
| rec_b32 | recursive_b32_max64_tf32 | 0.0 | 6 | 6 | 0.814 | 0.31 | — |
| rec_b32 | recursive_b32_max128_tf32 | 0.0 | 6 | 6 | 1.533 | 0.57 | — |

## diffusion 256² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 606 | 661 |  | 6.29 |  |
| **jacobi4** (GPU) |  | 0.0 | 255 | 275 |  | 3.64 |  |
| **chebyshev8** (GPU) |  | 0.0 | 119 | 129 |  | 2.53 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 7 | 7 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| final_2l2 | multi_b64_tf32 | 0.0 | 7 | 7 | 0.812 | 0.36 | — |
| final_2l2 | multi_b32_tf32 | 0.0 | 8 | 9 | 0.768 | 0.38 | — |
| final_2l2 | recursive_b32_max64_tf32 | 0.0 | 6 | 6 | 1.126 | 0.42 | — |
| final_2l2 | recursive_b32_max128_tf32 | 0.0 | 6 | 7 | 1.859 | 0.70 | — |
| final_2l4 | multi_b64_tf32 | 0.0 | 10 | 11 | 0.811 | 0.51 | — |
| final_2l4 | multi_b32_tf32 | 0.0 | 7 | 7 | 0.765 | 0.33 | — |
| final_2l4 | recursive_b32_max64_tf32 | 0.0 | 6 | 7 | 1.125 | 0.42 | — |
| final_2l4 | recursive_b32_max128_tf32 | 0.0 | 7 | 7 | 1.855 | 0.81 | — |
| ml_b32 | multi_b64_tf32 | 0.0 | 16 | 17 | 0.814 | 0.81 | — |
| ml_b32 | multi_b32_tf32 | 0.0 | 6 | 6 | 0.769 | 0.29 | — |
| ml_b32 | recursive_b32_max64_tf32 | 0.0 | 7 | 7 | 1.128 | 0.49 | — |
| ml_b32 | recursive_b32_max128_tf32 | 0.0 | 7 | 7 | 1.859 | 0.81 | — |
| rec_b32 | multi_b64_tf32 | 0.0 | 9 | 11 | 0.813 | 0.46 | — |
| rec_b32 | multi_b32_tf32 | 0.0 | 7 | 13 | 0.765 | 0.33 | — |
| rec_b32 | recursive_b32_max64_tf32 | 0.0 | 6 | 6 | 1.123 | 0.42 | — |
| rec_b32 | recursive_b32_max128_tf32 | 0.0 | 6 | 7 | 1.861 | 0.70 | — |

## diffusion 512² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 1234 | 1351 |  | 53.21 |  |
| **jacobi4** (GPU) |  | 0.0 | 517 | 558 |  | 31.35 |  |
| **chebyshev8** (GPU) |  | 0.0 | 241 | 262 |  | 20.13 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 6 | 7 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| final_2l2 | multi_b64_tf32 | 0.0 | 7 | 7 | 2.250 | 0.98 | — |
| final_2l2 | multi_b32_tf32 | 0.0 | 8 | 9 | 2.211 | 1.11 | — |
| final_2l2 | recursive_b32_max64_tf32 | 0.0 | 6 | 6 | 2.555 | 0.96 | — |
| final_2l2 | recursive_b32_max128_tf32 | 0.0 | 7 | 7 | 3.291 | 1.44 | — |
| final_2l4 | multi_b64_tf32 | 0.0 | 10 | 11 | 2.247 | 1.40 | — |
| final_2l4 | multi_b32_tf32 | 0.0 | 7 | 7 | 2.197 | 0.96 | — |
| final_2l4 | recursive_b32_max64_tf32 | 0.0 | 7 | 7 | 2.545 | 1.11 | — |
| final_2l4 | recursive_b32_max128_tf32 | 0.0 | 7 | 7 | 3.289 | 1.44 | — |
| ml_b32 | multi_b64_tf32 | 0.0 | 16 | 17 | 2.252 | 2.25 | — |
| ml_b32 | multi_b32_tf32 | 0.0 | 6 | 7 | 2.191 | 0.82 | — |
| ml_b32 | recursive_b32_max64_tf32 | 0.0 | 7 | 7 | 2.549 | 1.12 | — |
| ml_b32 | recursive_b32_max128_tf32 | 0.0 | 7 | 7 | 3.289 | 1.44 | — |
| rec_b32 | multi_b64_tf32 | 0.0 | 10 | 11 | 2.230 | 1.39 | — |
| rec_b32 | multi_b32_tf32 | 0.0 | 8 | 13 | 2.184 | 1.09 | — |
| rec_b32 | recursive_b32_max64_tf32 | 0.0 | 6 | 7 | 2.536 | 0.95 | — |
| rec_b32 | recursive_b32_max128_tf32 | 0.0 | 7 | 7 | 3.272 | 1.43 | — |

## diffusion 1024² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 2549 | 2759 |  | 400.94 |  |
| **jacobi4** (GPU) |  | 0.0 | 1076 | 1129 |  | 239.06 |  |
| **chebyshev8** (GPU) |  | 0.0 | 502 | 532 |  | 172.76 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 8 | 8 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| final_2l2 | multi_b64_tf32 | 0.0 | 7 | 7 | 9.377 | 4.10 | — |
| final_2l2 | multi_b32_tf32 | 0.0 | 9 | 9 | 9.343 | 5.26 | — |
| final_2l2 | recursive_b32_max64_tf32 | 0.0 | 6 | 7 | 9.785 | 3.67 | — |
| final_2l2 | recursive_b32_max128_tf32 | 0.0 | 7 | 7 | 10.537 | 4.61 | — |
| final_2l4 | multi_b64_tf32 | 0.0 | 10 | 11 | 9.382 | 5.86 | — |
| final_2l4 | multi_b32_tf32 | 0.0 | 7 | 7 | 9.344 | 4.09 | — |
| final_2l4 | recursive_b32_max64_tf32 | 0.0 | 7 | 7 | 9.730 | 4.26 | — |
| final_2l4 | recursive_b32_max128_tf32 | 0.0 | 7 | 7 | 10.542 | 4.61 | — |
| ml_b32 | multi_b64_tf32 | 0.0 | 16 | 17 | 9.370 | 9.37 | — |
| ml_b32 | multi_b32_tf32 | 0.0 | 7 | 7 | 9.351 | 4.09 | — |
| ml_b32 | recursive_b32_max64_tf32 | 0.0 | 7 | 7 | 9.698 | 4.24 | — |
| ml_b32 | recursive_b32_max128_tf32 | 0.0 | 7 | 7 | 10.513 | 4.60 | — |
| rec_b32 | multi_b64_tf32 | 0.0 | 10 | 11 | 9.352 | 5.84 | — |
| rec_b32 | multi_b32_tf32 | 0.0 | 8 | 14 | 9.307 | 4.65 | — |
| rec_b32 | recursive_b32_max64_tf32 | 0.0 | 6 | 7 | 9.668 | 3.63 | — |
| rec_b32 | recursive_b32_max128_tf32 | 0.0 | 7 | 7 | 10.476 | 4.58 | — |

## poisson 128² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 316 | 319 |  | 1.77 |  |
| **jacobi4** (GPU) |  | 0.0 | 136 | 137 |  | 0.85 |  |
| **chebyshev8** (GPU) |  | 0.0 | 64 | 65 |  | 0.52 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 6 | 7 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| final_2l2 | multi_b64_tf32 | 0.0 | 7 | 7 | 0.506 | 0.22 | — |
| final_2l2 | multi_b32_tf32 | 0.0 | 9 | 10 | 0.455 | 0.26 | — |
| final_2l2 | recursive_b32_max64_tf32 | 0.0 | 6 | 7 | 0.815 | 0.31 | — |
| final_2l2 | recursive_b32_max128_tf32 | 0.0 | 6 | 7 | 1.527 | 0.57 | — |
| final_2l4 | multi_b64_tf32 | 0.0 | 12 | 13 | 0.506 | 0.38 | — |
| final_2l4 | multi_b32_tf32 | 0.0 | 7 | 7 | 0.458 | 0.20 | — |
| final_2l4 | recursive_b32_max64_tf32 | 0.0 | 7 | 7 | 0.814 | 0.36 | — |
| final_2l4 | recursive_b32_max128_tf32 | 0.0 | 6 | 6 | 1.534 | 0.58 | — |
| ml_b32 | multi_b64_tf32 | 0.0 | 17 | 20 | 0.506 | 0.54 | — |
| ml_b32 | multi_b32_tf32 | 0.0 | 6 | 7 | 0.458 | 0.17 | — |
| ml_b32 | recursive_b32_max64_tf32 | 0.0 | 7 | 7 | 0.815 | 0.36 | — |
| ml_b32 | recursive_b32_max128_tf32 | 0.0 | 7 | 8 | 1.540 | 0.67 | — |
| rec_b32 | multi_b64_tf32 | 0.0 | 12 | 13 | 0.506 | 0.38 | — |
| rec_b32 | multi_b32_tf32 | 0.0 | 8 | 8 | 0.460 | 0.23 | — |
| rec_b32 | recursive_b32_max64_tf32 | 0.0 | 6 | 6 | 0.811 | 0.30 | — |
| rec_b32 | recursive_b32_max128_tf32 | 0.0 | 7 | 7 | 1.529 | 0.67 | — |

## poisson 256² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 632 | 639 |  | 7.20 |  |
| **jacobi4** (GPU) |  | 0.0 | 271 | 275 |  | 3.95 |  |
| **chebyshev8** (GPU) |  | 0.0 | 128 | 130 |  | 2.73 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 8 | 8 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| final_2l2 | multi_b64_tf32 | 0.0 | 7 | 7 | 0.816 | 0.36 | — |
| final_2l2 | multi_b32_tf32 | 0.0 | 9 | 10 | 0.768 | 0.43 | — |
| final_2l2 | recursive_b32_max64_tf32 | 0.0 | 7 | 7 | 1.131 | 0.49 | — |
| final_2l2 | recursive_b32_max128_tf32 | 0.0 | 7 | 7 | 1.861 | 0.81 | — |
| final_2l4 | multi_b64_tf32 | 0.0 | 11 | 12 | 0.811 | 0.56 | — |
| final_2l4 | multi_b32_tf32 | 0.0 | 7 | 7 | 0.760 | 0.33 | — |
| final_2l4 | recursive_b32_max64_tf32 | 0.0 | 7 | 7 | 1.116 | 0.49 | — |
| final_2l4 | recursive_b32_max128_tf32 | 0.0 | 7 | 7 | 1.851 | 0.81 | — |
| ml_b32 | multi_b64_tf32 | 0.0 | 16 | 19 | 0.813 | 0.81 | — |
| ml_b32 | multi_b32_tf32 | 0.0 | 7 | 7 | 0.768 | 0.34 | — |
| ml_b32 | recursive_b32_max64_tf32 | 0.0 | 7 | 7 | 1.125 | 0.49 | — |
| ml_b32 | recursive_b32_max128_tf32 | 0.0 | 7 | 7 | 1.858 | 0.81 | — |
| rec_b32 | multi_b64_tf32 | 0.0 | 12 | 12 | 0.813 | 0.61 | — |
| rec_b32 | multi_b32_tf32 | 0.0 | 8 | 8 | 0.762 | 0.38 | — |
| rec_b32 | recursive_b32_max64_tf32 | 0.0 | 6 | 7 | 1.119 | 0.42 | — |
| rec_b32 | recursive_b32_max128_tf32 | 0.0 | 7 | 7 | 1.848 | 0.81 | — |

## poisson 512² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 1270 | 1291 |  | 55.48 |  |
| **jacobi4** (GPU) |  | 0.0 | 546 | 555 |  | 32.32 |  |
| **chebyshev8** (GPU) |  | 0.0 | 258 | 261 |  | 21.10 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 7 | 7 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| final_2l2 | multi_b64_tf32 | 0.0 | 7 | 7 | 2.219 | 0.97 | — |
| final_2l2 | multi_b32_tf32 | 0.0 | 9 | 11 | 2.171 | 1.22 | — |
| final_2l2 | recursive_b32_max64_tf32 | 0.0 | 7 | 7 | 2.530 | 1.11 | — |
| final_2l2 | recursive_b32_max128_tf32 | 0.0 | 7 | 7 | 3.272 | 1.43 | — |
| final_2l4 | multi_b64_tf32 | 0.0 | 11 | 12 | 2.224 | 1.53 | — |
| final_2l4 | multi_b32_tf32 | 0.0 | 7 | 7 | 2.175 | 0.95 | — |
| final_2l4 | recursive_b32_max64_tf32 | 0.0 | 7 | 7 | 2.534 | 1.11 | — |
| final_2l4 | recursive_b32_max128_tf32 | 0.0 | 7 | 7 | 3.272 | 1.43 | — |
| ml_b32 | multi_b64_tf32 | 0.0 | 16 | 19 | 2.220 | 2.22 | — |
| ml_b32 | multi_b32_tf32 | 0.0 | 7 | 7 | 2.167 | 0.95 | — |
| ml_b32 | recursive_b32_max64_tf32 | 0.0 | 7 | 7 | 2.536 | 1.11 | — |
| ml_b32 | recursive_b32_max128_tf32 | 0.0 | 7 | 7 | 3.275 | 1.43 | — |
| rec_b32 | multi_b64_tf32 | 0.0 | 12 | 12 | 2.220 | 1.66 | — |
| rec_b32 | multi_b32_tf32 | 0.0 | 8 | 8 | 2.174 | 1.09 | — |
| rec_b32 | recursive_b32_max64_tf32 | 0.0 | 7 | 7 | 2.529 | 1.11 | — |
| rec_b32 | recursive_b32_max128_tf32 | 0.0 | 7 | 7 | 3.272 | 1.43 | — |

## poisson 1024² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 2569 | 2604 |  | 402.25 |  |
| **jacobi4** (GPU) |  | 0.0 | 1103 | 1122 |  | 240.11 |  |
| **chebyshev8** (GPU) |  | 0.0 | 521 | 529 |  | 176.39 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 9 | 9 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| final_2l2 | multi_b64_tf32 | 0.0 | 7 | 8 | 9.189 | 4.02 | — |
| final_2l2 | multi_b32_tf32 | 0.0 | 9 | 10 | 9.207 | 5.18 | — |
| final_2l2 | recursive_b32_max64_tf32 | 0.0 | 7 | 7 | 9.543 | 4.17 | — |
| final_2l2 | recursive_b32_max128_tf32 | 0.0 | 7 | 7 | 10.349 | 4.53 | — |
| final_2l4 | multi_b64_tf32 | 0.0 | 11 | 12 | 9.210 | 6.33 | — |
| final_2l4 | multi_b32_tf32 | 0.0 | 7 | 7 | 9.234 | 4.04 | — |
| final_2l4 | recursive_b32_max64_tf32 | 0.0 | 7 | 7 | 9.587 | 4.19 | — |
| final_2l4 | recursive_b32_max128_tf32 | 0.0 | 7 | 7 | 10.368 | 4.54 | — |
| ml_b32 | multi_b64_tf32 | 0.0 | 17 | 19 | 9.223 | 9.80 | — |
| ml_b32 | multi_b32_tf32 | 0.0 | 7 | 7 | 9.224 | 4.04 | — |
| ml_b32 | recursive_b32_max64_tf32 | 0.0 | 7 | 7 | 9.541 | 4.17 | — |
| ml_b32 | recursive_b32_max128_tf32 | 0.0 | 7 | 7 | 10.364 | 4.53 | — |
| rec_b32 | multi_b64_tf32 | 0.0 | 12 | 12 | 9.224 | 6.92 | — |
| rec_b32 | multi_b32_tf32 | 0.0 | 8 | 9 | 9.230 | 4.61 | — |
| rec_b32 | recursive_b32_max64_tf32 | 0.0 | 7 | 7 | 9.620 | 4.21 | — |
| rec_b32 | recursive_b32_max128_tf32 | 0.0 | 7 | 7 | 10.390 | 4.55 | — |
