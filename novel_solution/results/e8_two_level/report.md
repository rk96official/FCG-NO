# E8 two-level

## diffusion 128² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook wrapper_cheb2_tf32 |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 300 | 324 |  | 1.37 |  |
| **jacobi4** (GPU) |  | 0.0 | 126 | 137 |  | 0.66 |  |
| **chebyshev8** (GPU) |  | 0.0 | 59 | 64 |  | 0.41 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 6 | 6 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| h1_on_step | wrapper_cheb2_tf32 | 33.7 | 10 | — | 3.171 | 1.98 | 0.0170 |
| h1_on_step | two_level_c2_cheb2_tf32 | 0.0 | 56 | 111 | 1.381 | 4.83 | 0.0583 |
| h1_on_step | two_level_c2_cheb4_tf32 | 0.0 | 39 | 70 | 1.407 | 3.43 | 0.0089 |
| h1_on_step | two_level_c4_cheb2_tf32 | 0.0 | 89 | 117 | 1.208 | 6.72 | 0.0888 |
| h1_on_step | two_level_c4_cheb4_tf32 | 0.0 | 56 | 71 | 1.213 | 4.25 | 0.0164 |
| h1_on_step | two_level_c2_cheb2_safeguard_tf32 | 0.0 | 53 | 107 | 1.431 | 4.74 | 0.0490 |
| h1_on_step | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 87 | 109 | 1.213 | 6.59 | 0.0854 |
| h1_on_step | two_level_c4_cheb4_safeguard_tf32 | 0.0 | 52 | 65 | 1.212 | 3.94 | 0.0107 |
| m0_on_step | wrapper_cheb2_tf32 | 33.3 | 10 | — | 3.176 | 1.99 | 0.0034 |
| m0_on_step | two_level_c2_cheb2_tf32 | 0.0 | 83 | 100 | 1.370 | 7.11 | 0.0666 |
| m0_on_step | two_level_c2_cheb4_tf32 | 0.0 | 58 | 72 | 1.390 | 5.04 | 0.0154 |
| m0_on_step | two_level_c4_cheb2_tf32 | 0.0 | 100 | 115 | 1.195 | 7.47 | 0.0875 |
| m0_on_step | two_level_c4_cheb4_tf32 | 0.0 | 64 | 74 | 1.194 | 4.78 | 0.0153 |
| m0_on_step | two_level_c2_cheb2_safeguard_tf32 | 0.0 | 76 | 89 | 1.409 | 6.69 | 0.0288 |
| m0_on_step | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 92 | 106 | 1.192 | 6.86 | 0.0579 |
| m0_on_step | two_level_c4_cheb4_safeguard_tf32 | 0.0 | 58 | 67 | 1.196 | 4.33 | 0.0085 |
| notebook | wrapper_cheb2_tf32 | 33.3 | 7 | — | 3.160 | 1.38 | 1.0000 |
| notebook | two_level_c2_cheb2_tf32 | 0.0 | 87 | 118 | 1.358 | 7.39 | 0.0544 |
| notebook | two_level_c2_cheb4_tf32 | 0.0 | 67 | 77 | 1.364 | 5.71 | 0.0179 |
| notebook | two_level_c4_cheb2_tf32 | 0.0 | 117 | 127 | 1.207 | 8.82 | 0.0927 |
| notebook | two_level_c4_cheb4_tf32 | 0.0 | 71 | 78 | 1.210 | 5.37 | 0.0220 |
| notebook | two_level_c2_cheb2_safeguard_tf32 | 0.0 | 82 | 108 | 1.385 | 7.10 | 0.0457 |
| notebook | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 109 | 123 | 1.209 | 8.24 | 0.0919 |
| notebook | two_level_c4_cheb4_safeguard_tf32 | 0.0 | 65 | 73 | 1.207 | 4.90 | 0.0198 |

## diffusion 256² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook wrapper_cheb2_tf32 |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 606 | 661 |  | 6.29 |  |
| **jacobi4** (GPU) |  | 0.0 | 255 | 275 |  | 3.64 |  |
| **chebyshev8** (GPU) |  | 0.0 | 119 | 129 |  | 2.53 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 7 | 7 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| h1_on_step | wrapper_cheb2_tf32 | 33.3 | 17 | — | 10.772 | 11.44 | 0.0000 |
| h1_on_step | two_level_c2_cheb2_tf32 | 0.0 | 11 | 15 | 3.444 | 2.37 | 0.0000 |
| h1_on_step | two_level_c2_cheb4_tf32 | 0.0 | 12 | 26 | 3.542 | 2.66 | 0.0000 |
| h1_on_step | two_level_c4_cheb2_tf32 | 0.0 | 136 | 274 | 1.575 | 13.39 | 0.0998 |
| h1_on_step | two_level_c4_cheb4_tf32 | 0.0 | 84 | 157 | 1.662 | 8.73 | 0.0998 |
| h1_on_step | two_level_c2_cheb2_safeguard_tf32 | 0.0 | 11 | 14 | 3.529 | 2.43 | 0.0000 |
| h1_on_step | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 135 | 280 | 1.659 | 13.99 | 0.0998 |
| h1_on_step | two_level_c4_cheb4_safeguard_tf32 | 0.0 | 81 | 147 | 1.745 | 8.83 | 0.0998 |
| m0_on_step | wrapper_cheb2_tf32 | 33.3 | 17 | — | 10.752 | 11.42 | 0.0000 |
| m0_on_step | two_level_c2_cheb2_tf32 | 0.0 | 11 | 167 | 3.383 | 2.33 | 0.0000 |
| m0_on_step | two_level_c2_cheb4_tf32 | 0.0 | 13 | 105 | 3.477 | 2.82 | 0.0000 |
| m0_on_step | two_level_c4_cheb2_tf32 | 0.0 | 191 | 230 | 1.538 | 18.36 | 0.0998 |
| m0_on_step | two_level_c4_cheb4_tf32 | 0.0 | 128 | 154 | 1.624 | 12.99 | 0.0998 |
| m0_on_step | two_level_c2_cheb2_safeguard_tf32 | 0.0 | 11 | 151 | 3.465 | 2.38 | 0.0000 |
| m0_on_step | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 182 | 226 | 1.615 | 18.38 | 0.0998 |
| m0_on_step | two_level_c4_cheb4_safeguard_tf32 | 0.0 | 120 | 142 | 1.700 | 12.75 | 0.0998 |
| notebook | wrapper_cheb2_tf32 | 33.3 | 15 | — | 10.773 | 10.10 | 1.0000 |
| notebook | two_level_c2_cheb2_tf32 | 0.0 | 12 | 169 | 3.389 | 2.54 | 0.0000 |
| notebook | two_level_c2_cheb4_tf32 | 0.0 | 17 | 106 | 3.473 | 3.69 | 0.0000 |
| notebook | two_level_c4_cheb2_tf32 | 0.0 | 238 | 299 | 1.531 | 22.77 | 0.0998 |
| notebook | two_level_c4_cheb4_tf32 | 0.0 | 148 | 173 | 1.625 | 15.03 | 0.0998 |
| notebook | two_level_c2_cheb2_safeguard_tf32 | 0.0 | 11 | 151 | 3.455 | 2.38 | 0.0000 |
| notebook | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 233 | 311 | 1.616 | 23.53 | 0.0998 |
| notebook | two_level_c4_cheb4_safeguard_tf32 | 0.0 | 142 | 169 | 1.695 | 15.04 | 0.0998 |

## poisson 128² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook wrapper_cheb2_tf32 |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 316 | 319 |  | 1.77 |  |
| **jacobi4** (GPU) |  | 0.0 | 136 | 137 |  | 0.85 |  |
| **chebyshev8** (GPU) |  | 0.0 | 64 | 65 |  | 0.52 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 6 | 7 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| h1_on_step | wrapper_cheb2_tf32 | 33.3 | 9 | — | 0.858 | 0.48 | 0.0000 |
| h1_on_step | two_level_c2_cheb2_tf32 | 0.0 | 18 | 24 | 0.595 | 0.67 | 0.0000 |
| h1_on_step | two_level_c2_cheb4_tf32 | 0.0 | 19 | 28 | 0.592 | 0.70 | 0.0000 |
| h1_on_step | two_level_c4_cheb2_tf32 | 0.0 | 90 | 109 | 0.593 | 3.34 | 0.0000 |
| h1_on_step | two_level_c4_cheb4_tf32 | 0.0 | 60 | 69 | 0.580 | 2.17 | 0.0000 |
| h1_on_step | two_level_c2_cheb2_safeguard_tf32 | 0.0 | 18 | 25 | 0.594 | 0.67 | 0.0000 |
| h1_on_step | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 89 | 106 | 0.578 | 3.22 | 0.0000 |
| h1_on_step | two_level_c4_cheb4_safeguard_tf32 | 0.0 | 59 | 68 | 0.583 | 2.15 | 0.0000 |
| m0_on_step | wrapper_cheb2_tf32 | 33.3 | 10 | — | 0.834 | 0.52 | 0.0000 |
| m0_on_step | two_level_c2_cheb2_tf32 | 0.0 | 24 | 64 | 0.601 | 0.90 | 0.0000 |
| m0_on_step | two_level_c2_cheb4_tf32 | 0.0 | 24 | 56 | 0.600 | 0.90 | 0.0000 |
| m0_on_step | two_level_c4_cheb2_tf32 | 0.0 | 94 | 153 | 0.579 | 3.40 | 0.0000 |
| m0_on_step | two_level_c4_cheb4_tf32 | 0.0 | 61 | 95 | 0.572 | 2.18 | 0.0000 |
| m0_on_step | two_level_c2_cheb2_safeguard_tf32 | 0.0 | 24 | 62 | 0.603 | 0.90 | 0.0000 |
| m0_on_step | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 95 | 151 | 0.578 | 3.43 | 0.0000 |
| m0_on_step | two_level_c4_cheb4_safeguard_tf32 | 0.0 | 60 | 86 | 0.574 | 2.15 | 0.0000 |
| notebook | wrapper_cheb2_tf32 | 0.0 | 7 | 10 | 0.837 | 0.37 | 1.0000 |
| notebook | two_level_c2_cheb2_tf32 | 0.0 | 13 | 14 | 0.596 | 0.48 | 0.0000 |
| notebook | two_level_c2_cheb4_tf32 | 0.0 | 10 | 13 | 0.601 | 0.38 | 0.0000 |
| notebook | two_level_c4_cheb2_tf32 | 0.0 | 30 | 39 | 0.574 | 1.08 | 0.0000 |
| notebook | two_level_c4_cheb4_tf32 | 0.0 | 32 | 45 | 0.577 | 1.15 | 0.0000 |
| notebook | two_level_c2_cheb2_safeguard_tf32 | 0.0 | 13 | 14 | 0.603 | 0.49 | 0.0000 |
| notebook | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 29 | 39 | 0.572 | 1.04 | 0.0000 |
| notebook | two_level_c4_cheb4_safeguard_tf32 | 0.0 | 32 | 44 | 0.575 | 1.15 | 0.0000 |

## poisson 256² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook wrapper_cheb2_tf32 |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 632 | 639 |  | 7.20 |  |
| **jacobi4** (GPU) |  | 0.0 | 271 | 275 |  | 3.95 |  |
| **chebyshev8** (GPU) |  | 0.0 | 128 | 130 |  | 2.73 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 8 | 8 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| h1_on_step | wrapper_cheb2_tf32 | 33.3 | 19 | — | 2.458 | 2.92 | 0.0000 |
| h1_on_step | two_level_c2_cheb2_tf32 | 0.0 | 13 | 13 | 1.133 | 0.92 | 0.0000 |
| h1_on_step | two_level_c2_cheb4_tf32 | 0.0 | 11 | 12 | 1.225 | 0.84 | 0.0000 |
| h1_on_step | two_level_c4_cheb2_tf32 | 0.0 | 18 | 20 | 0.714 | 0.80 | 0.0031 |
| h1_on_step | two_level_c4_cheb4_tf32 | 0.0 | 17 | 20 | 0.795 | 0.84 | 0.0349 |
| h1_on_step | two_level_c2_cheb2_safeguard_tf32 | 0.0 | 13 | 13 | 1.206 | 0.98 | 0.0000 |
| h1_on_step | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 18 | 20 | 0.782 | 0.88 | 0.1691 |
| h1_on_step | two_level_c4_cheb4_safeguard_tf32 | 0.0 | 17 | 21 | 0.868 | 0.92 | 0.2416 |
| m0_on_step | wrapper_cheb2_tf32 | 33.3 | 19 | — | 2.434 | 2.89 | 0.0000 |
| m0_on_step | two_level_c2_cheb2_tf32 | 0.0 | 12 | 14 | 1.042 | 0.78 | 0.0000 |
| m0_on_step | two_level_c2_cheb4_tf32 | 0.0 | 11 | 12 | 1.135 | 0.78 | 0.0000 |
| m0_on_step | two_level_c4_cheb2_tf32 | 0.0 | 19 | 29 | 0.656 | 0.78 | 0.0000 |
| m0_on_step | two_level_c4_cheb4_tf32 | 0.0 | 22 | 50 | 0.748 | 1.03 | 0.0000 |
| m0_on_step | two_level_c2_cheb2_safeguard_tf32 | 0.0 | 12 | 14 | 1.119 | 0.84 | 0.0000 |
| m0_on_step | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 19 | 29 | 0.734 | 0.87 | 0.0000 |
| m0_on_step | two_level_c4_cheb4_safeguard_tf32 | 0.0 | 22 | 50 | 0.824 | 1.13 | 0.0000 |
| notebook | wrapper_cheb2_tf32 | 0.0 | 16 | 21 | 2.440 | 2.44 | 1.0000 |
| notebook | two_level_c2_cheb2_tf32 | 0.0 | 15 | 19 | 1.039 | 0.97 | 0.0000 |
| notebook | two_level_c2_cheb4_tf32 | 0.0 | 10 | 14 | 1.129 | 0.71 | 0.0000 |
| notebook | two_level_c4_cheb2_tf32 | 0.0 | 23 | 26 | 0.657 | 0.94 | 0.0000 |
| notebook | two_level_c4_cheb4_tf32 | 0.0 | 16 | 17 | 0.749 | 0.75 | 0.0000 |
| notebook | two_level_c2_cheb2_safeguard_tf32 | 0.0 | 14 | 19 | 1.118 | 0.98 | 0.0000 |
| notebook | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 21 | 24 | 0.737 | 0.97 | 0.0000 |
| notebook | two_level_c4_cheb4_safeguard_tf32 | 0.0 | 16 | 17 | 0.820 | 0.82 | 0.0000 |
