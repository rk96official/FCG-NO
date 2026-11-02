# E9 ablations

## diffusion 128² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 300 | 324 |  | 1.37 |  |
| **jacobi4** (GPU) |  | 0.0 | 126 | 137 |  | 0.66 |  |
| **chebyshev8** (GPU) |  | 0.0 | 59 | 64 |  | 0.41 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 6 | 6 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| abl_cg | two_level_c2_cheb2_tf32 | 0.0 | 7 | 7 | 0.505 | 0.22 | — |
| abl_cg | two_level_c4_cheb2_tf32 | 0.0 | 10 | 12 | 0.412 | 0.26 | — |
| abl_cg | two_level_c8_cheb2_tf32 | 0.0 | 95 | 105 | 0.379 | 2.25 | — |
| abl_cg | two_level_c16_cheb2_tf32 | 0.0 | 114 | 131 | 0.376 | 2.68 | — |
| abl_cg | multi_b64_tf32 | 0.0 | 7 | 7 | 0.503 | 0.22 | — |
| abl_m0 | two_level_c2_cheb2_tf32 | 0.0 | 6 | 6 | 0.592 | 0.22 | — |
| abl_m0 | two_level_c4_cheb2_tf32 | 0.0 | 10 | 11 | 0.569 | 0.36 | — |
| abl_m0 | two_level_c8_cheb2_tf32 | 0.0 | 45 | 95 | 0.586 | 1.65 | — |
| abl_m0 | two_level_c16_cheb2_tf32 | 0.0 | 108 | 128 | 0.572 | 3.86 | — |
| abl_m0 | multi_b64_tf32 | 0.0 | 6 | 6 | 0.598 | 0.22 | — |
| abl_notay | two_level_c2_cheb2_tf32 | 0.0 | 6 | 7 | 0.508 | 0.19 | — |
| abl_notay | two_level_c4_cheb2_tf32 | 0.0 | 10 | 11 | 0.415 | 0.26 | — |
| abl_notay | two_level_c8_cheb2_tf32 | 0.0 | 88 | 106 | 0.383 | 2.11 | — |
| abl_notay | two_level_c16_cheb2_tf32 | 0.0 | 109 | 127 | 0.380 | 2.59 | — |
| abl_notay | multi_b64_tf32 | 0.0 | 6 | 7 | 0.507 | 0.19 | — |
| abl_single | two_level_c2_cheb2_tf32 | 0.0 | 25 | 41 | 0.512 | 0.80 | — |
| abl_single | two_level_c4_cheb2_tf32 | 0.0 | 80 | 111 | 0.417 | 2.08 | — |
| abl_single | two_level_c8_cheb2_tf32 | 0.0 | 117 | 146 | 0.383 | 2.80 | — |
| abl_single | two_level_c16_cheb2_tf32 | 0.0 | 114 | 140 | 0.378 | 2.70 | — |
| abl_single | multi_b64_tf32 | 0.0 | 25 | 41 | 0.507 | 0.79 | — |

## diffusion 256² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 606 | 661 |  | 6.29 |  |
| **jacobi4** (GPU) |  | 0.0 | 255 | 275 |  | 3.64 |  |
| **chebyshev8** (GPU) |  | 0.0 | 119 | 129 |  | 2.53 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 7 | 7 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| abl_cg | two_level_c2_cheb2_tf32 | 0.0 | 13 | 14 | 1.119 | 0.91 | — |
| abl_cg | two_level_c4_cheb2_tf32 | 0.0 | 9 | 9 | 0.698 | 0.39 | — |
| abl_cg | two_level_c8_cheb2_tf32 | 0.0 | 15 | 17 | 0.597 | 0.56 | — |
| abl_cg | two_level_c16_cheb2_tf32 | 0.0 | 208 | 234 | 0.560 | 7.28 | — |
| abl_cg | multi_b64_tf32 | 0.0 | 7 | 7 | 0.815 | 0.36 | — |
| abl_m0 | two_level_c2_cheb2_tf32 | 0.0 | 13 | 14 | 1.032 | 0.84 | — |
| abl_m0 | two_level_c4_cheb2_tf32 | 0.0 | 9 | 10 | 0.647 | 0.36 | — |
| abl_m0 | two_level_c8_cheb2_tf32 | 0.0 | 15 | 16 | 0.551 | 0.52 | — |
| abl_m0 | two_level_c16_cheb2_tf32 | 0.0 | 71 | 224 | 0.525 | 2.33 | — |
| abl_m0 | multi_b64_tf32 | 0.0 | 7 | 7 | 0.717 | 0.31 | — |
| abl_notay | two_level_c2_cheb2_tf32 | 0.0 | 12 | 13 | 1.119 | 0.84 | — |
| abl_notay | two_level_c4_cheb2_tf32 | 0.0 | 8 | 8 | 0.696 | 0.35 | — |
| abl_notay | two_level_c8_cheb2_tf32 | 0.0 | 15 | 16 | 0.596 | 0.56 | — |
| abl_notay | two_level_c16_cheb2_tf32 | 0.0 | 214 | 260 | 0.560 | 7.49 | — |
| abl_notay | multi_b64_tf32 | 0.0 | 7 | 7 | 0.812 | 0.36 | — |
| abl_single | two_level_c2_cheb2_tf32 | 0.0 | 11 | 13 | 1.120 | 0.77 | — |
| abl_single | two_level_c4_cheb2_tf32 | 0.0 | 23 | 33 | 0.697 | 1.00 | — |
| abl_single | two_level_c8_cheb2_tf32 | 0.0 | 178 | 249 | 0.604 | 6.72 | — |
| abl_single | two_level_c16_cheb2_tf32 | 0.0 | 275 | 445 | 0.566 | 9.73 | — |
| abl_single | multi_b64_tf32 | 0.0 | 28 | 48 | 0.821 | 1.44 | — |

## diffusion 512² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 1234 | 1351 |  | 53.21 |  |
| **jacobi4** (GPU) |  | 0.0 | 517 | 558 |  | 31.35 |  |
| **chebyshev8** (GPU) |  | 0.0 | 241 | 262 |  | 20.13 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 6 | 7 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| abl_cg | two_level_c2_cheb2_tf32 | 0.0 | 33 | 38 | 3.869 | 7.98 | — |
| abl_cg | two_level_c4_cheb2_tf32 | 0.0 | 18 | 19 | 2.164 | 2.43 | — |
| abl_cg | two_level_c8_cheb2_tf32 | 0.0 | 14 | 14 | 1.659 | 1.45 | — |
| abl_cg | two_level_c16_cheb2_tf32 | 0.0 | 28 | 30 | 1.521 | 2.66 | — |
| abl_cg | multi_b64_tf32 | 0.0 | 7 | 7 | 2.238 | 0.98 | — |
| abl_m0 | two_level_c2_cheb2_tf32 | 0.0 | 39 | 43 | 3.550 | 8.65 | — |
| abl_m0 | two_level_c4_cheb2_tf32 | 0.0 | 21 | 22 | 1.949 | 2.56 | — |
| abl_m0 | two_level_c8_cheb2_tf32 | 0.0 | 14 | 15 | 1.500 | 1.31 | — |
| abl_m0 | two_level_c16_cheb2_tf32 | 0.0 | 27 | 29 | 1.407 | 2.37 | — |
| abl_m0 | multi_b64_tf32 | 0.0 | 7 | 7 | 1.825 | 0.80 | — |
| abl_notay | two_level_c2_cheb2_tf32 | 0.0 | 28 | 30 | 3.919 | 6.86 | — |
| abl_notay | two_level_c4_cheb2_tf32 | 0.0 | 17 | 18 | 2.178 | 2.31 | — |
| abl_notay | two_level_c8_cheb2_tf32 | 0.0 | 14 | 14 | 1.674 | 1.46 | — |
| abl_notay | two_level_c16_cheb2_tf32 | 0.0 | 27 | 29 | 1.535 | 2.59 | — |
| abl_notay | multi_b64_tf32 | 0.0 | 7 | 7 | 2.261 | 0.99 | — |
| abl_single | two_level_c2_cheb2_tf32 | 0.0 | 16 | 17 | 3.885 | 3.89 | — |
| abl_single | two_level_c4_cheb2_tf32 | 0.0 | 15 | 19 | 2.164 | 2.03 | — |
| abl_single | two_level_c8_cheb2_tf32 | 0.0 | 26 | 32 | 1.672 | 2.72 | — |
| abl_single | two_level_c16_cheb2_tf32 | 0.0 | 397 | 594 | 1.527 | 37.90 | — |
| abl_single | multi_b64_tf32 | 0.0 | 28 | 49 | 2.243 | 3.92 | — |

## diffusion 1024² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 2549 | 2759 |  | 400.94 |  |
| **jacobi4** (GPU) |  | 0.0 | 1076 | 1129 |  | 239.06 |  |
| **chebyshev8** (GPU) |  | 0.0 | 502 | 532 |  | 172.76 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 8 | 8 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| abl_cg | two_level_c2_cheb2_tf32 | 0.0 | 149 | 190 | 16.142 | 150.32 | — |
| abl_cg | two_level_c4_cheb2_tf32 | 0.0 | 48 | 62 | 8.758 | 26.27 | — |
| abl_cg | two_level_c8_cheb2_tf32 | 0.0 | 22 | 23 | 6.608 | 9.09 | — |
| abl_cg | two_level_c16_cheb2_tf32 | 0.0 | 26 | 27 | 5.687 | 9.24 | — |
| abl_cg | multi_b64_tf32 | 0.0 | 7 | 8 | 9.338 | 4.09 | — |
| abl_m0 | two_level_c2_cheb2_tf32 | 0.0 | 143 | 188 | 14.563 | 130.16 | — |
| abl_m0 | two_level_c4_cheb2_tf32 | 0.0 | 64 | 74 | 7.926 | 31.70 | — |
| abl_m0 | two_level_c8_cheb2_tf32 | 0.0 | 28 | 33 | 5.929 | 10.38 | — |
| abl_m0 | two_level_c16_cheb2_tf32 | 0.0 | 26 | 28 | 5.364 | 8.72 | — |
| abl_m0 | multi_b64_tf32 | 0.0 | 7 | 7 | 7.612 | 3.33 | — |
| abl_notay | two_level_c2_cheb2_tf32 | 0.0 | 112 | 143 | 16.191 | 113.34 | — |
| abl_notay | two_level_c4_cheb2_tf32 | 0.0 | 44 | 47 | 8.751 | 24.06 | — |
| abl_notay | two_level_c8_cheb2_tf32 | 0.0 | 21 | 22 | 6.627 | 8.70 | — |
| abl_notay | two_level_c16_cheb2_tf32 | 0.0 | 25 | 27 | 5.712 | 8.92 | — |
| abl_notay | multi_b64_tf32 | 0.0 | 7 | 7 | 9.382 | 4.10 | — |
| abl_single | two_level_c2_cheb2_tf32 | 0.0 | 38 | 44 | 16.201 | 38.48 | — |
| abl_single | two_level_c4_cheb2_tf32 | 0.0 | 20 | 21 | 8.749 | 10.94 | — |
| abl_single | two_level_c8_cheb2_tf32 | 0.0 | 20 | 25 | 6.689 | 8.36 | — |
| abl_single | two_level_c16_cheb2_tf32 | 0.0 | 41 | 45 | 5.705 | 14.62 | — |
| abl_single | multi_b64_tf32 | 0.0 | 28 | 50 | 9.439 | 16.52 | — |

## poisson 128² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 316 | 319 |  | 1.77 |  |
| **jacobi4** (GPU) |  | 0.0 | 136 | 137 |  | 0.85 |  |
| **chebyshev8** (GPU) |  | 0.0 | 64 | 65 |  | 0.52 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 6 | 7 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| abl_cg | two_level_c2_cheb2_tf32 | 0.0 | 7 | 8 | 0.507 | 0.22 | — |
| abl_cg | two_level_c4_cheb2_tf32 | 0.0 | 11 | 12 | 0.416 | 0.29 | — |
| abl_cg | two_level_c8_cheb2_tf32 | 0.0 | 50 | 127 | 0.382 | 1.19 | — |
| abl_cg | two_level_c16_cheb2_tf32 | 0.0 | 131 | 146 | 0.378 | 3.09 | — |
| abl_cg | multi_b64_tf32 | 0.0 | 7 | 8 | 0.507 | 0.22 | — |
| abl_m0 | two_level_c2_cheb2_tf32 | 0.0 | 7 | 7 | 0.482 | 0.21 | — |
| abl_m0 | two_level_c4_cheb2_tf32 | 0.0 | 11 | 13 | 0.391 | 0.27 | — |
| abl_m0 | two_level_c8_cheb2_tf32 | 0.0 | 65 | 116 | 0.366 | 1.49 | — |
| abl_m0 | two_level_c16_cheb2_tf32 | 0.0 | 130 | 148 | 0.359 | 2.91 | — |
| abl_m0 | multi_b64_tf32 | 0.0 | 7 | 7 | 0.477 | 0.21 | — |
| abl_notay | two_level_c2_cheb2_tf32 | 0.0 | 7 | 7 | 0.509 | 0.22 | — |
| abl_notay | two_level_c4_cheb2_tf32 | 0.0 | 11 | 11 | 0.415 | 0.29 | — |
| abl_notay | two_level_c8_cheb2_tf32 | 0.0 | 61 | 86 | 0.383 | 1.46 | — |
| abl_notay | two_level_c16_cheb2_tf32 | 0.0 | 132 | 150 | 0.376 | 3.11 | — |
| abl_notay | multi_b64_tf32 | 0.0 | 7 | 7 | 0.502 | 0.22 | — |
| abl_single | two_level_c2_cheb2_tf32 | 0.0 | 25 | 108 | 0.502 | 0.78 | — |
| abl_single | two_level_c4_cheb2_tf32 | 0.0 | 115 | 138 | 0.412 | 2.96 | — |
| abl_single | two_level_c8_cheb2_tf32 | 0.0 | 131 | 144 | 0.380 | 3.11 | — |
| abl_single | two_level_c16_cheb2_tf32 | 0.0 | 132 | 151 | 0.373 | 3.07 | — |
| abl_single | multi_b64_tf32 | 0.0 | 25 | 108 | 0.500 | 0.78 | — |

## poisson 256² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 632 | 639 |  | 7.20 |  |
| **jacobi4** (GPU) |  | 0.0 | 271 | 275 |  | 3.95 |  |
| **chebyshev8** (GPU) |  | 0.0 | 128 | 130 |  | 2.73 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 8 | 8 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| abl_cg | two_level_c2_cheb2_tf32 | 0.0 | 14 | 15 | 1.115 | 0.98 | — |
| abl_cg | two_level_c4_cheb2_tf32 | 0.0 | 10 | 11 | 0.697 | 0.44 | — |
| abl_cg | two_level_c8_cheb2_tf32 | 0.0 | 17 | 18 | 0.595 | 0.63 | — |
| abl_cg | two_level_c16_cheb2_tf32 | 0.0 | 120 | 283 | 0.557 | 4.17 | — |
| abl_cg | multi_b64_tf32 | 0.0 | 7 | 8 | 0.813 | 0.36 | — |
| abl_m0 | two_level_c2_cheb2_tf32 | 0.0 | 15 | 16 | 1.030 | 0.97 | — |
| abl_m0 | two_level_c4_cheb2_tf32 | 0.0 | 10 | 10 | 0.642 | 0.40 | — |
| abl_m0 | two_level_c8_cheb2_tf32 | 0.0 | 18 | 20 | 0.550 | 0.62 | — |
| abl_m0 | two_level_c16_cheb2_tf32 | 0.0 | 150 | 280 | 0.517 | 4.85 | — |
| abl_m0 | multi_b64_tf32 | 0.0 | 7 | 7 | 0.712 | 0.31 | — |
| abl_notay | two_level_c2_cheb2_tf32 | 0.0 | 13 | 14 | 1.120 | 0.91 | — |
| abl_notay | two_level_c4_cheb2_tf32 | 0.0 | 9 | 9 | 0.703 | 0.40 | — |
| abl_notay | two_level_c8_cheb2_tf32 | 0.0 | 17 | 18 | 0.599 | 0.64 | — |
| abl_notay | two_level_c16_cheb2_tf32 | 0.0 | 150 | 230 | 0.581 | 5.45 | — |
| abl_notay | multi_b64_tf32 | 0.0 | 7 | 7 | 0.818 | 0.36 | — |
| abl_single | two_level_c2_cheb2_tf32 | 0.0 | 13 | 19 | 1.129 | 0.92 | — |
| abl_single | two_level_c4_cheb2_tf32 | 0.0 | 19 | 215 | 0.707 | 0.84 | — |
| abl_single | two_level_c8_cheb2_tf32 | 0.0 | 250 | 299 | 0.607 | 9.48 | — |
| abl_single | two_level_c16_cheb2_tf32 | 0.0 | 307 | 331 | 0.589 | 11.30 | — |
| abl_single | multi_b64_tf32 | 0.0 | 26 | 95 | 0.822 | 1.34 | — |

## poisson 512² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 1270 | 1291 |  | 55.48 |  |
| **jacobi4** (GPU) |  | 0.0 | 546 | 555 |  | 32.32 |  |
| **chebyshev8** (GPU) |  | 0.0 | 258 | 261 |  | 21.10 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 7 | 7 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| abl_cg | two_level_c2_cheb2_tf32 | 0.0 | 35 | 41 | 3.821 | 8.36 | — |
| abl_cg | two_level_c4_cheb2_tf32 | 0.0 | 20 | 21 | 2.130 | 2.66 | — |
| abl_cg | two_level_c8_cheb2_tf32 | 0.0 | 16 | 16 | 1.642 | 1.64 | — |
| abl_cg | two_level_c16_cheb2_tf32 | 0.0 | 32 | 35 | 1.500 | 3.00 | — |
| abl_cg | multi_b64_tf32 | 0.0 | 7 | 8 | 2.221 | 0.97 | — |
| abl_m0 | two_level_c2_cheb2_tf32 | 0.0 | 43 | 53 | 3.528 | 9.48 | — |
| abl_m0 | two_level_c4_cheb2_tf32 | 0.0 | 23 | 25 | 1.926 | 2.77 | — |
| abl_m0 | two_level_c8_cheb2_tf32 | 0.0 | 16 | 17 | 1.495 | 1.49 | — |
| abl_m0 | two_level_c16_cheb2_tf32 | 0.0 | 33 | 35 | 1.387 | 2.86 | — |
| abl_m0 | multi_b64_tf32 | 0.0 | 7 | 7 | 1.816 | 0.79 | — |
| abl_notay | two_level_c2_cheb2_tf32 | 0.0 | 30 | 32 | 3.901 | 7.31 | — |
| abl_notay | two_level_c4_cheb2_tf32 | 0.0 | 20 | 21 | 2.157 | 2.70 | — |
| abl_notay | two_level_c8_cheb2_tf32 | 0.0 | 15 | 16 | 1.655 | 1.55 | — |
| abl_notay | two_level_c16_cheb2_tf32 | 0.0 | 32 | 34 | 1.516 | 3.03 | — |
| abl_notay | multi_b64_tf32 | 0.0 | 7 | 7 | 2.240 | 0.98 | — |
| abl_single | two_level_c2_cheb2_tf32 | 0.0 | 18 | 20 | 3.892 | 4.38 | — |
| abl_single | two_level_c4_cheb2_tf32 | 0.0 | 17 | 22 | 2.169 | 2.30 | — |
| abl_single | two_level_c8_cheb2_tf32 | 0.0 | 25 | 397 | 1.670 | 2.61 | — |
| abl_single | two_level_c16_cheb2_tf32 | 0.0 | 576 | 724 | 1.519 | 54.69 | — |
| abl_single | multi_b64_tf32 | 0.0 | 27 | 90 | 2.245 | 3.79 | — |

## poisson 1024² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 2569 | 2604 |  | 402.25 |  |
| **jacobi4** (GPU) |  | 0.0 | 1103 | 1122 |  | 240.11 |  |
| **chebyshev8** (GPU) |  | 0.0 | 521 | 529 |  | 176.39 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 9 | 9 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| abl_cg | two_level_c2_cheb2_tf32 | 0.0 | 112 | 155 | 16.216 | 113.51 | — |
| abl_cg | two_level_c4_cheb2_tf32 | 0.0 | 49 | 66 | 8.696 | 26.63 | — |
| abl_cg | two_level_c8_cheb2_tf32 | 0.0 | 25 | 27 | 6.574 | 10.27 | — |
| abl_cg | two_level_c16_cheb2_tf32 | 0.0 | 31 | 33 | 5.657 | 10.96 | — |
| abl_cg | multi_b64_tf32 | 0.0 | 8 | 8 | 9.238 | 4.62 | — |
| abl_m0 | two_level_c2_cheb2_tf32 | 0.0 | 148 | 239 | 14.468 | 133.83 | — |
| abl_m0 | two_level_c4_cheb2_tf32 | 0.0 | 72 | 89 | 7.802 | 35.11 | — |
| abl_m0 | two_level_c8_cheb2_tf32 | 0.0 | 31 | 34 | 5.840 | 11.31 | — |
| abl_m0 | two_level_c16_cheb2_tf32 | 0.0 | 31 | 33 | 5.298 | 10.27 | — |
| abl_m0 | multi_b64_tf32 | 0.0 | 7 | 8 | 7.522 | 3.29 | — |
| abl_notay | two_level_c2_cheb2_tf32 | 0.0 | 100 | 108 | 16.097 | 100.61 | — |
| abl_notay | two_level_c4_cheb2_tf32 | 0.0 | 48 | 51 | 8.668 | 26.00 | — |
| abl_notay | two_level_c8_cheb2_tf32 | 0.0 | 24 | 26 | 6.581 | 9.87 | — |
| abl_notay | two_level_c16_cheb2_tf32 | 0.0 | 31 | 32 | 5.603 | 10.86 | — |
| abl_notay | multi_b64_tf32 | 0.0 | 7 | 7 | 9.246 | 4.05 | — |
| abl_single | two_level_c2_cheb2_tf32 | 0.0 | 45 | 50 | 16.124 | 45.35 | — |
| abl_single | two_level_c4_cheb2_tf32 | 0.0 | 24 | 25 | 8.663 | 12.99 | — |
| abl_single | two_level_c8_cheb2_tf32 | 0.0 | 22 | 35 | 6.549 | 9.00 | — |
| abl_single | two_level_c16_cheb2_tf32 | 0.0 | 45 | 701 | 5.643 | 15.87 | — |
| abl_single | multi_b64_tf32 | 0.0 | 27 | 91 | 9.265 | 15.64 | — |
