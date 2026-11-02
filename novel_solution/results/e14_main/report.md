# E14 final

## diffusion 128² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 300 | 324 |  | 1.37 |  |
| **jacobi4** (GPU) |  | 0.0 | 126 | 137 |  | 0.66 |  |
| **chebyshev8** (GPU) |  | 0.0 | 59 | 64 |  | 0.41 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 6 | 6 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| final_2l2 | two_level_c2_cheb2_tf32 | 0.0 | 6 | 6 | 0.510 | 0.19 | — |
| final_2l2 | two_level_c4_cheb2_tf32 | 0.0 | 10 | 11 | 0.417 | 0.26 | — |
| final_2l2 | two_level_c8_cheb2_tf32 | 0.0 | 85 | 99 | 0.385 | 2.05 | — |
| final_2l2 | two_level_c16_cheb2_tf32 | 0.0 | 114 | 130 | 0.380 | 2.71 | — |
| final_2l2 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 10 | 10 | 0.450 | 0.28 | — |
| final_2l2 | two_level_c8_cheb2_safeguard_tf32 | 0.0 | 82 | 92 | 0.420 | 2.15 | — |
| final_2l4 | two_level_c2_cheb2_tf32 | 0.0 | 11 | 11 | 0.529 | 0.36 | — |
| final_2l4 | two_level_c4_cheb2_tf32 | 0.0 | 7 | 7 | 0.441 | 0.19 | — |
| final_2l4 | two_level_c8_cheb2_tf32 | 0.0 | 39 | 110 | 0.386 | 0.94 | — |
| final_2l4 | two_level_c16_cheb2_tf32 | 0.0 | 102 | 117 | 0.382 | 2.44 | — |
| final_2l4 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 7 | 7 | 0.475 | 0.21 | — |
| final_2l4 | two_level_c8_cheb2_safeguard_tf32 | 0.0 | 39 | 114 | 0.421 | 1.03 | — |

## diffusion 256² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 606 | 661 |  | 6.29 |  |
| **jacobi4** (GPU) |  | 0.0 | 255 | 275 |  | 3.64 |  |
| **chebyshev8** (GPU) |  | 0.0 | 119 | 129 |  | 2.53 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 7 | 7 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| final_2l2 | two_level_c2_cheb2_tf32 | 0.0 | 13 | 14 | 1.116 | 0.91 | — |
| final_2l2 | two_level_c4_cheb2_tf32 | 0.0 | 9 | 9 | 0.693 | 0.39 | — |
| final_2l2 | two_level_c8_cheb2_tf32 | 0.0 | 15 | 16 | 0.592 | 0.55 | — |
| final_2l2 | two_level_c16_cheb2_tf32 | 0.0 | 193 | 232 | 0.555 | 6.69 | — |
| final_2l2 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 8 | 9 | 0.766 | 0.38 | — |
| final_2l2 | two_level_c8_cheb2_safeguard_tf32 | 0.0 | 15 | 15 | 0.670 | 0.63 | — |
| final_2l4 | two_level_c2_cheb2_tf32 | 0.0 | 24 | 25 | 1.116 | 1.67 | — |
| final_2l4 | two_level_c4_cheb2_tf32 | 0.0 | 14 | 15 | 0.697 | 0.61 | — |
| final_2l4 | two_level_c8_cheb2_tf32 | 0.0 | 13 | 13 | 0.595 | 0.48 | — |
| final_2l4 | two_level_c16_cheb2_tf32 | 0.0 | 89 | 297 | 0.558 | 3.10 | — |
| final_2l4 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 14 | 15 | 0.773 | 0.68 | — |
| final_2l4 | two_level_c8_cheb2_safeguard_tf32 | 0.0 | 13 | 13 | 0.671 | 0.55 | — |

## diffusion 512² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 1234 | 1351 |  | 53.21 |  |
| **jacobi4** (GPU) |  | 0.0 | 517 | 558 |  | 31.35 |  |
| **chebyshev8** (GPU) |  | 0.0 | 241 | 262 |  | 20.13 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 6 | 7 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| final_2l2 | two_level_c2_cheb2_tf32 | 0.0 | 37 | 42 | 3.883 | 8.98 | — |
| final_2l2 | two_level_c4_cheb2_tf32 | 0.0 | 20 | 21 | 2.179 | 2.72 | — |
| final_2l2 | two_level_c8_cheb2_tf32 | 0.0 | 14 | 14 | 1.677 | 1.47 | — |
| final_2l2 | two_level_c16_cheb2_tf32 | 0.0 | 27 | 29 | 1.537 | 2.59 | — |
| final_2l2 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 19 | 21 | 2.918 | 3.46 | — |
| final_2l2 | two_level_c8_cheb2_safeguard_tf32 | 0.0 | 14 | 14 | 2.396 | 2.10 | — |
| final_2l4 | two_level_c2_cheb2_tf32 | 0.0 | 83 | 98 | 3.853 | 19.99 | — |
| final_2l4 | two_level_c4_cheb2_tf32 | 0.0 | 35 | 38 | 2.152 | 4.71 | — |
| final_2l4 | two_level_c8_cheb2_tf32 | 0.0 | 17 | 18 | 1.658 | 1.76 | — |
| final_2l4 | two_level_c16_cheb2_tf32 | 0.0 | 24 | 25 | 1.524 | 2.29 | — |
| final_2l4 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 35 | 37 | 2.898 | 6.34 | — |
| final_2l4 | two_level_c8_cheb2_safeguard_tf32 | 0.0 | 17 | 18 | 2.392 | 2.54 | — |

## diffusion 1024² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 2549 | 2759 |  | 400.94 |  |
| **jacobi4** (GPU) |  | 0.0 | 1076 | 1129 |  | 239.06 |  |
| **chebyshev8** (GPU) |  | 0.0 | 502 | 532 |  | 172.76 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 8 | 8 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| final_2l2 | two_level_c2_cheb2_tf32 | 0.0 | 127 | 158 | 16.217 | 128.72 | — |
| final_2l2 | two_level_c4_cheb2_tf32 | 0.0 | 60 | 69 | 8.765 | 32.87 | — |
| final_2l2 | two_level_c8_cheb2_tf32 | 0.0 | 25 | 30 | 6.696 | 10.46 | — |
| final_2l2 | two_level_c16_cheb2_tf32 | 0.0 | 26 | 27 | 5.718 | 9.29 | — |
| final_2l2 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 57 | 64 | 12.064 | 42.98 | — |
| final_2l2 | two_level_c8_cheb2_safeguard_tf32 | 0.0 | 25 | 29 | 9.657 | 15.09 | — |
| final_2l4 | two_level_c2_cheb2_tf32 | 0.0 | 509 | 1080 | 16.175 | 514.56 | — |
| final_2l4 | two_level_c4_cheb2_tf32 | 0.0 | 151 | 192 | 8.768 | 82.75 | — |
| final_2l4 | two_level_c8_cheb2_tf32 | 0.0 | 44 | 49 | 6.627 | 18.22 | — |
| final_2l4 | two_level_c16_cheb2_tf32 | 0.0 | 28 | 30 | 5.708 | 9.99 | — |
| final_2l4 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 149 | 192 | 12.079 | 112.48 | — |
| final_2l4 | two_level_c8_cheb2_safeguard_tf32 | 0.0 | 44 | 49 | 9.643 | 26.52 | — |

## poisson 128² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 316 | 319 |  | 1.77 |  |
| **jacobi4** (GPU) |  | 0.0 | 136 | 137 |  | 0.85 |  |
| **chebyshev8** (GPU) |  | 0.0 | 64 | 65 |  | 0.52 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 6 | 7 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| final_2l2 | two_level_c2_cheb2_tf32 | 0.0 | 7 | 7 | 0.510 | 0.22 | — |
| final_2l2 | two_level_c4_cheb2_tf32 | 0.0 | 11 | 12 | 0.417 | 0.29 | — |
| final_2l2 | two_level_c8_cheb2_tf32 | 0.0 | 68 | 122 | 0.386 | 1.64 | — |
| final_2l2 | two_level_c16_cheb2_tf32 | 0.0 | 135 | 154 | 0.381 | 3.22 | — |
| final_2l2 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 11 | 12 | 0.451 | 0.31 | — |
| final_2l2 | two_level_c8_cheb2_safeguard_tf32 | 0.0 | 68 | 113 | 0.424 | 1.80 | — |
| final_2l4 | two_level_c2_cheb2_tf32 | 0.0 | 12 | 13 | 0.507 | 0.38 | — |
| final_2l4 | two_level_c4_cheb2_tf32 | 0.0 | 7 | 7 | 0.415 | 0.18 | — |
| final_2l4 | two_level_c8_cheb2_tf32 | 0.0 | 70 | 91 | 0.388 | 1.70 | — |
| final_2l4 | two_level_c16_cheb2_tf32 | 0.0 | 128 | 148 | 0.380 | 3.04 | — |
| final_2l4 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 7 | 7 | 0.451 | 0.20 | — |
| final_2l4 | two_level_c8_cheb2_safeguard_tf32 | 0.0 | 69 | 86 | 0.418 | 1.80 | — |

## poisson 256² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 632 | 639 |  | 7.20 |  |
| **jacobi4** (GPU) |  | 0.0 | 271 | 275 |  | 3.95 |  |
| **chebyshev8** (GPU) |  | 0.0 | 128 | 130 |  | 2.73 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 8 | 8 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| final_2l2 | two_level_c2_cheb2_tf32 | 0.0 | 15 | 16 | 1.129 | 1.06 | — |
| final_2l2 | two_level_c4_cheb2_tf32 | 0.0 | 10 | 10 | 0.703 | 0.44 | — |
| final_2l2 | two_level_c8_cheb2_tf32 | 0.0 | 18 | 19 | 0.602 | 0.68 | — |
| final_2l2 | two_level_c16_cheb2_tf32 | 0.0 | 151 | 278 | 0.582 | 5.49 | — |
| final_2l2 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 10 | 10 | 0.785 | 0.49 | — |
| final_2l2 | two_level_c8_cheb2_safeguard_tf32 | 0.0 | 17 | 18 | 0.679 | 0.72 | — |
| final_2l4 | two_level_c2_cheb2_tf32 | 0.0 | 27 | 32 | 1.121 | 1.89 | — |
| final_2l4 | two_level_c4_cheb2_tf32 | 0.0 | 16 | 17 | 0.700 | 0.70 | — |
| final_2l4 | two_level_c8_cheb2_tf32 | 0.0 | 14 | 15 | 0.594 | 0.52 | — |
| final_2l4 | two_level_c16_cheb2_tf32 | 0.0 | 179 | 225 | 0.564 | 6.31 | — |
| final_2l4 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 16 | 17 | 0.774 | 0.77 | — |
| final_2l4 | two_level_c8_cheb2_safeguard_tf32 | 0.0 | 14 | 15 | 0.669 | 0.59 | — |

## poisson 512² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 1270 | 1291 |  | 55.48 |  |
| **jacobi4** (GPU) |  | 0.0 | 546 | 555 |  | 32.32 |  |
| **chebyshev8** (GPU) |  | 0.0 | 258 | 261 |  | 21.10 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 7 | 7 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| final_2l2 | two_level_c2_cheb2_tf32 | 0.0 | 42 | 53 | 3.836 | 10.07 | — |
| final_2l2 | two_level_c4_cheb2_tf32 | 0.0 | 22 | 25 | 2.146 | 2.95 | — |
| final_2l2 | two_level_c8_cheb2_tf32 | 0.0 | 16 | 17 | 1.645 | 1.65 | — |
| final_2l2 | two_level_c16_cheb2_tf32 | 0.0 | 33 | 36 | 1.510 | 3.11 | — |
| final_2l2 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 22 | 25 | 2.873 | 3.95 | — |
| final_2l2 | two_level_c8_cheb2_safeguard_tf32 | 0.0 | 16 | 17 | 2.372 | 2.37 | — |
| final_2l4 | two_level_c2_cheb2_tf32 | 0.0 | 88 | 129 | 3.876 | 21.32 | — |
| final_2l4 | two_level_c4_cheb2_tf32 | 0.0 | 38 | 47 | 2.169 | 5.15 | — |
| final_2l4 | two_level_c8_cheb2_tf32 | 0.0 | 20 | 21 | 1.666 | 2.08 | — |
| final_2l4 | two_level_c16_cheb2_tf32 | 0.0 | 28 | 29 | 1.523 | 2.67 | — |
| final_2l4 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 37 | 45 | 2.914 | 6.74 | — |
| final_2l4 | two_level_c8_cheb2_safeguard_tf32 | 0.0 | 19 | 21 | 2.397 | 2.85 | — |

## poisson 1024² at 1e-6 (pooled over seeds × 100 systems)

| model | variant | fail % | median its | p90 its | ms/it b16 | ms per system to 1e-6 (b16) | Wilcoxon p vs notebook none |
|---|---|---|---|---|---|---|---|
| **cg** (GPU) |  | 0.0 | 2569 | 2604 |  | 402.25 |  |
| **jacobi4** (GPU) |  | 0.0 | 1103 | 1122 |  | 240.11 |  |
| **chebyshev8** (GPU) |  | 0.0 | 521 | 529 |  | 176.39 |  |
| **AMG smoothed_aggregation** (CPU, 1 core) |  | 0.0 | 9 | 9 |  | not comparable |  |
| **AMG ruge_stuben** (CPU, 1 core) |  | 0.0 | 4 | 4 |  | not comparable |  |
| final_2l2 | two_level_c2_cheb2_tf32 | 0.0 | 163 | 231 | 16.093 | 163.95 | — |
| final_2l2 | two_level_c4_cheb2_tf32 | 0.0 | 79 | 92 | 8.703 | 42.97 | — |
| final_2l2 | two_level_c8_cheb2_tf32 | 0.0 | 30 | 34 | 6.585 | 12.35 | — |
| final_2l2 | two_level_c16_cheb2_tf32 | 0.0 | 31 | 33 | 5.656 | 10.96 | — |
| final_2l2 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 74 | 83 | 11.945 | 55.25 | — |
| final_2l2 | two_level_c8_cheb2_safeguard_tf32 | 0.0 | 30 | 34 | 9.573 | 17.95 | — |
| final_2l4 | two_level_c2_cheb2_tf32 | 0.0 | 534 | 1117 | 16.057 | 535.91 | — |
| final_2l4 | two_level_c4_cheb2_tf32 | 0.0 | 164 | 212 | 8.698 | 89.16 | — |
| final_2l4 | two_level_c8_cheb2_tf32 | 0.0 | 46 | 63 | 6.557 | 18.85 | — |
| final_2l4 | two_level_c16_cheb2_tf32 | 0.0 | 35 | 37 | 5.608 | 12.27 | — |
| final_2l4 | two_level_c4_cheb2_safeguard_tf32 | 0.0 | 162 | 212 | 11.971 | 121.20 | — |
| final_2l4 | two_level_c8_cheb2_safeguard_tf32 | 0.0 | 46 | 63 | 9.574 | 27.53 | — |
