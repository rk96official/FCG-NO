# E20 jump coefficients vs classical multigrid

## jump2 128² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 94 | 126 | 0.114 | 0 | 0.67 | 0.67 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 9 | 12 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 7 | 8 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 7 | 8 |  | 4.71 | 1.89 | 6.60 |
| **amgx_classical_fp32** (GPU, per system) | 0.0 | 7 | 8 |  | 4.62 | 1.80 | 6.42 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 25 | 27 |  | 4.25 | 7.08 | 11.32 |
| gmg_vcycle_fp32coarse | 0.0 | 10 | 13 | 0.597 | 0.42 | 0.37 | 0.79 |
| tl64_vcycle | 0.0 | 10 | 13 | 0.694 | 0.42 | 0.43 | 0.85 |
| tl64_exact | 0.0 | 8 | 9 | 8.918 | 4.06 | 4.46 | 8.52 |
| tl64_chebyshev8 | 0.0 | 33 | 43 | 0.188 | 0.16 | 0.39 | 0.55 |
| redisc_dense16 | 0.0 | 19 | 23 | 0.681 | 0.00 | 0.81 | 0.81 |
| tl2_jr_coef | 0.0 | 144 | 246 | 0.635 | 0.00 | 5.71 | 5.71 |
| fine_jr | 0.0 | 14 | 17 | 1.252 | 0.00 | 1.10 | 1.10 |
| fine_jr_coef_s0 | 0.0 | 16 | 21 | 1.242 | 0.00 | 1.24 | 1.24 |
| fine_jr_coef_s1 | 0.0 | 16 | 21 | 1.244 | 0.00 | 1.24 | 1.24 |
| fine_jr_coef_s2 | 0.0 | 14 | 19 | 1.298 | 0.00 | 1.14 | 1.14 |

## jump2 256² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 190 | 250 | 0.337 | 0 | 4.00 | 4.00 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 10 | 13 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 8 | 9 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 8 | 8 |  | 6.61 | 2.68 | 9.29 |
| **amgx_classical_fp32** (GPU, per system) | 0.0 | 8 | 8 |  | 6.48 | 2.55 | 9.03 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 39 | 43 |  | 5.72 | 14.29 | 20.01 |
| gmg_vcycle_fp32coarse | 0.0 | 10 | 13 | 0.836 | 0.72 | 0.52 | 1.24 |
| tl64_vcycle | 0.0 | 14 | 17 | 0.843 | 0.44 | 0.74 | 1.18 |
| tl64_exact | 0.0 | 12 | 14 | 8.954 | 4.13 | 6.72 | 10.84 |
| tl64_chebyshev8 | 0.0 | 36 | 50 | 0.362 | 0.20 | 0.81 | 1.02 |
| redisc_dense16 | 0.0 | 23 | 29 | 0.906 | 0.00 | 1.30 | 1.30 |
| tl2_jr_coef | 0.0 | 299 | 488 | 1.117 | 0.00 | 20.87 | 20.87 |
| fine_jr | 0.0 | 35 | 43 | 3.276 | 0.00 | 7.17 | 7.17 |
| fine_jr_coef_s0 | 0.0 | 39 | 58 | 3.286 | 0.00 | 8.01 | 8.01 |
| fine_jr_coef_s1 | 0.0 | 37 | 53 | 3.295 | 0.00 | 7.62 | 7.62 |
| fine_jr_coef_s2 | 0.0 | 32 | 46 | 3.291 | 0.00 | 6.58 | 6.58 |

## jump2 512² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 388 | 503 | 1.301 | 0 | 31.55 | 31.55 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 10 | 13 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 9 | 11 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 8 | 9 |  | 9.97 | 3.74 | 13.71 |
| **amgx_classical_fp32** (GPU, per system) | 0.0 | 8 | 9 |  | 9.76 | 3.16 | 12.92 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 52 | 56 |  | 7.48 | 24.63 | 32.11 |
| gmg_vcycle_fp32coarse | 0.0 | 10 | 14 | 2.096 | 2.26 | 1.31 | 3.57 |
| tl64_vcycle | 0.0 | 23 | 26 | 1.650 | 0.82 | 2.37 | 3.19 |
| tl64_exact | 0.0 | 21 | 24 | 9.751 | 4.50 | 12.80 | 17.30 |
| tl64_chebyshev8 | 0.0 | 43 | 64 | 1.216 | 0.56 | 3.27 | 3.83 |
| redisc_dense16 | 0.0 | 27 | 35 | 1.968 | 0.00 | 3.32 | 3.32 |
| tl2_jr_coef | 0.0 | 613 | 931 | 3.859 | 0.00 | 147.84 | 147.84 |
| fine_jr | 0.0 | 106 | 131 | 12.372 | 0.00 | 81.97 | 81.97 |
| fine_jr_coef_s0 | 0.0 | 115 | 180 | 12.380 | 0.00 | 88.98 | 88.98 |
| fine_jr_coef_s1 | 0.0 | 142 | 224 | 12.477 | 0.00 | 110.73 | 110.73 |
| fine_jr_coef_s2 | 0.0 | 106 | 190 | 12.455 | 0.00 | 82.51 | 82.51 |

## jump4 128² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 118 | 169 | 0.113 | 0 | 0.84 | 0.84 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 12 | 20 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 6 | 7 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 7 | 7 |  | 4.70 | 1.86 | 6.56 |
| **amgx_classical_fp32** (GPU, per system) | 0.0 | 7 | 7 |  | 4.59 | 1.77 | 6.36 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 21 | 23 |  | 4.21 | 5.86 | 10.06 |
| gmg_vcycle_fp32coarse | 0.0 | 16 | 24 | 0.580 | 0.35 | 0.58 | 0.93 |
| tl64_vcycle | 0.0 | 16 | 24 | 0.677 | 0.35 | 0.68 | 1.03 |
| tl64_exact | 0.0 | 8 | 12 | 8.797 | 4.06 | 4.40 | 8.45 |
| tl64_chebyshev8 | 0.0 | 51 | 131 | 0.179 | 0.11 | 0.57 | 0.68 |
| redisc_dense16 | 7.0 | 211 | 487 | 0.663 | 0.00 | 8.74 | 8.74 |
| tl2_jr_coef | 0.0 | 199 | 320 | 0.534 | 0.00 | 6.64 | 6.64 |
| fine_jr | 0.0 | 54 | 98 | 1.261 | 0.00 | 4.25 | 4.25 |
| fine_jr_coef_s0 | 3.0 | 212 | 346 | 1.229 | 0.00 | 16.29 | 16.29 |
| fine_jr_coef_s1 | 3.0 | 209 | 389 | 1.229 | 0.00 | 16.06 | 16.06 |
| fine_jr_coef_s2 | 0.0 | 75 | 165 | 1.228 | 0.00 | 5.76 | 5.76 |

## jump4 256² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 245 | 334 | 0.325 | 0 | 4.97 | 4.97 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 14 | 24 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 7 | 8 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 7 | 8 |  | 6.71 | 2.42 | 9.13 |
| **amgx_classical_fp32** (GPU, per system) | 0.0 | 7 | 8 |  | 6.46 | 2.25 | 8.71 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 31 | 36 |  | 5.72 | 11.37 | 17.09 |
| gmg_vcycle_fp32coarse | 0.0 | 16 | 26 | 0.855 | 0.73 | 0.85 | 1.58 |
| tl64_vcycle | 0.0 | 21 | 41 | 0.869 | 0.45 | 1.14 | 1.59 |
| tl64_exact | 0.0 | 13 | 21 | 9.082 | 4.20 | 7.38 | 11.58 |
| tl64_chebyshev8 | 0.0 | 61 | 170 | 0.357 | 0.20 | 1.36 | 1.56 |
| redisc_dense16 | 3.0 | 437 | 865 | 0.924 | 0.00 | 25.23 | 25.23 |
| tl2_jr_coef | 1.0 | 387 | 632 | 1.128 | 0.00 | 27.28 | 27.28 |
| fine_jr | 21.0 | 635 | — | 3.296 | 0.00 | 130.80 | 130.80 |
| fine_jr_coef_s0 | 2.0 | 392 | 720 | 3.272 | 0.00 | 80.16 | 80.16 |
| fine_jr_coef_s1 | 1.0 | 418 | 715 | 3.284 | 0.00 | 85.79 | 85.79 |
| fine_jr_coef_s2 | 14.0 | 439 | — | 3.293 | 0.00 | 90.35 | 90.35 |

## jump4 512² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 485 | 660 | 1.285 | 0 | 38.94 | 38.94 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 13 | 23 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 7 | 9 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 8 | 8 |  | 10.10 | 3.83 | 13.92 |
| **amgx_classical_fp32** (GPU, per system) | 0.0 | 8 | 8 |  | 9.89 | 3.26 | 13.15 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 41 | 46 |  | 7.63 | 20.51 | 28.14 |
| gmg_vcycle_fp32coarse | 0.0 | 17 | 27 | 2.103 | 2.27 | 2.23 | 4.51 |
| tl64_vcycle | 0.0 | 37 | 83 | 1.658 | 0.88 | 3.83 | 4.72 |
| tl64_exact | 0.0 | 25 | 38 | 9.767 | 4.54 | 15.26 | 19.80 |
| tl64_chebyshev8 | 0.0 | 85 | 229 | 1.210 | 0.58 | 6.43 | 7.01 |
| redisc_dense16 | 2.0 | 906 | 1630 | 1.973 | 0.00 | 111.73 | 111.73 |
| tl2_jr_coef | 1.0 | 788 | 1277 | 3.809 | 0.00 | 187.60 | 187.60 |
| fine_jr | 17.0 | 1197 | — | 12.313 | 0.00 | 921.19 | 921.19 |
| fine_jr_coef_s0 | 2.0 | 805 | 1481 | 12.242 | 0.00 | 615.92 | 615.92 |
| fine_jr_coef_s1 | 2.0 | 798 | 1545 | 12.320 | 0.00 | 614.48 | 614.48 |
| fine_jr_coef_s2 | 15.0 | 858 | — | 12.336 | 0.00 | 661.51 | 661.51 |

## jump6 128² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 62 | 131 | 0.114 | 0 | 0.44 | 0.44 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 13 | 23 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 5 | 6 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 7 | 7 |  | 4.70 | 1.86 | 6.56 |
| **amgx_classical_fp32** (GPU, per system) | 16.0 | 7 | — |  | 4.61 | 1.78 | 6.39 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 19 | 20 |  | 4.21 | 5.32 | 9.54 |
| gmg_vcycle_fp32coarse | 0.0 | 17 | 32 | 0.576 | 0.34 | 0.61 | 0.96 |
| tl64_vcycle | 0.0 | 18 | 33 | 0.671 | 0.34 | 0.76 | 1.10 |
| tl64_exact | 0.0 | 8 | 15 | 8.776 | 4.02 | 4.39 | 8.40 |
| tl64_chebyshev8 | 1.0 | 39 | 95 | 0.172 | 0.10 | 0.42 | 0.52 |
| redisc_dense16 | 1.0 | 104 | 326 | 0.658 | 0.00 | 4.28 | 4.28 |
| tl2_jr_coef | 6.0 | 73 | 337 | 0.505 | 0.00 | 2.30 | 2.30 |
| fine_jr | 0.0 | 68 | 130 | 1.230 | 0.00 | 5.23 | 5.23 |
| fine_jr_coef_s0 | 4.0 | 73 | 316 | 1.221 | 0.00 | 5.57 | 5.57 |
| fine_jr_coef_s1 | 0.0 | 69 | 121 | 1.224 | 0.00 | 5.28 | 5.28 |
| fine_jr_coef_s2 | 6.0 | 73 | 297 | 1.221 | 0.00 | 5.57 | 5.57 |

## jump6 256² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 107 | 298 | 0.337 | 0 | 2.25 | 2.25 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 15 | 29 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 6 | 7 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 7 | 7 |  | 6.55 | 2.32 | 8.87 |
| **amgx_classical_fp32** (GPU, per system) | 6.0 | 13 | 186 |  | 6.45 | 4.08 | 10.53 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 25 | 27 |  | 5.65 | 8.94 | 14.60 |
| gmg_vcycle_fp32coarse | 0.0 | 20 | 38 | 0.854 | 0.78 | 1.07 | 1.85 |
| tl64_vcycle | 0.0 | 28 | 68 | 0.869 | 0.51 | 1.52 | 2.03 |
| tl64_exact | 0.0 | 12 | 27 | 9.086 | 4.20 | 6.81 | 11.02 |
| tl64_chebyshev8 | 0.0 | 54 | 183 | 0.353 | 0.25 | 1.19 | 1.44 |
| redisc_dense16 | 1.0 | 304 | 665 | 0.927 | 0.00 | 17.62 | 17.62 |
| tl2_jr_coef | 11.0 | 192 | — | 1.128 | 0.00 | 13.54 | 13.54 |
| fine_jr | 2.0 | 152 | 436 | 3.308 | 0.00 | 31.43 | 31.43 |
| fine_jr_coef_s0 | 8.0 | 128 | 769 | 3.318 | 0.00 | 26.55 | 26.55 |
| fine_jr_coef_s1 | 2.0 | 281 | 799 | 3.312 | 0.00 | 58.16 | 58.16 |
| fine_jr_coef_s2 | 7.0 | 132 | 903 | 3.338 | 0.00 | 27.54 | 27.54 |

## jump6 512² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 336 | 594 | 1.284 | 0 | 26.96 | 26.96 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 16 | 29 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 6 | 8 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 7 | 8 |  | 10.08 | 3.39 | 13.48 |
| **amgx_classical_fp32** (GPU, per system) | 27.0 | 12 | — |  | 9.91 | 4.79 | 14.70 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 31 | 35 |  | 7.59 | 15.37 | 22.96 |
| gmg_vcycle_fp32coarse | 0.0 | 21 | 40 | 2.097 | 2.28 | 2.75 | 5.03 |
| tl64_vcycle | 0.0 | 52 | 127 | 1.659 | 0.84 | 5.39 | 6.23 |
| tl64_exact | 0.0 | 24 | 49 | 9.784 | 4.52 | 14.68 | 19.19 |
| tl64_chebyshev8 | 0.0 | 79 | 436 | 1.199 | 0.59 | 5.92 | 6.51 |
| redisc_dense16 | 1.0 | 633 | 1214 | 1.969 | 0.00 | 77.90 | 77.90 |
| tl2_jr_coef | 28.0 | 837 | — | 3.821 | 0.00 | 199.90 | 199.90 |
| fine_jr | 2.0 | 364 | 1023 | 12.305 | 0.00 | 279.95 | 279.95 |
| fine_jr_coef_s0 | 32.0 | 909 | — | 12.332 | 0.00 | 700.59 | 700.59 |
| fine_jr_coef_s1 | 1.0 | 679 | 1432 | 12.287 | 0.00 | 521.43 | 521.43 |
| fine_jr_coef_s2 | 31.0 | 819 | — | 12.379 | 0.00 | 633.66 | 633.66 |
