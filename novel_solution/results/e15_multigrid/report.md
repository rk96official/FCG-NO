# E15 smooth high-contrast coefficients vs classical multigrid

## grf2 128² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 77 | 84 | 0.113 | 0 | 0.54 | 0.54 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 6 | 7 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 5 | 5 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 8 | 8 |  | 4.70 | 2.12 | 6.82 |
| **amgx_classical_fp32** (GPU, per system) | 0.0 | 8 | 8 |  | 4.59 | 2.02 | 6.61 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 30 | 31 |  | 4.26 | 8.35 | 12.62 |
| gmg_vcycle_fp32coarse | 0.0 | 6 | 6 | 0.601 | 0.42 | 0.23 | 0.65 |
| tl64_vcycle | 0.0 | 6 | 7 | 0.677 | 0.39 | 0.25 | 0.64 |
| tl64_chebyshev8 | 0.0 | 24 | 26 | 0.188 | 0.17 | 0.28 | 0.45 |
| redisc_dense16 | 0.0 | 7 | 7 | 0.683 | 0.00 | 0.30 | 0.30 |
| tl64_hc | 0.0 | 11 | 12 | 0.594 | 0.00 | 0.41 | 0.41 |
| tl64_hc_coef | 0.0 | 8 | 8 | 0.637 | 0.00 | 0.32 | 0.32 |
| ml32_hc | 0.0 | 10 | 11 | 0.593 | 0.00 | 0.37 | 0.37 |
| tl64_e14_diffusion | 0.0 | 27 | 31 | 0.595 | 0.00 | 1.00 | 1.00 |
| ml64_e14_diffusion | 0.0 | 27 | 31 | 0.686 | 0.00 | 1.16 | 1.16 |

## grf2 256² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 155 | 170 | 0.338 | 0 | 3.27 | 3.27 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 8 | 8 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 5 | 6 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 9 | 9 |  | 9.45 | 3.83 | 13.28 |
| **amgx_classical_fp32** (GPU, per system) | 0.0 | 9 | 9 |  | 9.36 | 3.88 | 13.24 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 45 | 47 |  | 8.47 | 25.80 | 34.27 |
| gmg_vcycle_fp32coarse | 0.0 | 6 | 7 | 0.839 | 0.77 | 0.31 | 1.09 |
| tl64_vcycle | 0.0 | 7 | 8 | 0.847 | 0.44 | 0.37 | 0.81 |
| tl64_chebyshev8 | 0.0 | 27 | 29 | 0.371 | 0.20 | 0.63 | 0.83 |
| redisc_dense16 | 0.0 | 7 | 7 | 0.914 | 0.00 | 0.40 | 0.40 |
| tl64_hc | 0.0 | 15 | 16 | 0.705 | 0.00 | 0.66 | 0.66 |
| tl64_hc_coef | 0.0 | 11 | 12 | 0.702 | 0.00 | 0.48 | 0.48 |
| ml32_hc | 0.0 | 10 | 11 | 0.767 | 0.00 | 0.48 | 0.48 |
| tl64_e14_diffusion | 0.0 | 39 | 44 | 0.698 | 0.00 | 1.70 | 1.70 |
| ml64_e14_diffusion | 0.0 | 26 | 29 | 0.813 | 0.00 | 1.32 | 1.32 |

## grf2 512² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 313 | 344 | 1.329 | 0 | 26.00 | 26.00 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 7 | 7 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 6 | 6 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 9 | 9 |  | 13.57 | 5.05 | 18.62 |
| **amgx_classical_fp32** (GPU, per system) | 0.0 | 9 | 9 |  | 13.38 | 4.87 | 18.25 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 62 | 65 |  | 11.59 | 42.22 | 53.81 |
| gmg_vcycle_fp32coarse | 0.0 | 7 | 7 | 2.098 | 2.26 | 0.92 | 3.18 |
| tl64_vcycle | 0.0 | 15 | 16 | 1.656 | 0.83 | 1.55 | 2.38 |
| tl64_chebyshev8 | 0.0 | 28 | 31 | 1.224 | 0.57 | 2.14 | 2.72 |
| redisc_dense16 | 0.0 | 7 | 7 | 1.971 | 0.00 | 0.86 | 0.86 |
| tl64_hc | 0.0 | 22 | 24 | 1.653 | 0.00 | 2.27 | 2.27 |
| tl64_hc_coef | 0.0 | 17 | 18 | 1.667 | 0.00 | 1.77 | 1.77 |
| ml32_hc | 0.0 | 10 | 11 | 2.183 | 0.00 | 1.36 | 1.36 |
| tl64_e14_diffusion | 0.0 | 54 | 60 | 1.661 | 0.00 | 5.61 | 5.61 |
| ml64_e14_diffusion | 0.0 | 25 | 29 | 2.254 | 0.00 | 3.52 | 3.52 |

## grf4 128² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 84 | 98 | 0.114 | 0 | 0.60 | 0.60 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 7 | 7 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 7 | 8 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 8 | 8 |  | 6.75 | 2.86 | 9.61 |
| **amgx_classical_fp32** (GPU, per system) | 0.0 | 8 | 8 |  | 6.67 | 2.80 | 9.47 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 29 | 30 |  | 6.39 | 13.51 | 19.90 |
| gmg_vcycle_fp32coarse | 0.0 | 6 | 6 | 0.584 | 0.36 | 0.22 | 0.58 |
| tl64_vcycle | 0.0 | 7 | 7 | 0.685 | 0.35 | 0.30 | 0.65 |
| tl64_chebyshev8 | 0.0 | 27 | 31 | 0.174 | 0.12 | 0.29 | 0.41 |
| redisc_dense16 | 0.0 | 7 | 7 | 0.668 | 0.00 | 0.29 | 0.29 |
| tl64_hc | 0.0 | 16 | 18 | 0.508 | 0.00 | 0.51 | 0.51 |
| tl64_hc_coef | 0.0 | 127 | 158 | 0.510 | 0.00 | 4.05 | 4.05 |
| ml32_hc | 0.0 | 13 | 15 | 0.487 | 0.00 | 0.40 | 0.40 |
| tl64_e14_diffusion | 0.0 | 62 | 78 | 0.539 | 0.00 | 2.09 | 2.09 |
| ml64_e14_diffusion | 0.0 | 62 | 78 | 0.510 | 0.00 | 1.98 | 1.98 |

## grf4 256² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 169 | 197 | 0.341 | 0 | 3.61 | 3.61 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 8 | 9 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 7 | 8 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 8 | 9 |  | 9.49 | 3.53 | 13.02 |
| **amgx_classical_fp32** (GPU, per system) | 0.0 | 8 | 9 |  | 9.28 | 3.51 | 12.78 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 44 | 47 |  | 8.59 | 25.55 | 34.14 |
| gmg_vcycle_fp32coarse | 0.0 | 6 | 7 | 0.840 | 0.72 | 0.31 | 1.04 |
| tl64_vcycle | 0.0 | 8 | 8 | 0.847 | 0.44 | 0.42 | 0.86 |
| tl64_chebyshev8 | 0.0 | 30 | 34 | 0.346 | 0.19 | 0.65 | 0.84 |
| redisc_dense16 | 0.0 | 7 | 8 | 0.911 | 0.00 | 0.40 | 0.40 |
| tl64_hc | 0.0 | 24 | 28 | 0.700 | 0.00 | 1.05 | 1.05 |
| tl64_hc_coef | 0.0 | 275 | 348 | 0.699 | 0.00 | 12.02 | 12.02 |
| ml32_hc | 0.0 | 14 | 15 | 0.765 | 0.00 | 0.67 | 0.67 |
| tl64_e14_diffusion | 0.0 | 105 | 136 | 0.700 | 0.00 | 4.59 | 4.59 |
| ml64_e14_diffusion | 0.0 | 57 | 69 | 0.817 | 0.00 | 2.91 | 2.91 |

## grf4 512² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 343 | 396 | 1.334 | 0 | 28.59 | 28.59 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 7 | 8 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 7 | 8 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 9 | 9 |  | 13.30 | 4.98 | 18.28 |
| **amgx_classical_fp32** (GPU, per system) | 0.0 | 9 | 9 |  | 13.11 | 4.82 | 17.93 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 61 | 65 |  | 11.58 | 42.15 | 53.73 |
| gmg_vcycle_fp32coarse | 0.0 | 7 | 7 | 2.106 | 2.26 | 0.92 | 3.19 |
| tl64_vcycle | 0.0 | 15 | 16 | 1.668 | 0.83 | 1.56 | 2.39 |
| tl64_chebyshev8 | 0.0 | 32 | 38 | 1.235 | 0.56 | 2.47 | 3.03 |
| redisc_dense16 | 0.0 | 7 | 8 | 1.982 | 0.00 | 0.87 | 0.87 |
| tl64_hc | 0.0 | 41 | 48 | 1.647 | 0.00 | 4.22 | 4.22 |
| tl64_hc_coef | 0.0 | 628 | 802 | 1.671 | 0.00 | 65.57 | 65.57 |
| ml32_hc | 0.0 | 14 | 15 | 2.186 | 0.00 | 1.91 | 1.91 |
| tl64_e14_diffusion | 0.0 | 160 | 201 | 1.674 | 0.00 | 16.74 | 16.74 |
| ml64_e14_diffusion | 0.0 | 57 | 69 | 2.244 | 0.00 | 7.99 | 7.99 |

## grf6 128² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 92 | 113 | 0.113 | 0 | 0.65 | 0.65 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 8 | 9 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 8 | 10 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 8 | 8 |  | 6.85 | 2.89 | 9.74 |
| **amgx_classical_fp32** (GPU, per system) | 0.0 | 8 | 8 |  | 6.78 | 2.85 | 9.62 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 28 | 31 |  | 6.47 | 13.15 | 19.61 |
| gmg_vcycle_fp32coarse | 0.0 | 6 | 7 | 0.589 | 0.41 | 0.22 | 0.63 |
| tl64_vcycle | 0.0 | 7 | 7 | 0.683 | 0.41 | 0.30 | 0.71 |
| tl64_chebyshev8 | 0.0 | 30 | 40 | 0.189 | 0.16 | 0.35 | 0.51 |
| redisc_dense16 | 0.0 | 7 | 8 | 0.672 | 0.00 | 0.29 | 0.29 |
| tl64_hc | 0.0 | 67 | 87 | 0.605 | 0.00 | 2.54 | 2.54 |
| tl64_hc_coef | 0.0 | 135 | 183 | 0.608 | 0.00 | 5.13 | 5.13 |
| ml32_hc | 0.0 | 19 | 22 | 0.603 | 0.00 | 0.72 | 0.72 |
| tl64_e14_diffusion | 0.0 | 106 | 140 | 0.611 | 0.00 | 4.05 | 4.05 |
| ml64_e14_diffusion | 0.0 | 106 | 140 | 0.708 | 0.00 | 4.69 | 4.69 |

## grf6 256² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 186 | 229 | 0.345 | 0 | 4.01 | 4.01 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 9 | 11 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 8 | 10 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 8 | 8 |  | 9.32 | 3.49 | 12.81 |
| **amgx_classical_fp32** (GPU, per system) | 0.0 | 8 | 8 |  | 9.19 | 3.51 | 12.70 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 44 | 48 |  | 8.53 | 25.41 | 33.94 |
| gmg_vcycle_fp32coarse | 0.0 | 6 | 7 | 0.848 | 0.72 | 0.32 | 1.04 |
| tl64_vcycle | 0.0 | 8 | 8 | 0.859 | 0.45 | 0.43 | 0.88 |
| tl64_chebyshev8 | 0.0 | 35 | 51 | 0.347 | 0.19 | 0.76 | 0.95 |
| redisc_dense16 | 0.0 | 7 | 8 | 0.919 | 0.00 | 0.40 | 0.40 |
| tl64_hc | 0.0 | 149 | 206 | 0.701 | 0.00 | 6.53 | 6.53 |
| tl64_hc_coef | 0.0 | 294 | 389 | 0.701 | 0.00 | 12.88 | 12.88 |
| ml32_hc | 0.0 | 19 | 22 | 0.765 | 0.00 | 0.91 | 0.91 |
| tl64_e14_diffusion | 0.0 | 182 | 254 | 0.700 | 0.00 | 7.97 | 7.97 |
| ml64_e14_diffusion | 0.0 | 96 | 129 | 0.815 | 0.00 | 4.89 | 4.89 |

## grf6 512² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 377 | 461 | 1.357 | 0 | 31.98 | 31.98 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 8 | 9 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 8 | 10 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 8 | 9 |  | 13.61 | 4.54 | 18.15 |
| **amgx_classical_fp32** (GPU, per system) | 0.0 | 8 | 9 |  | 13.42 | 4.35 | 17.78 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 60 | 66 |  | 11.53 | 41.02 | 52.55 |
| gmg_vcycle_fp32coarse | 0.0 | 7 | 7 | 2.130 | 2.35 | 0.93 | 3.29 |
| tl64_vcycle | 0.0 | 14 | 15 | 1.676 | 0.84 | 1.47 | 2.30 |
| tl64_chebyshev8 | 0.0 | 39 | 63 | 1.243 | 0.60 | 3.03 | 3.63 |
| redisc_dense16 | 0.0 | 8 | 8 | 1.993 | 0.00 | 1.00 | 1.00 |
| tl64_hc | 0.0 | 337 | 460 | 1.677 | 0.00 | 35.32 | 35.32 |
| tl64_hc_coef | 0.0 | 693 | 956 | 1.668 | 0.00 | 72.23 | 72.23 |
| ml32_hc | 0.0 | 19 | 22 | 2.205 | 0.00 | 2.62 | 2.62 |
| tl64_e14_diffusion | 0.0 | 300 | 407 | 1.704 | 0.00 | 31.94 | 31.94 |
| ml64_e14_diffusion | 0.0 | 96 | 131 | 2.282 | 0.00 | 13.69 | 13.69 |
