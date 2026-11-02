# E13 classical multigrid vs the learned coarse solver

## diffusion 128² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 59 | 64 | 0.112 | 0 | 0.41 | 0.41 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 6 | 6 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 4 | 4 |  |  |  | not comparable |
| gmg_vcycle | 0.0 | 6 | 6 | 0.679 | 0.36 | 0.25 | 0.61 |
| gmg_vcycle_fp32coarse | 0.0 | 6 | 6 | 0.591 | 0.36 | 0.22 | 0.58 |
| tl64_smoothing_only | 0.0 | 93 | 100 | 0.106 | 0.11 | 0.62 | 0.72 |
| tl64_chebyshev8 | 0.0 | 19 | 21 | 0.171 | 0.11 | 0.20 | 0.31 |
| tl64_vcycle | 0.0 | 6 | 6 | 0.690 | 0.35 | 0.26 | 0.61 |
| tl64_vcycle_fp32 | 0.0 | 6 | 6 | 0.586 | 0.35 | 0.22 | 0.57 |
| tl64_exact | 0.0 | 6 | 6 | 8.913 | 4.06 | 3.34 | 7.41 |
| tl64_network | 0.0 | 6 | 6 | 1.390 | 0.00 | 0.52 | 0.52 |
| tl32_vcycle | 0.0 | 7 | 7 | 0.635 | 0.20 | 0.28 | 0.48 |
| tl32_exact | 0.0 | 7 | 7 | 2.137 | 0.30 | 0.93 | 1.23 |
| tl32_network | 0.0 | 7 | 7 | 1.015 | 0.00 | 0.44 | 0.44 |

## diffusion 256² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 119 | 129 | 0.340 | 0 | 2.53 | 2.53 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 7 | 7 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 4 | 4 |  |  |  | not comparable |
| gmg_vcycle | 0.0 | 6 | 6 | 1.011 | 0.71 | 0.38 | 1.09 |
| gmg_vcycle_fp32coarse | 0.0 | 6 | 6 | 0.833 | 0.71 | 0.31 | 1.03 |
| tl64_smoothing_only | 0.0 | 186 | 202 | 0.248 | 0.20 | 2.88 | 3.08 |
| tl64_chebyshev8 | 0.0 | 21 | 23 | 0.345 | 0.19 | 0.45 | 0.64 |
| tl64_vcycle | 0.0 | 7 | 7 | 0.839 | 0.43 | 0.37 | 0.80 |
| tl64_vcycle_fp32 | 0.0 | 7 | 7 | 0.742 | 0.44 | 0.32 | 0.76 |
| tl64_exact | 0.0 | 7 | 7 | 8.922 | 4.15 | 3.90 | 8.05 |
| tl64_network | 0.0 | 9 | 9 | 1.556 | 0.00 | 0.88 | 0.88 |
| tl32_vcycle | 0.0 | 13 | 13 | 0.785 | 0.28 | 0.64 | 0.92 |
| tl32_exact | 0.0 | 13 | 13 | 2.251 | 0.38 | 1.83 | 2.21 |
| tl32_network | 0.0 | 13 | 13 | 1.201 | 0.00 | 0.98 | 0.98 |

## diffusion 512² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 241 | 262 | 1.337 | 0 | 20.13 | 20.13 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 6 | 7 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 4 | 4 |  |  |  | not comparable |
| gmg_vcycle | 0.0 | 6 | 6 | 4.682 | 2.28 | 1.76 | 4.03 |
| gmg_vcycle_fp32coarse | 0.0 | 6 | 6 | 2.112 | 2.27 | 0.79 | 3.06 |
| tl64_smoothing_only | 0.0 | 383 | 408 | 1.027 | 0.60 | 24.58 | 25.17 |
| tl64_chebyshev8 | 0.0 | 23 | 24 | 1.234 | 0.57 | 1.77 | 2.35 |
| tl64_vcycle | 0.0 | 13 | 13 | 1.673 | 0.83 | 1.36 | 2.19 |
| tl64_vcycle_fp32 | 0.0 | 13 | 13 | 1.576 | 0.85 | 1.28 | 2.13 |
| tl64_exact | 0.0 | 13 | 13 | 9.873 | 4.56 | 8.02 | 12.59 |
| tl64_network | 0.0 | 14 | 14 | 2.516 | 0.00 | 2.20 | 2.20 |
| tl32_vcycle | 0.0 | 24 | 25 | 1.599 | 0.63 | 2.40 | 3.03 |
| tl32_exact | 0.0 | 24 | 25 | 3.099 | 0.74 | 4.65 | 5.39 |
| tl32_network | 0.0 | 24 | 25 | 2.082 | 0.00 | 3.12 | 3.12 |

## diffusion 1024² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 502 | 532 | 5.506 | 0 | 172.76 | 172.76 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 8 | 8 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 4 | 4 |  |  |  | not comparable |
| gmg_vcycle | 0.0 | 6 | 6 | 15.959 | 9.20 | 5.98 | 15.18 |
| gmg_vcycle_fp32coarse | 0.0 | 6 | 7 | 7.462 | 9.24 | 2.80 | 12.04 |
| tl64_smoothing_only | 0.0 | 800 | 828 | 4.586 | 1.67 | 229.30 | 230.97 |
| tl64_chebyshev8 | 0.0 | 31 | 33 | 5.257 | 1.66 | 10.19 | 11.85 |
| tl64_vcycle | 0.0 | 24 | 24 | 5.342 | 1.92 | 8.01 | 9.93 |
| tl64_vcycle_fp32 | 0.0 | 24 | 25 | 5.299 | 1.93 | 7.95 | 9.88 |
| tl64_exact | 0.0 | 24 | 25 | 13.349 | 5.56 | 20.02 | 25.58 |
| tl64_network | 0.0 | 26 | 27 | 6.731 | 0.00 | 10.94 | 10.94 |
| tl32_vcycle | 0.0 | 46 | 49 | 5.250 | 1.58 | 15.09 | 16.67 |
| tl32_exact | 0.0 | 47 | 49 | 6.582 | 1.67 | 19.33 | 21.00 |
| tl32_network | 0.0 | 47 | 50 | 6.133 | 0.00 | 18.02 | 18.02 |

## poisson 128² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 64 | 65 | 0.129 | 0 | 0.52 | 0.52 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 6 | 7 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 4 | 4 |  |  |  | not comparable |
| gmg_vcycle | 0.0 | 6 | 6 | 0.685 | 0.42 | 0.26 | 0.68 |
| gmg_vcycle_fp32coarse | 0.0 | 6 | 6 | 0.598 | 0.42 | 0.22 | 0.64 |
| tl64_smoothing_only | 0.0 | 99 | 100 | 0.123 | 0.16 | 0.76 | 0.92 |
| tl64_chebyshev8 | 0.0 | 20 | 20 | 0.188 | 0.17 | 0.23 | 0.40 |
| tl64_vcycle | 0.0 | 6 | 6 | 0.695 | 0.42 | 0.26 | 0.68 |
| tl64_vcycle_fp32 | 0.0 | 6 | 6 | 0.590 | 0.42 | 0.22 | 0.64 |
| tl64_exact | 0.0 | 6 | 6 | 8.910 | 4.07 | 3.34 | 7.42 |
| tl64_network | 0.0 | 6 | 7 | 0.618 | 0.00 | 0.23 | 0.23 |
| tl32_vcycle | 0.0 | 7 | 7 | 0.639 | 0.25 | 0.28 | 0.53 |
| tl32_exact | 0.0 | 7 | 7 | 2.144 | 0.35 | 0.94 | 1.29 |
| tl32_network | 0.0 | 7 | 7 | 0.593 | 0.00 | 0.26 | 0.26 |

## poisson 256² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 128 | 130 | 0.341 | 0 | 2.73 | 2.73 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 8 | 8 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 4 | 4 |  |  |  | not comparable |
| gmg_vcycle | 0.0 | 6 | 6 | 1.018 | 0.76 | 0.38 | 1.14 |
| gmg_vcycle_fp32coarse | 0.0 | 6 | 7 | 0.837 | 0.79 | 0.31 | 1.10 |
| tl64_smoothing_only | 0.0 | 198 | 200 | 0.255 | 0.26 | 3.15 | 3.41 |
| tl64_chebyshev8 | 0.0 | 22 | 22 | 0.352 | 0.25 | 0.48 | 0.73 |
| tl64_vcycle | 0.0 | 7 | 7 | 0.847 | 0.49 | 0.37 | 0.86 |
| tl64_vcycle_fp32 | 0.0 | 7 | 7 | 0.752 | 0.49 | 0.33 | 0.82 |
| tl64_exact | 0.0 | 7 | 7 | 8.955 | 4.19 | 3.92 | 8.11 |
| tl64_network | 0.0 | 10 | 10 | 0.708 | 0.00 | 0.44 | 0.44 |
| tl32_vcycle | 0.0 | 14 | 15 | 0.793 | 0.34 | 0.69 | 1.04 |
| tl32_exact | 0.0 | 14 | 14 | 2.264 | 0.44 | 1.98 | 2.42 |
| tl32_network | 0.0 | 14 | 15 | 0.608 | 0.00 | 0.53 | 0.53 |

## poisson 512² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 258 | 261 | 1.309 | 0 | 21.10 | 21.10 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 7 | 7 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 4 | 4 |  |  |  | not comparable |
| gmg_vcycle | 0.0 | 6 | 6 | 4.692 | 2.24 | 1.76 | 4.00 |
| gmg_vcycle_fp32coarse | 0.0 | 7 | 7 | 2.109 | 2.23 | 0.92 | 3.16 |
| tl64_smoothing_only | 0.0 | 398 | 405 | 1.011 | 0.58 | 25.15 | 25.74 |
| tl64_chebyshev8 | 0.0 | 23 | 24 | 1.225 | 0.57 | 1.76 | 2.33 |
| tl64_vcycle | 0.0 | 14 | 15 | 1.666 | 0.83 | 1.46 | 2.29 |
| tl64_vcycle_fp32 | 0.0 | 14 | 15 | 1.570 | 0.84 | 1.37 | 2.21 |
| tl64_exact | 0.0 | 14 | 14 | 9.871 | 4.54 | 8.64 | 13.17 |
| tl64_network | 0.0 | 17 | 17 | 1.646 | 0.00 | 1.75 | 1.75 |
| tl32_vcycle | 0.0 | 28 | 29 | 1.601 | 0.66 | 2.80 | 3.46 |
| tl32_exact | 0.0 | 28 | 29 | 3.096 | 0.75 | 5.42 | 6.17 |
| tl32_network | 0.0 | 28 | 30 | 1.517 | 0.00 | 2.65 | 2.65 |

## poisson 1024² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 521 | 529 | 5.417 | 0 | 176.39 | 176.39 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 9 | 9 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 4 | 4 |  |  |  | not comparable |
| gmg_vcycle | 0.0 | 6 | 6 | 15.853 | 9.10 | 5.94 | 15.04 |
| gmg_vcycle_fp32coarse | 0.0 | 7 | 7 | 7.385 | 9.19 | 3.23 | 12.42 |
| tl64_smoothing_only | 0.0 | 805 | 816 | 4.424 | 1.63 | 222.57 | 224.21 |
| tl64_chebyshev8 | 0.0 | 35 | 37 | 5.125 | 1.62 | 11.21 | 12.84 |
| tl64_vcycle | 0.0 | 28 | 29 | 5.235 | 1.89 | 9.16 | 11.06 |
| tl64_vcycle_fp32 | 0.0 | 28 | 29 | 5.236 | 1.90 | 9.16 | 11.06 |
| tl64_exact | 0.0 | 27 | 28 | 13.226 | 5.54 | 22.32 | 27.86 |
| tl64_network | 0.0 | 32 | 33 | 5.643 | 0.00 | 11.29 | 11.29 |
| tl32_vcycle | 0.0 | 56 | 58 | 5.182 | 1.57 | 18.14 | 19.70 |
| tl32_exact | 0.0 | 55 | 57 | 6.461 | 1.64 | 22.21 | 23.85 |
| tl32_network | 0.0 | 57 | 60 | 5.405 | 0.00 | 19.25 | 19.25 |
