# E17 NeuralIF and IC(0)

## diffusion 128² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 59 | 64 | 0.112 | 0 | 0.41 | 0.41 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 6 | 6 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 4 | 4 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 7 | 8 |  | 6.99 | 2.60 | 9.58 |
| **amgx_classical_fp32** (GPU, per system) | 0.0 | 7 | 8 |  | 6.88 | 2.55 | 9.43 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 26 | 27 |  | 6.52 | 12.16 | 18.69 |
| ic0 | 0.0 | 88 | 96 | 6.666 | 0.20 | 36.66 | 36.86 |
| neuralif_s0 | 0.0 | 100 | 109 | 6.626 | 0.33 | 41.41 | 41.74 |
| neuralif_s1 | 0.0 | 100 | 109 | 6.636 | 0.33 | 41.47 | 41.80 |
| neuralif_s2 | 0.0 | 100 | 109 | 6.629 | 0.33 | 41.43 | 41.76 |

## diffusion 256² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 119 | 129 | 0.340 | 0 | 2.53 | 2.53 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 7 | 7 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 4 | 4 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 8 | 8 |  | 9.35 | 3.49 | 12.84 |
| **amgx_classical_fp32** (GPU, per system) | 0.0 | 8 | 8 |  | 9.25 | 3.52 | 12.77 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 38 | 40 |  | 8.79 | 22.30 | 31.08 |
| ic0 | 0.0 | 178 | 192 | 13.353 | 0.43 | 148.55 | 148.98 |
| neuralif_s0 | 0.0 | 202 | 218 | 13.305 | 1.29 | 167.98 | 169.27 |
| neuralif_s1 | 0.0 | 202 | 218 | 13.318 | 1.30 | 168.14 | 169.44 |
| neuralif_s2 | 0.0 | 202 | 218 | 13.313 | 1.29 | 168.08 | 169.37 |

## diffusion 512² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 241 | 262 | 1.337 | 0 | 20.13 | 20.13 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 6 | 7 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 4 | 4 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 8 | 8 |  | 13.27 | 4.48 | 17.75 |
| **amgx_classical_fp32** (GPU, per system) | 0.0 | 8 | 8 |  | 13.07 | 4.30 | 17.37 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 52 | 55 |  | 11.17 | 35.28 | 46.45 |
| ic0 | 0.0 | 363 | 388 | 26.485 | 0.90 | 600.88 | 601.78 |
| neuralif_s0 | 0.0 | 412 | 441 | 26.672 | 5.04 | 686.79 | 691.83 |
| neuralif_s1 | 0.0 | 412 | 441 | 26.598 | 5.03 | 684.89 | 689.92 |
| neuralif_s2 | 0.0 | 412 | 441 | 26.579 | 5.04 | 684.40 | 689.43 |

## diffusion 1024² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 502 | 532 | 5.506 | 0 | 172.76 | 172.76 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 8 | 8 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 4 | 4 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 9 | 9 |  | 19.62 | 7.23 | 26.85 |
| **amgx_classical_fp32** (GPU, per system) | 0.0 | 9 | 9 |  | 19.02 | 6.12 | 25.15 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 81 | 85 |  | 12.03 | 62.82 | 74.85 |
| ic0 | 0.0 | 759 | 786 | 45.894 | 1.93 | 2177.07 | 2179.00 |
| neuralif_s0 | 0.0 | 858 | 892 | 45.656 | 20.15 | 2448.29 | 2468.44 |
| neuralif_s1 | 0.0 | 858 | 892 | 45.679 | 20.16 | 2449.53 | 2469.70 |
| neuralif_s2 | 0.0 | 858 | 892 | 45.687 | 20.16 | 2449.94 | 2470.10 |

## poisson 128² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 64 | 65 | 0.129 | 0 | 0.52 | 0.52 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 6 | 7 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 4 | 4 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 8 | 9 |  | 6.75 | 2.87 | 9.63 |
| **amgx_classical_fp32** (GPU, per system) | 0.0 | 8 | 9 |  | 6.66 | 2.82 | 9.48 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 29 | 30 |  | 6.23 | 13.39 | 19.61 |
| ic0 | 0.0 | 95 | 98 | 6.677 | 0.20 | 39.65 | 39.85 |
| neuralif_s0 | 0.0 | 108 | 109 | 6.580 | 0.33 | 44.42 | 44.75 |
| neuralif_s1 | 0.0 | 108 | 109 | 6.646 | 0.33 | 44.86 | 45.20 |
| neuralif_s2 | 0.0 | 108 | 109 | 6.618 | 0.33 | 44.67 | 45.00 |

## poisson 256² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 128 | 130 | 0.341 | 0 | 2.73 | 2.73 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 8 | 8 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 4 | 4 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 9 | 9 |  | 9.36 | 3.88 | 13.24 |
| **amgx_classical_fp32** (GPU, per system) | 0.0 | 9 | 9 |  | 9.23 | 3.91 | 13.14 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 45 | 47 |  | 8.40 | 26.01 | 34.40 |
| ic0 | 0.0 | 189 | 191 | 13.295 | 0.43 | 157.05 | 157.48 |
| neuralif_s0 | 0.0 | 214 | 217 | 13.272 | 1.30 | 177.52 | 178.82 |
| neuralif_s1 | 0.0 | 214 | 217 | 13.300 | 1.30 | 177.89 | 179.18 |
| neuralif_s2 | 0.0 | 214 | 217 | 13.320 | 1.29 | 178.16 | 179.45 |

## poisson 512² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 258 | 261 | 1.309 | 0 | 21.10 | 21.10 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 7 | 7 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 4 | 4 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 9 | 10 |  | 13.29 | 4.92 | 18.21 |
| **amgx_classical_fp32** (GPU, per system) | 0.0 | 9 | 10 |  | 13.17 | 4.81 | 17.98 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 60 | 62 |  | 10.55 | 40.59 | 51.14 |
| ic0 | 0.0 | 378 | 385 | 26.277 | 0.90 | 620.79 | 621.69 |
| neuralif_s0 | 0.0 | 430 | 437 | 26.298 | 5.04 | 706.77 | 711.82 |
| neuralif_s1 | 0.0 | 430 | 437 | 26.483 | 5.05 | 711.73 | 716.78 |
| neuralif_s2 | 0.0 | 430 | 437 | 26.297 | 5.04 | 706.73 | 711.77 |

## poisson 1024² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 521 | 529 | 5.417 | 0 | 176.39 | 176.39 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 9 | 9 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 4 | 4 |  |  |  | not comparable |
| **amgx_classical** (GPU, per system) | 0.0 | 10 | 10 |  | 23.13 | 8.43 | 31.56 |
| **amgx_classical_fp32** (GPU, per system) | 0.0 | 10 | 10 |  | 22.60 | 7.26 | 29.86 |
| **amgx_aggregation** (GPU, per system) | 0.0 | 92 | 96 |  | 15.48 | 82.49 | 97.97 |
| ic0 | 0.0 | 765 | 776 | 54.717 | 1.92 | 2616.17 | 2618.10 |
| neuralif_s0 | 0.0 | 869 | 882 | 54.250 | 20.17 | 2946.43 | 2966.60 |
| neuralif_s1 | 0.0 | 869 | 882 | 54.162 | 20.17 | 2941.65 | 2961.82 |
| neuralif_s2 | 0.0 | 869 | 882 | 54.072 | 20.17 | 2936.81 | 2956.98 |
