# E16 multi-level vs GMG

## diffusion 128² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 59 | 64 | 0.112 | 0 | 0.41 | 0.41 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 6 | 6 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 4 | 4 |  |  |  | not comparable |
| gmg_vcycle_fp32coarse | 0.0 | 6 | 6 | 0.583 | 0.35 | 0.22 | 0.57 |
| redisc_dense16 | 0.0 | 6 | 6 | 0.665 | 0.00 | 0.25 | 0.25 |
| redisc_exact64 | 0.0 | 6 | 6 | 9.811 | 0.00 | 3.68 | 3.68 |
| redisc_cheb8_64 | 0.0 | 28 | 30 | 0.136 | 0.00 | 0.24 | 0.24 |
| redisc_exact32 | 0.0 | 6 | 6 | 2.208 | 0.00 | 0.83 | 0.83 |
| ml64_e14 | 0.0 | 6 | 6 | 0.508 | 0.00 | 0.19 | 0.19 |
| ml32_e14 | 0.0 | 6 | 7 | 0.460 | 0.00 | 0.17 | 0.17 |
| ml32_trained | 0.0 | 6 | 6 | 0.459 | 0.00 | 0.17 | 0.17 |
| rec32_max64_trained | 0.0 | 6 | 6 | 0.817 | 0.00 | 0.31 | 0.31 |
| rec32_max128_trained | 0.0 | 6 | 6 | 1.561 | 0.00 | 0.59 | 0.59 |

## diffusion 256² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 119 | 129 | 0.340 | 0 | 2.53 | 2.53 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 7 | 7 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 4 | 4 |  |  |  | not comparable |
| gmg_vcycle_fp32coarse | 0.0 | 6 | 6 | 0.839 | 0.71 | 0.31 | 1.03 |
| redisc_dense16 | 0.0 | 6 | 6 | 0.906 | 0.00 | 0.34 | 0.34 |
| redisc_exact64 | 0.0 | 6 | 6 | 10.077 | 0.00 | 3.78 | 3.78 |
| redisc_cheb8_64 | 0.0 | 27 | 29 | 0.379 | 0.00 | 0.64 | 0.64 |
| redisc_exact32 | 0.0 | 6 | 6 | 2.444 | 0.00 | 0.92 | 0.92 |
| ml64_e14 | 0.0 | 7 | 7 | 0.815 | 0.00 | 0.36 | 0.36 |
| ml32_e14 | 0.0 | 7 | 7 | 0.770 | 0.00 | 0.34 | 0.34 |
| ml32_trained | 0.0 | 6 | 6 | 0.768 | 0.00 | 0.29 | 0.29 |
| rec32_max64_trained | 0.0 | 6 | 6 | 1.134 | 0.00 | 0.43 | 0.43 |
| rec32_max128_trained | 0.0 | 6 | 7 | 1.853 | 0.00 | 0.69 | 0.69 |

## diffusion 512² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 241 | 262 | 1.337 | 0 | 20.13 | 20.13 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 6 | 7 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 4 | 4 |  |  |  | not comparable |
| gmg_vcycle_fp32coarse | 0.0 | 6 | 6 | 2.122 | 2.28 | 0.80 | 3.07 |
| redisc_dense16 | 0.0 | 6 | 7 | 1.995 | 0.00 | 0.75 | 0.75 |
| redisc_exact64 | 0.0 | 6 | 6 | 11.217 | 0.00 | 4.21 | 4.21 |
| redisc_cheb8_64 | 0.0 | 29 | 30 | 1.547 | 0.00 | 2.80 | 2.80 |
| redisc_exact32 | 0.0 | 6 | 6 | 3.552 | 0.00 | 1.33 | 1.33 |
| ml64_e14 | 0.0 | 7 | 7 | 2.255 | 0.00 | 0.99 | 0.99 |
| ml32_e14 | 0.0 | 7 | 7 | 2.210 | 0.00 | 0.97 | 0.97 |
| ml32_trained | 0.0 | 6 | 7 | 2.219 | 0.00 | 0.83 | 0.83 |
| rec32_max64_trained | 0.0 | 6 | 7 | 2.557 | 0.00 | 0.96 | 0.96 |
| rec32_max128_trained | 0.0 | 7 | 7 | 3.298 | 0.00 | 1.44 | 1.44 |

## diffusion 1024² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 502 | 532 | 5.506 | 0 | 172.76 | 172.76 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 8 | 8 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 4 | 4 |  |  |  | not comparable |
| gmg_vcycle_fp32coarse | 0.0 | 6 | 7 | 7.415 | 9.08 | 2.78 | 11.86 |
| redisc_dense16 | 0.0 | 7 | 7 | 7.593 | 0.00 | 3.32 | 3.32 |
| redisc_exact64 | 0.0 | 6 | 6 | 16.055 | 0.00 | 6.02 | 6.02 |
| redisc_cheb8_64 | 0.0 | 30 | 31 | 7.394 | 0.00 | 13.86 | 13.86 |
| redisc_exact32 | 0.0 | 6 | 7 | 8.840 | 0.00 | 3.32 | 3.32 |
| ml64_e14 | 0.0 | 7 | 7 | 9.488 | 0.00 | 4.15 | 4.15 |
| ml32_e14 | 0.0 | 7 | 7 | 9.425 | 0.00 | 4.12 | 4.12 |
| ml32_trained | 0.0 | 7 | 7 | 9.587 | 0.00 | 4.19 | 4.19 |
| rec32_max64_trained | 0.0 | 6 | 7 | 9.833 | 0.00 | 3.69 | 3.69 |
| rec32_max128_trained | 0.0 | 7 | 7 | 10.595 | 0.00 | 4.64 | 4.64 |

## poisson 128² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 64 | 65 | 0.129 | 0 | 0.52 | 0.52 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 6 | 7 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 4 | 4 |  |  |  | not comparable |
| gmg_vcycle_fp32coarse | 0.0 | 6 | 6 | 0.577 | 0.35 | 0.22 | 0.57 |
| redisc_dense16 | 0.0 | 7 | 7 | 0.659 | 0.00 | 0.29 | 0.29 |
| redisc_exact64 | 0.0 | 6 | 6 | 9.735 | 0.00 | 3.65 | 3.65 |
| redisc_cheb8_64 | 0.0 | 29 | 29 | 0.137 | 0.00 | 0.25 | 0.25 |
| redisc_exact32 | 0.0 | 6 | 7 | 2.172 | 0.00 | 0.81 | 0.81 |
| ml64_e14 | 0.0 | 7 | 7 | 0.507 | 0.00 | 0.22 | 0.22 |
| ml32_e14 | 0.0 | 7 | 7 | 0.457 | 0.00 | 0.20 | 0.20 |
| ml32_trained | 0.0 | 6 | 6 | 0.458 | 0.00 | 0.17 | 0.17 |
| rec32_max64_trained | 0.0 | 6 | 6 | 0.814 | 0.00 | 0.31 | 0.31 |
| rec32_max128_trained | 0.0 | 7 | 7 | 1.530 | 0.00 | 0.67 | 0.67 |

## poisson 256² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 128 | 130 | 0.341 | 0 | 2.73 | 2.73 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 8 | 8 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 4 | 4 |  |  |  | not comparable |
| gmg_vcycle_fp32coarse | 0.0 | 6 | 7 | 0.840 | 0.72 | 0.32 | 1.04 |
| redisc_dense16 | 0.0 | 7 | 7 | 0.910 | 0.00 | 0.40 | 0.40 |
| redisc_exact64 | 0.0 | 6 | 6 | 10.056 | 0.00 | 3.77 | 3.77 |
| redisc_cheb8_64 | 0.0 | 28 | 29 | 0.380 | 0.00 | 0.66 | 0.66 |
| redisc_exact32 | 0.0 | 7 | 7 | 2.433 | 0.00 | 1.06 | 1.06 |
| ml64_e14 | 0.0 | 7 | 7 | 0.814 | 0.00 | 0.36 | 0.36 |
| ml32_e14 | 0.0 | 7 | 7 | 0.765 | 0.00 | 0.33 | 0.33 |
| ml32_trained | 0.0 | 6 | 7 | 0.765 | 0.00 | 0.29 | 0.29 |
| rec32_max64_trained | 0.0 | 6 | 7 | 1.120 | 0.00 | 0.42 | 0.42 |
| rec32_max128_trained | 0.0 | 7 | 7 | 1.850 | 0.00 | 0.81 | 0.81 |

## poisson 512² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 258 | 261 | 1.309 | 0 | 21.10 | 21.10 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 7 | 7 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 4 | 4 |  |  |  | not comparable |
| gmg_vcycle_fp32coarse | 0.0 | 7 | 7 | 2.113 | 2.28 | 0.92 | 3.20 |
| redisc_dense16 | 0.0 | 7 | 7 | 1.988 | 0.00 | 0.87 | 0.87 |
| redisc_exact64 | 0.0 | 7 | 7 | 11.186 | 0.00 | 4.89 | 4.89 |
| redisc_cheb8_64 | 0.0 | 28 | 29 | 1.505 | 0.00 | 2.63 | 2.63 |
| redisc_exact32 | 0.0 | 7 | 7 | 3.543 | 0.00 | 1.55 | 1.55 |
| ml64_e14 | 0.0 | 7 | 7 | 2.232 | 0.00 | 0.98 | 0.98 |
| ml32_e14 | 0.0 | 7 | 7 | 2.184 | 0.00 | 0.96 | 0.96 |
| ml32_trained | 0.0 | 7 | 7 | 2.188 | 0.00 | 0.96 | 0.96 |
| rec32_max64_trained | 0.0 | 7 | 7 | 2.545 | 0.00 | 1.11 | 1.11 |
| rec32_max128_trained | 0.0 | 7 | 7 | 3.289 | 0.00 | 1.44 | 1.44 |

## poisson 1024² at 1e-6 (100 systems; batch 16 timings)

| method | fail % | median its | p90 its | ms/it b16 | setup ms/system | solve ms/system | total ms/system |
|---|---|---|---|---|---|---|---|
| **chebyshev8** (GPU) | 0.0 | 521 | 529 | 5.417 | 0 | 176.39 | 176.39 |
| **AMG smoothed_aggregation** (CPU, 1 core) | 0.0 | 9 | 9 |  |  |  | not comparable |
| **AMG ruge_stuben** (CPU, 1 core) | 0.0 | 4 | 4 |  |  |  | not comparable |
| gmg_vcycle_fp32coarse | 0.0 | 7 | 7 | 7.287 | 9.01 | 3.19 | 12.19 |
| redisc_dense16 | 0.0 | 7 | 7 | 7.524 | 0.00 | 3.29 | 3.29 |
| redisc_exact64 | 0.0 | 7 | 7 | 16.014 | 0.00 | 7.01 | 7.01 |
| redisc_cheb8_64 | 0.0 | 28 | 29 | 7.207 | 0.00 | 12.61 | 12.61 |
| redisc_exact32 | 0.0 | 7 | 7 | 8.606 | 0.00 | 3.77 | 3.77 |
| ml64_e14 | 0.0 | 7 | 7 | 9.249 | 0.00 | 4.05 | 4.05 |
| ml32_e14 | 0.0 | 7 | 7 | 9.223 | 0.00 | 4.03 | 4.03 |
| ml32_trained | 0.0 | 7 | 7 | 9.452 | 0.00 | 4.14 | 4.14 |
| rec32_max64_trained | 0.0 | 7 | 7 | 9.644 | 0.00 | 4.22 | 4.22 |
| rec32_max128_trained | 0.0 | 7 | 7 | 10.398 | 0.00 | 4.55 | 4.55 |
