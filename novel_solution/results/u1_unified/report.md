# Unified comparison with the paper's methods

Same 20 test systems and initial guesses as the reproduction (seed 12 / guess seed 1). Cells: median iterations to the relative residual tolerance pooled over seeds and systems (SNO 3 seeds, ours 5 seeds, others 1), with the share of systems that never reach it. Paper methods come from the reproduction runs (classical ones on the CPU); GPU time per system (batch 1, one system at a time) is to 1e-6.


## poisson, tolerance 0.001

| method | source | 32² | 64² | 128² |
|---|---|---|---|---|
| CG | paper | 20 | 20 | 20 |
| Jacobi(4) | paper | 19 | 34 | 62 |
| SGS(1) | paper | 7 | 7 | 7 |
| SGS(4) | paper | 3 | 3 | 3 |
| ILU(1) | paper | 15 | 16 | 16 |
| ILU(8) | paper | 7 | 9 | 9 |
| FCG-NO (SNO), paper settings | paper | 6 | 9 | 10 (15% fail) |
| FCG-NO (SNO), authors' notebook settings | paper | 6 | 8 | 11 |
| FCG-FNO | paper | 3 | 6 | 10 |
| FCG-U-Net | paper | 4 | 3 | 4 |
| FCG-DilResNet | paper | 2 | 2 | 3 |
| PCG-GNN (Li et al. 2023) | paper | 11 | 20 | 37 |
| CG (GPU) | baseline | 48 | 98 | 196 |
| Jacobi(4)-PCG (GPU) | baseline | 21 | 41 | 83 |
| Chebyshev(8)-PCG (GPU) | baseline | 10 | 19 | 39 |
| HC-NO two-level (trained per grid) | ours | 3 | 4 | 4 |
| HC-NO multi-level (trained per grid) | ours | 3 | 5 | 7 |
| NO in-role, no history, two-level (trained per grid) | ours | 3 | 4 | 4 |
| NO in-role, no history, multi-level (trained per grid) | ours | 3 | 4 | 6 |

## poisson, tolerance 1e-06

| method | source | 32² | 64² | 128² | ms per system at 128² (batch 1) |
|---|---|---|---|---|---|
| CG | paper | 75 | 131 | 216 | — |
| Jacobi(4) | paper | 30 | 58 | 113 | — |
| SGS(1) | paper | 27 | 47 | 76 | — |
| SGS(4) | paper | 13 | 23 | 36 | — |
| ILU(1) | paper | 53 | 94 | 157 | — |
| ILU(8) | paper | 17 | 45 | 75 | — |
| FCG-NO (SNO), paper settings | paper | 11 | 20 | 187 (50% fail) | — |
| FCG-NO (SNO), authors' notebook settings | paper | 11 | 18 | 34 | — |
| FCG-FNO | paper | 7 | 12 | 23 | — |
| FCG-U-Net | paper | 7 | 7 | 8 | — |
| FCG-DilResNet | paper | 4 | 4 | 7 | — |
| PCG-GNN (Li et al. 2023) | paper | 21 | 34 | 64 | — |
| CG (GPU) | baseline | 81 | 160 | 317 | 12.994 |
| Jacobi(4)-PCG (GPU) | baseline | 35 | 69 | 136 | 6.177 |
| Chebyshev(8)-PCG (GPU) | baseline | 17 | 33 | 64 | 3.448 |
| HC-NO two-level (trained per grid) | ours | 6 | 6 | 7 | 2.233 |
| HC-NO multi-level (trained per grid) | ours | 6 | 8 | 13 | 4.159 |
| NO in-role, no history, two-level (trained per grid) | ours | 6 | 6 | 7 | 2.038 |
| NO in-role, no history, multi-level (trained per grid) | ours | 6 | 8 | 12 | 3.644 |

## poisson, tolerance 1e-12

| method | source | 32² | 64² | 128² |
|---|---|---|---|---|
| CG | paper | 123 | 234 | 445 |
| Jacobi(4) | paper | 45 | 89 | 173 |
| SGS(1) | paper | 50 | 91 | 170 |
| SGS(4) | paper | 25 | 45 | 84 |
| ILU(1) | paper | 128 (50% fail) | fail (80% fail) | 448 |
| ILU(8) | paper | 34 | 104 | 208 |
| FCG-NO (SNO), paper settings | paper | 22 | 41 | fail (98% fail) |
| FCG-NO (SNO), authors' notebook settings | paper | 21 | 36 | 76 (3% fail) |
| FCG-FNO | paper | 13 | 24 | 48 |
| FCG-U-Net | paper | 12 | 14 | 19 |
| FCG-DilResNet | paper | 7 | 8 | 15 |
| PCG-GNN (Li et al. 2023) | paper | 37 | 58 | 109 |
| CG (GPU) | baseline | 124 | 243 | 483 |
| Jacobi(4)-PCG (GPU) | baseline | 54 | 105 | 208 |
| Chebyshev(8)-PCG (GPU) | baseline | 25 | 49 | 98 |
| HC-NO two-level (trained per grid) | ours | 12 | 12 | 12 |
| HC-NO multi-level (trained per grid) | ours | 12 | 15 | 23 |
| NO in-role, no history, two-level (trained per grid) | ours | 12 | 12 | 13 |
| NO in-role, no history, multi-level (trained per grid) | ours | 12 | 14 | 23 |

## diffusion, tolerance 0.001

| method | source | 32² | 64² | 128² |
|---|---|---|---|---|
| CG | paper | 20 | 20 | 20 |
| Jacobi(4) | paper | 19 | 34 | 62 |
| SGS(1) | paper | 7 | 7 | 7 |
| SGS(4) | paper | 3 | 3 | 3 |
| ILU(1) | paper | 15 | 16 | 16 |
| ILU(8) | paper | 7 | 9 | 9 |
| FCG-NO (SNO), paper settings | paper | 6 | 8 | 12 (37% fail) |
| FCG-NO (SNO), authors' notebook settings | paper | 6 | 10 | 10 (33% fail) |
| FCG-FNO | paper | 4 | 5 | 8 |
| FCG-U-Net | paper | 4 | 3 | 5 |
| FCG-DilResNet | paper | 3 | 3 | 3 |
| CG (GPU) | baseline | 42 | 80 | 161 |
| Jacobi(4)-PCG (GPU) | baseline | 18 | 33 | 66 |
| Chebyshev(8)-PCG (GPU) | baseline | 8 | 16 | 31 |
| HC-NO two-level (trained per grid) | ours | 3 | 3 | 3 |
| HC-NO multi-level (trained per grid) | ours | 3 | 4 | 5 |
| NO in-role, no history, two-level (trained per grid) | ours | 3 | 3 | 3 |
| NO in-role, no history, multi-level (trained per grid) | ours | 3 | 4 | 5 |

## diffusion, tolerance 1e-06

| method | source | 32² | 64² | 128² | ms per system at 128² (batch 1) |
|---|---|---|---|---|---|
| CG | paper | 77 | 131 | 219 | — |
| Jacobi(4) | paper | 32 | 60 | 113 | — |
| SGS(1) | paper | 27 | 46 | 75 | — |
| SGS(4) | paper | 13 | 22 | 36 | — |
| ILU(1) | paper | 53 | 93 | 154 | — |
| ILU(8) | paper | 17 | 44 | 75 | — |
| FCG-NO (SNO), paper settings | paper | 11 | 19 | fail (65% fail) | — |
| FCG-NO (SNO), authors' notebook settings | paper | 13 | 20 | 25 (33% fail) | — |
| FCG-FNO | paper | 9 | 12 | 17 | — |
| FCG-U-Net | paper | 7 | 7 | 11 | — |
| FCG-DilResNet | paper | 5 | 5 | 7 | — |
| CG (GPU) | baseline | 78 | 150 | 300 | 12.094 |
| Jacobi(4)-PCG (GPU) | baseline | 33 | 64 | 125 | 5.582 |
| Chebyshev(8)-PCG (GPU) | baseline | 16 | 30 | 59 | 3.125 |
| HC-NO two-level (trained per grid) | ours | 6 | 6 | 6 | 1.798 |
| HC-NO multi-level (trained per grid) | ours | 6 | 7 | 11 | 3.473 |
| NO in-role, no history, two-level (trained per grid) | ours | 6 | 6 | 6 | 1.755 |
| NO in-role, no history, multi-level (trained per grid) | ours | 6 | 7 | 11 | 3.365 |

## diffusion, tolerance 1e-12

| method | source | 32² | 64² | 128² |
|---|---|---|---|---|
| CG | paper | fail (100% fail) | fail (100% fail) | 512 (45% fail) |
| Jacobi(4) | paper | 53 | 102 | 200 |
| SGS(1) | paper | 51 | 94 | 176 |
| SGS(4) | paper | 25 | 46 | 87 |
| ILU(1) | paper | 128 (40% fail) | fail (70% fail) | 444 |
| ILU(8) | paper | 35 | 105 | 204 |
| FCG-NO (SNO), paper settings | paper | 22 | 55 (28% fail) | fail (67% fail) |
| FCG-NO (SNO), authors' notebook settings | paper | 31 | 42 | 53 (33% fail) |
| FCG-FNO | paper | 19 | 28 | 201 (50% fail) |
| FCG-U-Net | paper | 13 | 14 | 26 |
| FCG-DilResNet | paper | 9 | 10 | 13 |
| CG (GPU) | baseline | 143 | 279 | 551 |
| Jacobi(4)-PCG (GPU) | baseline | 62 | 119 | 232 |
| Chebyshev(8)-PCG (GPU) | baseline | 28 | 56 | 110 |
| HC-NO two-level (trained per grid) | ours | 12 | 12 | 12 |
| HC-NO multi-level (trained per grid) | ours | 12 | 14 | 21 |
| NO in-role, no history, two-level (trained per grid) | ours | 12 | 12 | 13 |
| NO in-role, no history, multi-level (trained per grid) | ours | 12 | 14 | 21 |

## poisson: trained at 128, tested at 128-1024 (median iterations to 1e-6 / ms per system at batch 1)

| method | 128² | 256² | 512² | 1024² |
|---|---|---|---|---|
| FCG-NO (SNO), paper settings (FP64) | 75 (15% fail) | 58 (35% fail) | 211 (30% fail) | 2985 / 47849.01 ms |
| FCG-NO (SNO), paper settings (TF32) | 25 (10% fail) | 57 (10% fail) | 210 / 324.02 ms | fail (100% fail) |
| FCG-NO (SNO), notebook settings (FP64) | 35 / 28.74 ms | 83 / 103.30 ms | 360 / 1065.56 ms | fail (100% fail) |
| FCG-NO (SNO), notebook settings (TF32) | 35 / 11.25 ms | 83 / 30.34 ms | 424 / 322.15 ms | fail (100% fail) |
| FCG-FNO (FP64) | 23 / 11.00 ms | 95 / 96.42 ms | fail (100% fail) | fail (100% fail) |
| FCG-FNO (TF32) | fail (100% fail) | fail (100% fail) | fail (100% fail) | fail (100% fail) |
| FCG-U-Net (FP64) | 8 / 12.09 ms | 15 / 52.46 ms | 43 / 541.41 ms | fail (100% fail) |
| FCG-U-Net (TF32) | 8 / 5.66 ms | 14 / 6.54 ms | 43 / 42.70 ms | fail (100% fail) |
| FCG-DilResNet (FP64) | 7 / 11.12 ms | 13 / 74.61 ms | 24 / 535.47 ms | 1053 / 91965.68 ms |
| FCG-DilResNet (TF32) | 7 / 2.82 ms | 13 / 10.29 ms | 24 / 45.76 ms | 1871 / 13585.13 ms |
| HC-NO multi-level (ours, TF32, 5 seeds) | 7 / 2.11 ms | 7 / 3.01 ms | 7 / 3.25 ms | 7 / 6.15 ms |
| HC-NO recursive (ours, TF32, 5 seeds) | 6 / 4.09 ms | 6 / 3.49 ms | 7 / 4.91 ms | 7 / 7.79 ms |
| NO in-role, no history, multi-level (ours, TF32, 5 seeds) | 7 / 2.68 ms | 7 / 2.13 ms | 7 / 3.36 ms | 7 / 4.96 ms |

## diffusion: trained at 128, tested at 128-1024 (median iterations to 1e-6 / ms per system at batch 1)

| method | 128² | 256² | 512² | 1024² |
|---|---|---|---|---|
| FCG-NO (SNO), paper settings (FP64) | 25 / 46.12 ms | 50 / 137.28 ms | 133 / 810.08 ms | 1096 / 17382.86 ms |
| FCG-NO (SNO), paper settings (TF32) | 25 / 8.29 ms | 50 / 28.10 ms | 137 / 211.17 ms | 1304 / 7144.29 ms |
| FCG-NO (SNO), notebook settings (FP64) | 24 / 47.19 ms | 47 / 136.80 ms | 164 / 1054.45 ms | 2463 (20% fail) |
| FCG-NO (SNO), notebook settings (TF32) | 24 / 15.76 ms | 47 / 51.75 ms | 165 / 483.08 ms | 2523 / 25721.93 ms |
| FCG-FNO (FP64) | 17 / 15.29 ms | 44 / 102.03 ms | fail (100% fail) | fail (100% fail) |
| FCG-FNO (TF32) | 18 / 10.28 ms | fail (100% fail) | fail (100% fail) | fail (100% fail) |
| FCG-U-Net (FP64) | 11 / 16.47 ms | 16 / 56.11 ms | 60 / 756.97 ms | fail (100% fail) |
| FCG-U-Net (TF32) | 10 / 3.93 ms | 15 / 7.00 ms | 60 / 61.71 ms | fail (100% fail) |
| FCG-DilResNet (FP64) | 7 / 11.11 ms | 10 / 57.37 ms | 17 / 378.80 ms | 43 / 3749.93 ms |
| FCG-DilResNet (TF32) | 7 / 2.83 ms | 10 / 7.96 ms | 17 / 32.71 ms | 43 / 306.93 ms |
| HC-NO multi-level (ours, TF32, 5 seeds) | 6 / 2.24 ms | 7 / 3.01 ms | 7 / 3.44 ms | 7 / 6.17 ms |
| HC-NO recursive (ours, TF32, 5 seeds) | 6 / 3.34 ms | 6 / 3.45 ms | 6 / 4.16 ms | 6 / 6.64 ms |
| NO in-role, no history, multi-level (ours, TF32, 5 seeds) | 6 / 3.24 ms | 7 / 2.96 ms | 7 / 3.15 ms | 7 / 4.98 ms |
