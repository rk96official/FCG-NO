# E1 stall diagnosis

## Failures / median iterations to 1e-6 per solver variant

| run | train loss | base | full_orthogonalization | residual_swap_0.9 | angle_cap_0.9 | angle_cap_0.75 | share of steps capped at 0.9 |
|---|---|---|---|---|---|---|---|
| diffusion 128 s0 notebook | 0.096 | 0% / 24 | 0% / 24 | 0% / 24 | 0% / 24 | 0% / 24 | 0.000 |
| diffusion 128 s0 paper | 0.097 | 0% / 25 | 0% / 25 | 0% / 25 | 0% / 25 | 0% / 25 | 0.000 |
| diffusion 128 s1 notebook | 0.999 (collapsed) | 100% / — | 100% / — | 0% / 283 | 0% / 2 | 0% / 2 | 0.595 |
| diffusion 128 s1 paper | 0.999 (collapsed) | 100% / — | 100% / — | 0% / 283 | 0% / 2 | 0% / 2 | 0.595 |
| diffusion 128 s2 notebook | 0.090 | 0% / 24 | 0% / 24 | 0% / 24 | 0% / 24 | 0% / 24 | 0.000 |
| diffusion 128 s2 paper | 0.110 | 91% / — | 91% / — | 0% / 40 | 0% / 82 | 0% / 43 | 0.809 |
| diffusion 64 s0 notebook | 0.148 | 0% / 23 | 0% / 23 | 0% / 23 | 0% / 23 | 0% / 23 | 0.000 |
| diffusion 64 s0 paper | 0.102 | 0% / 18 | 0% / 18 | 0% / 18 | 0% / 18 | 0% / 18 | 0.021 |
| diffusion 64 s1 notebook | 0.153 | 0% / 20 | 0% / 20 | 0% / 20 | 0% / 20 | 0% / 20 | 0.000 |
| diffusion 64 s1 paper | 0.104 | 0% / 49 | 1% / 42 | 0% / 25 | 0% / 36 | 0% / 28 | 0.579 |
| diffusion 64 s2 notebook | 0.146 | 0% / 20 | 0% / 20 | 0% / 20 | 0% / 20 | 0% / 20 | 0.000 |
| diffusion 64 s2 paper | 0.093 | 0% / 18 | 0% / 18 | 0% / 18 | 0% / 18 | 0% / 18 | 0.047 |
| poisson 128 s0 notebook | 0.115 | 0% / 35 | 0% / 34 | 0% / 35 | 0% / 35 | 0% / 35 | 0.000 |
| poisson 128 s0 paper | 0.085 | 36% / 274 | 14% / 135 | 0% / 31 | 0% / 59 | 0% / 37 | 0.801 |
| poisson 128 s1 notebook | 0.092 | 0% / 67 | 0% / 47 | 0% / 33 | 0% / 58 | 0% / 36 | 0.532 |
| poisson 128 s1 paper | 0.083 | 35% / 24 | 36% / 24 | 0% / 24 | 0% / 24 | 0% / 24 | 0.476 |
| poisson 128 s2 notebook | 0.094 | 0% / 25 | 0% / 25 | 0% / 25 | 0% / 25 | 0% / 25 | 0.000 |
| poisson 128 s2 paper | 0.086 | 97% / — | 95% / — | 0% / 35 | 0% / 50 | 0% / 31 | 0.337 |
| poisson 64 s0 notebook | 0.105 | 0% / 18 | 0% / 18 | 0% / 18 | 0% / 18 | 0% / 18 | 0.009 |
| poisson 64 s0 paper | 0.123 | 0% / 20 | 0% / 20 | 0% / 20 | 0% / 20 | 0% / 20 | 0.013 |
| poisson 64 s1 notebook | 0.102 | 0% / 18 | 0% / 18 | 0% / 18 | 0% / 18 | 0% / 18 | 0.002 |
| poisson 64 s1 paper | 0.127 | 0% / 20 | 0% / 20 | 0% / 20 | 0% / 20 | 0% / 20 | 0.028 |
| poisson 64 s2 notebook | 0.108 | 0% / 17 | 0% / 17 | 0% / 17 | 0% / 17 | 0% / 17 | 0.010 |
| poisson 64 s2 paper | 0.125 | 0% / 19 | 0% / 19 | 0% / 19 | 0% / 19 | 0% / 19 | 0.008 |

## H1 shift: median sin² of the network output on CG residuals vs its own FCG residuals

| run | on CG (steps < 100) | on FCG (all) | on FCG (good systems) | on FCG (bad systems) | bad systems |
|---|---|---|---|---|---|
| diffusion 128 s0 notebook | 0.009 | 0.523 | 0.523 | nan | 0 |
| diffusion 128 s0 paper | 0.009 | 0.538 | 0.538 | nan | 0 |
| diffusion 128 s2 notebook | 0.008 | 0.535 | 0.535 | nan | 0 |
| diffusion 128 s2 paper | 0.011 | 1.000 | 1.000 | 1.000 | 95 |
| diffusion 64 s0 notebook | 0.018 | 0.461 | 0.461 | nan | 0 |
| diffusion 64 s0 paper | 0.009 | 0.413 | 0.413 | nan | 0 |
| diffusion 64 s1 notebook | 0.018 | 0.426 | 0.426 | nan | 0 |
| diffusion 64 s1 paper | 0.010 | 0.979 | 0.970 | 0.994 | 16 |
| diffusion 64 s2 notebook | 0.016 | 0.400 | 0.400 | nan | 0 |
| diffusion 64 s2 paper | 0.008 | 0.435 | 0.425 | 0.958 | 2 |
| poisson 128 s0 notebook | 0.013 | 0.653 | 0.653 | nan | 0 |
| poisson 128 s0 paper | 0.007 | 0.999 | 0.999 | 0.999 | 56 |
| poisson 128 s1 notebook | 0.008 | 0.981 | 0.972 | 0.998 | 8 |
| poisson 128 s1 paper | 0.007 | 1.000 | 1.000 | 1.000 | 36 |
| poisson 128 s2 notebook | 0.008 | 0.541 | 0.541 | nan | 0 |
| poisson 128 s2 paper | 0.007 | 1.000 | 1.000 | 1.000 | 98 |
| poisson 64 s0 notebook | 0.011 | 0.360 | 0.360 | nan | 0 |
| poisson 64 s0 paper | 0.015 | 0.410 | 0.410 | nan | 0 |
| poisson 64 s1 notebook | 0.010 | 0.357 | 0.357 | nan | 0 |
| poisson 64 s1 paper | 0.016 | 0.412 | 0.412 | nan | 0 |
| poisson 64 s2 notebook | 0.012 | 0.345 | 0.345 | nan | 0 |
| poisson 64 s2 paper | 0.016 | 0.393 | 0.393 | nan | 0 |

## H2/H3: AUC of statistics over the first 20 steps for predicting bad systems (0.5 = no signal)

| pool | bad/all | mean sin² of w | max sin² of w | share sin² of w > 0.9 | mean Notay ratio | max Notay ratio | share Notay ratio > 1 | mean error outside modes |
|---|---|---|---|---|---|---|---|---|
| diffusion 64 | 18/600 | 0.99 | 0.98 | 0.97 | 0.96 | 0.95 | 0.95 | 0.77 |
| diffusion 128 | 95/400 | 0.99 | 1.00 | 0.97 | 0.99 | 1.00 | 0.97 | 0.21 |
| poisson 64 | 0/600 | — | — | — | — | — | — | — |
| poisson 128 | 198/600 | 0.87 | 0.91 | 0.94 | 0.97 | 0.98 | 0.96 | 0.64 |

## H3: median share of error energy outside the 20 retained modes

| run | step 5 good | step 5 bad | step 20 good | step 20 bad | step 40 good | step 40 bad |
|---|---|---|---|---|---|---|
| diffusion 128 s0 notebook | 0.090 | — | 0.454 | — | 0.254 | — |
| diffusion 128 s0 paper | 0.071 | — | 0.443 | — | 0.397 | — |
| diffusion 128 s2 notebook | 0.134 | — | 0.502 | — | 0.403 | — |
| diffusion 128 s2 paper | 0.123 | 0.135 | 0.322 | 0.430 | 0.282 | 0.369 |
| diffusion 64 s0 notebook | 0.221 | — | 0.364 | — | 0.278 | — |
| diffusion 64 s0 paper | 0.208 | — | 0.418 | — | 0.287 | — |
| diffusion 64 s1 notebook | 0.102 | — | 0.290 | — | — | — |
| diffusion 64 s1 paper | 0.186 | 0.184 | 0.436 | 0.460 | 0.305 | 0.308 |
| diffusion 64 s2 notebook | 0.286 | — | 0.374 | — | — | — |
| diffusion 64 s2 paper | 0.251 | 0.390 | 0.483 | 0.529 | 0.410 | 0.381 |
| poisson 128 s0 notebook | 0.276 | — | 0.597 | — | 0.449 | — |
| poisson 128 s0 paper | 0.216 | 0.217 | 0.720 | 0.794 | 0.691 | 0.792 |
| poisson 128 s1 notebook | 0.153 | 0.130 | 0.637 | 0.652 | 0.512 | 0.556 |
| poisson 128 s1 paper | 0.385 | 0.418 | 0.806 | 0.797 | 0.825 | 0.778 |
| poisson 128 s2 notebook | 0.232 | — | 0.555 | — | 0.367 | — |
| poisson 128 s2 paper | 0.139 | 0.204 | 0.618 | 0.639 | 0.649 | 0.675 |
| poisson 64 s0 notebook | 0.266 | — | 0.478 | — | — | — |
| poisson 64 s0 paper | 0.249 | — | 0.343 | — | — | — |
| poisson 64 s1 notebook | 0.295 | — | 0.367 | — | — | — |
| poisson 64 s1 paper | 0.391 | — | 0.367 | — | — | — |
| poisson 64 s2 notebook | 0.212 | — | 0.371 | — | — | — |
| poisson 64 s2 paper | 0.266 | — | 0.333 | — | — | — |
