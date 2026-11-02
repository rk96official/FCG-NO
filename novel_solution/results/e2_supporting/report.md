# E2: safeguard, smoothing wrapper, precision (no retraining), at 1e-6

## diffusion, 128→128 notebook (3 seeds)

| variant | fail % (s0/s1/s2) | median its | p90 its | ms/it b1 | ms/it b16 | est. ms to 1e-6 per system (b16) | notes |
|---|---|---|---|---|---|---|---|
| cg | 0 / 0 / 0 | 300 / 300 / 300 | 324 / 324 / 324 | 0.043 | 0.082 | 1.539 |  |
| base | 0 / 100 / 0 | 24 / — / 24 | 24 / — / 25 | 1.970 | 6.730 | — |  |
| safeguard_fallbackidentity | 0 / 0 / 0 | 24 / 253 / 24 | 24 / 276 / 25 | 1.985 | 6.761 | 10.141 | fallback share 0.336 |
| safeguard_fallbackjacobi | 0 / 0 / 0 | 24 / 249 / 24 | 24 / 271 / 25 | 1.993 | 6.781 | 10.171 | fallback share 0.336 |
| safeguard_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 14 / 137 / 14 | 15 / 148 / 15 | 1.991 | 6.789 | 5.941 | fallback share 0.466 |
| safeguard_every2_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 14 / 137 / 14 | 15 / 148 / 15 | 1.058 | 3.469 | 3.035 | fallback share 0.717 |
| safeguard_every4_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 17 / 146 / 24 | 18 / 152 / 34 | 0.595 | 1.876 | 2.813 | fallback share 0.857 |
| wrapper_smootherjacobi_sweeps1 | 0 / 100 / 0 | 12 / — / 12 | 12 / — / 12 | 1.983 | 6.760 | — |  |
| wrapper_smootherjacobi_sweeps2 | 0 / 100 / 0 | 9 / — / 9 | 9 / — / 9 | 1.984 | 6.768 | — |  |
| wrapper_smootherchebyshev_sweeps2 | 0 / 100 / 0 | 7 / — / 7 | 7 / — / 115 | 1.984 | 6.769 | — |  |
| precision_dtypefloat32_matmulhighest | 0 / 100 / 0 | 24 / — / 24 | 24 / — / 25 | 0.794 | 4.994 | — | δ median 1.52e-04, max 1.96e-04, Lemma 3 holds on 100.0% of steps |
| precision_dtypefloat32 | 0 / 100 / 0 | 24 / — / 25 | 25 / — / 25 | 0.656 | 3.112 | — | δ median 1.58e-02, max 5.79e-02, Lemma 3 holds on 100.0% of steps |
| precision_dtypebfloat16 | 100 / 100 / 0 | — / — / 130 | — / — / 132 | 0.708 | 2.693 | — | δ median 1.13e+00, max 1.52e+00, Lemma 3 holds on 100.0% of steps |

## diffusion, 128→128 paper (3 seeds)

| variant | fail % (s0/s1/s2) | median its | p90 its | ms/it b1 | ms/it b16 | est. ms to 1e-6 per system (b16) | notes |
|---|---|---|---|---|---|---|---|
| cg | 0 / 0 / 0 | 300 / 300 / 300 | 324 / 324 / 324 | 0.042 | 0.084 | 1.569 |  |
| base | 0 / 100 / 91 | 25 / — / — | 25 / — / — | 1.846 | 6.319 | — |  |
| safeguard_fallbackidentity | 0 / 0 / 0 | 24 / 254 / 24 | 25 / 276 / 25 | 1.867 | 6.359 | 9.539 | fallback share 0.342 |
| safeguard_fallbackjacobi | 0 / 0 / 0 | 24 / 250 / 24 | 25 / 271 / 25 | 1.877 | 6.368 | 9.552 | fallback share 0.342 |
| safeguard_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 14 / 137 / 15 | 15 / 149 / 15 | 1.873 | 6.376 | 5.978 | fallback share 0.469 |
| safeguard_every2_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 14 / 137 / 15 | 15 / 149 / 16 | 1.005 | 3.266 | 3.062 | fallback share 0.698 |
| safeguard_every4_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 17 / 148 / 22 | 18 / 153 / 23 | 0.572 | 1.771 | 2.435 | fallback share 0.819 |
| wrapper_smootherjacobi_sweeps1 | 0 / 100 / 0 | 12 / — / 12 | 12 / — / 13 | 1.864 | 6.345 | — |  |
| wrapper_smootherjacobi_sweeps2 | 0 / 100 / 0 | 9 / — / 9 | 9 / — / 10 | 1.862 | 6.370 | — |  |
| wrapper_smootherchebyshev_sweeps2 | 0 / 100 / 0 | 7 / — / 8 | 8 / — / 8 | 1.865 | 6.354 | — |  |
| precision_dtypefloat32_matmulhighest | 0 / 100 / 91 | 25 / — / — | 25 / — / — | 0.793 | 4.244 | — | δ median 1.89e-04, max 1.65e-03, Lemma 3 holds on 100.0% of steps |
| precision_dtypefloat32 | 0 / 100 / 92 | 25 / — / — | 25 / — / — | 0.331 | 1.650 | — | δ median 3.69e-02, max 3.61e-01, Lemma 3 holds on 100.0% of steps |
| precision_dtypebfloat16 | 0 / 100 / 100 | 270 / — / — | 278 / — / — | 0.324 | 1.127 | — | δ median 1.19e+00, max 1.67e+00, Lemma 3 holds on 100.0% of steps |

## diffusion, 128→256 transfer (3 seeds)

| variant | fail % (s0/s1/s2) | median its | p90 its | ms/it b1 | ms/it b16 | est. ms to 1e-6 per system (b16) | notes |
|---|---|---|---|---|---|---|---|
| cg | 0 / 0 / 0 | 606 / 606 / 606 | 660 / 661 / 660 | 0.034 | 0.172 | 6.521 |  |
| base | 4 / 100 / 0 | 49 / — / 48 | 52 / — / 51 | 2.738 | 19.345 | — |  |
| safeguard_fallbackidentity | 0 / 0 / 0 | 54 / 574 / 51 | 56 / 607 / 53 | 2.755 | 19.473 | 65.720 | fallback share 0.383 |
| safeguard_fallbackjacobi | 0 / 0 / 0 | 54 / 557 / 51 | 56 / 586 / 53 | 2.755 | 19.436 | 65.595 | fallback share 0.376 |
| safeguard_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 31 / 305 / 32 | 32 / 321 / 34 | 2.757 | 19.457 | 38.913 | fallback share 0.416 |
| safeguard_every2_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 31 / 307 / 32 | 32 / 324 / 34 | 1.430 | 9.917 | 19.835 | fallback share 0.696 |
| safeguard_every4_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 32 / 335 / 34 | 34 / 350 / 35 | 0.784 | 5.299 | 11.260 | fallback share 0.833 |
| wrapper_smootherjacobi_sweeps1 | 0 / 100 / 0 | 22 / — / 22 | 23 / — / 23 | 2.742 | 19.419 | — |  |
| wrapper_smootherjacobi_sweeps2 | 0 / 36 / 0 | 17 / 950 / 16 | 18 / — / 17 | 2.739 | 19.437 | 20.651 |  |
| wrapper_smootherchebyshev_sweeps2 | 0 / 0 / 0 | 15 / 394 / 15 | 15 / 454 / 16 | 2.745 | 19.439 | 18.224 |  |
| precision_dtypefloat32_matmulhighest | 4 / 100 / 0 | 49 / — / 48 | 52 / — / 51 | 1.346 | 11.608 | — | δ median 1.08e-04, max 5.49e-04, Lemma 3 holds on 100.0% of steps |
| precision_dtypefloat32 | 3 / 100 / 0 | 50 / — / 48 | 52 / — / 51 | 0.568 | 5.475 | — | δ median 4.78e-02, max 1.44e-01, Lemma 3 holds on 100.0% of steps |
| precision_dtypebfloat16 | 100 / 100 / 100 | — / — / — | — / — / — | 0.435 | 3.514 | — | δ median 1.91e+00, max 2.07e+00, Lemma 3 holds on 100.0% of steps |

## diffusion, 32→128 transfer (3 seeds)

| variant | fail % (s0/s1/s2) | median its | p90 its | ms/it b1 | ms/it b16 | est. ms to 1e-6 per system (b16) | notes |
|---|---|---|---|---|---|---|---|
| cg | 0 / 0 / 0 | 300 / 300 / 300 | 324 / 324 / 324 | 0.042 | 0.084 | 1.575 |  |
| base | 0 / 0 / 8 | 60 / 87 / 87 | 63 / 98 / 346 | 0.985 | 3.856 | 20.964 |  |
| safeguard_fallbackidentity | 0 / 0 / 0 | 60 / 72 / 68 | 63 / 77 / 75 | 1.009 | 3.885 | 16.509 | fallback share 0.110 |
| safeguard_fallbackjacobi | 0 / 0 / 0 | 59 / 71 / 68 | 63 / 77 / 74 | 1.015 | 3.894 | 16.550 | fallback share 0.113 |
| safeguard_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 54 / 61 / 57 | 57 / 65 / 61 | 1.016 | 3.914 | 13.942 | fallback share 0.249 |
| safeguard_every2_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 48 / 50 / 50 | 52 / 56 / 52 | 0.580 | 2.032 | 6.351 | fallback share 0.571 |
| safeguard_every4_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 43 / 47 / 48 | 50 / 52 / 52 | 0.346 | 1.131 | 3.321 | fallback share 0.764 |
| wrapper_smootherjacobi_sweeps1 | 0 / 0 / 0 | 28 / 28 / 28 | 30 / 31 / 30 | 1.012 | 3.882 | 6.793 |  |
| wrapper_smootherjacobi_sweeps2 | 0 / 0 / 0 | 26 / 26 / 25 | 28 / 28 / 27 | 1.009 | 3.879 | 6.303 |  |
| wrapper_smootherchebyshev_sweeps2 | 0 / 0 / 0 | 26 / 27 / 25 | 28 / 28 / 27 | 1.016 | 3.891 | 6.323 |  |
| precision_dtypefloat32_matmulhighest | 0 / 0 / 10 | 60 / 87 / 87 | 62 / 97 / 317 | 0.596 | 2.438 | 13.255 | δ median 6.43e-05, max 4.35e-04, Lemma 3 holds on 100.0% of steps |
| precision_dtypefloat32 | 0 / 0 / 63 | 61 / 95 / — | 64 / 102 / — | 0.308 | 1.104 | — | δ median 6.74e-02, max 4.48e-01, Lemma 3 holds on 100.0% of steps |
| precision_dtypebfloat16 | 100 / 100 / 100 | — / — / — | — / — / — | 0.260 | 0.712 | — | δ median 1.40e+00, max 1.81e+00, Lemma 3 holds on 100.0% of steps |

## diffusion, 64→256 transfer (3 seeds)

| variant | fail % (s0/s1/s2) | median its | p90 its | ms/it b1 | ms/it b16 | est. ms to 1e-6 per system (b16) | notes |
|---|---|---|---|---|---|---|---|
| cg | 0 / 0 / 0 | 606 / 606 / 606 | 660 / 660 / 660 | 0.034 | 0.172 | 6.517 |  |
| base | 0 / 0 / 0 | 143 / 78 / 92 | 165 / 83 / 106 | 2.201 | 15.322 | 88.103 |  |
| safeguard_fallbackidentity | 0 / 0 / 0 | 89 / 70 / 76 | 91 / 78 / 80 | 2.219 | 15.524 | 73.741 | fallback share 0.048 |
| safeguard_fallbackjacobi | 0 / 0 / 0 | 88 / 70 / 74 | 90 / 77 / 79 | 2.224 | 15.488 | 71.630 | fallback share 0.049 |
| safeguard_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 66 / 58 / 57 | 69 / 62 / 60 | 2.224 | 15.531 | 56.298 | fallback share 0.180 |
| safeguard_every2_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 66 / 58 / 54 | 70 / 64 / 60 | 1.158 | 7.938 | 28.776 | fallback share 0.559 |
| safeguard_every4_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 62 / 56 / 52 | 67 / 60 / 56 | 0.640 | 4.240 | 14.840 | fallback share 0.761 |
| wrapper_smootherjacobi_sweeps1 | 0 / 0 / 0 | 34 / 32 / 33 | 36 / 34 / 35 | 2.205 | 15.476 | 31.919 |  |
| wrapper_smootherjacobi_sweeps2 | 0 / 0 / 0 | 29 / 28 / 28 | 31 / 30 / 31 | 2.207 | 15.469 | 27.070 |  |
| wrapper_smootherchebyshev_sweeps2 | 0 / 0 / 0 | 28 / 28 / 28 | 31 / 30 / 30 | 2.207 | 15.484 | 27.097 |  |
| precision_dtypefloat32_matmulhighest | 0 / 0 / 0 | 143 / 78 / 92 | 165 / 83 / 106 | 1.151 | 9.686 | 55.697 | δ median 2.04e-04, max 6.62e-03, Lemma 3 holds on 100.0% of steps |
| precision_dtypefloat32 | 0 / 0 / 0 | 148 / 80 / 93 | 170 / 86 / 103 | 0.471 | 4.460 | 25.921 | δ median 1.11e-01, max 6.52e-01, Lemma 3 holds on 100.0% of steps |
| precision_dtypebfloat16 | 100 / 100 / 100 | — / — / — | — / — / — | 0.341 | 2.522 | — | δ median 2.01e+00, max 3.47e+00, Lemma 3 holds on 100.0% of steps |

## poisson, 128→128 notebook (3 seeds)

| variant | fail % (s0/s1/s2) | median its | p90 its | ms/it b1 | ms/it b16 | est. ms to 1e-6 per system (b16) | notes |
|---|---|---|---|---|---|---|---|
| cg | 0 / 0 / 0 | 316 / 316 / 316 | 319 / 319 / 319 | 0.043 | 0.082 | 1.612 |  |
| base | 0 / 0 / 0 | 35 / 67 / 25 | 36 / 108 / 25 | 0.823 | 3.023 | 6.612 |  |
| safeguard_fallbackidentity | 0 / 0 / 0 | 34 / 25 / 24 | 34 / 25 / 25 | 0.850 | 3.068 | 4.793 | fallback share 0.007 |
| safeguard_fallbackjacobi | 0 / 0 / 0 | 34 / 25 / 24 | 34 / 25 / 25 | 0.856 | 3.076 | 4.806 | fallback share 0.007 |
| safeguard_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 20 / 15 / 15 | 20 / 15 / 15 | 0.857 | 3.084 | 2.892 | fallback share 0.202 |
| safeguard_every2_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 20 / 15 / 15 | 20 / 15 / 15 | 0.492 | 1.622 | 1.521 | fallback share 0.556 |
| safeguard_every4_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 22 / 19 / 18 | 22 / 19 / 19 | 0.305 | 0.921 | 1.094 | fallback share 0.742 |
| wrapper_smootherjacobi_sweeps1 | 0 / 0 / 0 | 16 / 12 / 12 | 17 / 12 / 12 | 0.844 | 3.054 | 2.290 |  |
| wrapper_smootherjacobi_sweeps2 | 0 / 0 / 0 | 13 / 9 / 10 | 14 / 10 / 10 | 0.847 | 3.058 | 1.911 |  |
| wrapper_smootherchebyshev_sweeps2 | 0 / 0 / 0 | 9 / 7 / 7 | 10 / 7 / 7 | 0.845 | 3.060 | 1.339 |  |
| precision_dtypefloat32_matmulhighest | 0 / 0 / 0 | 35 / 64 / 25 | 36 / 114 / 25 | 0.533 | 1.535 | 3.359 | δ median 6.23e-05, max 3.05e-04, Lemma 3 holds on 100.0% of steps |
| precision_dtypefloat32 | 0 / 16 / 0 | 35 / 59 / 26 | 36 / — / 26 | 0.270 | 0.810 | 1.771 | δ median 1.56e-01, max 4.04e-01, Lemma 3 holds on 100.0% of steps |
| precision_dtypebfloat16 | 0 / 0 / 100 | 180 / 105 / — | 184 / 108 / — | 0.258 | 0.566 | — | δ median 1.20e+00, max 1.72e+00, Lemma 3 holds on 100.0% of steps |

## poisson, 128→128 paper (3 seeds)

| variant | fail % (s0/s1/s2) | median its | p90 its | ms/it b1 | ms/it b16 | est. ms to 1e-6 per system (b16) | notes |
|---|---|---|---|---|---|---|---|
| cg | 0 / 0 / 0 | 316 / 316 / 316 | 319 / 319 / 319 | 0.044 | 0.083 | 1.644 |  |
| base | 36 / 35 / 97 | 274 / 24 / — | — / — / — | 1.854 | 6.339 | — |  |
| safeguard_fallbackidentity | 0 / 0 / 0 | 25 / 24 / 32 | 30 / 30 / 34 | 1.865 | 6.384 | 9.976 | fallback share 0.329 |
| safeguard_fallbackjacobi | 0 / 0 / 0 | 25 / 24 / 32 | 30 / 30 / 34 | 1.874 | 6.388 | 9.981 | fallback share 0.329 |
| safeguard_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 15 / 15 / 27 | 26 / 20 / 32 | 1.874 | 6.397 | 5.998 | fallback share 0.630 |
| safeguard_every2_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 15 / 15 / 22 | 17 / 17 / 30 | 1.003 | 3.275 | 3.070 | fallback share 0.715 |
| safeguard_every4_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 18 / 18 / 22 | 19 / 18 / 24 | 0.570 | 1.777 | 1.999 | fallback share 0.806 |
| wrapper_smootherjacobi_sweeps1 | 0 / 1 / 50 | 12 / 12 / 469 | 24 / 12 / — | 1.864 | 6.367 | 4.775 |  |
| wrapper_smootherjacobi_sweeps2 | 0 / 0 / 29 | 9 / 9 / 70 | 19 / 91 / — | 1.864 | 6.377 | 3.587 |  |
| wrapper_smootherchebyshev_sweeps2 | 0 / 0 / 71 | 7 / 19 / — | 16 / 36 / — | 1.867 | 6.376 | — |  |
| precision_dtypefloat32_matmulhighest | 37 / 35 / 96 | 283 / 24 / — | — / — / — | 0.783 | 4.251 | — | δ median 2.64e-05, max 1.61e-04, Lemma 3 holds on 100.0% of steps |
| precision_dtypefloat32 | 36 / 33 / 78 | 26 / 24 / — | — / — / — | 0.331 | 1.666 | — | δ median 2.27e-02, max 9.38e-02, Lemma 3 holds on 100.0% of steps |
| precision_dtypebfloat16 | 29 / 100 / 0 | 62 / — / 117 | — / — / 122 | 0.295 | 1.129 | — | δ median 6.77e-01, max 1.28e+00, Lemma 3 holds on 100.0% of steps |

## poisson, 128→256 transfer (3 seeds)

| variant | fail % (s0/s1/s2) | median its | p90 its | ms/it b1 | ms/it b16 | est. ms to 1e-6 per system (b16) | notes |
|---|---|---|---|---|---|---|---|
| cg | 0 / 0 / 0 | 632 / 632 / 632 | 639 / 639 / 639 | 0.034 | 0.172 | 6.792 |  |
| base | 0 / 0 / 0 | 64 / 45 / 106 | 65 / 46 / 123 | 2.740 | 19.344 | 77.374 |  |
| safeguard_fallbackidentity | 0 / 0 / 0 | 56 / 48 / 66 | 57 / 54 / 69 | 2.758 | 19.429 | 68.000 | fallback share 0.080 |
| safeguard_fallbackjacobi | 0 / 0 / 0 | 56 / 48 / 66 | 57 / 54 / 69 | 2.765 | 19.376 | 67.816 | fallback share 0.080 |
| safeguard_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 34 / 29 / 35 | 34 / 29 / 37 | 2.762 | 19.397 | 41.218 | fallback share 0.134 |
| safeguard_every2_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 34 / 30 / 36 | 34 / 36 / 36 | 1.431 | 9.903 | 21.044 | fallback share 0.592 |
| safeguard_every4_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 34 / 31 / 38 | 36 / 34 / 38 | 0.784 | 5.286 | 11.233 | fallback share 0.750 |
| wrapper_smootherjacobi_sweeps1 | 0 / 0 / 0 | 25 / 21 / 26 | 26 / 22 / 28 | 2.747 | 19.356 | 30.243 |  |
| wrapper_smootherjacobi_sweeps2 | 0 / 0 / 0 | 19 / 17 / 20 | 20 / 18 / 21 | 2.751 | 19.383 | 23.017 |  |
| wrapper_smootherchebyshev_sweeps2 | 0 / 0 / 0 | 16 / 15 / 18 | 17 / 16 / 18 | 2.749 | 19.382 | 19.382 |  |
| precision_dtypefloat32_matmulhighest | 0 / 0 / 0 | 64 / 45 / 107 | 65 / 46 / 123 | 1.344 | 11.676 | 46.705 | δ median 9.23e-05, max 7.10e-04, Lemma 3 holds on 100.0% of steps |
| precision_dtypefloat32 | 0 / 0 / 3 | 64 / 46 / 64 | 65 / 47 / 106 | 0.566 | 5.501 | 22.003 | δ median 1.35e-01, max 3.10e-01, Lemma 3 holds on 100.0% of steps |
| precision_dtypebfloat16 | 100 / 100 / 100 | — / — / — | — / — / — | 0.434 | 3.551 | — | δ median 2.00e+00, max 2.21e+00, Lemma 3 holds on 100.0% of steps |

## poisson, 32→128 transfer (3 seeds)

| variant | fail % (s0/s1/s2) | median its | p90 its | ms/it b1 | ms/it b16 | est. ms to 1e-6 per system (b16) | notes |
|---|---|---|---|---|---|---|---|
| cg | 0 / 0 / 0 | 316 / 316 / 316 | 319 / 319 / 319 | 0.042 | 0.083 | 1.647 |  |
| base | 100 / 100 / 100 | — / — / — | — / — / — | 0.823 | 3.022 | — |  |
| safeguard_fallbackidentity | 0 / 0 / 0 | 86 / 99 / 93 | 94 / 109 / 103 | 0.846 | 3.069 | 17.839 | fallback share 0.363 |
| safeguard_fallbackjacobi | 0 / 0 / 0 | 86 / 99 / 93 | 94 / 109 / 103 | 0.852 | 3.075 | 17.872 | fallback share 0.363 |
| safeguard_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 60 / 65 / 64 | 63 / 68 / 67 | 0.852 | 3.078 | 12.313 | fallback share 0.456 |
| safeguard_every2_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 60 / 64 / 64 | 63 / 66 / 66 | 0.492 | 1.623 | 6.493 | fallback share 0.632 |
| safeguard_every4_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 63 / 67 / 66 | 65 / 71 / 67 | 0.305 | 0.921 | 3.800 | fallback share 0.787 |
| wrapper_smootherjacobi_sweeps1 | 0 / 0 / 0 | 35 / 38 / 36 | 37 / 41 / 38 | 0.840 | 3.048 | 6.857 |  |
| wrapper_smootherjacobi_sweeps2 | 0 / 0 / 0 | 30 / 31 / 30 | 31 / 32 / 31 | 0.845 | 3.058 | 5.734 |  |
| wrapper_smootherchebyshev_sweeps2 | 0 / 0 / 0 | 29 / 30 / 30 | 30 / 31 / 31 | 0.843 | 3.055 | 5.727 |  |
| precision_dtypefloat32_matmulhighest | 100 / 100 / 100 | — / — / — | — / — / — | 0.519 | 1.520 | — | δ median 6.06e-05, max 8.68e-05, Lemma 3 holds on 100.0% of steps |
| precision_dtypefloat32 | 100 / 100 / 100 | — / — / — | — / — / — | 0.269 | 0.800 | — | δ median 8.99e-02, max 1.15e-01, Lemma 3 holds on 100.0% of steps |
| precision_dtypebfloat16 | 100 / 100 / 100 | — / — / — | — / — / — | 0.253 | 0.567 | — | δ median 9.01e-01, max 1.19e+00, Lemma 3 holds on 100.0% of steps |

## poisson, 64→256 transfer (3 seeds)

| variant | fail % (s0/s1/s2) | median its | p90 its | ms/it b1 | ms/it b16 | est. ms to 1e-6 per system (b16) | notes |
|---|---|---|---|---|---|---|---|
| cg | 0 / 0 / 0 | 632 / 632 / 632 | 639 / 639 / 639 | 0.035 | 0.174 | 6.854 |  |
| base | 0 / 0 / 0 | 113 / 126 / 181 | 116 / 131 / 184 | 1.244 | 9.423 | 74.207 |  |
| safeguard_fallbackidentity | 0 / 0 / 0 | 88 / 93 / 95 | 91 / 96 / 97 | 1.258 | 9.555 | 55.541 | fallback share 0.035 |
| safeguard_fallbackjacobi | 0 / 0 / 0 | 88 / 93 / 95 | 91 / 96 / 97 | 1.265 | 9.515 | 55.305 | fallback share 0.035 |
| safeguard_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 62 / 66 / 68 | 66 / 70 / 70 | 1.267 | 9.534 | 39.327 | fallback share 0.169 |
| safeguard_every2_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 62 / 66 / 70 | 64 / 68 / 72 | 0.683 | 4.941 | 20.383 | fallback share 0.560 |
| safeguard_every4_fallbackchebyshev_sweeps2 | 0 / 0 / 0 | 60 / 60 / 63 | 62 / 63 / 67 | 0.392 | 2.715 | 10.180 | fallback share 0.766 |
| wrapper_smootherjacobi_sweeps1 | 0 / 0 / 0 | 40 / 41 / 41 | 41 / 42 / 42 | 1.247 | 9.482 | 24.298 |  |
| wrapper_smootherjacobi_sweeps2 | 0 / 0 / 0 | 32 / 33 / 32 | 33 / 34 / 34 | 1.249 | 9.515 | 19.030 |  |
| wrapper_smootherchebyshev_sweeps2 | 0 / 0 / 0 | 32 / 33 / 32 | 33 / 34 / 34 | 1.251 | 9.518 | 19.036 |  |
| precision_dtypefloat32_matmulhighest | 0 / 0 / 0 | 113 / 126 / 181 | 116 / 131 / 185 | 0.757 | 5.221 | 41.112 | δ median 8.43e-05, max 1.16e-04, Lemma 3 holds on 100.0% of steps |
| precision_dtypefloat32 | 0 / 0 / 0 | 115 / 130 / 186 | 119 / 136 / 189 | 0.352 | 2.336 | 18.977 | δ median 1.51e-01, max 2.01e-01, Lemma 3 holds on 100.0% of steps |
| precision_dtypebfloat16 | 100 / 100 / 100 | — / — / — | — / — / — | 0.278 | 1.444 | — | δ median 1.99e+00, max 2.31e+00, Lemma 3 holds on 100.0% of steps |
