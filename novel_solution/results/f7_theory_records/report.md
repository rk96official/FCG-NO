# Theorem 3 check: learned coarse solve vs the exact Galerkin coarse solve

E14 `final_2l2` models (seeds [0, 1, 2, 3, 4]), network on the 64 grid, records along their own two-level FCG trajectories (20 test systems each). delta = sin^2 in the Galerkin coarse norm; efficiency = cos^2_A of the prolongated coarse direction with the remainder error (the share of the energy the coarse step removes). Theorem 3 predicts learned = (1 - delta) * exact; the last column is the largest relative deviation.

| kind | grid | records | median delta | p90 delta | median exact efficiency | median learned efficiency | identity max rel. dev. |
|---|---|---|---|---|---|---|---|
| poisson | 128 | 1046 | 0.2023 | 0.2764 | 0.706 | 0.548 | 2.5e-04 |
| poisson | 256 | 1673 | 0.3182 | 0.3770 | 0.850 | 0.571 | 1.2e-03 |
| diffusion | 128 | 1019 | 0.0731 | 0.2497 | 0.496 | 0.447 | 8.4e-06 |
| diffusion | 256 | 1504 | 0.2228 | 0.3936 | 0.829 | 0.611 | 1.1e-05 |
