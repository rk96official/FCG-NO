# e7_coefficient

## a1 32² at 1e-6 (cap 128)

| variant | seeds | fail % per seed | median per seed | pooled fail % | pooled median | pooled p90 | rounds: fail/median | Wilcoxon p vs m0_cg_notay | ms/it b16 |
|---|---|---|---|---|---|---|---|---|---|
| c1_cg_notay | 3 | 1 / 1 / 1 | 12 / 12 / 12 | 1.0 | 12 | 14 | r0: 1%/12 | 0.0000 | 1.153 |
| c1h1_on_step | 3 | 1 / 1 / 1 | 10 / 10 / 10 | 1.0 | 10 | 12 | r0: 2%/14; r1: 1%/13; r2: 1%/12; r3: 1%/10 | 0.0000 | 1.175 |
| h1_on_step | 3 | 0 / 0 / 0 | 12 / 13 / 12 | 0.0 | 13 | 19 | r0: 1%/24; r1: 1%/22; r2: 0%/15; r3: 0%/12 | 0.0000 | 1.172 |
| m0_cg_notay | 3 | 0 / 1 / 1 | 24 / 34 / 33 | 0.7 | 31 | 42 | r0: 1%/33 | — | 1.151 |

### a1 32²: do failures coincide with high contrast?

| variant | failed/all | AUC of contrast | Spearman(contrast, iterations) |
|---|---|---|---|
| c1_cg_notay | 3/300 | 1.00 | 0.51 |
| c1h1_on_step | 3/300 | 1.00 | 0.75 |
| h1_on_step | 0/300 | — | 0.75 |
| m0_cg_notay | 2/300 | 1.00 | -0.16 |

## a1 64² at 1e-6 (cap 256)

| variant | seeds | fail % per seed | median per seed | pooled fail % | pooled median | pooled p90 | rounds: fail/median | Wilcoxon p vs m0_cg_notay | ms/it b16 |
|---|---|---|---|---|---|---|---|---|---|
| c1_cg_notay | 3 | 4 / 6 / 8 | 30 / 26 / 33 | 6.0 | 29 | 104 | r0: 6%/30 | 0.0000 | 2.359 |
| c1h1_on_step | 3 | 0 / 1 / 1 | 14 / 14 / 15 | 0.7 | 14 | 17 | r0: 10%/82; r1: 1%/32; r2: 1%/18; r3: 1%/14 | 0.0000 | 2.368 |
| h1_on_step | 3 | 0 / 0 / 0 | 19 / 18 / 19 | 0.0 | 19 | 22 | r0: 9%/121; r1: 1%/32; r2: 0%/21; r3: 0%/19 | 0.0000 | 2.368 |
| m0_cg_notay | 3 | 3 / 6 / 7 | 49 / 49 / 52 | 5.3 | 51 | 140 | r0: 5%/49 | — | 2.360 |

### a1 64²: do failures coincide with high contrast?

| variant | failed/all | AUC of contrast | Spearman(contrast, iterations) |
|---|---|---|---|
| c1_cg_notay | 18/300 | 1.00 | 0.72 |
| c1h1_on_step | 2/300 | 1.00 | 0.47 |
| h1_on_step | 0/300 | — | 0.30 |
| m0_cg_notay | 16/300 | 1.00 | 0.22 |

## a2 32² at 1e-6 (cap 128)

| variant | seeds | fail % per seed | median per seed | pooled fail % | pooled median | pooled p90 | rounds: fail/median | Wilcoxon p vs m0_cg_notay | ms/it b16 |
|---|---|---|---|---|---|---|---|---|---|
| c1_cg_notay | 3 | 1 / 1 / 1 | 12 / 12 / 12 | 1.0 | 12 | 14 | r0: 1%/12 | 0.0000 | 1.154 |
| c1h1_on_step | 3 | 0 / 1 / 0 | 10 / 10 / 10 | 0.3 | 10 | 11 | r0: 2%/13; r1: 1%/14; r2: 0%/12; r3: 0%/10 | 0.0000 | 1.176 |
| h1_on_step | 3 | 0 / 0 / 0 | 15 / 12 / 12 | 0.0 | 13 | 16 | r0: 0%/21; r1: 0%/16; r2: 0%/14; r3: 0%/12 | 0.0000 | 1.162 |
| m0_cg_notay | 3 | 1 / 0 / 0 | 27 / 20 / 19 | 0.3 | 22 | 34 | r0: 0%/20 | — | 1.153 |

### a2 32²: do failures coincide with high contrast?

| variant | failed/all | AUC of contrast | Spearman(contrast, iterations) |
|---|---|---|---|
| c1_cg_notay | 3/300 | 1.00 | 0.66 |
| c1h1_on_step | 1/300 | 1.00 | 0.37 |
| h1_on_step | 0/300 | — | 0.43 |
| m0_cg_notay | 1/300 | 1.00 | 0.09 |

## a2 64² at 1e-6 (cap 256)

| variant | seeds | fail % per seed | median per seed | pooled fail % | pooled median | pooled p90 | rounds: fail/median | Wilcoxon p vs m0_cg_notay | ms/it b16 |
|---|---|---|---|---|---|---|---|---|---|
| c1_cg_notay | 3 | 3 / 3 / 3 | 32 / 23 / 25 | 3.0 | 27 | 50 | r0: 3%/25 | 0.0000 | 2.360 |
| c1h1_on_step | 3 | 0 / 0 / 0 | 14 / 14 / 14 | 0.0 | 14 | 16 | r0: 10%/86; r1: 1%/31; r2: 0%/18; r3: 0%/14 | 0.0000 | 2.369 |
| h1_on_step | 3 | 0 / 0 / 0 | 21 / 19 / 19 | 0.0 | 20 | 31 | r0: 11%/77; r1: 0%/33; r2: 0%/19; r3: 0%/19 | 0.0000 | 2.363 |
| m0_cg_notay | 3 | 3 / 1 / 3 | 55 / 36 / 41 | 2.3 | 44 | 94 | r0: 2%/41 | — | 2.356 |

### a2 64²: do failures coincide with high contrast?

| variant | failed/all | AUC of contrast | Spearman(contrast, iterations) |
|---|---|---|---|
| c1_cg_notay | 9/300 | 1.00 | 0.60 |
| c1h1_on_step | 0/300 | — | 0.45 |
| h1_on_step | 0/300 | — | 0.73 |
| m0_cg_notay | 7/300 | 1.00 | 0.38 |
