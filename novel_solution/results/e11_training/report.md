# e11_training

## grf2 64² at 1e-6 (cap 256)

| variant | seeds | fail % per seed | median per seed | pooled fail % | pooled median | pooled p90 | rounds: fail/median | Wilcoxon p vs m0_cg_notay | ms/it b16 |
|---|---|---|---|---|---|---|---|---|---|
| c1h1_on_step | 3 | 16 / 0 / 15 | 19 / 15 / 19 | 10.3 | 17 | — | r0: 4%/79; r1: 2%/36; r2: 8%/25; r3: 10%/19 | 0.0000 | 1.729 |
| h1_on_step | 3 | 0 / 0 / 0 | 27 / 22 / 28 | 0.0 | 25 | 30 | r0: 100%/—; r1: 13%/157; r2: 0%/42; r3: 0%/27 | 0.0000 | 1.747 |
| m0_cg_notay | 3 | 9 / 2 / 100 | 162 / 128 / — | 37.0 | 174 | — | r0: 37%/162 | — | 1.716 |

## grf4 64² at 1e-6 (cap 256)

| variant | seeds | fail % per seed | median per seed | pooled fail % | pooled median | pooled p90 | rounds: fail/median | Wilcoxon p vs m0_cg_notay | ms/it b16 |
|---|---|---|---|---|---|---|---|---|---|
| c1h1_on_step | 3 | 0 / 47 / 32 | 27 / 123 / 79 | 26.3 | 39 | — | r0: 100%/—; r1: 87%/—; r2: 31%/182; r3: 26%/79 | 0.0000 | 1.728 |
| h1_on_step | 3 | 99 / 92 / 99 | — / — / — | 96.7 | — | — | r0: 100%/—; r1: 100%/—; r2: 88%/—; r3: 97%/— | 0.0051 | 1.749 |
| m0_cg_notay | 3 | 100 / 100 / 100 | — / — / — | 100.0 | — | — | r0: 100%/— | — | 1.716 |

## grf6 64² at 1e-6 (cap 256)

| variant | seeds | fail % per seed | median per seed | pooled fail % | pooled median | pooled p90 | rounds: fail/median | Wilcoxon p vs m0_cg_notay | ms/it b16 |
|---|---|---|---|---|---|---|---|---|---|
| c1h1_on_step | 3 | 73 / 98 / 92 | — / — / — | 87.7 | — | — | r0: 100%/—; r1: 100%/—; r2: 99%/—; r3: 88%/— | 0.0000 | 1.733 |
| h1_on_step | 3 | 100 / 100 / 100 | — / — / — | 100.0 | — | — | r0: 100%/—; r1: 100%/—; r2: 100%/—; r3: 100%/— | 1.0000 | 1.749 |
| m0_cg_notay | 3 | 100 / 100 / 100 | — / — / — | 100.0 | — | — | r0: 100%/— | — | 1.713 |
