# e14_final_training

## diffusion 128² at 1e-6 (cap 512)

| variant | seeds | fail % per seed | median per seed | pooled fail % | pooled median | pooled p90 | rounds: fail/median | Wilcoxon p vs m0_cg_notay | ms/it b16 |
|---|---|---|---|---|---|---|---|---|---|
| final_2l2 | 5 | 0 / 0 / 0 / 0 / 0 | 6 / 6 / 6 / 6 / 6 | 0.0 | 6 | 6 | r0: 0%/7; r1: 0%/6; r2: 0%/6; r3: 0%/6 | — | 1.372 |
| final_2l4 | 5 | 0 / 0 / 0 / 0 / 0 | 7 / 7 / 7 / 7 / 7 | 0.0 | 7 | 7 | r0: 0%/7; r1: 0%/7; r2: 0%/7; r3: 0%/7 | — | 0.998 |

## poisson 128² at 1e-6 (cap 512)

| variant | seeds | fail % per seed | median per seed | pooled fail % | pooled median | pooled p90 | rounds: fail/median | Wilcoxon p vs m0_cg_notay | ms/it b16 |
|---|---|---|---|---|---|---|---|---|---|
| final_2l2 | 5 | 0 / 0 / 0 / 0 / 0 | 7 / 6 / 7 / 7 / 6 | 0.0 | 7 | 7 | r0: 0%/7; r1: 0%/7; r2: 0%/7; r3: 0%/7 | — | 1.382 |
| final_2l4 | 5 | 0 / 0 / 0 / 0 / 0 | 7 / 7 / 7 / 7 / 7 | 0.0 | 7 | 7 | r0: 0%/7; r1: 0%/7; r2: 0%/7; r3: 0%/7 | — | 1.002 |
