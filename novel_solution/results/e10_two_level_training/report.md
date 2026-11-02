# e10_two_level_training

## diffusion 128² at 1e-6 (cap 512)

| variant | seeds | fail % per seed | median per seed | pooled fail % | pooled median | pooled p90 | rounds: fail/median | Wilcoxon p vs m0_cg_notay | ms/it b16 |
|---|---|---|---|---|---|---|---|---|---|
| h1_on_step_2l2 | 3 | 0 / 0 / 0 | 6 / 6 / 6 | 0.0 | 6 | 6 | r0: 0%/7; r1: 0%/6; r2: 0%/6; r3: 0%/6 | — | 3.271 |
| h1_on_step_2l4 | 3 | 0 / 0 / 0 | 7 / 7 / 7 | 0.0 | 7 | 7 | r0: 0%/7; r1: 0%/7; r2: 0%/7; r3: 0%/7 | — | 2.295 |
| m0_on_step_2l4 | 3 | 0 / 0 / 0 | 7 / 7 / 7 | 0.0 | 7 | 7 | r0: 0%/7; r1: 0%/7; r2: 0%/7; r3: 0%/7 | — | 2.276 |

## poisson 128² at 1e-6 (cap 512)

| variant | seeds | fail % per seed | median per seed | pooled fail % | pooled median | pooled p90 | rounds: fail/median | Wilcoxon p vs m0_cg_notay | ms/it b16 |
|---|---|---|---|---|---|---|---|---|---|
| h1_on_step_2l2 | 3 | 0 / 0 / 0 | 6 / 6 / 6 | 0.0 | 6 | 7 | r0: 0%/7; r1: 0%/7; r2: 0%/7; r3: 0%/6 | — | 1.384 |
| h1_on_step_2l4 | 3 | 0 / 0 / 0 | 7 / 7 / 7 | 0.0 | 7 | 7 | r0: 0%/7; r1: 0%/7; r2: 0%/7; r3: 0%/7 | — | 1.004 |
| m0_on_step_2l4 | 3 | 0 / 0 / 0 | 7 / 7 / 7 | 0.0 | 7 | 7 | r0: 0%/7; r1: 0%/7; r2: 0%/7; r3: 0%/7 | — | 0.964 |
