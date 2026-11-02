# e12_width_training

## diffusion 128² at 1e-6 (cap 512)

| variant | seeds | fail % per seed | median per seed | pooled fail % | pooled median | pooled p90 | rounds: fail/median | Wilcoxon p vs m0_cg_notay | ms/it b16 |
|---|---|---|---|---|---|---|---|---|---|
| h1_on_step_w32 | 3 | 0 / 0 / 0 | 18 / 18 / 21 | 0.0 | 18 | 21 | r0: 7%/42; r1: 0%/30; r2: 0%/19; r3: 0%/18 | — | 3.046 |
| h1_on_step_w48 | 3 | 0 / 0 / 0 | 17 / 18 / 21 | 0.0 | 18 | 22 | r0: 100%/—; r1: 1%/67; r2: 0%/22; r3: 0%/18 | — | 3.904 |
