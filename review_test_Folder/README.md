# Failure review — the seven runs in `results/`

What each run did, why it did it, and what to change. Written failure-first: each folder
derives the idea from scratch, shows the wreckage, then explains why the second followed
from the first.

Nothing here has been applied. No file under `results/` or `model/` was modified.

---

## The scoreboard

All runs: 200 epochs, identical validation set (230 samples, 159 bone-eligible).

| run | λ_L1 | λ_NCE | best `mae_norm` | @ep | drift to final | `dice_bone` | what went wrong |
|---|---|---|---|---|---|---|---|
| [exp1_pix2pix](exp1_pix2pix/) ¹ | 100 | 0 | **0.0763** | 72 | +2.2% | 0.364 | measured on a different pipeline |
| [exp2_paper](exp2_paper/) | 100 | 1 | 0.1166 | **18** | **+8.4%** | 0.369 | overfits at epoch 18; hallucinates bone |
| [exp3b_nce_6_pix48](exp3b_nce_6_pix48/) | 48 | 6 | 0.1203 | 121 | **+0.9%** | **0.385** | the best run — still hallucinates, still 119 HU on bone |
| [exp3b_nce_20_5](exp3b_nce_20_5/) | 20 | 5 | 0.1585 | 199 | 0.0% | **0.038** | contrast collapse — bone gone |
| [exp4_nce_max](exp4_nce_max/) | 10 | 5 | 0.1666 | 199 | 0.0% | **0.029** | contrast collapse, harder |
| [exp8_cyclegan](exp8_cyclegan/) | 0 | 0 | 0.2208 | **7** | +1.2% | 0.078 | never trained; best epoch is the identity map |
| [exp7_reggan](exp7_reggan/) | 0 | 0 | 0.2897 | **0** | +6.0% | 0.196 | never trained; learned to exploit the warp as a blur |

¹ **not comparable** — trained pre-ROI with `hflip: true`, before commit `4d98c6c`.

Two runs peaked at epoch **0** and epoch **7**. Two produced essentially **no bone**. One
peaked at epoch 18 and degraded for the next 181. **One run in seven behaved like something
you could ship.**

---

## Start here

**[`_comparison/`](_comparison/README.md)** — the five findings that only exist across
runs, and the honest status of the ladder's six questions. If you read one thing, read
this.

Then, by how much there is to learn from it:

| folder | the lesson |
|---|---|
| [exp7_reggan](exp7_reggan/) | a learned warp is a learned filter — and a loss curve can only tell you the number you wrote down is going down |
| [exp4_nce_max](exp4_nce_max/) | a contrastive loss is provably blind to intensity; L1's median discards minority modes; bone is both |
| [exp3b_nce_6_pix48](exp3b_nce_6_pix48/) | why the middle of the λ range is stable when both ends are not |
| [exp2_paper](exp2_paper/) | 54.4 M parameters against **32 patients**; and why an adversary invents texture |
| [exp8_cyclegan](exp8_cyclegan/) | cycle consistency buys injectivity, not correctness |
| [exp1_pix2pix](exp1_pix2pix/) | a masked pixel metric is not portable across preprocessing changes |
| [exp3b_nce_20_5](exp3b_nce_20_5/) | SSIM and PatchNCE share a blind spot — don't use one to score the other |

Every run folder has the same shape: **`01_the_idea`** (first principles, no repo code) →
**`02_what_happened`** (evidence, no explanation) → **`03_the_mechanism`** (the derivation
joining them) → **`04_what_to_change`**.

---

## The exp8 deep-dive

The five numbered documents at this level are the detailed exp8 diagnosis, written from
three sample panels *before* `results/` existed:

| doc | |
|---|---|
| [01_reading_the_panels.md](01_reading_the_panels.md) | the three epochs as evidence — **now scored against the real metrics** |
| [02_the_lambda_arithmetic.md](02_the_lambda_arithmetic.md) | why epoch 9 is a copy: the 15:1 λ ratio |
| [03_the_undefined_mapping.md](03_the_undefined_mapping.md) | why no λ can fix it: the target relation is not a function |
| [04_the_split_is_lopsided.md](04_the_split_is_lopsided.md) | the seed-1337 partition, measured |
| [05_the_background_artifacts.md](05_the_background_artifacts.md) | the corner blob and the streaks |
| [06_what_to_change.md](06_what_to_change.md) | ranked fixes |

Five of their six predictions held; the sixth was wrong and is corrected in place in 01.
[exp8_cyclegan/](exp8_cyclegan/) is the condensed version plus the full scorecard.

---

## Reproducing the numbers

Two read-only scripts. Neither writes anything.

```
python review_test_Folder/_comparison/ladder_audit.py     # every table in _comparison/
python review_test_Folder/evidence/split_audit.py         # the exp8 partition, from the manifest
```

`ladder_audit.py` takes `--table scoreboard|region|inversion|cliff`; `split_audit.py` takes
`--seed`.

---

## The three things that generalise beyond this project

**Every metric here has a blind spot, and two of them share one.** SSIM and PatchNCE are
both built on locally-normalised, ranking-style comparisons, so both are indifferent to
absolute intensity — which is the one thing a synthetic CT is for. In this ladder SSIM ranks
a model that produced *no bone* above the model with the best bone. Only `mae_hu` and
`dice_bone` have an opinion about absolute values; at least one belongs in every table.

**A healthy-looking training run is not evidence of anything.** exp7's `train/G_corr` fell
smoothly by 22× — the cleanest loss curve of all seven runs — while its validation error
rose the whole time. exp8's adversarial game was perfectly balanced for 200 epochs and
produced nothing.

**Four of the seven failures were visible within 20 epochs**, in columns `metrics.csv`
already contains. Two rules would have caught them: stop if `is_best` is still below epoch 5
once you pass epoch 20, and alert if `val/dice_bone` is exactly 0.0000 after epoch 5.
