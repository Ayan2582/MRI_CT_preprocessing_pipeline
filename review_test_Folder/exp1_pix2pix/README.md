# exp1_pix2pix

**Verdict: the best-scoring run in the ladder, and the score is not comparable to any other
run in it.**

`λ_GAN: 1`, `λ_L1: 100`, `λ_NCE: 0` — the standard pix2pix recipe, and by the numbers the
clear winner: `val/mae_norm` **0.0763**, 35% better than anything else. It is also the only
run trained on a **different preprocessing pipeline**, and that difference makes its task
measurably easier.

This is not a modelling failure. It is a measurement failure, and it is the one most likely
to end up in a writeup as a wrong claim.

---

## What is different about this run

| setting | exp1 | every other run |
|---|---|---|
| `data.use_roi` | **absent from `config.resolved.yaml`** | `true` |
| `data.augment.hflip` | **`true`** | `false` |

ROI cropping landed in commit `4d98c6c` ("Added ROI crop"), **after** this run was
executed. `exp2_paper` and everything since has `use_roi: True, hflip: False`.

So exp1 differs from the rest by three things at once — the objective (no NCE), the
**field of view**, and the **augmentation**. Two of those three favour exp1, and neither is
the variable the ladder was designed to test.

---

## The numbers

| | exp1_pix2pix | exp2_paper | exp3b_nce_6_pix48 |
|---|---|---|---|
| pipeline | **pre-ROI, hflip on** | current | current |
| best `val/mae_norm` | **0.0763** @ ep72 | 0.1166 @ ep18 | 0.1203 @ ep121 |
| `val/mae_norm` at **epoch 0** | **0.1024** | 0.1468 | 0.1627 |
| `val/ssim` | **0.615** | 0.386 | 0.355 |
| `val/mae_hu/brain` | **5.5 HU** | 9.4 HU | 10.3 HU |
| `val/dice_bone` | 0.364 | 0.369 | 0.385 |

Look at the epoch-0 row. After a single epoch, before any model has learned much, exp1 is
already 43% ahead of exp2 — which has the *same* `λ_L1: 100`. 01 explains what that number
is measuring.

---

## Reading order

| doc | question it answers |
|---|---|
| [01_the_measurement_problem.md](01_the_measurement_problem.md) | Why a masked pixel metric is not portable across preprocessing changes — and what an epoch-0 score actually tells you |
| [02_what_to_change.md](02_what_to_change.md) | What to re-run, and what you may and may not claim meanwhile |

---

## The one-paragraph version

`mae_norm` is a mean absolute error over the *valid* pixels of a slice. "Valid" excludes
zero-padding but **includes in-frame background** — the black air around the patient. ROI
cropping tightens the field of view onto the anatomy, which removes easy near-zero-error
background pixels from the average and leaves a higher proportion of hard ones. Same model
quality, larger number. exp1 was never ROI-cropped, so its denominator contains more easy
pixels than every other run's. It also trained with horizontal-flip augmentation that the
others did not have. The 0.0763 is real, and it is not a number you can put in the same
column as 0.1166.
