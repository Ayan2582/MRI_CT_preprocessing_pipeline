# 01 — What happened, and the metric it breaks

Assumes [exp4_nce_max/01_the_idea.md](../exp4_nce_max/01_the_idea.md) and
[03_the_mechanism.md](../exp4_nce_max/03_the_mechanism.md).

---

## 1. Bone was absent for a hundred consecutive epochs

`val/dice_bone`:

| epoch | 0 | 5 | 10 | 20 | 50 | 100 | 150 | 199 |
|---|---|---|---|---|---|---|---|---|
| | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0347 | 0.0378 |

Exactly zero from **epoch 6 through epoch 102** — 97 consecutive epochs. (Epoch 5 is
0.000004, which is zero for any practical purpose.) Recall from
[exp4/03 §2](../exp4_nce_max/03_the_mechanism.md) what zero requires: `bone_dice`
(`evaluation/metrics.py:228-236`) returns `None` when *neither* image contains bone, so a
reported 0.0 means the target **did** have bone and the prediction had **none** — no pixel
anywhere above 0.70 normalised (MSK/spine) or 0.775 (abdomen), across 159 bone-eligible
slices.

The model then discovered bone around epoch 103 and got as far as 0.038 by the
end, still rising. It was 5% of the way to the ladder's demonstrated ceiling when the
learning rate ran out.

## 2. The L1 term was never driven

| epoch | 0 | 5 | 20 | 50 | 100 | 199 |
|---|---|---|---|---|---|---|
| `train/G_L1` | 0.449 | 0.373 | 0.347 | 0.330 | 0.319 | **0.297** |

A 1.5× reduction over 200 epochs. exp3b_nce_6_pix48, at λ_L1 48, reaches **0.097** — three
times lower. exp2, at λ_L1 100, reaches 0.084.

The pattern across the ladder is stark: at 8:1 and above the L1 term lands near 0.09; at
4:1 and below it stalls near 0.30. There is nothing in between. That is the cliff, and this
run sits on the wrong side of it by one step.

Meanwhile `train/G_NCE` fell 2.809 → 1.823, perfectly healthily. As in exp4, the
contrastive objective was satisfied throughout. It was not the objective that needed to
hold.

`val/mae_norm` fell monotonically to 0.1585 with `is_best` at **epoch 199** — under-trained
and mis-weighted at once, exactly as exp4.

## 3. The SSIM problem

Here is the finding that makes this run worth its own folder.

| run | `val/dice_bone` | `val/ssim` |
|---|---|---|
| exp3b_nce_6_pix48 (λ_L1 48) | **0.385** | 0.355 |
| **exp3b_nce_20_5** (λ_L1 20) | **0.038** | **0.367** |

**SSIM ranks the run with no bone above the run with the best bone in the ladder.**

This is not noise — it is what SSIM is built to do. The structural term of SSIM is

```
             2·σ_xy + C₂
s(x, y) = ─────────────────
           σ_x² + σ_y² + C₂
```

a *correlation* of local variances, computed in a sliding window and normalised by the
local variances themselves. Two properties follow:

- **It is largely insensitive to contrast scale.** Compress the whole image toward
  mid-grey and `σ_x` shrinks — but it shrinks in both numerator and denominator, so the
  structural term is substantially preserved. SSIM asks "does the local structure
  co-vary?", not "is it the right magnitude?"
- **It rewards smoothness at boundaries.** A hard bone edge is a large local variance. Get
  its position slightly wrong and `σ_xy` collapses while `σ_x²` and `σ_y²` stay large,
  which is heavily punished. Render the same edge as a soft ramp and you lose far less.

So a model that draws correct anatomy in a compressed grey range — exactly what
[exp4/02 §4](../exp4_nce_max/02_what_happened.md) describes — is a model SSIM is
constitutionally inclined to like. And a model that commits to bright bone and misplaces
some of it by a pixel gets punished for the commitment.

Notice that this is the **same structural blind spot as PatchNCE**, arriving from a
different direction. Both are built on normalised, ranking-style comparisons. Both are
therefore indifferent to the absolute intensity scale. If you use PatchNCE as a loss and
SSIM as a metric, you have chosen a training signal and a scoring signal that share an
identical blindness — and the thing they are both blind to is the one thing a synthetic CT
is for.

`val/dice_bone` and `val/mae_hu` are the columns that see it. In this run they disagree
with SSIM completely.

## 4. Where the collapse hurts most: brain

`val/mae_norm` per region at each run's best epoch:

| run | brain | abdomen | musculoskeletal | spine |
|---|---|---|---|---|
| exp3b_nce_6_pix48 | 0.1284 | 0.1303 | 0.0834 | 0.1806 |
| **exp3b_nce_20_5** | **0.2016** | 0.1509 | 0.1070 | 0.2193 |
| **exp4_nce_max** | **0.2131** | 0.1573 | 0.1154 | 0.2160 |

Brain degrades most — 1.6× worse than the 8:1 run, against 1.2–1.3× for the other regions.
It is also the only region where the collapsed runs change the *ordering*: for
exp3b_6_48 and exp2, brain is the second-best region; for exp3b_20_5 and exp4 it drops to
third.

This follows from the windows. Brain CT is normalised over **0–80 HU**, the narrowest
window in the set (`Preprocessing/pipeline_config.py:208-229`). A narrow window means the
real image *uses the full [0,1] range* — skull saturated at 1.0, brain parenchyma spread
across the middle. A model that hedges toward mid-grey is therefore maximally wrong in
normalised terms precisely where the window is tightest.

The reverse is also true and worth noting: musculoskeletal, on a 500 HU window, looks like
the *best* region in this table while carrying 53.5 HU of error. That inversion is general
across all seven runs — see [`_comparison`](../_comparison/README.md).

---

## Summary

Nothing mechanistically new: this is [exp4](../exp4_nce_max/)'s failure at 4:1 instead of
2:1, and the derivation there covers it completely.

What it contributes is calibration and a warning:

- The cliff is between **4:1 and 8:1**, not out in some extreme corner. 4:1 is a setting a
  reasonable person would try.
- **SSIM and PatchNCE share a blind spot.** Do not use one to score the other.
