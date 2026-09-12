# 02 — What the metrics confirmed, and what they broke

The five top-level documents were written from three sample panels, before
`results/exp8_cyclegan_results/metrics.csv` existed. [01](../01_reading_the_panels.md) §
"Check this against your own metrics.csv" made six explicit predictions so they could be
checked.

Here is the check. Five held; one was wrong, and the wrong one is interesting.

---

## The scorecard

| predicted | actual | |
|---|---|---|
| `val/mae_norm` roughly flat from ~epoch 110 | **flat the entire run** — total range 0.2208–0.2283 across 200 epochs | **held, and understated** |
| spine the worst region by a wide margin | 0.3903 vs 0.1641 musculoskeletal — **2.4×** | **held** |
| `val/dice_bone` stays low | never exceeds **0.080**; 0.059 at the best epoch | **held** |
| `train/G_cycle` falls steadily while the images do not improve | `G_cycle_A` 0.170 → 0.036 (**4.7×**), `G_cycle_B` 0.149 → 0.024 (**6.1×**) | **held** |
| musculoskeletal regresses after ~epoch 50 | *panel* MSK regressed; *validation* MSK improved slightly | **partly wrong** |
| `train/D_acc_fake` high and stable — D winning comfortably | **0.543 → 0.869, rising slowly, never saturating** | **wrong** |

---

## 1. The prediction that was wrong, and why it matters

[01](../01_reading_the_panels.md) predicted a saturated discriminator. That is not what
happened.

| epoch | `D_acc_real_B` | `D_acc_fake_B` | `G_GAN_A2B` |
|---|---|---|---|
| 0 | 0.555 | 0.543 | 0.302 |
| 20 | 0.699 | 0.723 | 0.362 |
| 50 | 0.731 | 0.786 | 0.400 |
| 100 | 0.759 | 0.794 | 0.430 |
| 199 | 0.850 | 0.869 | 0.430 |

`D` climbs slowly from chance to about 87% and stops there. It never saturates. Compare
[exp7_reggan](../exp7_reggan/), where `D_acc` reaches exactly **1.0000** in most epochs past 20 and the
generator's adversarial gradient dies completely.

**By every standard diagnostic, exp8's adversarial game is healthy.** `D` is winning but not
dominating — textbook. Neither network collapsed. There is no mode collapse, no oscillation,
no vanishing gradient.

And the model produced nothing. Its best epoch was 7 and its validation metric moved 3.4%
in 200 epochs.

That correction is worth more than the prediction would have been. **A balanced adversarial
equilibrium is not evidence of a working run.** It only tells you the two networks are
matched. Two well-matched networks can sit in a stable equilibrium at a solution that is
completely wrong — here, because the terms that were supposed to force correctness
([01 §3](01_the_idea.md)) do not measure correctness, so nothing in the game is pulling
toward it.

There is a second detail in that table: **`G_GAN_A2B` rises**, 0.302 → 0.430. The generator
got *worse* at fooling the discriminator over 200 epochs, while its cycle loss fell 4.7×.
The two objectives were pulling against each other, and the reconstruction terms — at 15
against 1 — won.

## 2. The prediction that was partly wrong

Panel musculoskeletal MAE went 0.059 → 0.070 between epochs 9 and 199, and
[01](../01_reading_the_panels.md) read that as a regression. Validation MSK `mae_norm`
improved slightly over the same span.

Both are true. The panel is two fixed slices scored **unmasked** over the full padded frame
(`training/trainer.py:388`); validation is 230 slices scored **masked**
(`evaluation/metrics.py:127-135`). A regression on two slices is not a regression on the
region, and the doc should have said "these two slices" rather than "MSK".

The underlying point survives — the epoch-9 identity output was competitive with the
epoch-199 trained output on those slices — but it was stated more broadly than the evidence
supported.

## 3. The caveat that was flagged, and was right

The top-level [README](../README.md) warned that the panel MAE is unmasked and therefore
diluted by background, and that it should not be compared against `val/mae_norm`.

| | panel mean (unmasked) | `val/mae_norm` (masked) |
|---|---|---|
| epoch 199 | 0.0968 | **0.2235** |

**A factor of 2.3.** The warning was correct and the size of the effect was larger than
implied. Every panel number in the top-level documents should be read as roughly half the
real error.

## 4. What the metrics add that the panels could not show

**The best epoch was 7.** The panels could show that epoch 9 looked like a copy of the
input; only `is_best` shows that this copy was *the best the model ever did*. `val/mae_norm`
at epoch 0 is 0.2283 and at epoch 7 is 0.2208 — the entire useful progress of the run
happened in seven epochs and consisted of learning to leave the image roughly alone.

**The identity terms fell 12×.** `train/G_idt_A`: 0.148 → 0.012; `G_idt_B`: 0.135 → 0.010.
The generators became *more* nearly the identity on their own domain as training went on —
the direct measurement of the trap derived in [01 §4](01_the_idea.md) and
[02_the_lambda_arithmetic.md](../02_the_lambda_arithmetic.md).

**The errors in Hounsfield units:**

| region | `mae_hu` |
|---|---|
| brain | 21.6 |
| abdomen | 80.4 |
| musculoskeletal | 82.1 |
| **spine** | **195.2** |
| bone pixels only (`mae_band_bone`) | **209.9** |

209.9 HU on bone. For reference, the difference between cortical bone and soft tissue is
about 900 HU, so the model is placing bone density with an error of roughly a quarter of
the entire soft-tissue-to-bone range.

---

## Summary

The panel-only diagnosis was right about the mechanism, right about the trajectory, right
about spine, and right about its own caveat. It was wrong about the discriminator, and that
error is instructive: it assumed a bad result implied a broken adversarial game. exp8's
adversarial game was fine. **The objective was the problem, and a healthy-looking GAN is
perfectly capable of optimising a wrong objective very stably for 200 epochs.**
