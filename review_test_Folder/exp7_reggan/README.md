# exp7_reggan

**Verdict: the best epoch was epoch 0. The run got worse every epoch for 200 epochs, and
its training loss got better the whole time.**

This is the most instructive failure in the set, because nothing about it looks like a
bug. No NaN, no divergence, no crash. A loss curve that falls smoothly by a factor of 22,
and a model that ends up worse than its own initialisation.

---

## The numbers

| | epoch 0 | epoch 199 |
|---|---|---|
| `train/G_corr` (what was optimised) | 0.2040 | **0.0092** — 22× better |
| `val/mae_norm` (what was measured) | **0.2897** | 0.3071 — 6% worse |
| `val/mae_hu` | 92.6 HU | 100.2 HU |
| `val/ssim` | 0.122 | 0.222 |
| `val/psnr` | 7.70 | 7.22 |
| `train/R_flow_px` | 0.592 px | 0.699 px |
| `train/G_GAN` | (no D yet) | 0.8095 |

`is_best` is `True` at epoch 0 and never again.

For scale: `G_corr = 0.0092` is **nine times lower** than the best training L1 any paired
run in this ladder ever reached (exp2 finishes at 0.084). No model producing the epoch-199
panel earns that number honestly. Doc 03 shows how it was earned.

---

## Reading order

| doc | question it answers |
|---|---|
| [01_the_idea.md](01_the_idea.md) | What is RegGAN for? Why would you ever take your loss through a warp? |
| [02_what_happened.md](02_what_happened.md) | The evidence, epoch by epoch — no explanation yet |
| [03_the_mechanism.md](03_the_mechanism.md) | The derivation: why this objective had to produce that picture |
| [04_what_to_change.md](04_what_to_change.md) | Ranked fixes, and the one experiment that would confirm 03 |

Doc 03 is the point of this folder. 01 and 02 exist so that 03 lands.

---

## The one-paragraph version

RegGAN scores the generator *after* a learned deformation, so that residual misalignment
between the MRI and CT is not charged to the generator. exp7 set `lambda_l1: 0`, which
removed the only term measured in the un-warped frame. From that moment the generator was
optimised in a coordinate system the registration network controls, and evaluated in the
pixel grid — **two different measurements, with nothing connecting them.** Worse, the warp
is a bilinear resample at sub-pixel offsets, which is arithmetically a blur. The generator
learned to emit high-frequency garbage that the warp filters away before the loss sees it.
The flow field settled at 0.699 px — within rounding distance of √(0.5²+0.5²) = 0.707, the
offset at which bilinear interpolation blurs *most* — and stayed there for 197 epochs.
