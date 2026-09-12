# 02 — What happened

Evidence only. No explanation until 03. Everything here is from
`results/exp7_reggan_results/metrics.csv` and `samples/`.

---

## 1. The two curves point in opposite directions

| epoch | `train/G_corr` | `val/mae_norm` | `train/R_flow_px` | `is_best` |
|---|---|---|---|---|
| 0 | 0.2040 | **0.2897** | 0.592 | **True** |
| 1 | 0.0900 | 0.2906 | 0.697 | False |
| 2 | 0.0676 | 0.2928 | 0.709 | False |
| 3 | 0.0565 | 0.2950 | 0.712 | False |
| 4 | 0.0528 | 0.2971 | 0.712 | False |
| 5 | 0.0473 | 0.2993 | 0.712 | False |
| 10 | 0.0351 | 0.3004 | 0.716 | False |
| 20 | 0.0274 | 0.2992 | 0.715 | False |
| 50 | 0.0180 | 0.2971 | 0.712 | False |
| 100 | 0.0133 | 0.2976 | 0.710 | False |
| 150 | 0.0123 | 0.3023 | 0.699 | False |
| 199 | **0.0092** | **0.3071** | 0.698 | False |

The optimised quantity improved by 22×. The measured quantity got worse. There is no
epoch at which they agree.

**The divergence is immediate.** Between epoch 0 and epoch 1, `G_corr` more than halves
(0.204 → 0.090) and `val/mae_norm` *rises*. Whatever happened, happened in the first
epoch.

## 2. It happened before the discriminator existed

`train.gan_warmup_epochs: 5`. Check `train_log.jsonl`: the records for epochs 0–4 contain
**no `train/D_*` keys at all**. They first appear at epoch 5.

So for epochs 0 through 4 the entire generator objective was

```
100 · G_corr  +  10 · G_smooth
```

and in that window `val/mae_norm` went 0.2897 → 0.2993, monotonically worse, while
`G_corr` fell 4.3×.

**The discriminator cannot be the cause.** It was not switched on yet.

## 3. The flow field froze in the first three epochs

`train/R_flow_px` — the mean displacement magnitude, in millimetres, since 1 px = 1 mm:

```
epoch 0: 0.592     epoch 3:  0.712     epoch 100: 0.710
epoch 1: 0.697     epoch 5:  0.712     epoch 150: 0.699
epoch 2: 0.709     epoch 10: 0.716     epoch 199: 0.699
```

It rises for three epochs, reaches ~0.71, and then does not move for the remaining 197.
`R_flow_max` behaves the same way: 1.82 → 1.51, drifting slightly down and settling.

**Note this number: 0.699.** It matters in 03.

Note also what did *not* happen. `model/README.md:116-118` names the failure to watch
for as "`R_flow_max` growing without bound while `G_corr` keeps falling." `G_corr` kept
falling. `R_flow_max` did not grow — it *shrank*, from 1.82 to 1.51. **The documented
diagnostic never fired.**

## 4. By the end, the regulariser outweighs the objective

Multiply out the generator loss at epoch 199:

| term | weight | value | contribution |
|---|---|---|---|
| `G_corr` | 100 | 0.00924 | **0.924** |
| `G_smooth` | 10 | 0.10181 | **1.018** |
| `G_GAN` | 1 | 0.80951 | 0.810 |
| | | | **2.752** |

which matches the logged `train/G_total` = 2.7515 exactly. At epoch 0 the same
arithmetic gives 100(0.2040) + 10(0.1752) = 22.15, matching `G_total` = 22.1543.

So by the end, **the flow-smoothness penalty is the largest single term in the objective
the generator is descending.** The image-fidelity term is second, and shrinking.

## 5. The discriminator won completely, and then stopped mattering

From epoch 20 onward `D` is effectively perfect — **exactly 1.0000 in 70% of the
remaining epochs**, never below 0.93, with `D_score_fake` at 0.0001 by the end:

```
epoch 199:  train/D_acc_real = 1.0000   train/D_acc_fake = 1.0000
            train/D_score_fake = 0.0001  train/G_GAN = 0.8095
```

`0.8095` is not an arbitrary number. Under LSGAN with `label_smoothing.real_target: 0.9`,
the generator's loss is `(D(fake) − 0.9)²`. When D is certain and outputs 0, that is
**`0.81` exactly**. From epoch 10 to the end, `G_GAN` has a mean of **0.809** and stays
within 0.76–0.85 apart from a handful of excursions — i.e. it sits at the value it takes
when the discriminator is completely certain, and does not move off it. The generator
received essentially **no usable adversarial gradient for 190 epochs**.

## 6. What the panels show

**Epoch 4** (`samples/epoch_0004.png`) — during warm-up, before D exists:

- A featureless mid-grey blob in the rough silhouette of the body. No bone, no internal
  anatomy, no tissue contrast whatsoever.
- A dense, fine **checkerboard/grid pattern with a pitch of one to two pixels**, covering
  the entire frame — inside the body and out.
- The background is not black. It is a soft grey halo around the silhouette, plus the
  grid.

**Epoch 199** (`samples/epoch_0199.png`) — the same failure, slightly sharper silhouette:

- Still a featureless grey blob. The brain axial slice is a plain grey oval where the real
  CT has a saturated white skull and grey/white matter differentiation.
- Still the fine grid, now reading as heavy horizontal striping in the brain and
  musculoskeletal panels.
- Panel MAE per slice: 0.195, 0.171, 0.204, 0.175, 0.130, 0.139, 0.145, 0.144.

The failure at epoch 4 and the failure at epoch 199 are **the same picture**. 195 epochs
of training, with a loss that improved 5× further across them, changed nothing visible.

## 7. Where the error is, in HU

| region | `val/mae_hu` at epoch 199 |
|---|---|
| brain | 24.4 HU |
| musculoskeletal | 117.4 HU |
| abdomen | 135.4 HU |
| spine | 224.5 HU |

And `val/mae_band_bone` = 222.9 HU — the mean error, restricted to pixels that are bone in
the real CT, is over 220 Hounsfield units.

---

## The four facts 03 has to explain

1. `G_corr` reached 0.0092 — **9× lower than the best paired L1 anywhere in this ladder**
   (exp2 finishes at 0.084) — while producing that panel.
2. The failure was fully formed by epoch 4, with no discriminator involved.
3. `R_flow_px` converged to **0.698** and stayed there for 197 epochs.
4. The visible artifact is specifically a **one-to-two-pixel checkerboard**, not blur, not
   noise, not saturation.

Facts 3 and 4 are the ones that give it away.
