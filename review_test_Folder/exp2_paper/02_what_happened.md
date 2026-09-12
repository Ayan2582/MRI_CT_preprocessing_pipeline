# 02 — What happened

All five predictions from 01 hold. From `results/exp2_paper_results/`.

---

## 1. The turn came at epoch 18

| epoch | `train/G_L1` | `val/mae_norm` | `val/ssim` | `val/dice_bone` |
|---|---|---|---|---|
| 0 | 0.358 | 0.1468 | 0.345 | 0.0002 |
| 5 | 0.232 | 0.1302 | 0.402 | 0.275 |
| 10 | 0.198 | 0.1193 | 0.430 | 0.374 |
| **18** | — | **0.1166** ← best | — | — |
| 20 | 0.166 | 0.1168 | **0.435** | 0.392 |
| 50 | 0.139 | 0.1201 | 0.411 | 0.378 |
| 100 | 0.108 | 0.1239 | 0.395 | 0.381 |
| 150 | 0.092 | 0.1247 | 0.387 | 0.374 |
| 199 | **0.084** | **0.1264** | 0.386 | 0.369 |

Predictions 1, 2 and 3 in one table. After epoch 20:

- `train/G_L1` continues to fall, by a further 2×.
- `val/mae_norm` rises monotonically, by 8.4%.
- `val/ssim` falls from its peak.
- `val/dice_bone` peaks and drifts down.

**Every validation signal turns at the same point, and the training loss does not notice.**
That is memorisation, and it took 18 epochs out of 200. The remaining 91% of the run made
the model worse.

Note also that the learning-rate schedule (`lr_decay_start_frac: 0.5`) does not begin
decaying until epoch 100 — so from epoch 20 to epoch 100, the model trained at full
learning rate, moving steadily away from its best state.

## 2. The GAN itself was healthy

This is worth stating, because it rules out the obvious alternative explanation.

| epoch | `D_acc_real` | `D_acc_fake` | `G_GAN` |
|---|---|---|---|
| 5 | 0.506 | 0.522 | 0.284 |
| 20 | 0.896 | 0.893 | 0.582 |
| 100 | 0.894 | 0.894 | 0.630 |
| 199 | 0.922 | 0.941 | 0.661 |

`D` sits around 89–94% accuracy for the entire run — winning, but not saturating. Compare
[exp7](../exp7_reggan/), where `D_acc` is exactly 1.0000 in most epochs past 20 and `G_GAN` holds at
≈0.81, killing the adversarial gradient entirely.

exp2's adversarial game is well-balanced throughout. **The failure is not a GAN failure.**
It is overfitting plus the texture prior doing what 01 §2 says it does.

## 3. What the epoch-199 panel shows

`samples/epoch_0199.png`. Look at the four regions separately, because they tell different
stories.

**Brain axial — excellent.** Panel MAE 0.019. The skull is bright and continuous, brain
parenchyma is correctly grey, the ventricles are in the right place with the right density.
This is a genuinely good synthetic CT.

**Brain coronal — mixed.** Panel MAE 0.081. The vault is right. But the skull base and
sinus region is a chaotic field of bright and dark speckle where the real CT has structured
bone and air. The model knows something bright and complicated goes there and has produced
something bright and complicated.

**Abdomen coronal — hallucination, plainly.** Panel MAE 0.069. The real CT shows smooth
muscle with the femoral heads and pelvic ring as the only bright structures. The synth
scatters **bright bone-density speckle through the soft tissue** across the whole
abdominal wall and gluteal muscle — regions that in the real CT contain no bone at all.

**Spine — the worst.** Panel MAE 0.060/0.053, but look at the image. The real CT is a clean
sagittal column: bright vertebral bodies, dark discs, clear cortical margins. The synth is a
bright granular mess in roughly the right envelope, with the disc spaces largely filled in.

Prediction 4, and prediction 5, both confirmed — and the ordering is the ordering of the
training-slice counts from 01 §5.

## 4. Why `dice_bone` looks fine anyway

`val/dice_bone` = 0.369, the second-best in the ladder. From a model that scatters invented
bone through soft tissue.

The mechanism is in the definition (`evaluation/metrics.py:230-236`):

```python
pred_bone = ((pred >= threshold).float() * mask)
target_bone = ((target >= threshold).float() * mask)
return float(2.0 * (pred_bone * target_bone).sum() / denominator)
```

Dice is `2|A ∩ B| / (|A| + |B|)`. Spraying extra bone-bright pixels does two things: it
raises `|A ∩ B|` wherever the spray happens to land on real bone, and it raises `|A|`. The
numerator gains twice what the denominator does, so **up to a point, over-producing bone
raises Dice.** A model that hedges — [exp4](../exp4_nce_max/) — scores 0.029. A model that
sprays scores 0.369.

So `dice_bone` is doing its stated job (`metrics.py:216-220`: catching a model that
"fabricates or erases" bone) in only one direction. It catches **erasure** decisively. It
is much softer on **fabrication**, and can reward it.

Nothing in the current metric set separates "bone in the right place" from "bone in
roughly the right region". `val/mae_band_bone` — error restricted to pixels that are bone
in the *target* — is the closest thing, and at epoch 199 it is 128.9 HU (117.5 HU at the epoch-18 peak). That number is where the
hallucination shows up; Dice hides it.

## 5. The region ranking, and a caution about reading it

`val/mae_norm` at epoch 18:

| brain | abdomen | musculoskeletal | spine |
|---|---|---|---|
| 0.1177 | 0.1317 | 0.0755 | **0.2045** |

Spine is worst by 2.7×, matching prediction 5 and matching every other run in the ladder
(see [`_comparison`](../_comparison/README.md)).

But the apparent winner — musculoskeletal at 0.0755 — is an artifact of the normalisation.
In HU:

| brain | abdomen | musculoskeletal | spine |
|---|---|---|---|
| **9.4 HU** | 52.7 HU | 37.8 HU | 102.2 HU |

Brain is four times more accurate than musculoskeletal, not 1.6× worse. The per-region HU
windows differ 6.25× and `mae_norm` divides that out. This inversion holds in all seven
runs and is documented in [`_comparison`](../_comparison/README.md).

---

## What 03 has to address

Two independent problems, and they need different fixes:

1. **Overfitting** — 54.4 M parameters against 32 subjects, turning at epoch 18. This is a
   data and regularisation problem, not a λ problem.
2. **Hallucination** — the adversarial texture prior filling in where L1 is uncertain,
   worst where training data is thinnest, and partly rewarded by the metric meant to catch
   it.
