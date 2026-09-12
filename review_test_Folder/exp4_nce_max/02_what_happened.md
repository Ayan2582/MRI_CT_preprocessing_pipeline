# 02 — What happened

Evidence only. From `results/exp4_nce_max_results/metrics.csv` and `samples/`.

All six predictions at the end of 01 can be checked here. They all hold.

---

## 1. Bone did not merely degrade — it disappeared

`val/dice_bone`, the overlap between predicted and real pixels above 150 HU:

| epoch | 0 | 5 | 10 | 20 | 50 | 100 | 150 | 199 |
|---|---|---|---|---|---|---|---|---|
| `dice_bone` | 0.0003 | 0.0001 | **0.0000** | **0.0000** | 0.0028 | 0.0532 | 0.0355 | **0.0290** |

It is **exactly zero for epochs 13 through 48**, and below 0.0002 from epoch 5 to 50. Zero Dice does not mean
misplaced bone; the numerator is `2·|pred ∩ true|`. It means the prediction contained
**no pixel above the bone threshold at all**, anywhere in 159 bone-eligible validation
slices.

The comparison that isolates the cause — same architecture, same data, same 200 epochs,
only λ changed:

| run | λ_L1 | λ_NCE | final `dice_bone` |
|---|---|---|---|
| exp2_paper | 100 | 1 | 0.369 |
| exp3b_nce_6_pix48 | 48 | 6 | **0.385** |
| exp3b_nce_20_5 | 20 | 5 | 0.038 |
| **exp4_nce_max** | **10** | **5** | **0.029** |

## 2. The L1 term never got driven down

`train/G_L1` over the run:

| epoch | 0 | 5 | 20 | 50 | 100 | 199 |
|---|---|---|---|---|---|---|
| exp4 (`λ_L1` 10) | 0.479 | 0.400 | 0.369 | 0.356 | 0.341 | **0.322** |
| exp2 (`λ_L1` 100) | 0.358 | 0.232 | 0.166 | 0.139 | 0.108 | **0.084** |

exp2 reduces its L1 by 4.3×. exp4 reduces its by 1.5× and finishes **3.8× worse**.

This is prediction 5, and it is the mechanical heart of the run: with a weight of 10, the
L1 term is simply not what the optimiser is following.

## 3. Meanwhile the NCE term was satisfied normally

`train/G_NCE`: 2.799 → 1.784. It fell steadily and behaved exactly as it does in the runs
that worked. Per layer at epoch 199:

```
L0 5.089    L1 1.441    L2 1.185    L3 0.825    L4 0.379
```

Layers 1–4 are learning. **Layer 0 is not** — 5.342 → 5.089, against a chance level of
`ln(256 + 1) = 5.549` for a 257-way classification. It starts near chance and stays near
chance. The same is true in all four NCE runs (see
[`_comparison`](../_comparison/README.md)).

The contrastive objective was healthy. It was just not the objective that needed to hold.

## 4. The panel: right anatomy, no dynamic range

`samples/epoch_0199.png`. This is the clearest picture in the whole ladder of what a
failure mode actually *is*.

**What is right:**

- Every structure is in the correct place. The pelvis, the muscle compartments, the
  vertebral column, the brain, the sinuses — all present and correctly positioned.
- Edges are crisp. This is not a blurry image.
- The abdomen coronal and the musculoskeletal slices are anatomically excellent.

**What is wrong:**

- **The entire image is rendered in a narrow band of mid-grey.** The brain axial synth is
  a uniform grey oval. The real CT beside it has a saturated white skull ring, and clearly
  separated grey and white matter. The synth has neither.
- The musculoskeletal cortical bone — brilliant white in the real CT — is *barely
  distinguishable* from the muscle around it.
- The spine vertebrae are visible as faint texture, not as bright bone.
- **The error maps outline the bone.** Every bright structure in column 4 is a skull, a
  cortical rim, or a vertebral body. The soft tissue is dark, i.e. correct.

The model knows exactly where the bone is. It declines to make it bright.

Predictions 1, 2, 3 and 4 from 01, in one image.

## 5. The global metrics barely notice

| metric | exp2 (best) | exp4 (best) | ratio |
|---|---|---|---|
| `val/mae_norm` | 0.1166 | 0.1666 | 1.43× worse |
| `val/ssim` | 0.435 | 0.296 | 1.47× worse |
| **`val/dice_bone`** | **0.369** | **0.029** | **12.7× worse** |

Prediction 6. The averaged metrics register a moderate degradation. The metric that looks
at bone registers a collapse. Read `mae_norm` alone and you would file exp4 as
"noticeably worse"; it is in fact clinically unusable, and `dice_bone` is the only column
that says so.

`evaluation/metrics.py:216-220` says this is exactly what `bone_dice` was added to catch.
It worked.

## 6. The run was still improving when it stopped

`val/mae_norm` fell monotonically to the last epoch — 0.2235 → 0.1917 → 0.1743 → 0.1666,
with `is_best` at **epoch 199**.

So exp4 is not overfit. It is not diverged. It was still learning, slowly, and the
learning-rate schedule ran out. It is **under-trained and mis-weighted at the same time**,
and no amount of extra epochs would have fixed the second problem.

---

## The three things 03 has to prove

1. That PatchNCE **cannot** express a preference about absolute intensity — not "does not
   in practice", but cannot, from the form of the loss.
2. That a weak L1 term **provably** fails to reach bone, with the threshold arithmetic
   worked out in the units this repo actually uses.
3. Why the collapse is so much worse at `λ_L1: 10` and `λ_L1: 20` than at `λ_L1: 48` —
   i.e. why there is a **cliff** rather than a gradual decline.
