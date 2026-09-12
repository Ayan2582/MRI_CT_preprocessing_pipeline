# 01 — Why 8:1 is the stable point

Three runs bracket this one, and each fails in a different direction. Understanding why
this one sits between them is the most useful thing in the ladder.

---

## 1. The three failures it is between

| | λ_L1 : λ_NCE | how it fails |
|---|---|---|
| exp2_paper | 100 : 1 | **overfits** — peaks at epoch 18, then 181 epochs of decline |
| **exp3b_nce_6_pix48** | **48 : 6** | — |
| exp3b_nce_20_5 | 20 : 5 | **collapses** — bone gone, `dice` 0.038 |
| exp4_nce_max | 10 : 5 | collapses harder |

Two different failure modes, on opposite sides. That shape — bad, good, bad — means there
is a real trade being made, not just a monotone "more L1 is better".

## 2. What the trade is

From [exp4/01 §5](../exp4_nce_max/01_the_idea.md), the three terms do different jobs:

```
λ_L1 · L₁          absolute intensity  — the only term that knows a Hounsfield unit
λ_NCE · L_NCE      correspondence      — provably indifferent to intensity
λ_GAN · L_adv      texture plausibility — indifferent to correctness
```

Now think about what each does to **generalisation**, which is the axis exp2 fails on.

**L1 is the term that can be memorised.** It is a per-pixel regression against a specific
target image. With 54.4 M parameters and 32 independent subjects
([exp2/01 §4](../exp2_paper/01_the_idea.md)), driving L1 to zero on the training set is
achievable, and achieving it *is* memorisation. exp2's `train/G_L1` reaches 0.084, and its
validation error rises for the last 181 epochs while it does so.

**PatchNCE is much harder to memorise.** Its negatives are drawn from within the same image
(`losses/patch_nce.py:104-113`), and the sampled locations are re-drawn every step
(`networks/patch_sampler.py`). So the task changes every iteration: it is never the same
257-way classification twice. There is no fixed target to store. What it can learn is a
*general* rule about which encoder features correspond — which is the thing that transfers
to a new patient.

So the two terms differ not just in what they constrain but in **how they fail with too
much weight**:

- Too much L1 → the model memorises the training set, and validation turns early.
- Too little L1 → nothing carries intensity, and bone collapses
  ([exp4/03](../exp4_nce_max/03_the_mechanism.md)).

**λ_NCE is acting as a regulariser on λ_L1.** Not by penalising weights, but by spending
gradient budget on an objective that cannot be memorised. That is why the middle of the
range is stable and both ends are not.

## 3. The evidence that this is what happened

**`train/G_L1` was driven down, but not all the way.**

| epoch | 0 | 20 | 50 | 100 | 199 |
|---|---|---|---|---|---|
| exp2 (λ 100) | 0.358 | 0.166 | 0.139 | 0.108 | **0.084** |
| exp3b_6_48 (λ 48) | 0.405 | 0.184 | 0.145 | 0.123 | **0.097** |
| exp3b_20_5 (λ 20) | 0.449 | 0.347 | 0.330 | 0.319 | **0.297** |

exp3b_6_48 lands at 0.097 — only 15% above exp2's 0.084, and **three times** below the
collapsed run. It is clearly on the same side of the cliff as exp2. The λ_L1 reduction from
100 to 48 barely changed how well the L1 term was fitted; it changed how much of the
optimiser's attention went to a memorisable objective.

**And the validation curve stopped turning over.**

| epoch | 20 | 50 | 100 | **121** | 150 | 199 |
|---|---|---|---|---|---|---|
| `val/mae_norm` | 0.1218 | 0.1214 | 0.1211 | **0.1203** | 0.1208 | 0.1214 |

Flat to four decimal places from epoch 20 onward, with a shallow minimum at 121. Compare
exp2, which rises 8.4% over the same span. This is what a run that has stopped memorising
looks like: it converges and then sits there.

**Bone appeared fast and mostly stayed.** `val/dice_bone`: 0.073 at epoch 5, **0.405 by
epoch 10**, then 0.401 / 0.405 / 0.405 / 0.391 / 0.385 through to the end. Ten epochs to
learn bone, and a slow 5% erosion over the remaining 190 — which is the residual
overfitting that λ_NCE reduced but did not eliminate.

## 4. Why the peak accuracy is slightly lower, and why that is fine

exp2's best is 0.1166; this run's is 0.1203. A 3% gap.

That is the cost of the trade. Some of exp2's peak accuracy at epoch 18 is genuine, and
some of it is a model that has begun to fit its 32 subjects specifically. exp3b_6_48 spends
part of its budget on a term that cannot fit those subjects specifically, so it does not
reach quite as low.

But look at what you actually get to *use*:

| | exp2_paper | exp3b_nce_6_pix48 |
|---|---|---|
| best `mae_norm` | 0.1166 @ ep18 | 0.1203 @ ep121 |
| final `mae_norm` | **0.1264** | **0.1214** |
| `dice_bone` at final | 0.369 | **0.385** |

**At the last epoch — the checkpoint you get if you simply train the run and take what
comes out — exp3b_6_48 is better on both metrics.** exp2 only wins if you know to reach
back to epoch 18, and knowing that requires having watched a validation curve that was
already telling you the run was going wrong.

A configuration whose final checkpoint is its best checkpoint is worth real accuracy,
because it removes a decision you can get wrong.

## 5. The transferable rule

> When one term in a multi-term loss is memorisable and another is not, the ratio between
> them is a regularisation hyperparameter, whether or not you were treating it as one.

`λ_L1 : λ_NCE` was framed in this ladder as a question about *what to constrain* —
`model/README.md:73` calls exp3 "lean on NCE — open question". The runs say it is also, and
maybe mostly, a question about *how much memorisation to permit*. The cliff at 4:1 is where
intensity supervision fails; the drift at 100:1 is where memorisation begins; and 8:1 is
between them.

That is a real answer to the open question, and it was worth the four runs it took.
