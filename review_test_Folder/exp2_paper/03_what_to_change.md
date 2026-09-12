# 03 — What to change

Two independent problems needing two different fixes. Recommendations only; nothing
applied.

---

## 1. Overfitting: stop trying to fix it with λ

The ratio from 01 §4 — 54.4 M parameters against 32 independent subjects — is not something
a loss weight repairs. Four things that do help, roughly in order of effort:

**Use the best checkpoint, and say which one.** `eval.selection_metric: mae_norm` with
`selection_mode: min` already writes `best.pt`, and for exp2 that is epoch 18. Quote 0.1166,
not 0.1264. This costs nothing and is the difference between reporting exp2 as 8.4% worse
than it is and reporting it accurately.

**Stop early.** 200 epochs is 180 epochs of harm here. Even a crude patience rule — stop
after 40 epochs with no improvement in `val/mae_norm` — would have ended this run around
epoch 60 and saved two thirds of the compute.

**Shrink the model.** `ngf: 32` instead of 64 quarters the generator to ~14 M parameters.
On 32 subjects that is very unlikely to cost accuracy and may help. This is the single
highest-value untried experiment in the whole ladder, and it is one config line.

**Augment more.** `augment.hflip` is `false` in the current base config. Left-right flip is
anatomically legitimate for brain, spine and most musculoskeletal slices, roughly doubles
the effective dataset, and costs nothing. (Note it was `true` in exp1 — see
[exp1_pix2pix](../exp1_pix2pix/) for why that makes exp1 non-comparable, but it is not a
reason to leave it off going forward.) Small rotations are the obvious next addition,
tempered by the note in `base.yaml` that in-plane rotation residual is exactly what the QC
could not correct.

## 2. Hallucination: the λs are close to right; the metric is not

exp2's `λ_L1: 100, λ_NCE: 1` is not the problem — it is above the cliff described in
[exp4](../exp4_nce_max/), and it produces bright, plentiful bone.

The problem is that nothing in the metric set distinguishes bone in the right place from
bone in roughly the right region, and `dice_bone` actively rewards over-production (02 §4).
Three additions, none of which requires retraining anything:

**Report a false-positive bone rate.** Alongside Dice, report
`|pred_bone \ target_bone| / |target_bone|` — the fraction of invented bone relative to real
bone. That is the number that separates exp2 from a model that places bone correctly, and
it is two lines next to the existing `bone_dice` in `evaluation/metrics.py:212-236`.

**Promote `mae_band_bone`.** It already exists and already sees this: 128.9 HU at epoch 199
against 117.5 HU at the epoch-18 peak. It is currently reported and ignored.

**Add precision and recall, not just Dice.** Dice is their harmonic mean, and it conceals
which of the two is failing. exp2 and [exp4](../exp4_nce_max/) fail in opposite directions
— exp2 over-produces, exp4 under-produces — and Dice alone cannot tell you which you are
looking at.

## 3. Compare against exp3b_nce_6_pix48 before deciding anything

| | exp2_paper | exp3b_nce_6_pix48 |
|---|---|---|
| λ_L1 : λ_NCE | 100 : 1 | 48 : 6 |
| best `val/mae_norm` | **0.1166** @ ep18 | 0.1203 @ ep121 |
| drift, best → final | **+8.4%** | **+0.9%** |
| `val/dice_bone` | 0.369 | **0.385** |

exp2 wins on peak accuracy by 3%. exp3b_6_48 wins on bone, and does not degrade — its peak
is at epoch 121 rather than 18, so it is a run you can actually train to completion and
ship the last checkpoint of.

For a model you intend to *use*, that stability is worth more than 3% of `mae_norm`.
See [exp3b_nce_6_pix48](../exp3b_nce_6_pix48/) for why the heavier NCE weight buys it.

## 4. Spine: stop reporting it as a model result

`val/mae_norm/spine` is 0.2045 — 102.2 HU. Spine is **27 of 1687 training slices (1.6%)**,
and it is the worst region in all seven runs in this ladder without exception
([`_comparison`](../_comparison/README.md)).

That is a dataset fact, not a finding about pix2pix. No λ, architecture or objective change
in this ladder moved it. Either collect more spine data, exclude spine from the headline
number and say so, or report it separately as a known-underpowered region. Quoting a
whole-ladder average that includes it makes every model look worse for the same reason.

---

## What a successful re-run looks like

| signal | target |
|---|---|
| `is_best` | past epoch 60 — evidence the model is still generalising, not memorising |
| drift, best → final | under 2%, as exp3b_6_48 achieves |
| `val/mae_band_bone` | below ~110 HU |
| false-positive bone rate (if added) | falling while Dice holds — the direct test that hallucination is reducing |
| the abdomen coronal panel | smooth muscle, with bone only where the real CT has bone |

The first two are free: they come from stopping early and from `ngf: 32`. Try those before
touching a single λ.
