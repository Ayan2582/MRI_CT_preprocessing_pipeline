# 01 — Why this score is not portable

---

## 1. What `mae_norm` actually averages over

`evaluation/metrics.py:127-135`:

```python
def metric_mae_norm(pred, target, mask, hu_min, hu_max):
    return float(_masked_mean((pred - target).abs(), mask))
```

A mean of `|pred − target|` over the pixels where `mask` is 1.

Now: what is `mask`? From `data/dataset.py:65-79`, it is 1 on **original pixels** and 0 on
**zero-padding** added to reach a common size. That is the right thing to exclude — padding
is identical in prediction and target, so including it would make a small slice look better
than a large one for reasons having nothing to do with the model.

But note what the mask does **not** exclude: the **air around the patient inside the
original frame**. That is real acquired image data, so `mask = 1` there, and it enters the
average.

Air is the easiest pixel in the image. It is black in the MRI and black in the CT. Any model
past its first few hundred steps predicts it essentially perfectly.

## 2. So the denominator is a mixture, and its composition matters

Write the mean out as a weighted combination:

```
mae_norm  =  f_bg · e_bg  +  f_anat · e_anat
```

where `f_bg` is the fraction of valid pixels that are background, `e_bg ≈ 0` is the error
there, and `e_anat` is the error on anatomy. So

```
mae_norm  ≈  (1 − f_bg) · e_anat
```

**The score is the model's actual error, scaled down by however much background happened to
be in frame.**

That is fine — desirable, even — as long as `f_bg` is the same for every run you compare.
It is not a property of the model; it is a property of the *cropping*.

## 3. What ROI cropping does to `f_bg`

`use_roi: true` resolves to mode `"crop"`, which physically slices each slice down to the
QC reviewer's bounding box (`data/dataset.py:239-249`) — a box drawn to exclude scanner
table rails and empty air.

That is exactly a reduction in `f_bg`. It removes easy background pixels from the average
and raises the proportion of hard anatomical ones.

**Same model, same predictions, higher `mae_norm`.** The metric got harder; the model did
not get worse.

exp1 predates that change. Every other run has it.

## 4. Measuring the size of the effect

Here is the trick that separates task difficulty from model quality: **look at epoch 0.**

After one epoch the models have barely learned anything and are all doing roughly the same
crude thing. Whatever difference remains is mostly the difficulty of the measurement, not
the skill of the model.

| run | λ_L1 | pipeline | `val/mae_norm` at epoch 0 |
|---|---|---|---|
| exp1_pix2pix | 100 | pre-ROI, hflip on | **0.1024** |
| exp2_paper | 100 | current | **0.1468** |

Same λ_L1, same architecture, same 32 subjects, same 230 validation samples. The gap is
**+43%**, and it is present before either model has done anything interesting.

Scale exp1's best score by that factor:

```
0.0763 × 1.43  ≈  0.109
```

which sits between exp2 (0.1166) and exp3b_nce_6_pix48 (0.1203) — not 35% ahead of them.

**Treat 0.109 as an estimate with real uncertainty, not a corrected number.** Two caveats,
both of which should keep you from quoting it:

- exp2 also has `λ_NCE: 1`, which exp1 does not. At epoch 0 that should matter little, but
  it is not zero, so 43% is an *upper* bound on the pipeline effect.
- The relationship between `f_bg` and the final converged error is not exactly linear.
  Multiplying by 1.43 assumes it is.

The honest statement is: **exp1's lead is largely or entirely explained by the measurement
change, and the residual is within the noise of this estimate.** That is a much weaker claim
than "exp1 wins", and it is the one the data supports.

## 5. The second difference, which also favours exp1

`augment.hflip: true` in exp1; `false` everywhere else.

Horizontal flip roughly doubles the effective training set. Given
[exp2/01 §4](../exp2_paper/01_the_idea.md) — 54.4 M parameters against **32 independent
subjects** — doubling the effective data is not a minor perturbation. It is a direct attack
on the binding constraint.

And it shows. exp1 is the only run whose validation curve behaves like a well-regularised
model: best at epoch **72** with only 2.2% drift to the end, against exp2's epoch 18 and
8.4% drift. exp2 and exp1 have *identical* λs; the entire difference in overfitting
behaviour is the pipeline.

So exp1 had an easier metric **and** more augmentation. Both push the same way.

## 6. What is still true about exp1

None of this makes the run worthless. Two claims survive intact:

**`val/mae_hu/brain` = 5.5 HU.** HU is an absolute physical unit — it does not depend on how
much background was in frame, and it does not depend on the region's normalisation window.
This is the best brain accuracy in the ladder and it is directly comparable to published
synthetic-CT work.

**`val/ssim` = 0.615**, against 0.386 for exp2 and 0.355 for exp3b_6_48. SSIM is computed
locally over sliding windows, so it is far less sensitive to `f_bg` than a global mean is.
A gap that large is unlikely to be entirely artifactual.

Both are consistent with exp1 being a good run. Neither establishes that it is the *best*
run, because no other run has been measured under the same conditions.

---

## The transferable rule

> A metric averaged over a region is only comparable across runs that define the region the
> same way.

`mae_norm` is masked, which correctly handles padding — and that correctness is exactly what
makes it easy to assume it handles everything. It does not normalise for field of view, and
nothing in the number announces which field of view produced it.

The general defence is the one used in §4: **check the epoch-0 score.** If two runs disagree
before they have learned anything, they are not solving the same problem, and their final
numbers do not belong in the same column.
