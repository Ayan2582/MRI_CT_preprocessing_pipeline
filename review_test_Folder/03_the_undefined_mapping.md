# 03 — The mapping exp8 asks for is not a function

**Finding 2, and the one that matters.** The generator is asked to produce an output
whose correct value depends on information it is never given. No amount of λ tuning,
longer training, or architecture swapping fixes this, because the target is not
well defined.

---

## The failure

Your CT is normalised with a **per-region HU window**, frozen into the manifest at
preprocessing time. `Preprocessing/pipeline_config.py:208-229`:

```python
REGION_PROFILES = {
    "brain":           {"ct_win_min":    0.0, "ct_win_max":  80.0},
    "abdomen":         {"ct_win_min": -160.0, "ct_win_max": 240.0},
    "musculoskeletal": {"ct_win_min": -200.0, "ct_win_max": 300.0},
    "spine":           {"ct_win_min": -200.0, "ct_win_max": 300.0},
}
```

Confirmed against `model/data/manifest.csv` — every row of a region carries that
region's window and no other. The clip-and-rescale is
`Preprocessing/normalization.py:3-18`, then `dataset.py` maps `[0,1] → [-1,1]`.

So one normalised unit is:

| region | HU span across the full [0,1] range | 1 normalised unit = |
|---|---|---|
| brain | 80 HU | 80 HU |
| abdomen | 400 HU | 400 HU |
| musculoskeletal / spine | 500 HU | 500 HU |

**The same normalised pixel value means a different tissue in each region**, and a
region's own appearance changes with it. A brain CT windowed 0–80 has its skull
saturated hard at 1.0 — look at the real-CT column of the brain panels, the bone is
pure flat white with no internal structure. An abdomen CT windowed −160–240 does not
saturate: its cortical bone is bright but still textured.

## Why that is fatal specifically for exp8

Ask what the generator knows. In exp8 it is handed `real_A` — one MRI slice, one
channel — and nothing else:

- `lambda_l1: 0.0`, so there is no target image supplying the answer.
- `model.discriminator.conditional: false`, and `training/cyclegan.py:63-83` **raises a
  `ValueError` if you try to set it true**. So D_B never sees the MRI either; it judges
  the CT alone.
- `body_region` *is* in the batch (`data/dataset.py:402`) and *is* used by the metrics
  — and never reaches any network.

So: two anatomically similar MRI inputs, one from a brain series and one from a
musculoskeletal series, require outputs on scales differing by a factor of 6.25, and the
generator has no way to tell them apart. **The relation it is being asked to learn is
one-to-many. It is not a function, and a deterministic network cannot represent it.**

What a network does when trained on a one-to-many relation is average. That is exactly
what epoch 199 shows: the brain axial synth CT is flat mid-grey with a weak rim, where
the real brain CT has a saturated white skull and clearly separated grey/white matter.
The generator hedged between windows it cannot distinguish.

## Why every other experiment is fine

This is worth being precise about, because it is not a preprocessing bug — the
per-region windows are the right call, and the rest of the ladder depends on them.

| experiment | how it resolves the ambiguity |
|---|---|
| exp0 | `lambda_l1: 100` — the target *is* the answer, at the right window |
| exp1–exp4 | L1 as above, **plus** a conditional D seeing `cat[MRI, CT]` |
| exp5, exp6 | conditional D; exp6 also has L1 |
| exp7 (RegGAN) | `lambda_corr: 100` on the registered target — still paired supervision |
| **exp8** | **nothing** |

exp8 is the only rung that removes *both* the paired target and the conditional
discriminator. Those were the only two channels through which region information ever
reached the generator, and the ladder removed them in the same step.

The `conditional: false` refusal at `cyclegan.py:63-83` is correct on its own terms —
its comment says a conditional D "would be trained on a relationship absent from the
data," which is true, the MRI and CT come from different patients. The problem is that
closing that door also closed the only remaining path for the region signal, and nothing
was put in its place.

## The knock-on effect on your metrics

`val/mae_norm` is described in `evaluation/metrics.py:127-135` as "comparable across
regions in a way that no HU-denominated number is, because it does not carry a
region-dependent scale factor." For *ranking model checkpoints* that is right.

For reading exp8's failure it hides the severity. A panel MAE of 0.10 is:

- **8 HU** on a brain slice — clinically negligible.
- **50 HU** on a musculoskeletal slice — the entire difference between muscle (~50 HU)
  and fat (−100 HU), or between muscle and trabecular bone.

The two musculoskeletal slices are the ones that got *worse* over training (0.059 →
0.070), and they are on the 500 HU window. In HU that regression is +5.5 HU of error on
top of an already-large 35 HU. `val/mae_hu/musculoskeletal` in your `metrics.csv` is the
column that shows this honestly; `mae_norm` does not.

## What this means for the fix

Findings 1, 3, 4 and 5 are all worth fixing and all improve the run. This one gates
them: **until the generator is told which region it is translating, exp8 cannot succeed
at any λ.** 06 §2 gives the two ways out, and they are cheap — the information is
already in the batch.
