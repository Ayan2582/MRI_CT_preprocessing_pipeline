# 04 — What to change

Recommendations only; nothing here has been applied.

---

## 1. First, run the experiment that confirms 03

Ten lines, no retraining, uses the checkpoint you already have. Do this before changing
anything, because it separates "the warp is a filter" from "something else is wrong".

```python
# load exp7's checkpoint into netG and netR, eval mode
fake_B = netG(mri)                                   # what validation scores
flow   = netR(torch.cat([fake_B, real_ct], dim=1))   # reggan.py:124
warped = stn(fake_B, flow)                           # reggan.py:130

print("unwarped L1:", masked_l1(fake_B, real_ct, mask, l1).item())   # expect ~0.30
print("warped   L1:", masked_l1(warped, real_ct, mask, l1).item())   # expect ~0.01
print("flow frac  :", (flow.abs() % 1.0).mean().item())              # expect ~0.5
```

Three predictions, each of which can falsify 03:

- **The warped L1 is ~30× lower than the unwarped one.** If they are close, the frame
  mismatch is not the story.
- **The checkerboard disappears from `warped`** while remaining in `fake_B`. Save both as
  PNGs and look. This is the direct visual test.
- **The per-axis fractional part of the flow clusters near 0.5.** This is the claim 03 §3
  labels as circumstantial; `(flow.abs() % 1.0)` settles it. If it clusters near 0.0
  instead, the maximum-blur reading is wrong and the mechanism needs revisiting.

Also worth plotting: a radially-averaged power spectrum of `fake_B` and of `warped`. The
prediction is a sharp notch at Nyquist in `warped` that is absent in `fake_B`.

## 2. Put the tether back — the one change that matters

```yaml
loss:
  lambda_l1: 100.0     # was 0.0
  lambda_corr: 100.0
  lambda_smooth: 10.0
```

This is the whole fix for the structural problem. RegGAN's warped term answers "is the
anatomy right, allowing for misalignment?"; the plain L1 term answers "is the image right
in the coordinate frame you will actually be graded in?". You need both, because you train
with `R` and you deploy without it.

With `λ_L1: 100` alongside, the half-pixel offset stops being free: any blur R introduces
is now charged in full by the un-warped term. R can still absorb genuine misalignment,
which is the point of the experiment, but it can no longer buy loss reduction with a
filter.

**This also restores exp7's stated purpose.** The ladder describes exp7 as the run that
"produces a measurement rather than just a score" — the measurement being `R_flow_px` in
millimetres. As it stands that number is 0.699 mm of *interpolation exploit*, not 0.699 mm
of anatomical residual, so the experiment produced no measurement at all. With the tether
in place, `R_flow_px` means what it was supposed to mean.

## 3. Penalise the magnitude of the field, not only its roughness

The gap derived in 01 §3 and confirmed in 03 §2: `flow_smoothness` prices raggedness, and
a constant offset of any size is free. Add a magnitude term:

```
L_R = λ_smooth · S(φ)  +  λ_mag · mean(|φ|²)
```

`flow_magnitude` (`losses/registration.py:62-78`) already computes almost exactly this —
it just runs under `torch.no_grad()` and is only logged. Making a differentiable sibling of
it is a few lines.

Start `λ_mag` small (0.1–1.0). The prior you are encoding is "the residual misalignment is
a few millimetres at most", which your QC record supports, and which is a much more
specific claim than "the field is smooth."

## 4. Fix the monitor

`model/README.md:116-118` tells the reader to watch `R_flow_max` for unbounded growth.
That alarm cannot fire on this failure — the dangerous field was small and smooth. Replace
it with the two checks that would have caught it within five epochs:

- **`train/G_corr` falling while `val/mae_norm` rises.** True from epoch 1. This is the
  frame mismatch made visible and costs nothing to log; the columns already exist.
- **`is_best` still pointing at an epoch below ~5 once you are past epoch 20.** exp7's
  `is_best` was `True` at epoch 0 and never again. A run whose best checkpoint is its
  first is not converging slowly; it is not converging.

Neither needs new instrumentation — both are in `metrics.csv` already.

## 5. Then reconsider the architecture

Only after 2–4. The checkerboard is a decoder artifact (`networks/unet.py:91-111`, 4×4
stride-2 `ConvTranspose2d`) that `λ_L1: 100` normally suppresses. Once the tether is back
it should vanish on its own. If it does not, replace the transposed convolutions with
nearest-neighbour or bilinear upsampling followed by a 3×3 convolution — the standard
remedy, and it removes the uneven kernel overlap that causes the pattern rather than
training against it.

Do not do this first. If you change the decoder now you will suppress the *symptom*, the
panels will look plausible, and the frame mismatch will still be there — silently costing
accuracy instead of announcing itself.

---

## What a successful re-run looks like

Not a low `G_corr`. exp7 already has the lowest reconstruction loss in the ladder and it
means nothing.

| signal | what to expect |
|---|---|
| `is_best` | lands somewhere past epoch 50 |
| `val/mae_norm` vs `train/G_corr` | fall **together** |
| `R_flow_px` | settles at a value that is *not* ≈0.707 — and whatever it settles at is then a real measurement in mm |
| the panels | bone visible, no 1–2 px grid |
| vs exp1 | the honest question this experiment exists to ask — does correcting residual misalignment beat not correcting it? |

If `R_flow_px` converges near 0 with the tether in place, that is a *result*: the QC pairs
really are aligned, exp3's hedge was against nothing, and exp2 stays the target. That
outcome is worth just as much as the alternative, and right now you have neither.
