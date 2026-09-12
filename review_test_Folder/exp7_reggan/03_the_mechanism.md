# 03 — The mechanism

01 predicted three weaknesses from the objective alone. 02 produced four facts. This
document joins them.

The short version: **the warp is a low-pass filter, and the generator was trained through
it.** Everything else follows.

---

## 1. First, confirm the frame mismatch in the code

`RegGANModel` computes its loss on the warped prediction —
`model/training/reggan.py:124-138`:

```python
flow = self.netR(torch.cat([fake_B, real_B], dim=1))
warped = self.stn(fake_B, flow)
warped_mask = self.stn(mask, flow, mode="nearest") * mask
g_corr = masked_l1(warped, real_B, warped_mask, self.criteria["l1"])
g_smooth = flow_smoothness(flow, mask)
```

Now find where validation gets its image. `RegGANModel` **does not define
`generator_for_eval`** — grep the repo: `netR` appears only in `reggan.py`,
`smoke_test.py` and docstrings. So it inherits `pix2pix_nce.py:472-482`:

```python
def generator_for_eval(self):
    use_ema = self.cfg.get_path("eval.use_ema", True)
    if use_ema and self.ema is not None:
        return self.ema.module
    return self.netG
```

A bare generator. No `netR`, no `stn`. And `Trainer.validate`
(`training/trainer.py:236-254`) and `render_samples` (`:376-388`) both call it and compare
`net(real_A)` against `real_B` directly.

**Confirmed:** training scores `warp(G(x), φ)`; validation scores `G(x)`.

That is correct behaviour, not a bug — `R` needs the real CT as input
(`reggan.py:124` passes `cat[fake_B, real_B]`), and at deployment there is no real CT. As
01 §4 argued, the tether that reconciles the two frames is an un-warped L1 term. And
`configs/exp7_reggan.yaml` sets:

```yaml
lambda_l1: 0.0
```

so the `if self.plan.use_l1` branch at `pix2pix_nce.py:381-387` **never executes**. There
is zero un-warped pixel supervision in the entire run. The tether was cut.

## 2. Why the smoothness penalty did not stop it

`model/losses/registration.py:43-44`:

```python
d_x = flow[:, :, :, 1:] - flow[:, :, :, :-1]
d_y = flow[:, :, 1:, :] - flow[:, :, :-1, :]
```

Exactly the finite-difference roughness measure from 01 §3, with the property derived
there: a **constant** field has `d_x = d_y = 0` and costs nothing.

And there is no other constraint. `flow_magnitude` — the function that would notice a
large field — runs inside `with torch.no_grad():` (`registration.py:73`) and its output
goes only to `stats` (`reggan.py:147-148`). **It is a log line, never a loss.** Nowhere in
the codebase is the magnitude of `φ` penalised.

`reggan.py:93-103` argues that keeping R on the generator's own backward prevents
degeneracy:

> R and G minimise the SAME scalar. […] Here it can only reduce `L_corr` by finding a
> deformation that genuinely improves the match, because the smoothness term is in the
> same sum.

The premise is true — one scalar, one backward, merged optimisers. The conclusion does not
follow, because the smoothness term does not price the deformation this run actually
found. §3 shows what that deformation was.

## 3. The derivation: bilinear resampling is a blur

This is the part that explains the picture.

`SpatialTransformer.forward` (`networks/registration.py:81-101`) resamples with
`F.grid_sample(..., mode="bilinear")`. Take one axis and one pixel. To read the image at
position `i + a` for a fractional offset `a ∈ [0, 1)`, bilinear interpolation returns

```
out[i] = (1 − a) · in[i]  +  a · in[i+1]
```

That is a two-tap FIR filter with weights `(1−a, a)`. It has a frequency response. Put in
a sinusoid `in[i] = e^{iωi}`:

```
H(ω) = (1 − a) + a·e^{iω}
|H(ω)|² = (1−a)² + a² + 2a(1−a)·cos ω
```

Evaluate at the **Nyquist frequency** `ω = π` — a signal that alternates every pixel,
`+1, −1, +1, −1`:

```
|H(π)|² = (1−a)² + a² − 2a(1−a) = (1 − 2a)²
|H(π)|  = |1 − 2a|
```

Read that off:

| offset `a` | `|H(π)|` | effect on a 2-px pattern |
|---|---|---|
| 0.0 | 1.0 | untouched — no resampling |
| 0.25 | 0.5 | halved |
| **0.50** | **0.0** | **annihilated completely** |
| 0.75 | 0.5 | halved |
| 1.0 | 1.0 | untouched — a pure integer shift |

**A half-pixel offset places an exact zero at Nyquist.** And it is not only Nyquist. At
`a = 0.5` the whole response collapses to

```
|H(ω)| = |cos(ω/2)|
```

| spatial period | attenuation |
|---|---|
| 2 px | **100%** — gone |
| 3 px | 50% |
| 4 px | 29% |
| 8 px | 8% |

A half-pixel resample is a strong, non-optional low-pass filter. In 2-D the axes multiply,
so an offset of `(0.5, 0.5)` annihilates the diagonal checkerboard and heavily attenuates
everything else fine.

### The measurement

An offset of half a pixel in *each* axis has displacement magnitude

```
|φ| = √(0.5² + 0.5²) = 0.7071
```

`train/R_flow_px` — the mask-weighted mean of `|φ|` in pixels
(`losses/registration.py:62-78`), and 1 px = 1 mm here — converged in three epochs and
then held for 197:

```
0.592 → 0.697 → 0.709 → … → 0.710 → 0.699 → 0.699
```

**0.699 against a predicted 0.707. Within 1.1%.**

The registration network did not converge to the anatomical misalignment. It converged to
**the sub-pixel offset at which bilinear interpolation destroys the most high-frequency
content**, and stopped there.

Treat this as strong circumstantial evidence rather than proof: `R_flow_px` is a mean over
pixels, and a mean of 0.699 is consistent with per-axis offsets clustered near ±0.5 but
does not by itself demonstrate it. 04 §1 gives the ten-line experiment that settles it.

## 4. Why both networks wanted this

Neither network was "cheating". Both were descending the same scalar, and this is where it
went.

**For R.** Early in training `G(x)` is wrong, and much of that error is high-frequency —
mismatched edges, texture in the wrong place. `|warp(G(x), φ) − y|₁` falls if the warp
attenuates that error. Gradient descent on `φ` therefore has a direct downhill path toward
the half-pixel offset, and the smoothness penalty does not object because a smooth,
near-constant half-pixel field is almost perfectly smooth. R took that path in **three
epochs** and never left.

**For G.** Once the loss is computed through `|cos(ω/2)|`, the top of the spectrum is in
the loss's **null space**. The gradient with respect to the Nyquist component of `G`'s
output is exactly zero, and the gradient for everything above ~4 px period is heavily
attenuated. So:

- **Bone edges get almost no gradient.** A cortical bone boundary is a step — its energy
  is concentrated at exactly the frequencies the filter removes. The generator is not
  being told it got them wrong, because after the warp, it did not. Hence the featureless
  grey blob: the model is a good fit to a *blurred* target.
- **The checkerboard is free.** A 1–2 px grid sits precisely on the null. It costs the
  training loss nothing. And a U-Net decoder built from 4×4 stride-2 transposed
  convolutions (`networks/unet.py:91-111`) has a well-known intrinsic bias toward
  producing exactly that pattern — uneven kernel overlap, the classic checkerboard
  artifact. In every other experiment `λ_L1 = 100` penalises it into nonexistence. Here
  nothing does, so the architecture's own bias expresses itself unopposed.

**The artifact you can see is not something the model learned. It is something nothing
stopped.** That is why it is fully formed by epoch 4 and unchanged at epoch 199 (02 §6):
it appeared as soon as the filter did, and no later gradient ever touched it.

## 5. This closes every fact from 02

| fact | explanation |
|---|---|
| `G_corr` = 0.0092, 9× better than any paired L1 in the ladder | it is measured on the **post-filter** image. A blurred prediction against a blurred target is an easy comparison. The number is real and means nothing. |
| Fully formed by epoch 4, no discriminator | the filter is established in 3 epochs, before `gan_warmup_epochs: 5` ends. D is irrelevant to the cause. |
| `R_flow_px` froze at 0.699 | = √(0.5²+0.5²) = 0.707, the maximum-attenuation offset. R reached the bottom of its well and stayed. |
| The artifact is a 1–2 px checkerboard specifically | that is the exact frequency the filter nulls, and the exact artifact the decoder architecture already tends to produce. |
| `val/mae_norm` best at epoch 0 | the head is initialised to `N(0, 1e-5)` with zero bias (`registration.py:165-166`), so the first steps really are plain L1 — `registration.py:160-164` says so. By the end of epoch 0, `R_flow_px` is already 0.592 and the well has been entered. There was never a later epoch that could beat it. |
| D effectively perfect, `G_GAN` held at ≈0.81 | consequence, not cause. Distinguishing a grey checkerboard from a real CT is trivial, so D wins immediately and its gradient vanishes. It could not have rescued the run. |

## 6. What this run actually teaches

Three transferable lessons, in increasing order of how much they should change your
practice.

**One.** `model/README.md:116-118` says the failure to watch for is "`R_flow_max` growing
without bound." It did not grow — it *shrank*, 1.82 → 1.51 — and the run failed anyway.
The dangerous field here was **small, smooth, and constant**: the one shape the smoothness
penalty is blind to and the one a bounded-magnitude alarm would never catch. A monitor
written against the wrong failure is worse than none, because it reads as reassurance.

**Two.** Any loss taken through a learned resampling is a loss taken through a learned
filter. If you let a network choose the sampling grid, you have let it choose which
frequencies you are allowed to be graded on — and it will choose the ones it is already
bad at. This applies to every spatial-transformer, deformable-registration or
learned-warp objective, not just RegGAN.

**Three.** `train/G_corr` fell smoothly by 22× over 200 epochs. There is no wobble, no
plateau, no spike — it is the healthiest-looking loss curve of all seven runs in this
ladder. **A loss curve can only tell you that the number you wrote down is going down.**
The only thing that caught this was validating in the frame you will actually deploy in,
and even that only caught it because someone read `is_best` and noticed it said epoch 0.
