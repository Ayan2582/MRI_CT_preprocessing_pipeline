# RegGAN — taking the loss in a learned frame

pix2pix compares pixel `(i, j)` of the prediction to pixel `(i, j)` of the target.
RegGAN inserts a learned deformation between the two, so the comparison happens in a
frame where they actually correspond. This document explains dense displacement
fields and spatial transformers, then walks `networks/registration.py`,
`losses/registration.py` and `training/reggan.py`.

RegGAN is also the cleanest example in this repo of **extending** a model rather than
replacing it: three additions, no overridden training step.

---

## 1. The failure, and how it differs from PatchNCE's

The MRI/CT pairs here were aligned by hand. Manual QC corrected translation but could
not correct in-plane rotation, so **a rotational residual remains** — and L1 scored on
a mis-framed target teaches blur, for the conditional-median reason in
`02_gan_from_zero.md` section 2.

PatchNCE hedges against this: it compares *features* rather than pixels, so it
tolerates small misalignment. RegGAN does something more direct — it **measures and
removes** the misalignment.

That distinction is the reason `exp7` exists, and why `kaggle_workflow.md` §9 puts it
before `exp3` in the run order:

> Run **exp7 before exp3**, not after. exp3 reweights its objective as a hedge
> against a residual misalignment nobody has measured; exp7 measures it.

`R_flow_px` — the mean displacement RegGAN's network predicts, in millimetres — is a
direct readout of how misaligned the pairs actually are. That number is what the
PatchNCE hypothesis was arguing about without evidence.

---

## 2. What a dense displacement field is

A **flow field** is a two-channel image the same size as the picture:

```
flow[0, y, x] = dx      how far to move horizontally to reach the source pixel
flow[1, y, x] = dy      how far to move vertically
```

Warping means: for each output pixel `(x, y)`, sample the input at
`(x + dx, y + dy)`. Since that lands between pixels, you interpolate — and bilinear
interpolation is differentiable, which is what makes the whole thing trainable.

That resampling operation is a **spatial transformer**. In PyTorch it is
`F.grid_sample`, and the only real work is converting coordinates into the normalised
`[-1, 1]` form it expects.

### Minimal version — build it yourself

```python
import torch, torch.nn.functional as F

def warp(x, flow):
    """x: [N,1,H,W].  flow: [N,2,H,W] in PIXELS, (dx, dy)."""
    N, _, H, W = x.shape
    ys, xs = torch.meshgrid(torch.arange(H, dtype=x.dtype),
                            torch.arange(W, dtype=x.dtype), indexing="ij")
    base = torch.stack((xs, ys), dim=0)          # [2,H,W]
    coords = base.unsqueeze(0) + flow            # absolute source positions

    # pixel centres -> normalised, for align_corners=False
    nx = 2.0 * (coords[:, 0] + 0.5) / W - 1.0
    ny = 2.0 * (coords[:, 1] + 0.5) / H - 1.0
    return F.grid_sample(x, torch.stack((nx, ny), dim=-1),
                         mode="bilinear", padding_mode="border", align_corners=False)

x = torch.arange(25., ).reshape(1, 1, 5, 5)
flow = torch.zeros(1, 2, 5, 5); flow[:, 0] = 1.0          # shift one pixel in x
print(warp(x, flow)[0, 0])        # each row shifted left by one, edge repeated
```

Run it: a flow of `dx = 1` everywhere pulls each row's content one pixel left, and
`padding_mode="border"` repeats the edge. That is the entire mechanism.

---

## 3. The real one: `SpatialTransformer` (`networks/registration.py:45-101`)

Line for line the same, with three decisions made explicit.

**The field is in pixels, not normalised coordinates** (`:48-54`):

> A displacement in normalised units means a different physical distance on a 180x180
> slice than on a 430x430 one, so the smoothness penalty and the reported flow
> magnitude would both **silently depend on slice size**. In pixels — and 1 px = 1 mm
> throughout this project — they mean millimetres on every slice.

This is why `R_flow_px` is directly interpretable. A field in normalised units would
give you a number that changes meaning between body regions.

**`padding_mode="border"`, not `"zeros"`** (`:56-58`): a field that samples just past
the edge should pick up the nearest real tissue, not a black pixel that L1 would then
read as a large error **attributable to the generator**. The generator would be
punished for the registration network's excursion.

**`mode` is a forward argument, not a constructor setting** (`:81-88`): the same
transformer warps images bilinearly and validity masks with **nearest**. Interpolating
a mask produces fractional validity, and a pixel is either real or padding.

The identity grid is cached per `(H, W, device, dtype)` in a plain dict (`:67`) and
deliberately **not registered as a buffer** — it is derived geometry, not learned state,
and does not belong in checkpoints.

---

## 4. `RegistrationUNet` — small on purpose (`:116-190`)

```python
widths = [min(nrf * 2 ** i, 256) for i in range(num_downs + 1)]   # [32,64,128,256,256]
self.stem  = _conv_block(in_channels, widths[0], ...)
self.downs = [Sequential(MaxPool2d(2), _conv_block(w[i], w[i+1])) for i ...]
self.ups   = [_conv_block(w[i+1] + w[i], w[i]) for i in reversed(...)]
self.head  = nn.Conv2d(widths[0], out_channels, 3, padding=1)
```

A conventional U-Net with `MaxPool2d` downsampling and `F.interpolate` upsampling
(rather than strided/transposed conv as in `networks/unet.py`) — appropriate because it
predicts a smooth field, not texture. See `01_building_blocks.md` section 3.

**Upsampling is sized to the skip, not by a fixed factor** (`:181-185`):

```python
h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
```

which survives odd spatial sizes — validation slices are not all powers of two.

### The zero-initialised head is the single most important line here

```python
init_weights(self)                                   # :151

nn.init.normal_(self.head.weight, 0.0, 1e-5)         # :165-166
nn.init.constant_(self.head.bias, 0.0)
```

**The ordering is the whole point** — this runs *after* `init_weights`, which would
otherwise overwrite it with `N(0, 0.02)` like every other conv.

> ⚠️ **The failure it prevents.** With a normally-initialised head, step 0 warps the
> target by a random field of tens of pixels. The generator's first gradients point
> toward matching a scrambled version of the CT, and **it does not recover — while the
> loss curve falls exactly as it would in a healthy run**, because R is simultaneously
> learning to undo its own noise. There is no symptom to notice.

Starting at zero displacement makes epoch 0 identical to plain pix2pix, and the field
grows only insofar as the data asks it to.

This is the third silent-failure guard in this codebase built the same way — see
`07_patchnce.md` section 6 and `01_building_blocks.md` section 8. The pattern is worth
naming: **when a component can be wrong in a way that leaves the loss curve looking
healthy, constrain it structurally rather than trusting yourself to notice.**

---

## 5. The two loss terms (`losses/registration.py`)

### `flow_smoothness` (`:25-59`)

```python
d_x = flow[:, :, :, 1:] - flow[:, :, :, :-1]
d_y = flow[:, :, 1:, :] - flow[:, :, :-1, :]
return d_x.pow(2).mean() + d_y.pow(2).mean()          # unmasked case
```

First-order finite differences, squared. It penalises the field for *changing quickly*
across space, which is what makes it a registration rather than an arbitrary
per-pixel reshuffle.

**Masking is subtler than it looks** (`:51-52`):

```python
m_x = mask[..., 1:] * mask[..., :-1]
```

A difference is valid only where **both** pixels are, so the mask is ANDed with itself
shifted by one. `_masked_mean` then divides by `(m.sum() * diff.shape[1])` so the two
flow channels are counted in the denominator.

### `flow_magnitude` (`:62-78`)

```python
norm = flow.pow(2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
```

Under `torch.no_grad()`. **Diagnostic only, never a loss** — this is the `R_flow_px`
and `R_flow_max` that appear in the log, in millimetres.

### Why the two are inseparable

`LossPlan` refuses to run one without the other (`losses/builder.py:72-80`):

> The correction loss warps the prediction by a learned field before comparing it to
> the target, and **an unpenalised field can warp almost any prediction onto almost any
> target — the loss would fall to zero while the generator learned nothing.**

Three guards keep the deformation honest, and it takes all three:

1. **The smoothness penalty**, which `LossPlan` mandates.
2. **R's small capacity** — `nrf: 32`, `num_downs: 4` (`networks/registration.py:129`).
   A network that cannot represent a high-frequency field cannot cheat with one.
3. **The zero-initialised head**, section 4.

---

## 6. RegGAN as an extension: three additions, no overridden step

`training/reggan.py:51` — `class RegGANModel(Pix2PixNCEModel)`. It does **not** override
`backward_G`, `backward_D` or `optimize_parameters`. It adds:

| addition | line | plugs into |
|---|---|---|
| `netR` + `self.stn` | `:68`, `:70` | — |
| `optimizer_R` | `:72-79` | `g_step_optimizers()` (`:93-103`) |
| the correction and smoothness terms | `:105-151` | `extra_G_terms(stats)` (`:105`) |

Those two hooks are declared in the base class at `pix2pix_nce.py:236` and `:250`
precisely so a subclass can do this. `13_adding_your_own.md` covers the pattern.

**R is stepped by the generator's backward**, in the same `scaler.step` loop —
`g_step_optimizers` returns `super()'s list + [optimizer_R]`. There is no alternating
R step: **R and G minimise the same scalar**, and nothing between them is detached.
They share one graph on purpose.

`in_channels` is derived, not configured (`:67`): `2 * model.generator.out_channels`,
because R sees `cat[fake_CT, real_CT]`.

---

## 7. `extra_G_terms` — the four decisions (`:105-151`)

```python
with torch.autocast(device_type=self.device.type, enabled=False):
    fake_B, real_B, mask = self.fake_B.float(), self.real_B.float(), self.mask.float()

    flow = self.netR(torch.cat([fake_B, real_B], dim=1))
    warped = self.stn(fake_B, flow)
    warped_mask = self.stn(mask, flow, mode="nearest") * mask

    g_corr = masked_l1(warped, real_B, warped_mask, self.criteria["l1"])
    g_smooth = flow_smoothness(flow, mask)
    mean_flow, max_flow = flow_magnitude(flow, mask)

return self.plan.lambda_corr * g_corr + self.plan.lambda_smooth * g_smooth
```

**It runs in fp32, deliberately** (`:113-117`):

> The field is a displacement in pixels, and at half precision a value near 400 — the
> size of the largest validation slices — quantises to steps of about 0.25 px. **The
> residual this whole experiment exists to measure is itself only a few pixels**, so an
> fp16 field would put the measurement noise at the same order as the measurement.

Note what kind of AMP exception this is: a **disabled-autocast region inside the
ordinary scaled backward**, not a separate unscaled step. There is no double backward
here, so nothing needs splitting off — unlike R1 and path-length. That is regime two of
the three in `08_stabilisers_in_code.md` section 6.

**The generated image is warped, never the target** (`:126-129`):

> Warping the target instead would make the supervision signal itself mobile, and G
> could then be rewarded for producing an image that is **easy to warp** rather than one
> that is correct.

**The valid region moves with the warp** (`:132-136`):

```python
warped_mask = self.stn(mask, flow, mode="nearest") * mask
```

Score only pixels valid in *both* frames. Using the unwarped mask alone lets the
deformation drag zero-padding into the scored area, **where it agrees with the target's
padding perfectly and reads as accuracy.**

**Nothing is detached.** G and R share one graph, which is what lets R's gradient reach
back into what G produces.

---

## 8. What `exp7` measures

```yaml
model.name: reggan
loss: { lambda_gan: 1, lambda_l1: 0, lambda_nce: 0,
        lambda_corr: 100, lambda_smooth: 10 }
model.registration: { nrf: 32, num_downs: 4 }
```

`lambda_l1: 0` and `lambda_corr: 100` — the L1 term is not *removed*, it is **moved
into the warped frame**. Against `exp1_pix2pix` (λ_gan 1, λ_L1 100, everything else
zero) this is a clean one-variable comparison: same architecture, same objective, only
the frame the reconstruction loss is taken in has changed.

Watch two log fields (`training_log_reference.md` decodes both):

| field | reading |
|---|---|
| `R_flow_px` | mean predicted displacement, in mm. **This is the number the PatchNCE argument was missing.** |
| `R_flow_max` | worst-case displacement — a large value with a small mean means R is straining somewhere specific |

If `R_flow_px` settles near zero, the pairs are well aligned, RegGAN reduces to
pix2pix, and the misalignment premise behind `exp3` is wrong. If it settles at several
millimetres, the residual is real and worth correcting. **Either outcome is a result** —
which is what makes this the right experiment to run early.

---

## See also

- `05_the_training_step.md` — the `extra_G_terms` and `g_step_optimizers` hooks
- `08_stabilisers_in_code.md` — the three AMP regimes, of which this is the second
- `13_adding_your_own.md` — extending via hooks versus writing a sibling model
- `11_cyclegan.md` — the alternative model that could *not* use these hooks
- `training_log_reference.md` — reading `R_flow_px` and `R_flow_max` during a run
