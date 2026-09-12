# The PatchGAN discriminator

The discriminator's job is to give the generator a gradient that says "this looks
fake". This document explains why it emits a *grid* of verdicts rather than one
number, derives the 70x70 receptive field, and walks `networks/patchgan.py` — a
138-line file with four decisions in it that are each easy to get wrong.

---

## 1. The failure a whole-image discriminator has

The obvious design is a classifier: convolutions down to a single scalar,
"real or fake?". Two things go wrong.

**It duplicates work L1 already does.** In this objective, `lambda_l1: 100` against
`lambda_gan: 1` means the L1 term carries roughly 99% of the gradient magnitude, and
L1 already enforces global anatomy — the liver is in the right place because L1 put
it there. A discriminator policing global structure is spending its capacity on a
question that is already answered.

What L1 *cannot* do is prevent blur, for the conditional-median reason in
`02_gan_from_zero.md` section 2. Blur is a **local texture** property. So the useful
division of labour is: L1 owns global anatomy, D owns local texture.

**It has too many parameters for the data.** A full-image discriminator needs to
reduce 256x256 to one number, which means a deep stack ending in large fully
connected layers. On 1687 training slices, that memorises.

## 2. The fix: judge overlapping windows

A PatchGAN just *stops downsampling early*. With `n_layers: 3` the output is a
`[B, 1, 30, 30]` map from a 256x256 input, and each of those 900 values judges one
overlapping 70x70 window of the input.

The loss then averages over the grid — which is exactly what `F.mse_loss(pred, target)`
already does, with no special code:

```python
# losses/gan_loss.py:82, unchanged for a grid-shaped pred
loss_real = F.mse_loss(pred_real, self._target_like(pred_real, True))
```

`_target_like` uses `torch.full_like`, so the target is a `[B,1,30,30]` tensor of
`0.9`, and the mean reduction averages 900 verdicts per image. **Nothing in
`gan_loss.py` knows the discriminator is patch-based.** That is the point — the
shape carries the design.

Three consequences worth naming:

| property | consequence |
|---|---|
| far fewer parameters | 2.8M vs the generator's 54.4M (`exp2_paper.yaml` records both) |
| fully convolutional | applies unchanged to any input size — necessary here, where validation slices range from 256 to 512 after padding |
| local by construction | D physically cannot reason about global layout, so it cannot drift into L1's job |

---

## 3. Deriving the 70

The stack for `n_layers=3` is five 4x4 convolutions with strides `2, 2, 2, 1, 1`.
Working backwards with `r_in = (r_out − 1)·stride + kernel`:

| layer | channels | kernel | stride | output (from 256) | receptive field |
|---|---|---|---|---|---|
| conv 1 | 2 → 64 | 4 | 2 | 128 | 70 |
| conv 2 | 64 → 128 | 4 | 2 | 64 | 34 |
| conv 3 | 128 → 256 | 4 | 2 | 32 | 16 |
| conv 4 | 256 → 512 | 4 | 1 | 31 | 7 |
| conv 5 | 512 → 1 | 4 | 1 | 30 | 4 |

Read the receptive-field column bottom-up: 4, 7, 16, 34, **70**.

The two stride-1 layers at the end are doing something specific: they **widen the
receptive field without shrinking the grid further**. Three stride-2 layers alone
would give a 32x32 grid with a 22-pixel window; the stride-1 pair takes the window
to 70 while only trimming the grid to 30. That is the trade the architecture is
making, and it is why `n_layers` is a receptive-field knob rather than a depth knob.

At this project's 1 px = 1 mm, 70 px is a 70 mm window — roughly the scale of an
organ, which is a reasonable unit for "does this texture belong to this tissue".

`01_building_blocks.md` section 2 has a five-line script that measures this
empirically rather than trusting the table.

---

## 4. Why conditional

D receives `cat[MRI, CT]` on the channel axis, not the CT alone.

**The failure this prevents:** an unconditional D only asks "is this a plausible
CT?" — and a generator can satisfy that by producing a convincing CT *of the wrong
patient*. The adversarial term would then be actively fighting L1, which is trying
to make it the right patient.

Feeding both modalities changes the question to "is this a plausible CT **of this
MRI**", so the adversarial term reinforces correspondence instead of competing with
it. The argument is in the module docstring at `networks/patchgan.py:18-22`.

> **The one experiment that turns this off is `exp8_cyclegan`.** In unpaired
> training the MRI and CT in a batch come from different patients, so conditioning
> would train D on a correspondence that does not exist. `training/cyclegan.py:63-83`
> raises an error rather than letting you configure it. See `11_cyclegan.md`.

---

## 5. Minimal version — build it yourself

```python
import torch, torch.nn as nn

def patchgan(in_ch=2, ndf=64):
    def block(c_in, c_out, stride, norm=True):
        layers = [nn.Conv2d(c_in, c_out, 4, stride, 1, bias=not norm)]
        if norm:
            layers.append(nn.InstanceNorm2d(c_out, affine=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    return nn.Sequential(
        *block(in_ch,   ndf,   2, norm=False),   # first layer: NO norm
        *block(ndf,     ndf*2, 2),
        *block(ndf*2,   ndf*4, 2),
        *block(ndf*4,   ndf*8, 1),               # stride 1: widen, don't shrink
        nn.Conv2d(ndf*8, 1, 4, 1, 1),            # NO sigmoid
    )

x = torch.randn(2, 2, 256, 256)                  # cat[MRI, CT]
print(patchgan()(x).shape)                       # -> torch.Size([2, 1, 30, 30])
```

That is the whole discriminator. The repo version differs only in generalising the
middle loop over `n_layers` and making spectral norm and the norm type
configurable.

---

## 6. The real one: `networks/patchgan.py:40-108`

### The inner `conv` helper (`:67-70`)

```python
def conv(c_in, c_out, stride, use_bias):
    layer = nn.Conv2d(c_in, c_out, kernel_size=4, stride=stride, padding=1, bias=use_bias)
    return apply_spectral_norm(layer) if spectral else layer
```

Every convolution in the network goes through this, so spectral normalisation is
applied uniformly or not at all — there is no way to half-apply it by editing one
layer.

### The channel cap (`:79`, `:88`)

```python
prev, mult = mult, min(2 ** i, 8)
```

Same `min(..., 8)` cap as the generator, so widths go 64 → 128 → 256 → 512 and stop.

### Four decisions to notice

**The first conv has no norm** (`:72-75`). This is not the "first layer is special"
convention it looks like; the reason is domain-specific:

```
Both modalities are calibrated to a fixed [0,1] range before training,
so absolute intensity is meaningful. Normalising it away in layer 1
would discard real information.
```

This has a downstream consequence: `exp6_stylegan2_fitted.yaml` drops `color` from
its DiffAugment policy and cites `patchgan.py:73-75` as the reason. Randomising
brightness destroys exactly the signal layer 1 was left un-normalised to see.

**There is no sigmoid** (`:95-98`). LSGAN and hinge consume raw scores; vanilla uses
`BCEWithLogits`, which applies its own sigmoid in a numerically stable way. Adding
one would break all three objectives at once.

**Spectral normalisation bounds D's Lipschitz constant** (`:48-58`). This is the
cheapest stabiliser available and the one most worth understanding:

> It divides each weight matrix by its largest singular value, which bounds how fast
> D's output can change with its input. A D with a bounded Lipschitz constant
> **cannot become an arbitrarily confident classifier** — and it is exactly that
> runaway confidence that saturates the adversarial loss and leaves the generator
> with no usable gradient direction. Costs one extra power iteration per forward pass.

The "Lipschitz constant" is just a ceiling on the slope: if `|D(a) − D(b)| ≤ K·|a − b|`
for all inputs, `K` is the Lipschitz constant. Bounding it bounds how steep D's
decision surface can get, and a gradient that G can use is a gradient that is not
vertical.

**`init_weights(self)` at `:101`** — the `N(0, 0.02)` init from
`01_building_blocks.md` section 7, which matters more for D than for G.

---

## 7. The builder derives more than it reads: `:111-138`

```python
gen = cfg_model.generator
in_channels = gen.get("out_channels", 1)
if disc.get("conditional", True):
    in_channels += gen.get("in_channels", 1)
```

Three things here are deliberate:

1. **`in_channels` is derived, never configured** (`:128-130`). It is
   `gen.out_channels + gen.in_channels` when conditional. There is no
   `discriminator.in_channels` key to get wrong, and no way for D's input width to
   drift out of sync with G's output.
2. **`norm` is read from the *generator's* config** (`:136`):
   `norm=gen.get("norm", "instance")`. One normalisation choice for the whole model,
   not two that could disagree.
3. **`spectral` is a function argument, not a config read** (`:111-117`). It comes
   from `cfg.stabilizers.spectral_norm_d` because it is a training-stability choice
   rather than an architectural one, and it belongs beside the other stabilisers. See
   `08_stabilisers_in_code.md`.

That last one is why `build_discriminator` takes `cfg_model` (the whole `model`
block) while `build_generator` takes only `cfg.model.generator` — the asymmetry is
load-bearing, and `13_adding_your_own.md` covers it.

---

## See also

- `02_gan_from_zero.md` — what `pred_real` and `pred_fake` are fed into
- `03_unet_generator.md` — the network being judged
- `05_the_training_step.md` — when D is updated relative to G, and what is detached
- `08_stabilisers_in_code.md` — spectral norm, R1 and DiffAugment in code
- `training_strategies.md` — Part 2 argues these design choices from the imaging side
