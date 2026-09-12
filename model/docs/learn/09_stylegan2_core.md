# StyleGAN2 — the ideas and the primitives

StyleGAN2 is a genuinely different way to build a generator. Instead of carrying
spatial detail down an encoder and back up through skips, it starts from a learned
4x4 constant and paints the image using a **style vector** that says *what kind of
thing* to draw. This document explains W space, equalized learning rate, modulated
convolution, noise injection and minibatch-stddev, and walks the primitives in
`networks/stylegan2.py:100-467`.

`10_stylegan2_adapted.md` covers what changes when you use it for *translation*
rather than sampling. Read this one first.

---

## 1. Why a second architecture exists at all

The `exp0`–`exp4` ladder varies the loss on top of one fixed pair of networks. So it
can only ever answer questions about the **objective**. This module supplies a second
architecture so that `exp6_stylegan2_fitted` can be compared against `exp2_paper`
with the loss held constant and the architecture as the only variable.

That is the whole justification, and it is worth keeping in view: StyleGAN2 here is
a *measurement*, not an upgrade.

---

## 2. The core idea

The U-Net decides each output pixel by carrying spatial detail down through an
encoder and back up through skip connections. StyleGAN2 splits that job in two:

```
w      — a style vector: WHAT KIND of thing is being drawn, globally and per-layer
noise  — per-pixel Gaussian: the stochastic fine grain
```

The synthesis network starts from a **learned 4x4 constant** and knows nothing about
the input except through `w`. Every block doubles the resolution and applies `w`
again, so `w` is consulted at every scale — coarse layers get pose and layout, fine
layers get texture.

> ⚠️ **The consequence to state plainly, because it is the main risk here.** There are
> **no encoder-to-decoder skip connections.** The only route from the MRI to the output
> is a single global vector. The `to_image` sums between resolutions are StyleGAN2's
> own "skip" generator — *image* residuals, not feature skips — and they do not restore
> spatial correspondence. **Anatomical drift is the expected failure mode, and it will
> show up in the sample panels before it shows up in any metric.** The module docstring
> at `networks/stylegan2.py:19-25` says the same.

### What W space is

`z` is a sample from a simple prior — a Gaussian. `w = MappingNetwork(z)` is a learned
transformation of it, and the difference matters.

A Gaussian is spherical, but the space of *plausible images* is not. Forcing a
spherical prior to map directly onto image space means the generator must warp it
severely, and warping is exactly what makes a mapping ill-conditioned — some
directions become hypersensitive, others flat. That is the failure path-length
regularisation exists to measure (`08_stabilisers_in_code.md` section 5).

The mapping network absorbs that warp into eight cheap fully connected layers, so the
synthesis network receives an already-shaped `w`. W space ends up **disentangled**:
moving along one direction tends to change one property.

> **In this repo there is no `z` at all.** `w = MappingNetwork(StyleEncoder(MRI))` —
> this is paired translation, not sampling. `10_stylegan2_adapted.md` covers what that
> costs.

---

## 3. Equalized learning rate

**The failure.** Adam normalises each parameter's update by its own gradient standard
deviation. If two layers hold weights with very different dynamic ranges, they take
different *effective* step sizes even at the same nominal learning rate — so the
learning rate means something different in each layer.

**The fix.** Hold weights at `N(0, 1)` and apply the fan-in scaling **at runtime**:

```python
# networks/stylegan2.py:125-137
self.weight = nn.Parameter(torch.randn(out_channels, in_channels, k, k))
self.scale  = 1.0 / math.sqrt(in_channels * k * k)

def forward(self, x):
    return F.conv2d(x, self.weight * self.scale, self.bias, ...)
```

Every *stored* weight now has the same scale, so Adam sees the same dynamic range
everywhere, and the per-layer scaling that a normal initialisation would have baked
in is applied on the fly instead.

`EqualizedLinear` (`:100-119`) adds one more knob:

```python
self.weight = nn.Parameter(torch.randn(out_features, in_features) / lr_mul)
self.scale  = lr_mul / math.sqrt(in_features)
```

`lr_mul` below 1 slows a layer down relative to the rest of the network. The mapping
network uses **0.01**, because a mapping net trained at the same rate as the synthesis
net destabilises W early in training. Note it appears in both the initialisation and
the scale, so the *stored* weights are correspondingly larger and the forward pass is
unchanged — only the gradient scale moves.

> ⚠️ **This is why `init_weights` must never touch these modules.** Equalized LR
> requires an `N(0,1)` init; applying `N(0, 0.02)` on top leaves every weight ~50x too
> small and then runtime-scales it. The network trains without complaint and learns
> nothing. See `networks/stylegan2.py:41-47` and `01_building_blocks.md` section 8.

### The activation gain

```python
ACT_GAIN = math.sqrt(2.0)                                  # :80
def leaky(x): return F.leaky_relu(x, 0.2) * ACT_GAIN       # :83-84
```

`LeakyReLU(0.2)` shrinks activation variance; multiplying by `sqrt(2)` puts it back.
Under equalized LR, where every scale is deliberate, an unrestored variance would
compound through the depth of the network.

---

## 4. Modulated convolution — the heart of it

**What StyleGAN1 did, and why it broke.** StyleGAN1 used AdaIN: normalise the
activations, then rescale them by the style. That destroys information carried in the
*relative magnitudes* between feature maps — and the generator learned to smuggle that
information past the normaliser as a large localised spike. The result was the
notorious **water-droplet artifact**: a bright blob that appears in essentially every
StyleGAN1 sample, because it is the network's channel for information the architecture
would otherwise have erased.

**The fix.** Never touch the activations. Let the style scale the convolution
**weights** instead:

```
modulate:    w'[i,j] = s[i] · w[i,j]           s from the style, per INPUT channel
demodulate:  w''[i,j] = w'[i,j] / sqrt( Σ_i,k w'[i,j,k]² + ε )
```

Modulation applies the style; demodulation restores unit output variance by dividing
the weights by their own L2 norm. Same statistical control as AdaIN, applied to
weights rather than activations, so **there is no spike to hide behind**.

### Minimal version — build it yourself

```python
import torch, torch.nn as nn, torch.nn.functional as F

class ModConv(nn.Module):
    def __init__(self, c_in, c_out, k, w_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, c_out, c_in, k, k))
        self.affine = nn.Linear(w_dim, c_in)
        nn.init.ones_(self.affine.bias)          # untrained style == identity
        self.scale, self.pad, self.c_out, self.k = 1 / (c_in * k * k) ** 0.5, k // 2, c_out, k

    def forward(self, x, w):
        B, C, H, W = x.shape
        style  = self.affine(w).view(B, 1, C, 1, 1)
        weight = self.weight * self.scale * style                     # modulate
        inv    = torch.rsqrt(weight.float().pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * inv.to(weight.dtype).view(B, -1, 1, 1, 1)    # demodulate

        # every sample now has its OWN weight tensor -> grouped conv
        x = x.reshape(1, B * C, H, W)
        weight = weight.reshape(B * self.c_out, C, self.k, self.k)
        return F.conv2d(x, weight, padding=self.pad, groups=B).reshape(B, self.c_out, H, W)

print(ModConv(64, 128, 3, 512)(torch.randn(4, 64, 16, 16), torch.randn(4, 512)).shape)
# -> torch.Size([4, 128, 16, 16])
```

**The grouped-convolution trick is the part that looks strange.** After modulation
every sample in the batch has a *different* weight tensor, and `F.conv2d` applies one
weight tensor to a whole batch. Folding the batch into the channel axis and setting
`groups=batch` makes one call do `B` independent convolutions.

### The real one: `networks/stylegan2.py:250-297`

Identical, plus three details:

**`bias_init=1.0` on the affine layer** (`:273`):

```python
self.affine = EqualizedLinear(w_dim, in_channels, bias_init=1.0)
```

An untrained style is then the **identity modulation** rather than multiplying every
feature map by roughly zero. Initialise this at 0 and the network starts by
annihilating its own signal.

**The demodulation runs in fp32, deliberately** (`:281-286`):

```python
inv = torch.rsqrt(weight.float().pow(2).sum(dim=[2, 3, 4]) + 1e-8)
```

> This sums squares over an entire kernel — 512 input channels by 3x3 — and **overflows
> fp16 under AMP, which shows up as a loss that goes to nan a few hundred steps in.**

A `.float()` that looks like noise is load-bearing. It is also a good example of why
AMP failures are hard: the symptom appears far from the cause.

**Upsampling is nearest + blur** (`:288-290`), not transposed convolution — see
`01_building_blocks.md` section 4.

---

## 5. The supporting primitives

### `Blur` (`:140-159`)

A separable `[1,3,3,1]` FIR low-pass, applied around every resampling step.
Nearest-neighbour upsampling and strided convolution both alias badly; this filter is
what stops the aliasing being baked into the texture, **and its absence is visible as
a faint checkerboard**.

```python
k = torch.tensor([1.0, 3.0, 3.0, 1.0])
k = k[:, None] * k[None, :]                     # outer product -> 4x4
self.register_buffer("kernel", (k / k.sum())[None, None])
...
return F.conv2d(F.pad(x, (1, 2, 1, 2)), kernel, groups=channels)
```

The **asymmetric padding `(1, 2, 1, 2)`** is what keeps an even-length 4-tap kernel
size-preserving — an even kernel has no centre tap, so the padding must be lopsided.
`groups=channels` makes it depthwise: the same fixed filter on every channel, no
learned parameters.

### `PixelNorm` (`:162-167`)

```python
return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-8)
```

Normalises each position's feature vector to unit RMS. Applied only to the mapping
network's input, where it keeps W from drifting in overall scale.

### `NoiseInjection` (`:170-215`)

```python
self.strength = nn.Parameter(torch.zeros(()))     # ONE learned scalar
...
noise = torch.randn(B, 1, H, W, ...)              # one channel, broadcast
return x + self.strength * noise
```

**What it buys:** without it the generator must synthesise stochastic detail — grain,
texture — deterministically from `w`, which burns capacity and makes texture visibly
repeat.

The strength starts at **zero**, so the network begins with no noise and learns how
much it wants, per layer. Two gates here are specific to medical synthesis and are
covered in `10_stylegan2_adapted.md` section 4 — including one design decision about
reading `nn.Module.training` rather than an external flag that is worth reading as a
general lesson.

### `MinibatchStdDev` (`:218-247`)

**The failure it detects: mode collapse.** If the generator produces near-identical
outputs for different inputs, no *single-image* discriminator can tell — each output
looks fine on its own. The give-away is only visible across a batch.

So compute the standard deviation of each feature across a group of samples, average
it into one number, and append it as an extra constant channel. Real batches vary;
collapsed fake batches do not, and D can now see the difference directly.

```python
while n % group != 0:        # :238-240  shrink until the group divides the batch
    group -= 1
```

That guard is why `build_stylegan2_discriminator` raises when
`batch_size % mbstd_group_size != 0` (`:754-760`) — better a startup error than a
silently shrunken group.

> **This layer deliberately couples samples in a batch**, which is the same objection
> `01_building_blocks.md` section 5 raises against BatchNorm. It is confined to D, at
> training only, and diversity is precisely what is being measured — so the coupling is
> the feature.

### `ResDownBlock` (`:300-319`)

The discriminator's rung:

```python
residual = self.skip(F.avg_pool2d(self.blur(x), 2))
h = leaky(self.conv1(self.blur(leaky(self.conv0(x)))))
return (h + residual) / math.sqrt(2)
```

The `/ sqrt(2)` holds activation variance stable across the residual sum: adding two
roughly independent unit-variance signals gives variance 2, so dividing by `sqrt(2)`
restores it. Under equalized LR, where nothing else renormalises, this is not optional.

---

## 6. `MappingNetwork` and `SynthesisBlock`

### `MappingNetwork` (`:324-337`)

```python
PixelNorm(), then n_layers x [EqualizedLinear(w_dim, w_dim, lr_mul=0.01), leaky]
```

Eight layers by default, **same width in and out**. Note that consequence: it maps
`w_dim → w_dim`, so whatever produces its input must already emit `w_dim` values. In
the original that is `z`; here it is the encoder.

### `SynthesisBlock` (`:424-467`)

One resolution level:

```
[upsample+modconv, noise, leaky]   (skipped for the first block)
 modconv,          noise, leaky
 to_image (1x1 modconv, demodulate=False)
```

Two details:

**`to_image` is not demodulated** (`:447-450`). It writes intensities, and
renormalising them would fight the final `tanh`.

**Image accumulation is a residual in image space** (`:465-466`):

```python
image = F.interpolate(image, scale_factor=2, mode="bilinear", align_corners=False) + contribution
```

Every block contributes an image at its own resolution; the running image is upsampled
and added to. This is the StyleGAN2 "skip" generator, and it gives gradients a short
path to every resolution. **It is not a U-Net skip** — no encoder features are involved.

`num_ws = 2 if is_first else 3` (`:451`) counts how many style vectors the block
consumes: two modulated convs plus `to_image`, or one fewer for the first block, which
has no upsampling conv. For a 4→256 generator that totals `2 + 6·3 = 20`.

---

## See also

- `10_stylegan2_adapted.md` — the encoder, the conditional D, tiling, and the invariants
- `01_building_blocks.md` — the other initialisation regime, and why they are incompatible
- `08_stabilisers_in_code.md` — R1 and path-length, the two regularisers this architecture assumes
- `training_strategies.md` — where these appear as choices with costs
