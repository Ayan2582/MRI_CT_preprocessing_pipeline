# StyleGAN2 adapted for translation

StyleGAN2 was designed to *sample* faces from a Gaussian prior. This project needs it
to *translate* one medical image into another. Every difference between the two
follows from that, and each one costs something. This document walks the encoder that
replaces `z`, the conditional discriminator, the noise gates, sliding-window
inference, and the three invariants checked at startup.

Read `09_stylegan2_core.md` first — it covers W space, modulated convolution and the
primitives this builds on.

---

## 1. The five deviations, and what each costs

| deviation | why it is forced | what it costs |
|---|---|---|
| `w = Mapping(StyleEncoder(MRI))` instead of `Mapping(z)` | this is paired translation; there is no prior to sample from | every `w` is a real patient, so path-length regularisation loses its free samples |
| conditional D, seeing `cat[MRI, CT]` | StyleGAN2's D is unconditional; left that way, **nothing whatsoever would tie the output to the input** | D's input width doubles |
| nearest/average resampling + `[1,3,3,1]` blur | the paper's fused `upfirdn2d` is a custom CUDA kernel | slightly slower, same filter and intent |
| sliding-window inference | a fixed-size synthesis net cannot emit the 512x512 slices validation produces | tile-boundary artifacts to feather over |
| noise capped by resolution | fine noise at 128/256 px invents structure a reader would interpret as a finding | less stochastic texture |

The first is the deep one. The rest follow.

---

## 2. `StyleEncoder` — the whole conditioning burden in one vector

```python
# networks/stylegan2.py:365-376
self.from_image = EqualizedConv2d(in_channels, ch(native_size), 1)   # 1x1
for _ in range(self.num_downs):
    blocks.append(ResDownBlock(ch(resolution), ch(resolution // 2)))
    resolution //= 2
self.to_w = EqualizedLinear(ch(const_size) * const_size * const_size, w_dim)
```

A 1x1 conv to lift the image into feature space, `num_downs` residual downsampling
blocks (6, taking 256 → 4), then a flatten of the 4x4 map into one `w_dim`-length
vector.

> ⚠️ **Read the docstring at `:342-347`.** "This carries the whole conditioning burden.
> StyleGAN2 samples `z` from a Gaussian prior; there is no prior here, so `w` has to
> come from the input image, and **every spatial fact the generator will ever know
> about the MRI has to survive the trip through this bottleneck**."

Compare with the U-Net: there, the bottleneck is also 1x1, but eight skip connections
carry spatial detail around it. Here there are none. A 512-number vector must encode
everything about a 256x256 slice that the output should reflect.

**That is why anatomical drift is the predicted failure mode**, and why it appears in
sample panels before it appears in `mae_norm` — a slightly-wrong liver boundary
changes the mean absolute error very little and changes the image a lot.

### The tap protocol is copied deliberately

`StyleEncoder` implements `n_taps`, `tap_channels`, `tap_spatial` and
`forward(x, tap_layers, encode_only)` with **exactly** the semantics `UnetGenerator`
uses (`:378-421`). The reason is stated at `:349-351`: so that
`Pix2PixNCEModel.compute_nce` needs no branch.

This is a good pattern to notice. Rather than adding `if isinstance(netG, ...)` to the
training code, the second architecture was made to satisfy the first one's contract.
The training step does not know which generator it has. `13_adding_your_own.md`
returns to this.

---

## 3. `StyleGAN2Generator` — assembling it

```python
# networks/stylegan2.py:512-533
self.encoder = StyleEncoder(...)
self.mapping = MappingNetwork(w_dim, n_mapping)
self.const   = nn.Parameter(torch.randn(1, ch(const_size), const_size, const_size))
# blocks: while resolution <= native_size, doubling
self.num_ws  = sum(block.num_ws for block in self.blocks)
self.register_buffer("w_avg", torch.zeros(w_dim))
```

For 4 → 256 that is **7 synthesis blocks** (4, 8, 16, 32, 64, 128, 256) and
`num_ws = 2 + 6·3 = 20` style vectors.

### `synthesise_with_styles` (`:581-607`)

```python
ws = self._styles(self.mapping(self.encoder(x)))

h = self.const.to(x.dtype).expand(x.shape[0], -1, -1, -1)
image, consumed = None, 0
for block in self.blocks:
    h, image = block(h, ws[:, consumed:consumed + block.num_ws], image)
    consumed += block.num_ws

return torch.tanh(image), ws
```

The learned constant is expanded to the batch — **every sample starts from the same
4x4 tensor**, and all the variation enters through `ws` and the noise.

This method exists separately from `forward` for one reason: path-length
regularisation must differentiate the image with respect to `ws`, so `ws` has to be a
returned node in the graph, and it is **deliberately not detached** (`:583-588`).

### `_styles` (`:555-579`) — three things at once

**A running average of W** is maintained during training:

```python
self.w_avg.copy_(self.w_avg.lerp(w.detach().float().mean(0), 0.005))
```

**Truncation** uses it at eval: `w = w_avg.lerp(w, truncation_psi)`. Pulling `w` toward
the average trades diversity for reliability. `truncation_psi: 1.0` (the default here)
disables it — in a translation model, "diversity" is patient variation you actually
want.

**Style mixing** (`:567-577`):

```python
cutoff = int(torch.randint(1, self.num_ws, ()))
other  = w[torch.randperm(batch, device=w.device)]
keep   = (torch.arange(self.num_ws, device=w.device) < cutoff)
ws     = torch.where(keep.view(1, -1, 1), ws, other)
```

Use one `w` for layers below `cutoff` and another above it, so the network cannot
assume adjacent layers see correlated styles.

> **In the original, the second `w` is a fresh draw from the prior. Here it is
> `w[torch.randperm(batch)]` — another patient in the batch**, because there is no
> prior. That makes style mixing much more aggressive than the paper intends, which is
> why `exp5_stylegan2_vanilla` sets `style_mixing_prob: 0.9` (paper-faithful) and
> `exp6_stylegan2_fitted` sets it to `0.0` (this task's answer).

**`torch.where`, not in-place slice assignment** (`:572-575`). `ws` is a non-leaf
tensor inside the autograd graph, and writing into it in place is the kind of thing
that works until a version bump turns it into a "modified by an inplace operation"
error.

---

## 4. The noise gates — two decisions worth reading twice

```python
# networks/stylegan2.py:200-215
self.gated_off = bool(max_resolution) and resolution > int(max_resolution)

def forward(self, x):
    if self.gated_off or not self.enabled:
        return x
    if not self.training and not self.at_eval:
        return x
    noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], ...)
    return x + self.strength * noise
```

**`max_resolution` is a clinical safety decision, not a hyperparameter** (`:191-197`):

> Noise at 128 and 256 px synthesises fine texture that has no counterpart anywhere in
> the source MRI. That is **hallucinated detail a reader would interpret as
> structure**. Capping it at 64 keeps the mechanism for coarse stochastic variation and
> denies it the scale at which it invents findings.

`exp6_stylegan2_fitted` sets `noise_max_resolution: 64`. `exp5_stylegan2_vanilla` sets
`0` — no cap, the paper's behaviour — precisely so the difference is measurable. Note
also that `gated_off` is computed **once at construction**, not per forward.

**The eval gate reads `nn.Module.training`, on purpose** (`:184-190`):

> The gate is read off `nn.Module.training` rather than from an externally-set flag ON
> PURPOSE. A flag flipped by `set_eval_mode` would stay flipped when the trainer went
> back to training, **silently disabling noise for the rest of the run after the first
> validation pass.**

This is a general lesson about state. `net.train()` / `net.eval()` is a single
mechanism PyTorch already restores correctly; a parallel flag is a second source of
truth that only has to get out of sync once.

---

## 5. `StyleGAN2Discriminator` (`:637-688`)

```
from_image (1x1 conv)  →  6 x ResDownBlock (256 → 4)  →  MinibatchStdDev
                       →  final_conv  →  final_linear  →  out (1 scalar)
```

Two differences from the PatchGAN that matter:

**No normalisation layers and no spectral norm anywhere** (`:641-648`). Equalized
learning rate plus R1 is the entire stability story. `build_stylegan2_discriminator`
**warns and ignores** `spectral` if you set it (`:743-748`) rather than silently
applying something the architecture was not designed with.

**The output is one scalar per image, not a patch grid.** That is a real change in
what the adversarial term polices — global plausibility rather than local texture.
Combined with `lambda_l1: 0` in `exp5`, nothing at all constrains local correspondence,
which is exactly why `exp5` is a deliberate upper bound rather than a candidate. See
`model/README.md`'s experiment ladder.

---

## 6. Tiling — the price of a fixed output size

**The failure.** The U-Net is fully convolutional: hand it a 512x512 slice and it
returns 512x512. A StyleGAN2 synthesis network grows a learned 4x4 constant through a
fixed number of blocks, so it emits **exactly one resolution and nothing else**.

The numbers are not marginal (`networks/tiling.py:12-16`):

```
45% of validation slices (103 of 230) pad to 512.
That is EVERY abdomen slice (94) and EVERY spine slice (9).
Nothing exceeds 512.
```

Without tiling those slices could not be generated at all, `mae_norm` would be
computed on brain and MSK slices only, and **no StyleGAN2 run would be comparable to
the exp0–exp4 ladder that scored all 230.**

**The fix** (`tiled_forward`, `:79-132`): cut the input into overlapping
`native`-sized windows, generate each, and sum them back weighted by a 2-D raised
cosine, dividing at the end by the accumulated weight.

```python
out[:, :, top:top+native, left:left+native]        += pred * window
weight_sum[:, :, top:top+native, left:left+native] += window
...
return out / weight_sum
```

Four implementation details:

**The fast path is bit-identical** (`:103-104`). If `h <= native and w <= native`,
`fn(x)` is returned directly — no blending, no extra cost. That covers every training
crop and the 55% of validation slices that pad to exactly 256.

**The Hann window is clamped to `1e-3`** (`:52-54`). A textbook Hann is exactly zero at
both ends, so the outermost row and column would accumulate weight 0 and the final
division would produce **NaN**. The floor is load-bearing.

**The last origin is snapped flush to the edge** (`:61-70`), so the final window ends
at `extent - native` rather than leaving a strip uncovered.

**The window is built lazily on the first tile** (`:119-121`), so the output channel
count comes from `fn` itself rather than from an assumption about it.

> **Why feathering is not optional here** (`:26-31`). In ordinary tiled inference the
> tiles are nearly consistent and a hard seam is a minor artifact. Here **each tile is
> encoded to its own style vector `w`**, so neighbouring tiles are generated under
> genuinely different global styles — brightness and texture really do differ across a
> seam. The cosine weights make the transition gradual instead of a visible line down
> the middle of the abdomen. **It hides the discontinuity; it does not remove it.**

`fn` must be the **single-tile** forward (`self._synthesise`), not the public one, or
it recurses forever — see `:86-87` and the call site at `:625-628`.

---

## 7. Three invariants checked at startup

Each of these would otherwise be a silent or late failure:

| check | line | what it prevents |
|---|---|---|
| `num_downs == log2(native_size / const_size)` | `:693-711` | `data/dataset.py:317` reads `num_downs` to set the validation padding multiple; a disagreement produces wrongly padded validation inputs |
| `batch_size % mbstd_group_size == 0` | `:754-760` | `MinibatchStdDev` would silently shrink its group rather than error |
| input is exactly `native_size` inside `synthesise_with_styles` | `:590-596` | catches `data.crop_size` and `model.generator.native_size` disagreeing, with an error that names both |

The third one's message is worth copying as a style:

```
StyleGAN2Generator synthesises 256x256 only, got (192, 192). Larger inputs are
handled by tiling; a smaller one means data.crop_size and
model.generator.native_size disagree.
```

It states what happened, what the legitimate larger case is, and what the likely cause
of the smaller case is.

---

## 8. The two configurations, and why both exist

| | `exp5_stylegan2_vanilla` | `exp6_stylegan2_fitted` |
|---|---|---|
| λ_L1 / λ_NCE | 0 / 0 | 100 / 1 |
| `channel_base` / `channel_max` | 32768 / 512 (config-f) | 16384 / 256 |
| params (G + D) | 56.5M + 28.9M = **85.4M** | 17.9M + 7.2M = **25.1M** |
| `style_mixing_prob` | 0.9 | 0.0 |
| `noise_max_resolution` | 0 (uncapped) | 64 |
| R1 γ / path-length | 3.3 / on | 1.6 / off |
| batch | 4 | 8 |

`exp5` is StyleGAN2 as published — purely adversarial, full capacity, every paper
default. It is a **ceiling measurement**, not a candidate: with no reconstruction term
there is nothing anchoring the anatomy.

`exp6` is the one-variable comparison against `exp2_paper`: same loss, same λ, same
warm-up, different architecture. That is the question this module was written to
answer. Compare `exp2` (57.2M params) with `exp6` (25.1M) and note the fitted version
is deliberately the *smaller* network — 1687 training slices do not support config-f.

---

## See also

- `09_stylegan2_core.md` — W space, equalized LR, modulated convolution, the primitives
- `03_unet_generator.md` — the architecture this is being compared against
- `07_patchnce.md` — the tap protocol `StyleEncoder` was written to satisfy
- `08_stabilisers_in_code.md` — R1 and path-length, which this architecture assumes
- `model/README.md` — the experiment ladder, and why exp5 is a bound rather than a candidate
