# The U-Net generator

The generator has to turn a 256x256 MRI slice into a 256x256 CT slice. This
document explains why that job wants an encoder-decoder with skip connections,
builds a minimal U-Net you can run, and then walks `networks/unet.py` — including
the tap protocol, which is this repo's one real departure from stock pix2pix.

---

## 1. The failure a plain encoder-decoder has

Take the obvious architecture: downsample the image to a small bottleneck, then
upsample back. The bottleneck forces the network to build an abstract, global
representation — which is what you want for deciding *what tissue is where*.

With `num_downs: 8` on a 256x256 crop, the bottleneck is **1x1**. Everything the
decoder produces must pass through a single spatial location holding 512 numbers.

**That is fine for semantics and catastrophic for geometry.** A 1x1x512 vector can
say "abdomen, this patient, this slice level". It cannot say "the cortex boundary
is at pixel (137, 92), not (138, 92)". The precise position of every edge has to be
reconstructed from nothing, and it comes back approximately — which is a blur, from
a second and independent cause to the one in `02_gan_from_zero.md`.

## 2. The fix: hand the decoder what the encoder saw

A **skip connection** takes the encoder activation at some resolution and
concatenates it onto the decoder activation at the same resolution:

```
encoder block 2 output  [B, 256, 64, 64]  ─────────────┐
                                                       ▼
decoder block 5 input   [B, 256, 64, 64]  ──── concat ──→  [B, 512, 64, 64]
```

The decoder now has both: the abstract description from the bottleneck, and the
exact spatial detail from the encoder at the resolution where it needs it. The
bottleneck no longer has to carry positions, because positions are available
directly.

This is the U-Net, and its shape on paper is the reason for the name — encoder
down the left, decoder up the right, skips as the horizontal bars.

> **Why concatenate rather than add.** Adding forces the two signals into the same
> channel space and lets them cancel. Concatenating keeps them separate and lets the
> next convolution learn how to combine them. The cost is that the decoder block's
> input width doubles, which is exactly what you see at `networks/unet.py:94`.

---

## 3. Minimal version — build it yourself

A working 3-level U-Net, small enough to read in one go:

```python
import torch, torch.nn as nn

class TinyUNet(nn.Module):
    def __init__(self, ch=(1, 32, 64, 128)):
        super().__init__()
        # encoder: three halvings
        self.e1 = nn.Conv2d(ch[0], ch[1], 4, 2, 1)                     # 64 -> 32
        self.e2 = nn.Sequential(nn.LeakyReLU(0.2),
                                nn.Conv2d(ch[1], ch[2], 4, 2, 1))      # 32 -> 16
        self.e3 = nn.Sequential(nn.LeakyReLU(0.2),
                                nn.Conv2d(ch[2], ch[3], 4, 2, 1))      # 16 ->  8
        # decoder: three doublings; inputs double where a skip arrives
        self.d1 = nn.Sequential(nn.ReLU(), nn.ConvTranspose2d(ch[3],   ch[2], 4, 2, 1))
        self.d2 = nn.Sequential(nn.ReLU(), nn.ConvTranspose2d(ch[2]*2, ch[1], 4, 2, 1))
        self.d3 = nn.Sequential(nn.ReLU(), nn.ConvTranspose2d(ch[1]*2, ch[0], 4, 2, 1),
                                nn.Tanh())

    def forward(self, x):
        s1 = self.e1(x)            # keep every encoder output
        s2 = self.e2(s1)
        h  = self.e3(s2)           # bottleneck
        h  = self.d1(h)
        h  = self.d2(torch.cat([h, s2], dim=1))     # skip
        h  = self.d3(torch.cat([h, s1], dim=1))     # skip
        return h

print(TinyUNet()(torch.randn(2, 1, 64, 64)).shape)   # -> torch.Size([2, 1, 64, 64])
```

Three things in there are the entire architecture: keep the encoder outputs,
concatenate them on the way up, and double the decoder's input width to match. The
repo's version is this loop generalised to `num_downs` levels, plus normalisation,
dropout and the tap protocol.

---

## 4. The real one: `networks/unet.py:41-192`

### The channel schedule (`:61-65`)

```python
chans = []
for i in range(self.num_downs):
    chans.append(min(ngf * (2 ** i), ngf * 8))
self.enc_channels = chans
```

Widths double each level and then **cap at `8·ngf`**. With `ngf=64, num_downs=8`:

```
[64, 128, 256, 512, 512, 512, 512, 512]
```

The cap is what keeps the parameter count finite — without it the eighth level
would be 8192 channels. The resulting generator is 54.4M parameters
(`exp2_paper.yaml` records the measurement).

Spatial sizes fall in lockstep, and this table is worth internalising because the
PatchNCE taps are indexed against it:

| tap | source | channels | spatial (from 256) |
|---|---|---|---|
| 0 | the input image | 1 | 256 |
| 1 | encoder block 0 | 64 | 128 |
| 2 | encoder block 1 | 128 | 64 |
| 3 | encoder block 2 | 256 | 32 |
| 4 | encoder block 3 | 512 | 16 |
| 5 | encoder block 4 | 512 | 8 |
| 6 | encoder block 5 | 512 | 4 |
| 7 | encoder block 6 | 512 | 2 |
| 8 | encoder block 7 | 512 | 1 |

### The encoder has three cases, not one (`:67-85`)

Every block is `Conv2d(k=4, s=2, p=1, bias=uses_bias(norm))`, but what wraps it
differs:

```python
if i == 0:
    block = nn.Sequential(conv)                                    # no act, no norm
elif i == self.num_downs - 1:
    block = nn.Sequential(nn.LeakyReLU(0.2, inplace=True), conv)   # no norm
else:
    block = nn.Sequential(nn.LeakyReLU(0.2, inplace=True), conv, norm_layer(c_out))
```

| case | line | why |
|---|---|---|
| **outermost** — no activation, no norm | `:73-76` | it consumes the raw image; there is nothing to activate yet, and normalising would destroy absolute intensity, which is meaningful because both modalities are calibrated to a fixed `[0,1]` range before scaling |
| **innermost** — no norm | `:77-80` | at 1x1 there is exactly one spatial location, so InstanceNorm would divide a single value by its own zero variance |
| **everything else** | `:81-83` | the standard rung |

The innermost case is a genuine numerical bug waiting to happen, not a stylistic
choice. `InstanceNorm2d` normalises over `(H, W)`; when `H = W = 1` the variance is
identically zero.

### The decoder, and where the ×2 comes from (`:87-111`)

```python
for j in range(self.num_downs):
    enc_idx = self.num_downs - 1 - j
    c_in = chans[enc_idx] if j == 0 else chans[enc_idx] * 2
```

Decoder block `j` undoes encoder block `num_downs-1-j`. The `* 2` at `:94` is the
concatenated skip — and `j == 0` is the exception because the bottleneck has no
skip to receive (it *is* the deepest encoder output).

The last block is special (`:95-100`):

```python
up = nn.ConvTranspose2d(c_in, out_channels, 4, 2, 1, bias=True)
block = nn.Sequential(nn.ReLU(inplace=True), up, nn.Tanh())
```

`Tanh` for the reason in `01_building_blocks.md` section 6: the dataset produces
`[−1, 1]`, so the generator must too.

### Dropout is in exactly three blocks (`:107-108`)

```python
if use_dropout and 1 <= j <= 3:
    layers.append(nn.Dropout(0.5))
```

The three *innermost* decoder blocks, per pix2pix. The reason is stated at `:49-51`:
**pix2pix has no noise vector, so dropout is the generator's only stochasticity.**
Without it, G is a deterministic function and the "distribution" it models is a
single point.

> **This project turns it off at evaluation anyway.** `model.generator.dropout_at_eval`
> defaults to `false` (`training/pix2pix_nce.py:484-500`), diverging from the pix2pix
> convention, so that validating the same checkpoint twice gives the same number.
> `training_strategies.md` Part 2 argues that trade.

### `forward` has three return contracts (`:145-192`)

```
out            when tap_layers is None
(out, feats)   when tap_layers is given
feats          when encode_only=True
```

The encoder loop is where all three are served:

```python
for i, block in enumerate(self.encoder):
    h = block(h)
    skips.append(h)                        # every output, for the decoder
    if (i + 1) in taps:
        feats.append(h)                    # requested outputs, for PatchNCE
    if encode_only and (i + 1) >= deepest:
        return feats                       # early exit
```

The early exit at `:178-181` is a real optimisation, not tidiness. PatchNCE encodes
the real MRI purely to read shallow features; running the remaining encoder blocks
and the entire decoder would compute a full synthetic CT and discard it.

---

## 5. The tap protocol — why this file is not the reference implementation

The stock pix2pix U-Net is built by **recursive nesting**: an outermost
`UnetSkipConnectionBlock` wraps a submodule, which wraps a submodule, down to the
bottleneck. It is elegant and it is opaque — by the time `forward()` returns, the
intermediate encoder activations exist only inside nested closures, with no clean
way to read them out.

PatchNCE needs exactly those activations. So this implementation lays the encoder
and decoder out as two explicit `nn.ModuleList`s, which makes `forward()` a readable
loop and a tap a list index. The full argument is in the module docstring at
`networks/unet.py:1-29`.

Three methods define the contract:

| method | line | contract |
|---|---|---|
| `n_taps` | `:120-123` | `num_downs + 1` — tap 0 is the input image, tap k is encoder block k−1 |
| `tap_channels(tap)` | `:125-128` | channel count, used to size the PatchNCE MLP heads |
| `tap_spatial(tap, size)` | `:138-141` | `size // 2**tap` — how many spatial locations that tap offers |

**`tap_spatial` is the one that bites.** `loss.nce.num_patches` defaults to 256, and
a tap can only supply as many patches as it has locations:

```
tap 0 → 256² = 65536 locations
tap 1 → 128² = 16384
tap 2 →  64² =  4096
tap 3 →  32² =  1024
tap 4 →  16² =   256   ← exactly enough, and the practical floor
tap 5 →   8² =    64   ← silently clamped to 64 patches
tap 8 →   1  =     1   ← useless
```

The default `loss.nce.layers: [0,1,2,3,4]` stops at 4 for precisely this reason.
`configs/base.yaml:216-222` records the same arithmetic. Going deeper does not crash
— `networks/patch_sampler.py:116-119` clamps and logs a debug line — it just quietly
contributes almost nothing.

`_check_tap` (`:130-136`) raises a `ValueError` that names the valid range rather
than letting an out-of-range index fail somewhere downstream.

---

## 6. The leaf builder (`:195-209`)

```python
def build_generator(cfg_generator):
    gen_type = cfg_generator.get("type", "unet")
    if gen_type != "unet":
        raise NotImplementedError(...)
    return UnetGenerator(
        in_channels=cfg_generator.get("in_channels", 1),
        ...
        use_dropout=cfg_generator.get("dropout", True),
    )
```

Two things to note. First, the config key is **`dropout`** and the constructor
argument is `use_dropout` — they are not the same name. Second, there are **two
functions called `build_generator`** in this repo: this leaf one, and the dispatcher
at `networks/builder.py:25` that chooses between `unet` and `stylegan2` and then
calls this. Cite the right one; `13_adding_your_own.md` covers the split.

---

## See also

- `01_building_blocks.md` — convolution, transposed convolution, norm and init
- `04_patchgan_discriminator.md` — the network that judges this one's output
- `07_patchnce.md` — what the taps are actually for
- `10_stylegan2_adapted.md` — the alternative generator, which has no skips at all
- `training_strategies.md` — why InstanceNorm, and why dropout is off at eval
