# The pieces every network here is made of

Before any GAN, the vocabulary. This document explains convolution, receptive field,
up- and downsampling, normalisation, activations and weight initialisation — the
seven things every network in `model/networks/` is assembled from. It ends by
walking `networks/init.py`, the smallest real file in the repo and the right one
to read first.

If you already know what a strided convolution does, skip to section 6.

---

## 1. A convolution is a sliding dot product

A 2-D convolution has a small weight tensor — the **kernel** — of shape
`[out_channels, in_channels, k, k]`. It slides across the image, and at every
position computes the dot product of the kernel with the patch underneath. Every
output channel is one such kernel, so `Conv2d(1, 64, 4)` learns 64 different 4x4
feature detectors and produces 64 output maps.

Three numbers control the geometry:

| parameter | what it does |
|---|---|
| `kernel_size` (k) | how wide the patch is |
| `stride` (s) | how far the kernel moves between positions |
| `padding` (p) | how many zero rows/columns are added around the border first |

The output size follows directly:

```
out = floor((in + 2·p − k) / s) + 1
```

Every downsampling layer in this repo is `k=4, s=2, p=1`, which gives exactly half:

```
in = 256  →  (256 + 2 − 4) / 2 + 1  =  128
```

That is why the U-Net's `num_downs: 8` reaches a 1x1 bottleneck from a 256x256
crop — eight halvings of 256 is 1.

> **Why 4x4 and not 3x3.** A 3x3 kernel at stride 2 does not tile the input
> evenly: some input pixels are read twice, some once. A 4x4 kernel at stride 2
> gives every input pixel exactly two reads. This matters far more on the way *up*
> than on the way down — see section 4.

---

## 2. Receptive field — how much of the image one output pixel sees

One output value in layer 1 sees a 4x4 window. One output value in layer 2 sees a
4x4 window *of layer-1 outputs*, each of which saw 4x4 pixels — so it sees more
than 4x4, and how much more depends on the strides in between.

Work it out backwards from the output, one layer at a time:

```
r_out = 1
r_in  = (r_out − 1) · stride + kernel
```

Applied to the discriminator's five convolutions, whose strides are `2, 2, 2, 1, 1`:

| layer (from the end) | kernel | stride | receptive field |
|---|---|---|---|
| conv 5 | 4 | 1 | (1−1)·1 + 4 = **4** |
| conv 4 | 4 | 1 | (4−1)·1 + 4 = **7** |
| conv 3 | 4 | 2 | (7−1)·2 + 4 = **16** |
| conv 2 | 4 | 2 | (16−1)·2 + 4 = **34** |
| conv 1 | 4 | 2 | (34−1)·2 + 4 = **70** |

That is where the "70x70 PatchGAN" name comes from. It is not a configured number;
it is the arithmetic of three stride-2 layers followed by two stride-1 layers.
Change `model.discriminator.n_layers` and the 70 changes with it.

**Receptive field is the most useful number to keep in your head** when reading
these architectures — it tells you the physical scale at which each part of the
network reasons. At this project's 1 px = 1 mm, 70 px is a 70 mm window, about the
width of a kidney.

You can measure it rather than trusting the table:

```python
import torch, torch.nn as nn

net = nn.Sequential(
    nn.Conv2d(1, 1, 4, 2, 1), nn.Conv2d(1, 1, 4, 2, 1),
    nn.Conv2d(1, 1, 4, 2, 1), nn.Conv2d(1, 1, 4, 1, 1),
    nn.Conv2d(1, 1, 4, 1, 1),
)
x = torch.zeros(1, 1, 256, 256, requires_grad=True)
net(x)[0, 0, 15, 15].backward()          # one output pixel
rows = (x.grad[0, 0].abs().sum(1) > 0).nonzero()
print(int(rows[-1] - rows[0]) + 1)       # -> 70
```

The gradient of one output value is non-zero at exactly the input pixels that
influenced it. Counting them *is* measuring the receptive field.

---

## 3. Going down: strided convolution, not pooling

There are two ways to halve a feature map: pool it, or convolve it with stride 2.
This repo uses both, in different places, and the choice is not arbitrary.

| network | downsampling | file |
|---|---|---|
| U-Net generator | strided conv `k=4 s=2` | `networks/unet.py:72` |
| PatchGAN discriminator | strided conv `k=4 s=2` | `networks/patchgan.py:67-70` |
| StyleGAN2 blocks | blur, then `avg_pool2d` on the skip | `networks/stylegan2.py:308-319` |
| Registration U-Net | `MaxPool2d(2)` | `networks/registration.py:138-142` |

A strided convolution *learns* how to summarise the 2x2 neighbourhood. Max pooling
hard-codes "keep the brightest". For image synthesis you want the learned version.
For the registration network — which predicts a smooth displacement field, not
texture — the cheaper fixed pooling is fine, and it keeps the parameter count down,
which is itself load-bearing (see `12_reggan.md`).

---

## 4. Going up: transposed convolution, and the checkerboard artefact

`ConvTranspose2d` is the inverse geometry of `Conv2d`: it takes each input value,
multiplies the whole kernel by it, and *adds* the result into a larger output.

```
out = (in − 1) · s − 2·p + k
```

With `k=4, s=2, p=1` that doubles: `(128 − 1)·2 − 2 + 4 = 256`.

**The failure this shape prevents.** Because each output pixel accumulates
contributions from several kernel placements, output pixels can receive *unequal
numbers of contributions*. Where the kernel size is not divisible by the stride —
`k=3, s=2` is the classic offender — some output pixels get two contributions and
their neighbours one, producing a fixed high-frequency grid: the **checkerboard
artefact**. With `k=4, s=2` the overlap is even and the geometric cause is removed.

That does not make checkerboarding impossible — the network can still learn weights
that produce it — but it removes the structural reason. If you see a regular grid in
a synthetic CT, `gan_evaluation_guide.md` section 6 is the place to start; it is
more often a discriminator that found a shortcut than the upsampling geometry.

The alternative, used by StyleGAN2 here, separates the two jobs — interpolate
first, then filter:

```python
# networks/stylegan2.py:288-290, in effect
x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
x = blur(x)      # a fixed [1,3,3,1] FIR filter, smoothing the nearest-neighbour steps
```

No accumulation, so no uneven overlap, at the cost of one extra operation.

---

## 5. Normalisation: what BatchNorm and InstanceNorm actually do

Both rescale a feature map to roughly zero mean and unit variance. They differ only
in **what they average over**. Given activations of shape `[N, C, H, W]`:

```
BatchNorm2d      statistics over (N, H, W)  →  one mean/var per channel,
                                               shared across the batch
InstanceNorm2d   statistics over (H, W)     →  one mean/var per channel,
                                               per image
```

**The failure InstanceNorm prevents here.** BatchNorm's statistics couple the
samples in a batch: image 3's normalisation depends on images 1, 2 and 4. At batch
size 8 those statistics are noisy, and — more importantly for this project — a CPU
smoke test at batch 2 would not behave like a Kaggle run at batch 8. That
reproducibility argument is written into the code:

```
networks/init.py:56-63
  "batch statistics are noisy at the batch sizes this project trains at, and
   they couple samples within a batch, so a CPU smoke test at batch 2 would not
   behave like the Kaggle run at batch 8."
```

`training_strategies.md` Part 2 argues the same choice from the imaging side.

There is one deliberate exception: the StyleGAN2 discriminator's `MinibatchStdDev`
layer (`networks/stylegan2.py:218`) *does* couple samples in a batch, on purpose —
it is measuring batch diversity, which is the whole point. See `09_stylegan2_core.md`.

---

## 6. Activations, and why the generator ends in `tanh`

| activation | where | why |
|---|---|---|
| `LeakyReLU(0.2)` | every encoder block, every discriminator layer | a plain ReLU zeroes the gradient for all negative inputs; in a discriminator being pushed hard, whole channels can go permanently dead. The 0.2 slope keeps a gradient path alive. |
| `ReLU` | every decoder block | by the decoder, activations are the network's own construction rather than raw signal, and the harder nonlinearity is the pix2pix convention |
| `Tanh` | the generator's final layer only | bounds the output to `[−1, 1]` |

**The `tanh` is a contract with the dataset, not a free parameter.**
`data/dataset.py` scales both modalities to `[−1, 1]`, so the generator's output
range must match the target's. Change the dataset to `[0, 1]` and leave the `tanh`,
and half the output range becomes unreachable — the model then spends capacity
learning never to go negative.

That is also why the discriminator has **no** sigmoid on its output
(`networks/patchgan.py:95-97`): LSGAN and hinge consume raw scores, and the vanilla
objective uses `binary_cross_entropy_with_logits`, which applies the sigmoid
internally in a numerically stable way. Adding one would break all three.

---

## 7. Initialisation is load-bearing

The part most easily dismissed as boilerplate, and it is not.

**The failure it prevents.** PyTorch's default `Conv2d` initialisation is Kaiming,
tuned so activations keep unit variance through a deep ReLU stack. Applied to a
discriminator it makes the very first outputs large — which means D is *confident*
before it has learned anything. A confident D produces a saturated adversarial
gradient, the generator gets almost no signal, and the run dies in the first epoch.
Published pix2pix and CycleGAN results use `N(0, 0.02)` instead, and this is the
most common cause of "my GAN collapsed immediately".

### Minimal version — build it yourself

```python
import torch.nn as nn

def init_weights(net, gain=0.02):
    def f(m):
        name = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in name or "Linear" in name):
            nn.init.normal_(m.weight.data, 0.0, gain)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias.data, 0.0)
    net.apply(f)          # apply() walks every submodule, depth-first
    return net
```

`nn.Module.apply` is the whole mechanism: it calls your function on every module in
the tree, so you never enumerate layers yourself.

### The real one: `networks/init.py:22-51`

```python
def init_weights(net, init_type="normal", init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, "weight") and ("Conv" in classname or "Linear" in classname):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                ...
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif "BatchNorm2d" in classname or "InstanceNorm2d" in classname:
            if getattr(m, "weight", None) is not None:
                nn.init.normal_(m.weight.data, 1.0, init_gain)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias.data, 0.0)
```

Four things worth noticing:

1. **Matching is on the class *name*, as a substring.** `"Conv" in classname` catches
   `Conv2d`, `ConvTranspose2d` and `EqualizedConv2d` alike. Convenient, and also a
   trap — see section 8.
2. **Norm weights are centred on 1.0, not 0.0** (`:44`). A norm layer's weight is a
   multiplicative scale; initialising it at zero would erase the signal.
3. **`getattr(m, "weight", None) is not None`** (`:43`, `:45`) rather than `hasattr`.
   `InstanceNorm2d` is built here with `affine=False`, so the attribute exists but is
   `None`. `hasattr` would return `True` and the call would crash.
4. **Biases go to exactly zero**, always.

### The two companion helpers

`get_norm_layer` (`:54-70`) returns a **factory**, not a class:

```python
if norm_type == "instance":
    return lambda c: nn.InstanceNorm2d(c, affine=False, track_running_stats=False)
```

so a caller writes `norm_layer = get_norm_layer("instance")` once and then
`norm_layer(256)` wherever it needs a layer of a given width. `"none"` returns
`nn.Identity`, which is how the StyleGAN2 path gets no normalisation without any
`if` statements in the block code.

`uses_bias` (`:73-81`) answers one question: should the conv before this norm layer
carry a bias?

```
instance / none  →  True    InstanceNorm(affine=False) applies no shift,
                            so the conv must supply one
batch            →  False   BatchNorm has its own beta; a conv bias is
                            redundant and its gradient degenerate
```

You will see `bias=uses_bias(norm)` threaded through `unet.py`, `patchgan.py` and
`registration.py`. It is not a micro-optimisation — a redundant bias in front of
BatchNorm has no well-defined gradient direction, because any value of it produces
the same normalised output.

---

## 8. There are two initialisation regimes here, and they are incompatible

Everything above applies to the U-Net, the PatchGAN, the PatchNCE projection heads
and the registration network. It **must never be applied to StyleGAN2**.

StyleGAN2 uses *equalized learning rate*: weights are drawn from `N(0, 1)` and then
scaled at runtime by a constant computed from the fan-in
(`networks/stylegan2.py:114`, `:131`). That runtime scale assumes the stored weights
are unit-variance.

> ⚠️ **Passing a StyleGAN2 module to `init_weights` leaves every weight about 50x too
> small, and the network trains without complaining and learns nothing.** The warning
> is in the module docstring at `networks/stylegan2.py:41-47`. Every class in that
> file self-initialises; none is ever handed to `init_weights`.

This is a first example of a pattern that recurs throughout this codebase: the
dangerous failures are the *silent* ones. A crash is cheap. A run that completes 200
epochs and produces nothing is not.

---

## See also

- `02_gan_from_zero.md` — the next document: from an L1 regressor to a GAN
- `03_unet_generator.md` — where all of this is assembled into a generator
- `09_stylegan2_core.md` — the other initialisation regime, and why it exists
- `training_strategies.md` — the design rationale for these choices, argued from the imaging side
