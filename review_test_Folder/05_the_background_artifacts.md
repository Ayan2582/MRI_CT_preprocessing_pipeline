# 05 — The corner blob, the streaks, and the generator built for a different job

**Findings 4 and 5.** Two things that are not the main cause but explain most of what
looks alarming in the epoch-109 and epoch-199 panels — and one of them is a hint about
how the cycle loss is being satisfied.

---

## 1. The failure: an artifact that does not depend on the input

At epoch 109 a bright blob appears near the top-left corner of the brain coronal, the
brain axial, and both musculoskeletal panels. Four different anatomies, four different
patients, one blob, one position. At epoch 199 it is still on the musculoskeletal slice.
The brain panels also pick up thin vertical streaks near the frame edges.

Anything that appears in the same place regardless of input is not a translation error.
It is something being *added*.

## 2. Why the background is unconstrained

Both non-adversarial terms in exp8 go through `masked_l1`
(`losses/builder.py:260-273`):

```python
per_pixel = criterion(pred, target)
denom = mask.sum().clamp(min=1.0)
return (per_pixel * mask).sum() / denom
```

and the mask is **zero on padding** (`data/dataset.py:65-79`):

```python
mask = np.zeros_like(out, dtype=np.float32)
mask[top:top + h, left:left + w] = 1.0
```

Multiply by zero, and no gradient reaches the generator's weights through those pixels.
So in exp8 — where `masked_l1` is the entire non-adversarial objective, cycle and
identity both — **the padding region is trained by the discriminator alone**.

That is fine in exp0–exp4: `lambda_l1: 100` also skips the padding, but the conditional
D there sees `cat[MRI, CT]` and can flag a blob that has no counterpart in the MRI.
exp8's D is unconditional and cannot.

And what little supervision D could offer there, DiffAugment removes. Every real and
every fake passes through `translation` (±12.5% of each dimension, zero-filling) and
`cutout` (a zeroed rectangle covering 25% of the area) before D sees it —
`training/diffaug.py:66-105`, policy `color,translation,cutout` from `base.yaml`. A
small fixed-position corner artifact is translated out of position or zeroed outright on
a large fraction of steps. It is close to the ideal place to hide something from this
discriminator.

Nothing in the panel MAE penalises it much either: `trainer.py:388` averages over the
full padded frame, so a small blob barely moves the number. The error maps show it
plainly — look at the top-left of the epoch-199 musculoskeletal error map.

## 3. What the blob probably is

This is the one part of this review that is inference rather than a code citation, so it
is flagged as such — but the shape of the evidence is specific.

`model/docs/learn/11_cyclegan.md` §2 already names the mechanism:

> A generator can learn a reversible *encoding* — hiding information in imperceptible
> high-frequency patterns — and satisfy the cycle while producing anatomically wrong
> output.

The cycle term requires `G_B2A(G_A2B(mri))` to approximate `mri`. It does **not**
require `G_A2B(mri)` to be a correct CT. If `G_A2B` writes a compact code into a region
no loss is watching, `G_B2A` can read it back and reconstruct the MRI — cycle loss
falls, and the visible anatomy is free to drift wherever the adversarial prior sends it.

Two observations fit: the artifact sits exactly where the reconstruction terms are blind
(§2), and it appeared at the same time the anatomy started going wrong (epoch 109, doc
01).

**The check that would settle it**, using your saved checkpoint, no retraining:

1. Take one MRI, compute `fake_B = G_A2B(mri)` and `rec_A = G_B2A(fake_B)`. Record the
   masked L1 of `rec_A` against `mri`.
2. Zero the padding region of `fake_B` — or overwrite the corner block with the
   background value — and recompute `rec_A`.
3. If the cycle loss jumps sharply, the reconstruction depends on the background and the
   encoding is real. If it barely moves, the blob is only an unpenalised decoder
   artifact and finding 5 reduces to §2 alone.

Either way §2's fix applies; this only decides how much it buys.

## 4. Why D cannot police global tone either

Related, and worth naming because it interacts with doc 02.

DiffAugment's `color` policy is two functions (`training/diffaug.py:49-56`):

```python
def rand_brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, ...) - 0.5)     # +/- 0.5 offset

def rand_contrast(x):
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    factor = torch.rand(x.size(0), 1, 1, 1, ...) + 0.5          # scale in [0.5, 1.5]
    return (x - mean) * factor + mean
```

Images live in `[-1, 1]` (tanh output, `networks/unet.py:100`). A ±0.5 offset is
**±25% of the full dynamic range**, and contrast is scaled by up to 1.5×. Applied to
reals and fakes alike, this deliberately makes D **invariant to global brightness and
contrast**. That is the intended behaviour, and it is the right trade in exp0–exp4,
where `lambda_l1: 100` pins absolute intensity and D only needs to judge texture.

In exp8, `lambda_l1: 0` and the adversary is the *only* cross-domain signal. So the
policy blinds D to precisely the global intensity mapping — MRI intensity to CT HU —
that the generator most needs to learn from it. The remaining tone anchor is the
identity loss, which `11_cyclegan.md` §6 justifies on exactly these grounds — and which,
per doc 02, has its minimum at the identity function.

That is the loop:

```
DiffAugment removes D's authority over global tone
        -> the identity term becomes the sole tone anchor
        -> the identity term's minimum is G = I
        -> epoch 9
```

## 5. Finding 4: the U-Net is the wrong generator for this job

`base.yaml` gives every experiment the same generator, and exp8 reuses it deliberately —
`cyclegan.py:86-89` notes this is what keeps exp8 differing from exp1 "in its objective
and its data pairing, not in its capacity." Good discipline for the comparison. But the
generator is `type: unet, num_downs: 8, ngf: 64` (`networks/unet.py:53-116`), which at a
256 px crop reaches a **1×1 bottleneck**, with skip connections on every level.

Reference CycleGAN uses a 9-block ResNet with two downsamplings. That is not incidental:

- **A 1×1 bottleneck means all spatial structure must arrive through the skips.** The
  architecture is biased toward "output ≈ input, recoloured." That is exactly what
  `lambda_l1: 100` wants in pix2pix, and exactly wrong when the task is to change what
  the tissue *is*. It makes `G = I` cheap in parameter space, not just in loss — which
  is why epoch 9 is such a clean copy rather than a muddle.
- **The skips also hand the cycle a free ride.** `rec_A = G_B2A(G_A2B(A))` can be
  satisfied largely through low-level skip paths carrying the MRI's fine detail through
  both networks. `train/G_cycle` falls without the anatomy at the bottleneck ever
  becoming correct — the same disconnect §3 describes, by a second route.

This does not by itself explain the failure, and swapping the generator alone would not
fix it. It is listed because it makes findings 1 and 3 easier to fall into, and because
"exp8 uses the same generator as exp1" — which reads as rigour — is also what stops exp8
from being a fair test of CycleGAN. Doc 06 §4 covers the trade.

---

## Not a cause here, but it will bite you

`train.image_pool_size: 0` is documented as the ablation setting in `base.yaml:325` and
called "the right setting for an ablation" at `training/image_pool.py:41-43`. **It would
crash exp8.**

`cyclegan.py:211` has no explicit detach:

```python
fake_in = self.diffaug(pool.query(fake))
```

It relies on `query()` detaching internally — which it does at `image_pool.py:67`, but
only after the short-circuit at `image_pool.py:62-63`:

```python
if self.pool_size == 0:
    return images
```

With `pool_size == 0` the fakes reach D still carrying the generator's graph.
`backward_D` traverses it, then `backward_G` goes through the same graph again and
raises *"Trying to backward through the graph a second time."* Compare
`training/pix2pix_nce.py:332`, which detaches explicitly and does not have the problem.

Unrelated to this run — exp8 used the default `image_pool_size: 50` — but the documented
ablation path is broken.
