# CycleGAN — translation without pairs

Everything so far assumed you have matched MRI/CT pairs. CycleGAN assumes you do not.
This document explains why unpaired translation is underdetermined, derives cycle
consistency as the constraint that fixes it, and walks `training/cyclegan.py` and
`training/image_pool.py`.

In this repo `exp8_cyclegan` is not primarily a candidate model — it is a
**measurement of what the QC pairing bought**. Keep that in view while reading.

---

## 1. The failure: unpaired translation is underdetermined

Give a generator a pile of MRIs and an unrelated pile of CTs, and ask it to make the
first look like the second. An adversarial loss alone can score that: D says whether
the output looks like real CT.

The problem is that **there are infinitely many mappings that satisfy it**. A generator
that ignores its input entirely and emits one memorised, perfectly convincing CT slice
scores perfectly. So does one that maps every MRI to a random CT. Nothing in the
adversarial objective connects the output to *this particular* input.

With pairs, L1 supplies that connection. Without them, you need something else.

## 2. The fix: make the translation reversible

Train a second generator that goes the other way, and require the round trip to return
the original:

```
F(G(mri)) ≈ mri          and          G(F(ct)) ≈ ct
```

**Why this constrains anything:** if `G` discarded the input and emitted a memorised
CT, `F` would have no way to recover *which* MRI it came from. The round trip can only
close if `G(mri)` retains enough information to reconstruct `mri`. Cycle consistency
buys injectivity — the mapping must be information-preserving — which is not the same
as correctness, but it is enough to rule out the degenerate solutions.

```
L_cycle = |F(G(mri)) − mri| + |G(F(ct)) − ct|
```

> **What it does not buy.** Cycle consistency does not force anatomical accuracy. A
> generator can learn a reversible *encoding* — hiding information in imperceptible
> high-frequency patterns — and satisfy the cycle while producing anatomically wrong
> output. This is a documented CycleGAN failure mode, and it is exactly why paired data
> is worth the QC effort. `exp8` measures that gap.

---

## 3. Minimal version — build it yourself

```python
import torch

def cyclegan_g_step(G_A2B, G_B2A, D_A, D_B, opt_G, gan, l1, real_A, real_B,
                    lam_cycle=10.0):
    fake_B = G_A2B(real_A);  rec_A = G_B2A(fake_B)      # A -> B -> A
    fake_A = G_B2A(real_B);  rec_B = G_A2B(fake_A)      # B -> A -> B

    loss = (gan.g_loss(D_B(fake_B)) + gan.g_loss(D_A(fake_A))
            + lam_cycle * (l1(rec_A, real_A) + l1(rec_B, real_B)))

    opt_G.zero_grad(set_to_none=True)
    loss.backward()
    opt_G.step()
    return float(loss)
```

Four generator passes per step: two forward translations and two reconstructions.
That is the intrinsic cost — CycleGAN is roughly four times the generator compute of
pix2pix per step, before the identity term.

---

## 4. The real one: four networks, two optimizers

```python
# training/cyclegan.py:88-94
self.netG_A2B = build_generator(cfg.model.generator)      # MRI -> CT
self.netG_B2A = build_generator(cfg.model.generator)      # CT  -> MRI
self.netD_B   = build_discriminator(cfg.model, spectral)  # judges CT
self.netD_A   = build_discriminator(cfg.model, spectral)  # judges MRI
```

All four come from the **same builders and the same config block** as the paired
experiments. Nothing architecturally new — only the wiring.

**One optimizer per *player*, not per network** (`:96-113`):

```python
self.optimizer_G = torch.optim.Adam(
    itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()), lr, betas)
self.optimizer_D = torch.optim.Adam(
    itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr_d, betas)
```

This is not just tidiness. `training/trainer.py` builds exactly two LR schedulers, one
for `.optimizer_G` and one for `.optimizer_D` (`:93-95`). Keeping those two attributes
meaning what the trainer expects is what lets a four-network model run through an
unmodified trainer. It is the same principle as the tap protocol in
`10_stylegan2_adapted.md` section 2: **satisfy the existing contract instead of
special-casing the caller.**

### `CycleGANModel` is a sibling of `Pix2PixNCEModel`, not a subclass

`training/cyclegan.py:55` — the step structure is too different for the
`extra_G_terms` hook to express. It reimplements `backward_D`, `backward_G` and
`optimize_parameters`, and reuses `build_generator`, `build_discriminator`, `GANLoss`,
`masked_l1`, `DiffAugment`, `ModelEMA` and the whole trainer. `12_reggan.md` shows the
other case, where a hook *is* enough.

### Three constructor errors (`:63-83`)

| condition | why |
|---|---|
| `lambda_gan == 0` | the only cross-domain link is adversarial; without it there is no reason for `fake_B` to resemble CT at all |
| `lambda_cycle == 0` | back to the underdetermined problem in section 1 |
| `model.discriminator.conditional` is true | **the MRI and CT in a batch come from different patients**, so a conditional D would be trained on a correspondence that does not exist |

That third one is a genuine trap: `conditional: true` is the base default and is right
for every other experiment. `exp8_cyclegan.yaml` must set it false, and the model
refuses to start otherwise rather than silently training D on noise.

---

## 5. `forward` — all four passes (`:186-191`)

```python
self.fake_B = self.netG_A2B(self.real_A)     # synthetic CT
self.rec_A  = self.netG_B2A(self.fake_B)     # back to MRI
self.fake_A = self.netG_B2A(self.real_B)     # synthetic MRI
self.rec_B  = self.netG_A2B(self.fake_A)     # back to CT
```

Computed once and reused by both `backward_D` and `backward_G`, as in
`05_the_training_step.md` section 3.

`set_input` (`:176-184`) carries **two** masks, with a fallback:

```python
self.mask_A = batch.get("mask_A", batch["mask"]).to(...)
```

so the model still runs on the paired dataset — which is what makes `exp8` vs `exp1`
a controlled comparison rather than two unrelated runs.

---

## 6. `backward_G` — three families, one backward (`:234-278`)

```python
if self.gan_active(epoch):
    g_a2b = gan.g_loss(self.netD_B(self.diffaug(self.fake_B)))
    g_b2a = gan.g_loss(self.netD_A(self.diffaug(self.fake_A)))
    loss_G = loss_G + self.plan.lambda_gan * (g_a2b + g_b2a)

cyc_A = masked_l1(self.rec_A, self.real_A, self.mask_A, self.criteria["l1"])
cyc_B = masked_l1(self.rec_B, self.real_B, self.mask_B, self.criteria["l1"])
loss_G = loss_G + self.plan.lambda_cycle * (cyc_A + cyc_B)

if self.plan.use_identity:
    idt_B = masked_l1(self.netG_A2B(self.real_B), self.real_B, self.mask_B, ...)
    idt_A = masked_l1(self.netG_B2A(self.real_A), self.real_A, self.mask_A, ...)
    weight = self.plan.lambda_cycle * self.plan.lambda_identity
    loss_G = loss_G + weight * (idt_A + idt_B)
```

**The cycle term is not gated on warm-up.** The adversarial term is; cycle consistency
is computed every step from epoch 0. That is what warm-up *is* here — train on the
reconstruction terms while D is held back.

**The cycle terms use the fresh `fake_*`, not the pool.** The image pool is a D-side
device only (section 7).

### The identity loss

A generator handed an image already in its target domain must leave it alone:
`G_A2B(real_B) ≈ real_B`. Both domains are single-channel here, so this is well
defined.

**Why it helps:** without it, nothing stops `G_A2B` from applying a global shift to
everything — a systematic brightness or contrast change that D tolerates and the cycle
undoes. The identity term pins the mapping to be the identity on inputs that need no
translation, which anchors overall tone.

**The weight is a product** (`:263`):

```python
weight = self.plan.lambda_cycle * self.plan.lambda_identity
```

`lambda_identity: 0.5` against `lambda_cycle: 10` gives an **effective weight of 5.0**.
This is the reference CycleGAN convention, kept because every published value for it is
quoted in those terms. Read as an absolute weight it would be twenty times weaker than
intended and the term would do nothing at all. See `losses/builder.py:55-59` and
`05_the_training_step.md` section 7.

**It costs two extra generator forwards** per step, taking the total to six.

---

## 7. `ImagePool` — training D against G's recent history

**The failure.** G and D oscillate: G finds an artifact that fools the current D, D
learns to catch it, G moves on, and D — trained only on G's most recent output —
*forgets* the earlier artifact. G can then cycle back to it. The pair orbits instead of
converging.

**Why pix2pix does not need this** (`training/image_pool.py:19-22`): there, L1 pins the
output to a specific target, so G cannot wander far between steps and there is little
history to forget. CycleGAN has no such anchor — nothing says what the output should
look like pixel by pixel — which is exactly why the oscillation has room to develop.

**The fix** (`:53-79`): keep a buffer of past generated images. Per image, with
probability 1/2 hand D the new one and store it; otherwise hand D a random stored one
and put the new one in its place.

```python
if len(self.images) < self.pool_size:
    self.images.append(image); out.append(image)
elif random.random() > 0.5:
    index = random.randrange(self.pool_size)
    out.append(self.images[index])
    self.images[index] = image                 # replace, so the buffer stays fresh
else:
    out.append(image)
```

D is now trained against a moving window of G's recent history rather than a single
instant of it, so an artifact it learned to catch stays caught.

**Used on the D side only** (`:210-211`):

```python
fake_in = self.diffaug(pool.query(fake))
```

That is also where the detach comes from — `query` returns detached tensors (`:67`),
so there is no explicit `.detach()` anywhere in `cyclegan.py`. The reason it is
mandatory:

> A stored tensor that still carried its graph would keep the generator's activations
> from several steps ago alive — **a slow memory leak that ends in an OOM tens of
> epochs in, long after the change that caused it.**

`pool_size: 0` makes `query` the identity, which is the right setting for an ablation.

**Pools are deliberately not checkpointed** (`:298-339`, `image_pool.py:25-29`): a few
dozen images of transient state that refills in well under an epoch, against ~50
tensors in every checkpoint. The consequence is that **bit-exact resume does not hold
for CycleGAN**, and `scripts/smoke_test.py` CHECK 6 asserts structure and step instead.

---

## 8. What `exp8` is actually measuring

```yaml
model.name: cyclegan
data.unpaired: true                # train split shuffled apart, seed 1337
data.unpaired_split_seed: 1337
model.discriminator.conditional: false
loss: { lambda_gan: 1, lambda_l1: 0, lambda_nce: 0,
        lambda_cycle: 10, lambda_identity: 0.5 }
train: { batch_size: 4, gan_warmup_epochs: 0 }
```

**Validation and test stay paired.** The train split is shuffled apart to simulate
unpaired data, but the model is scored against real pairs on the same metrics as every
other experiment.

That is what makes `exp8` vs `exp1_pix2pix` a clean one-variable comparison: same
architecture, same adversarial objective, **the data is the only thing that changed**.
The gap between them is the value of the pairing — which is the value of the manual QC
work in `qc_app/`. `model/README.md`'s ladder describes it as varying *the data*, in
the same way `exp7` varies *the frame* and `exp6` varies *the architecture*.

`gan_warmup_epochs: 0` because there is no paired reconstruction term to warm up on —
`LossPlan` would refuse a non-zero value (`losses/builder.py:100-111`).

---

## See also

- `05_the_training_step.md` — the paired step this one reshapes
- `12_reggan.md` — the other alternative model, which extends rather than replaces
- `13_adding_your_own.md` — when to subclass and when to write a sibling
- `model/README.md` — the experiment ladder and what each rung varies
