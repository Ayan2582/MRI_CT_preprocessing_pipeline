# The stabilisers, in code

GAN training is a game between two networks, and games do not converge the way loss
minimisation does. A stabiliser is a modification that prevents one specific way the
game goes wrong. `training_strategies.md` Part 3 argues *which* to use and *when*;
this document shows the code, and covers one thing documented nowhere else — the
mixed-precision discipline that three of them force on the training step.

---

## 1. The seven stabilisers and where they live

| stabiliser | failure it prevents | code | on by default |
|---|---|---|---|
| spectral norm on D | D becomes an arbitrarily confident classifier and saturates the gradient | `networks/patchgan.py:67-70` | yes |
| label smoothing | same, milder | `losses/gan_loss.py:170-173` | yes |
| TTUR | same, by giving D a slower learning rate | `training/pix2pix_nce.py:98-103` | no |
| DiffAugment | D memorises 1687 slices | `training/diffaug.py` | yes |
| R1 | same, by flattening D's decision spikes | `losses/gan_loss.py:147` | no |
| generator EMA | everything oscillates step to step | `training/ema.py` | yes |
| path-length reg | G's style-to-image map is ill-conditioned | `training/pix2pix_nce.py:285` | no |

Spectral norm is covered in `04_patchgan_discriminator.md` section 6, and label
smoothing in `02_gan_from_zero.md` section 6. This document takes the remaining five.

---

## 2. Generator EMA — the one that changes every number you report

**The failure.** G and D are chasing each other, so G's weights oscillate around a
good solution rather than settling into it. The weights at the end of epoch 47 are
not better than epoch 46's in any stable sense; they are a different point on the
same orbit. Evaluate them and you measure the orbit, not the progress.

**The fix.** Keep a second copy of the generator that is an exponential moving average
of the live one:

```
ema = decay · ema + (1 − decay) · live
```

Averaging over the orbit lands nearer its centre than any single point on it.

### Minimal version — build it yourself

```python
import copy, torch

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)          # never trained, only averaged into

    @torch.no_grad()
    def update(self, model):
        ema_params = dict(self.ema.named_parameters())
        for name, p in model.named_parameters():
            ema_params[name].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
```

### The real one: `training/ema.py:39-105`

Same three lines, plus two rules that matter.

**Before `start_epoch`, it copies rather than averages** (`:65-70`):

```python
if epoch < self.start_epoch:
    self.copy_from(model)
    return
```

> Before `start_epoch` the live weights are still leaving their random
> initialisation and there is nothing worth remembering about where they have been.

With `decay=0.999` the shadow has a ~1000-step memory. Averaging in the random
initialisation would contaminate the shadow for that entire window — roughly five
epochs of validation numbers measured on a model that is partly noise.

**Buffers are copied, not averaged** (`:81-84`). Running statistics are already
running averages; averaging them a second time biases them toward staleness. With
`InstanceNorm(track_running_stats=False)` there are none here, but the rule is
correct for anyone who switches to BatchNorm.

**Picking `decay`.** The window is `1 / (1 − decay)` steps, which the constructor logs:

```
0.999  →  ~1000 steps  →  ~5 epochs at ~210 iterations/epoch
0.9999 →  ~10000 steps →  ~48 epochs — too slow, it lags real progress
```

`update` is called once per generator step from `optimize_parameters`
(`training/pix2pix_nce.py:465`).

> **This is not a diagnostic — it is what gets evaluated.** `eval.use_ema` defaults to
> true, so `generator_for_eval()` (`:472-482`) returns `ema.module` and **every
> validation metric and every sample panel in this project is measured on the averaged
> weights.** `gan_evaluation_guide.md` §2 explains why.

---

## 3. DiffAugment — augmentation that does not leak into the generator

**The failure**, stated precisely in `training/diffaug.py:6-13`: with 1687 training
slices, a PatchGAN has ample capacity to memorise the texture of 33 patient folders.
Once it has, it is no longer answering "does this look like real CT?" — it is
answering "have I seen this exact patch before?". Its gradient stops carrying
information about realism, and G starts chasing artifacts that game a critic that is
no longer measuring anything.

The tell: **D near 100% accurate on training data while sample quality plateaus or
drifts backwards.** That is failure C in `gan_evaluation_guide.md`.

**Why ordinary augmentation does not work here.** Augment the reals and you have
taught G to reproduce the augmentation — G's target distribution has changed. Two
properties fix that:

1. **Applied to both real and fake, immediately before D.** The transformation cancels
   out of the objective, so D's job gets harder without changing what G aims for.
2. **Differentiable.** Gradients flow back through the augmentation into G. A
   non-differentiable augmentation would sever that path and silently turn the
   adversarial term into noise.

That second property is why this lives coupled to the discriminator step and not in
the dataset transform pipeline.

### The three policies (`:110-114`)

| policy | functions | note |
|---|---|---|
| `color` | brightness, saturation, contrast | **saturation is a no-op on one channel** (`:60-62`) — it interpolates toward the channel mean, which for a single channel is itself |
| `translation` | up to ±12.5% shift, zero-filled | implemented as a 1-pixel `F.pad` plus a gather on a meshgrid — differentiable, unlike a slice-and-copy |
| `cutout` | zero one random rectangle at 50% of each dimension | a multiplicative mask, so differentiable by construction |

Unknown policy names raise (`:133-137`) rather than being ignored.

### Three application sites, and one that is easy to miss

```python
training/pix2pix_nce.py:334-335   real and fake, in backward_D
training/pix2pix_nce.py:375       the fake, in backward_G
training/cyclegan.py:210-211, :241-242    the same pattern, both directions
```

**The `backward_G` site is the one people forget.** If D judges augmented images
during its own step but clean images during G's step, G is being critiqued by a
discriminator operating outside its training distribution.

**Real and fake are always two separate calls**, each drawing its own random
parameters (`diffaug.py:121-125`). They share the *distribution*, not the *sample* —
applying the identical shift to both would just translate the whole comparison.

### Two medical-specific policy notes

`configs/base.yaml` defaults to `color,translation,cutout`, but
`exp6_stylegan2_fitted.yaml` **drops `color`**, citing `patchgan.py:73-75`: that
discriminator's first conv has no norm specifically so it can see absolute intensity,
and both modalities are calibrated to a fixed range. Randomising brightness destroys
exactly the signal the architecture was shaped to preserve.

Rotation is absent for a data reason (`diffaug.py:33-38`): manual QC corrected
translation on these pairs but could not correct in-plane rotation, so a rotational
residual already remains. There is no sense in adding more of it.

---

## 4. R1 — flattening D's memorisation spikes

Covered mechanically in `02_gan_from_zero.md` section 7. What matters here is *how it
is applied*.

```python
# training/pix2pix_nce.py:322-349
do_r1 = self.r1_enabled and (self.global_step % self.r1_every == 0)
ctx = torch.autocast(device_type=self.device.type, enabled=False) if do_r1 else amp_ctx

with ctx:
    ...
    if do_r1:
        real_in.requires_grad_(True)
    pred_real = self.netD(real_in)
    ...
    if do_r1:
        penalty = r1_penalty(pred_real, real_in)
        loss_D = loss_D + (self.r1_gamma / 2.0) * penalty * self.r1_every
```

Three things:

**`real_in.requires_grad_(True)`** — R1 differentiates D's output with respect to its
*input*, so the input has to be a leaf that tracks gradients. Forgetting this raises
rather than failing silently, which is the good outcome.

**Lazy regularisation.** Applied every `r1_every` steps (default 16) and multiplied by
that factor. Running an expensive penalty at 1/16th the frequency and scaling it up
recovers most of the benefit at a fraction of the cost.

**The whole D step drops to fp32 when R1 runs** (`:328`). See section 6.

`γ` is worth a note: `training_strategies.md` gives the heuristic
`γ = 0.0002 · N² / M` for resolution N and batch M, which is where `exp5`'s 3.3 (batch
4) and `exp6`'s 1.6 (batch 8) come from. It is not a number to guess.

---

## 5. Path-length regularisation — and how it differs here

**The failure.** In a style-based generator, a fixed-size step in W should move the
image by a roughly fixed amount. If the mapping is wildly sensitive in some directions
and flat in others, it is ill-conditioned: small style changes cause large image
changes in unpredictable places, and optimisation through it is unstable.

**The measurement** (`training/pix2pix_nce.py:285-317`):

```python
fake, ws = self.netG.synthesise_with_styles(self.real_A)

noise = torch.randn_like(fake) / math.sqrt(fake.shape[2] * fake.shape[3])
grad = torch.autograd.grad(outputs=(fake * noise).sum(), inputs=ws,
                           create_graph=True, only_inputs=True)[0]

lengths = grad.square().sum(dim=2).mean(dim=1).sqrt()
mean = self.pl_mean.lerp(lengths.detach().mean(), self.pl_decay)
return (lengths - mean).square().mean(), lengths.detach().mean()
```

Push a random unit direction in *image* space back through the Jacobian into W space,
measure the resulting length, and penalise its deviation from a running mean of past
lengths. The `/ sqrt(H·W)` makes the expected magnitude resolution-independent.

The target is a **running mean, not a constant**. The objective is not "make the
Jacobian a specific size"; it is "make it the *same* size everywhere", and the running
mean supplies the moving target that expresses that.

> **How this differs from the paper here** (`:296-300`). StyleGAN2 draws `w` from the
> mapping network's Gaussian prior, so it has unlimited free samples. **There is no
> prior in a translation model — every `w` is the encoding of a real patient** — so the
> estimate comes from whatever batch is in flight, which makes it noisier than the
> published version at these batch sizes. This is a genuine adaptation cost of using
> StyleGAN2 for translation, and `10_stylegan2_adapted.md` covers the family of them.

`synthesise_with_styles` exists as a separate method (`networks/stylegan2.py:581-607`)
purely so this penalty can reach `ws`. `forward()` would not return it, and `ws` is
deliberately **not** detached.

---

## 6. Mixed precision: why two of these get their own step

Autocast runs eligible ops in fp16; `GradScaler` multiplies the loss by a large factor
so small fp16 gradients do not flush to zero, then divides it out before stepping.

R1 and path-length both need a **double backward** — differentiating a quantity that is
itself a gradient. In fp16 that overflows. So both leave the scaler entirely:

```python
# R1, training/pix2pix_nce.py:353-357
loss_D.backward()                    # plain, not scaler.scale(...)
if self.grad_clip > 0:
    nn.utils.clip_grad_norm_(self.netD.parameters(), self.grad_clip)
self.optimizer_D.step()              # plain, not scaler.step(...)
```

And path-length is a **separate optimisation step**, not another term in `loss_G`
(`:437-449`). The reason is at `:432-435`:

> **Mixing a scaled and an unscaled backward into one `GradScaler` step is exactly
> where silent gradient corruption lives.**

The three regimes, collected:

| regime | example | mechanism |
|---|---|---|
| normal | ordinary G and D steps | `scaler.scale(loss).backward()` → `scaler.step(opt)` |
| fp32 *region* inside a scaled backward | RegGAN's correction term (`training/reggan.py:119`) | `autocast(enabled=False)` around the computation, still inside the ordinary `scaler.step` — no double backward, nothing to split |
| fully unscaled fp32 *step* | R1, path-length | plain `.backward()` + `optimizer.step()` |

**The rule to carry away: any penalty needing `create_graph=True` needs its own fp32
step.** If you add one (`13_adding_your_own.md`), copy the path-length block's shape,
not the L1 term's.

---

## 7. TTUR, in one paragraph

Two Time-scale Update Rule: give D a different learning rate from G.

```python
# training/pix2pix_nce.py:98-103
lr_d = cfg.get_path("stabilizers.ttur.lr_d", lr) if ttur_enabled else lr
self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr_d, betas=betas)
```

That is the entire implementation. It is off by default because the `0.5` in
`gan_loss.py:95` already halves D's effective rate, and stacking two handicaps on the
same player without measuring is how you get failure B — a collapsed discriminator —
instead of failure A. `training_strategies.md` Part 3 gives the order to try them in.

---

## See also

- `training_strategies.md` — which stabiliser to reach for, and what each costs
- `gan_evaluation_guide.md` — the four failure modes these are named against
- `05_the_training_step.md` — where each of these hooks into the step
- `09_stylegan2_core.md` — the architecture path-length regularisation was designed for
