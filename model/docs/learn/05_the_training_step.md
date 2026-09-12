# One training step, start to finish

Two networks, two optimizers, one batch. What runs, in what order, and what is
detached where? This document walks a single call to `optimize_parameters` in
`training/pix2pix_nce.py` — the 15 lines that all five models in this repo are
organised around — and then explains `LossPlan`, the object that decides which
terms exist at all.

---

## 1. The failure that fixes the ordering

G and D are optimising against each other. That means **each one's loss depends on
the other's current weights**, and the order you update them in is not arbitrary.

The specific hazard is this: D's loss involves `G(x)`. If you compute D's gradient
without care, backpropagation walks straight through the generator and accumulates
gradients in G's parameters — gradients that came from D's objective, which is the
*opposite* of G's. Step the generator afterwards and you have trained G to be
better at being detected.

The fix is one call:

```python
fake_in = self._d_input(self.real_A, self.fake_B.detach())
```

`.detach()` returns a tensor with the same values and no history. The graph stops
there, D's backward reaches D's parameters and nothing further.

**The mirror-image rule applies in G's step:** there, `fake_B` is used *without*
detaching, because the whole point is for the gradient to flow through D into G.

```python
# training/pix2pix_nce.py:332   D step   — detached
fake_in = self._d_input(self.real_A, self.fake_B.detach())

# training/pix2pix_nce.py:375   G step   — not detached
fake_in = self.diffaug(self._d_input(self.real_A, self.fake_B))
```

Both lines pass `fake_B` to `netD`. The one-word difference is the entire
separation between the two players.

---

## 2. Minimal version — build it yourself

```python
import torch

def train_step(G, D, opt_G, opt_D, gan_loss, l1, x, y, lambda_l1=100.0):
    fake = G(x)                                  # ONE generator pass, reused below

    # ── D step ──────────────────────────────────────────────────────
    opt_D.zero_grad(set_to_none=True)
    pred_real = D(torch.cat([x, y], 1))
    pred_fake = D(torch.cat([x, fake.detach()], 1))       # detached
    loss_D, _ = gan_loss.d_loss(pred_real, pred_fake)
    loss_D.backward()
    opt_D.step()

    # ── G step ──────────────────────────────────────────────────────
    opt_G.zero_grad(set_to_none=True)
    pred_fake = D(torch.cat([x, fake], 1))                # NOT detached
    loss_G = gan_loss.g_loss(pred_fake) + lambda_l1 * l1(fake, y)
    loss_G.backward()
    opt_G.step()

    return float(loss_D), float(loss_G)
```

That is a complete pix2pix training step. Note `fake = G(x)` is computed **once** and
used in both halves — recomputing it would double the cost and give D and G slightly
different generators to argue about.

Everything the repo adds to this is mixed precision, regularisers that need their own
passes, and the machinery for turning terms off.

---

## 3. The real one: `optimize_parameters` (`training/pix2pix_nce.py:453-468`)

```python
def optimize_parameters(self, batch, epoch, scaler, amp_ctx):
    self.set_input(batch)
    self.forward()                     # fake_B = netG(real_A), once

    stats = {}
    if self.gan_active(epoch):
        stats.update(self.backward_D(scaler, amp_ctx))     # D first
    stats.update(self.backward_G(epoch, scaler, amp_ctx))  # then G
    scaler.update()                    # ONCE, after both

    if self.ema is not None:
        self.ema.update(self.netG, epoch)

    self.global_step += 1
    return {k: float(v) for k, v in stats.items()}
```

Five things this fixes:

| decision | line | why |
|---|---|---|
| one `forward()`, reused | `:456` | D and G judge the same generated image; also halves the cost |
| D before G | `:459-461` | G is then critiqued by a D that has already seen this batch |
| the D step is *skipped* during warm-up | `:459` | `gan_active(epoch)` is `epoch >= warmup_epochs` (`:210-212`) |
| `scaler.update()` exactly once | `:462` | the AMP scale factor is per *step*, not per backward — see section 6 |
| everything returned as `float` | `:467` | nothing that could hold a graph reference escapes into the logger |

**`gan_active` is a real branch, not a multiply by zero.** During the first
`train.gan_warmup_epochs` (default 5), `backward_D` never runs, `optimizer_D` is
never stepped, and no `D_*` key appears in the stats. That absence has consequences
downstream — it is the reason `training/trainer.py` rebuilds `metrics.csv` from
scratch every epoch rather than appending, which `06_the_trainer.md` covers.

---

## 4. `backward_D` (`:321-366`)

Reading it in order:

```python
with ctx:
    real_in = self._d_input(self.real_A, self.real_B)
    fake_in = self._d_input(self.real_A, self.fake_B.detach())

    real_in = self.diffaug(real_in)
    fake_in = self.diffaug(fake_in)          # two SEPARATE calls

    pred_real = self.netD(real_in)
    pred_fake = self.netD(fake_in)
    loss_D, stats = self.criteria["gan"].d_loss(pred_real, pred_fake)

self.optimizer_D.zero_grad(set_to_none=True)
scaler.scale(loss_D).backward()
if self.grad_clip > 0:
    scaler.unscale_(self.optimizer_D)
    nn.utils.clip_grad_norm_(self.netD.parameters(), self.grad_clip)
scaler.step(self.optimizer_D)
```

`_d_input` (`:204-208`) is where conditioning happens — `cat([source, target], 1)`
when `model.discriminator.conditional`, else the target alone.

**The two `diffaug` calls are deliberately separate** (`:334-335`). Real and fake
must each draw their *own* random augmentation parameters, under the same policy.
They share the distribution, not the sample. `08_stabilisers_in_code.md` explains why
that is what makes the augmentation cancel out of the objective.

**`zero_grad(set_to_none=True)`** rather than the default. Setting gradients to `None`
instead of zeroing them in place skips a kernel launch per parameter and lets the
optimizer skip parameters that received no gradient at all — which matters here,
because during warm-up whole networks receive none.

The `if do_r1:` branches are R1 regularisation, which forces this step into fp32 and
out of the scaler entirely. Section 6 and `08_stabilisers_in_code.md`.

---

## 5. `backward_G` (`:368-451`) — assembling a composite loss

The pattern is: start at zero, add each active term, one backward.

```python
with amp_ctx:
    loss_G = torch.zeros((), device=self.device)

    if self.gan_active(epoch):
        ...
        loss_G = loss_G + self.plan.lambda_gan * g_gan
        stats["G_GAN"] = g_gan.detach()

    if self.plan.use_l1:
        g_l1 = masked_l1(self.fake_B, self.real_B, self.mask, self.criteria["l1"])
        loss_G = loss_G + self.plan.lambda_l1 * g_l1
        stats["G_L1"] = g_l1.detach()

    if self.plan.use_nce:
        g_nce, per_layer = self.compute_nce(self.real_A, self.fake_B)
        loss_G = loss_G + self.plan.lambda_nce * g_nce
        ...

    loss_G = loss_G + self.extra_G_terms(stats)
```

Which realises the objective from `model/README.md`:

```
L_total = λ_gan · L_cGAN + λ_L1 · L_L1 + λ_NCE · L_PatchNCE
```

Four details:

**Every logged term is `.detach()`ed at creation.** `stats["G_L1"] = g_l1.detach()`,
not `g_l1`. A stats dict holding live tensors keeps the entire graph alive until the
dict is dropped — a slow leak that ends in OOM tens of epochs later.

**`extra_G_terms` (`:406`, defined `:236-248`) is an extension hook**, returning a
zero scalar in the base class. RegGAN overrides it to add the registration terms,
and that is the *entire* mechanism by which RegGAN differs in the G step. See
`12_reggan.md`.

**Optimizers are collected *after* the autocast block** (`:410`), never before:

```python
# Collected AFTER the autocast block, never before: optimizer_F comes
# into existence during the NCE forward pass above.
optimizers = self.g_step_optimizers()
```

`optimizer_F` optimises the PatchNCE projection heads, whose widths depend on the
encoder's channel counts, so it cannot be constructed until the first NCE forward has
run. Collecting the list one line earlier would silently drop it, and the heads would
never train. `07_patchnce.md` covers the lazy-construction chain.

**Only `netG`'s parameters are clipped** (`:416-418`), and only `optimizer_G` is
unscaled. Documented at `:245-247`.

---

## 6. Mixed precision: three regimes in one function

This is the subtlest thing in the file. Autocast runs eligible ops in fp16 for speed;
`GradScaler` multiplies the loss by a large factor so small gradients do not flush to
zero in fp16, then divides it back out before the optimizer steps.

There are exactly three regimes in this codebase:

| regime | where | mechanism |
|---|---|---|
| **normal** | the usual G and D steps | `scaler.scale(loss).backward()` → optional `scaler.unscale_(opt)` + clip → `scaler.step(opt)` |
| **fp32 region inside a scaled backward** | RegGAN's correction term (`training/reggan.py:119`) | `torch.autocast(enabled=False)` around the computation, but still inside the ordinary `scaler.step` — no double backward, so nothing needs splitting |
| **fully unscaled fp32 step** | R1 (`:328`, `:353-357`) and path-length (`:437-449`) | plain `.backward()` + `optimizer.step()`, the scaler bypassed entirely |

The reason for the third is written into the source at `:432-435`:

> It needs a double backward through G, which is fragile and overflows in fp16 under
> autocast, so it has to run unscaled in fp32 — and **mixing a scaled and an unscaled
> backward into one `GradScaler` step is exactly where silent gradient corruption
> lives.**

That is why path-length is a *separate optimisation step* rather than another term
added to `loss_G`. It is not stylistic.

**Lazy regularisation** appears twice with the identical shape:

```python
if global_step % every == 0:
    loss = weight * penalty * every        # scaled by the interval
```

R1 at `:349`, path-length at `:440`. Running an expensive penalty every 16 steps and
multiplying it by 16 recovers most of the benefit at a fraction of the cost.

---

## 7. `LossPlan` — how a zero λ removes a code path

`losses/builder.py:39` holds every weight and answers one kind of question:

```python
@property
def use_gan(self):
    """False means: build no discriminator, run no D pass, save no D state."""
    return self.lambda_gan > EPS
```

`EPS = 1e-12` (`:36`) guards against a config written as `1e-12` by accident and
against float noise.

**The design rule is that a zero lambda removes the entire code path, not just the
contribution.** `build_losses` (`:225-257`) returns a dict where inactive keys are
*absent*, so `if "gan" in modules` is a structural statement rather than a numeric
one. `exp0_l1_only` does not build a discriminator at all
(`training/pix2pix_nce.py:70-77`), does not run a D pass, and does not save D state.

That matters for more than tidiness: a `lambda_gan: 0` run that still built and
stepped a discriminator would be slower, would produce misleading `D_*` logs, and
would write checkpoints that are not comparable to a real ablation.

### Three errors it raises at construction

| condition | line | why |
|---|---|---|
| `lambda_corr > 0` with `lambda_smooth == 0` | `:72-80` | an unpenalised deformation field can warp almost any prediction onto almost any target — the loss falls to zero while the generator learns nothing |
| every lambda zero | `:82-89` | the generator has no training signal at all |
| `gan_warmup_epochs > 0` with no reconstruction term | `:100-111` | warm-up trains on the reconstruction terms while D is held back; with none, `backward_G` calls `.backward()` on a fresh zero scalar that never entered the graph, and PyTorch raises "element 0 of tensors does not require grad" five layers from the config that caused it |

All three are configurations that would otherwise fail late, confusingly, or not at
all. This is the same principle as `01_building_blocks.md` section 8: **make the
silent failures loud.**

### One convention that is easy to misread

```python
# losses/builder.py:55-59
self.lambda_identity = float(cfg.get_path("loss.lambda_identity", 0.0))
```

`lambda_identity` is a **fraction of `lambda_cycle`**, not an absolute weight — the
reference CycleGAN convention, kept because every published value for it (0.5) is
quoted in those terms. Against `lambda_cycle: 10`, the effective weight is 5.0. Read
as absolute it would be twenty times weaker than intended and the term would do
nothing. See `11_cyclegan.md`.

---

## 8. `masked_l1` (`losses/builder.py:260-273`)

```python
per_pixel = criterion(pred, target)          # criterion has reduction="none"
denom = mask.sum().clamp(min=1.0)
return (per_pixel * mask).sum() / denom
```

`build_losses` constructs `nn.L1Loss(reduction="none")` specifically so this is
possible. The reason is a dataset property:

> The dataset zero-pads variable-sized slices up to a common size. Padded pixels are
> identical in prediction and target, so including them contributes near-zero error
> over a large area and **dilutes the loss by however much padding a given slice
> happened to need** — making a 180x180 slice look better than a 430x430 one for
> reasons that have nothing to do with the model.

`clamp(min=1.0)` on the denominator guards the degenerate all-padding case.

> **Note the `or`s in `build_losses`** (`:240-250`): the L1 *criterion* is built when
> `use_l1 or use_corr or use_cycle`. `exp7_reggan` and `exp8_cyclegan` both run with
> `lambda_l1: 0` and still need an L1 criterion — for the warped comparison and the
> cycle terms respectively.

---

## See also

- `06_the_trainer.md` — the epoch loop that calls `optimize_parameters`
- `07_patchnce.md` — what `compute_nce` does, and the lazy `optimizer_F`
- `08_stabilisers_in_code.md` — EMA, DiffAugment, R1 and path-length in detail
- `11_cyclegan.md`, `12_reggan.md` — the two models that reshape this step
- `loss_function_guide.md` — what each λ does to the result, and how to tune it
