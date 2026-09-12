# From a regressor to a GAN

Why does adding a second network that only ever outputs a number make synthetic CT
look sharper? This document builds the adversarial objective from the ground up —
starting with a plain L1 regressor, showing precisely why it blurs, and deriving
the discriminator as the fix. It then walks `losses/gan_loss.py` in full.

The three objectives here (`lsgan`, `vanilla`, `hinge`) are argued *as choices* in
`training_strategies.md` Part 1. This document explains what they are.

---

## 1. Start with no GAN at all — `exp0_l1_only`

The simplest possible MRI→CT model is a regressor. Feed the U-Net an MRI slice,
compare its output to the real CT, minimise the mean absolute error:

```
L_L1 = mean(|G(mri) − ct|)
```

That is exactly what `exp0_l1_only.yaml` runs: `lambda_gan: 0`, `lambda_nce: 0`,
`lambda_l1: 100`. There is no discriminator — `training/pix2pix_nce.py:70-77` does
not build one, and logs "no discriminator built at all".

**This works.** It converges, it is stable, and it will produce the *best* raw
`mae_norm` of any experiment in the ladder. It is also the one you must not ship.

---

## 2. Why L1 blurs — the conditional-median argument

Here is the mechanism, and it is not a quirk of implementation.

Given one MRI slice, the CT is not fully determined. Bone edges land a fraction of
a pixel differently, soft-tissue texture varies, the residual misregistration
between the pair shifts structures slightly. So for a fixed input `x`, there is a
*distribution* of plausible targets `y`.

The regressor must emit one image. Ask which single image minimises the expected
loss:

```
argmin_ŷ  E_y[ |ŷ − y| ]   →  the conditional MEDIAN of y given x
argmin_ŷ  E_y[ (ŷ − y)² ]  →  the conditional MEAN   of y given x
```

Both are *averages over plausible outputs*. Where the plausible outputs agree —
the interior of a bone, the middle of a soft-tissue region — the average is sharp.
Where they disagree — every edge — the average is a blend of an edge in several
positions, which is a blur.

**So L1 does not blur because it is a weak loss. It blurs because a blurred image
is genuinely the optimal answer to the question L1 asks.** Turning up `lambda_l1`
makes this worse, not better. You need to ask a different question.

> **Implementation note.** L1 also has a property this project actively wants: it
> anchors anatomy to the right place. The failure mode of removing it entirely
> (`exp5_stylegan2_vanilla`, `lambda_l1: 0`) is a network that produces
> CT-*looking* images with invented anatomy. The ladder is built to find the
> trade-off, not to declare a winner. See `loss_function_guide.md`.

---

## 3. The question a discriminator asks instead

Rather than "how close is this output to the target, pixel by pixel?", ask:

> **Does this output look like it came from the real CT distribution?**

You cannot write that down as a formula, so you *learn* it. Train a second network
D to classify images as real or generated. Then train G to make D wrong.

The critical difference: a blurred edge is close to the target in L1 *and* trivially
identifiable as fake. There is no averaging that satisfies D, because D is not
comparing to one target — it is comparing to what real CT looks like in general.
Blur has a signature, and once D learns that signature, G is penalised for it.

This is the whole idea. Everything that follows is making it trainable.

---

## 4. The minimax game, and the gradient that actually matters

The original formulation is a two-player minimax:

```
min_G  max_D   E_x[ log D(x) ] + E_z[ log(1 − D(G(z))) ]
```

D maximises: push `D(real)` up, push `D(fake)` down. G minimises the second term:
make `D(G(z))` large.

**What G actually receives.** G never sees the real CT in this term. It receives
`∂L/∂G(z)` — a gradient flowing backwards *through D* into the generated image. D's
job is to convert "this looks fake" into a per-pixel direction: brighten here,
sharpen this edge, remove that texture. D is a learned, differentiable critic, and
its gradient is the only channel through which the adversarial signal reaches G.

That is also why D's health matters so much. A D that is always right gives a
saturated gradient. A D that is always wrong gives a meaningless one. Neither
teaches G anything, which is why `gan_evaluation_guide.md` section 3 spends so long
on reading `D_acc_real` and `D_acc_fake`.

### The non-saturating trick

Minimising `log(1 − D(G(z)))` has a nasty property: when G is bad, `D(G(z)) ≈ 0`,
and the gradient of `log(1 − D)` there is nearly flat. **The gradient vanishes
exactly when G is doing worst and most needs a signal.**

The fix is to maximise `log D(G(z))` instead. Same fixed point, opposite curvature
— large gradient when G is losing. Every implementation uses this form, including
this one:

```python
# losses/gan_loss.py:141-143
return F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))
```

Note it targets a hard `torch.ones_like`, not `self.real_target`. That is deliberate
— label smoothing is a handicap applied to D, and it must not leak into G's target.

---

## 5. Three objectives, one interface

### Minimal version — build it yourself

```python
import torch, torch.nn.functional as F

def d_loss_lsgan(pred_real, pred_fake):
    return 0.5 * (F.mse_loss(pred_real, torch.ones_like(pred_real)) +
                  F.mse_loss(pred_fake, torch.zeros_like(pred_fake)))

def g_loss_lsgan(pred_fake):
    return F.mse_loss(pred_fake, torch.ones_like(pred_fake))
```

That is a complete, working LSGAN objective in six lines. D regresses real scores
toward 1 and fake scores toward 0; G regresses fake scores toward 1. There is no
sigmoid anywhere — `pred_*` are raw network outputs.

### The real one: `losses/gan_loss.py:48-145`

`GANLoss` is an `nn.Module` with two methods and one diagnostic:

| method | line | returns |
|---|---|---|
| `d_loss(pred_real, pred_fake)` | `:79` | `(loss, stats_dict)` |
| `g_loss(pred_fake)` | `:134` | scalar |
| `decision_threshold()` | `:112` | float, log-only |

The three objectives differ only inside those methods:

| mode | D loss | G loss |
|---|---|---|
| `lsgan` (default) | `mse(real, t_real) + mse(fake, t_fake)` | `mse(fake, t_real)` |
| `vanilla` | `bce_with_logits` against the same targets | `bce_with_logits(fake, 1)` — non-saturating |
| `hinge` | `relu(1 − real).mean() + relu(1 + fake).mean()` | `−fake.mean()` |

All three consume **raw scores on a `[B, 1, h, w]` grid**, not a single number. The
mean reduction averages over every patch position — see `04_patchgan_discriminator.md`.

---

## 6. The three details in this file that are easy to miss

### The `0.5` on D's loss (`:95`)

```python
loss = (loss_real + loss_fake) * 0.5
```

This halves D's effective learning rate relative to G. It is the pix2pix convention,
and the reasoning is written in the source: *a mild handicap on the player that
usually wins*. D has the easier job — recognising is easier than generating — and
an unchecked D is failure mode A in `gan_evaluation_guide.md`.

### Label smoothing is one-sided (`:170-173`)

```python
smoothing   = cfg.get_path("stabilizers.label_smoothing.enabled", False)
real_target = cfg.get_path("stabilizers.label_smoothing.real_target", 0.9) if smoothing else 1.0
return GANLoss(gan_mode=cfg.loss.gan_mode, real_target=real_target)
```

Only `real_target` moves, to 0.9. `fake_target` stays at 0.0 and is never smoothed.
The docstring at `:48-57` gives the reason: smoothing *both* targets has been shown
to encourage G to match a blurred version of the data distribution — which is the
exact failure section 2 was trying to escape.

The constructor also warns when smoothing is set with `gan_mode: hinge` (`:69-73`),
because hinge has no regression target to smooth; its margin already limits D's
confidence. This is a config that silently does nothing, so it says so.

### `decision_threshold()` is not zero for LSGAN (`:112-132`)

This is the subtlest thing in the file, and it caused a real bug in this repo.

D's raw score has no fixed meaning — where "real" starts depends on the objective:

```
lsgan     D regresses toward real_target (0.9 or 1.0) and fake_target (0.0),
          so the boundary is the MIDPOINT — about 0.45, not zero.
hinge     D pushes reals above +1 and fakes below −1, so zero IS the midpoint.
vanilla   raw logits; logit 0 is probability 0.5, the natural boundary.
```

A healthy LSGAN discriminator might score fakes at +0.35 — comfortably on the fake
side of 0.45. Compare that against a threshold of 0 and you record `D_acc_fake ≈ 0`
and conclude D has collapsed, when it is working fine.

> **This affects logging only.** `D_acc_real` and `D_acc_fake` are `.detach()`ed
> (`:104-105`) and never enter the loss. The consequence of getting it wrong is a
> wrong diagnosis, not a wrong model — which is arguably worse, because you act on
> it. `training_log_reference.md` documents the before/after of the fix.

---

## 7. R1 — the one regulariser that lives in this file

`r1_penalty` (`:147-165`) is a module-level function, not a `GANLoss` method,
because it needs D's *input* and not just its output:

```python
grad = torch.autograd.grad(
    outputs=pred_real.sum(), inputs=real_input,
    create_graph=True, retain_graph=True, only_inputs=True)[0]
return grad.pow(2).flatten(1).sum(1).mean()
```

**The failure it prevents**, in the source's own geometric framing: a discriminator
that has memorised its training set carries very sharp decision boundaries around
each individual real image — tall spikes in a mostly flat landscape — and a spike
has a large gradient. Penalising the gradient at real points flattens those spikes,
forcing D to separate real from fake with general rules rather than per-image
lookups. On a 1687-slice training set, memorisation is a real risk.

`create_graph=True` is what makes this differentiable — the penalty must itself be
backpropagated, which means a **second backward pass through D**. That cost is why
it is applied lazily, and that laziness interacts badly with mixed precision. Both
are covered in `08_stabilisers_in_code.md`.

---

## 8. Where these get called

Nothing in this file decides *when* to compute anything. `GANLoss` is constructed by
`losses/builder.py:build_losses` and stored in a dict, and it is only built at all
when `lambda_gan > 0`:

```python
# losses/builder.py, in effect
if plan.use_gan:
    modules["gan"] = build_gan_loss(cfg)
```

So `exp0_l1_only` does not get a zeroed-out GAN loss — it gets **no `"gan"` key**.
The training step then reads `if plan.use_gan:` rather than multiplying by zero.
That design rule (a zero λ removes the whole code path) is the subject of
`05_the_training_step.md`.

---

## See also

- `03_unet_generator.md` — the network that produces `G(x)`
- `04_patchgan_discriminator.md` — the network that produces `pred_real` and `pred_fake`
- `05_the_training_step.md` — where `d_loss` and `g_loss` are actually called, and in what order
- `training_strategies.md` — how to *choose* between lsgan, vanilla and hinge
- `gan_evaluation_guide.md` — reading `D_acc_real` / `D_acc_fake` during a run
