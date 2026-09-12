# 02 — Why epoch 9 is a copy

**Finding 1.** The generator loss is weighted 15:1 against the only term that pushes
the output toward CT, and the identity function is an exact minimum of the other 15.

---

## The failure

At epoch 9 `G_A2B(mri) ≈ mri`. Not blurred, not tinted — passed through. A generator
that has learned nothing after 9 epochs is normally a sign of a broken gradient path or
a dead learning rate. Here both are fine. **The generator found a good solution to the
loss you wrote.**

## The arithmetic

Three terms, from `training/cyclegan.py:234-278`:

```python
loss_G = lambda_gan   * ( g_loss(D_B(fake_B)) + g_loss(D_A(fake_A)) )     # :243
       + lambda_cycle * ( cyc_A + cyc_B )                                 # :251
       + lambda_cycle * lambda_identity * ( idt_A + idt_B )               # :263-264
```

The identity weight is a **product**, not an absolute value — `cyclegan.py:263`:

```python
weight = self.plan.lambda_cycle * self.plan.lambda_identity
```

`exp8_cyclegan.yaml` sets `lambda_cycle: 10.0` and `lambda_identity: 0.5`. So:

| term | weight | what it wants |
|---|---|---|
| adversarial | **1** | output should look like *a* CT |
| cycle | **10** | `G_B2A(G_A2B(A)) = A` |
| identity | **5** | `G_A2B(B) = B` and `G_B2A(A) = A` |

Now substitute `G_A2B = G_B2A = I`:

- cycle: `G_B2A(G_A2B(A)) = A` — **exactly zero**, not approximately.
- identity: `G_A2B(B) = B` — **exactly zero**.

The identity pair is a *global* minimum of 15 units of loss, reachable from anywhere,
with a smooth downhill path. The only thing objecting is 1 unit of adversarial loss.
Fifteen to one. Epoch 9 is not the model failing to learn; it is the model at the
bottom of the well you dug.

## The part that makes this sharp

`exp8_cyclegan.yaml` sets `gan_warmup_epochs: 0`, and gives this reason:

> Nothing paired to warm up on, and the cycle term is already available from step 0 — a
> warm-up here would just train two generators to be each other's inverse with no
> pressure toward either domain, which is a worse starting point than random.

That reasoning is exactly right, and the λ weights reintroduce the warm-up anyway. With
`gan_warmup_epochs: 0` the adversarial term is switched on from step 0 — at 1/15th of
the gradient budget. Functionally that is a soft warm-up of unbounded length, and it
produced precisely the failure the comment predicted: **two generators trained to be
each other's inverse.**

The config disabled the mechanism and left the effect in place.

## The codebase already knew this

`training/cyclegan.py:63-69` refuses to start if `lambda_gan` is 0, and the error text
is the diagnosis of this run:

> CycleGAN's only link between the two domains is adversarial — with no discriminators
> the cycle loss is minimised perfectly by making both generators the identity, and the
> run would learn nothing.

That is epoch 9, word for word. The guard catches `lambda_gan == 0` exactly, and misses
`lambda_gan` being one fifteenth of the reconstruction weight — which produces the same
attractor, just with a slow escape instead of no escape.

Worth noting as an instance of the theme in `model/docs/learn/README.md`: *"when a
component can be wrong in a way that leaves the loss curve looking healthy, constrain it
structurally."* The structural constraint here was written against the wrong variable.
A `LossPlan` check on the **ratio** — refuse when `lambda_cycle * (1 + lambda_identity)`
exceeds some multiple of `lambda_gan` — would have caught it.

## Why the identity term is the sharper half of the problem

Cycle at weight 10 is the published CycleGAN value and is not, on its own, unusual.
Identity at an effective 5.0 is also the published value — but published for
photographs, where the two domains are *the same scene* under different rendering, and
"leave an already-correct image alone" is a mild anchor.

MRI and CT are not that. `G_A2B(real_CT) = real_CT` asks the MRI→CT generator to be the
identity on CT inputs, and `G_B2A(real_MRI) = real_MRI` asks the CT→MRI generator to be
the identity on MRI inputs. Both are pulling each network toward `I` on half its
possible inputs, on images whose intensity statistics are wildly different from the
other half. In a photo-to-photo task the identity term costs you a little tonal
conservatism. Here it is 5 units of gradient pointing at the degenerate solution.

`model/docs/learn/11_cyclegan.md` §6 justifies the term as a tone anchor — *"without it,
nothing stops `G_A2B` from applying a global shift to everything."* That is a real
problem and the term does solve it. But 05 §4 shows that the reason D cannot police
global tone by itself is another config choice, and once that is fixed the identity term
is not carrying the weight it was given.

## What this does *not* explain

Epoch 109 and 199. By then the adversarial term had won and the output does look like
CT — the identity trap is escapable, it just costs a hundred epochs. What it does not
explain is why the escape landed on the *wrong* CT.

That is 03, and it is not a weighting problem.
