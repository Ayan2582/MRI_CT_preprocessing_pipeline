# GAN training strategies

Why this model is configured the way it is: the adversarial objective, the
discriminator design, and the six stabilisers — each of which fixes one named
failure mode.

---

## Part 1 — The adversarial objective

A GAN is two networks competing. Training is a **game**, not an optimisation.
There is no single loss descending toward a minimum; D's improvement is G's loss
increasing. Everything below follows from that.

`loss.gan_mode` selects between three formulations. All three consume raw
discriminator scores (there is no sigmoid in D) and average over the PatchGAN's
score grid.

### `lsgan` — least squares (default)

```
D:  (D(real) − 1)² + D(fake)²
G:  (D(fake) − 1)²
```

Penalises **how far** a sample sits from the decision boundary, not merely which
side of it it is on. The consequence that matters: the gradient stays informative
even after D is comfortably winning. This is the pix2pix default and the reason
to prefer it is entirely about reliable gradient supply.

### `vanilla` — binary cross-entropy

The original Goodfellow objective. Included for completeness and because its
failure is instructive: once D is confident, the sigmoid saturates, its gradient
goes to zero, and G stops learning **while the loss numbers still look busy**.
G uses the non-saturating form (maximise `log D(fake)` rather than minimise
`log(1 − D(fake))`) because the naive form has vanishing gradient exactly when G
is doing worst — precisely when it most needs one.

Not recommended. Use it to see the failure, not to train.

### `hinge`

```
D:  relu(1 − D(real)) + relu(1 + D(fake))
G:  −D(fake)
```

The modern large-scale default (SAGAN, BigGAN). D is penalised only until each
sample clears a margin of 1, after which that sample contributes nothing — a
built-in brake on over-confidence that pairs well with spectral normalisation.
Roughly as stable as lsgan, sometimes crisper. Worth trying as a variation once
the ladder is done.

### What was not implemented, and why

**WGAN-GP.** Its headline attraction is real — the loss value actually correlates
with sample quality, which would help exactly the problem this project has. But
it needs `n_critic = 5` discriminator steps per generator step (~3× the compute)
and it interacts badly with a `λ_L1 = 100` term that dominates the gradient
anyway. The cost/benefit does not work here.

---

## Part 2 — Discriminator design

### Why 70×70 patches rather than a whole-image verdict

D outputs a **grid** of scores, each judging one overlapping 70×70 receptive
field, not one number per image. This follows from the division of labour:

- L1 already enforces global anatomy and carries ~99% of the gradient magnitude.
- What L1 *cannot* do is prevent blur, since a blur is its optimum under
  uncertainty.
- Local texture realism is exactly what a patch-level critic can enforce.

Restricting D to a 70×70 window keeps it on the job L1 cannot do. Three further
benefits: far fewer parameters (2.8M vs tens of millions), which matters on 1687
training slices; it applies unchanged to any input size, useful when validation
images range 256–512 after padding; and 70 px = 70 mm at this dataset's
1 mm/pixel, roughly organ scale — a sensible unit for "does this texture belong
to this tissue".

### Why conditional

D receives `cat[MRI, CT]`. An unconditional D asks only "is this a plausible
CT?", which G can satisfy with a convincing CT of the wrong patient. Both
modalities make the question "is this a plausible CT **of this MRI**", so the
adversarial term reinforces correspondence rather than competing with it.

### Why InstanceNorm, not BatchNorm

Batch statistics are noisy at small batch sizes and they **couple samples within
a batch** — one image's normalisation depends on its neighbours. InstanceNorm is
per-image and batch-size-independent, so the CPU smoke test at batch 2 behaves
like the Kaggle run at batch 8, and results do not shift when you drop the batch
size to fit memory.

### Why dropout is OFF at evaluation

pix2pix conventionally leaves dropout on at test time, using it as its only
source of output variation (there is no noise vector `z`). That is the wrong
default for medical image synthesis: two runs of the same checkpoint on the same
slice would disagree, so a reported metric could not be reproduced and a
clinician could not be shown a stable image. `model.generator.dropout_at_eval`
defaults to `false`.

---

## Part 3 — The seven stabilisers

Each fixes a specific failure. Presented failure-first, because that is how you
will encounter them.

### Failure 1 — The discriminator wins too hard

D becomes a perfect classifier and outputs "fake, maximum confidence" for
everything G makes. The gradient reaching G approaches zero: D says *wrong* but
no longer says *wrong in which direction*. G stops learning from the adversarial
term entirely.

**Symptom:** `D_total` → 0, `D_acc` → 1.0, `G_GAN` climbs then flattens, samples
stop sharpening.

#### Spectral normalisation — `stabilizers.spectral_norm_d` (default ON)

Divides each weight matrix by its largest singular value, bounding how fast D's
output can change with its input — its **Lipschitz constant**. A Lipschitz-bounded
D *cannot* become arbitrarily confident, so it always keeps supplying a usable
gradient direction.

Cost: one power iteration per forward, ~2%. **The best stability-per-effort
technique available**, and on by default for that reason.

#### TTUR — `stabilizers.ttur` (default OFF)

Two Time-scale Update Rule: give D a lower learning rate (1e-4) than G (2e-4).
A direct handicap on the player that tends to win. Free.

Off by default because it is a response to an observed problem, not a
prophylactic — reach for it when the curves show Failure 1.

#### One-sided label smoothing — `stabilizers.label_smoothing` (default ON)

Train D toward **0.9** for reals instead of 1.0. It sounds trivial, but it
removes D's incentive to push confidence toward infinity, which is the mechanism
behind the saturation above. Free — it is a changed constant.

Only the *real* target is smoothed. Smoothing the fake target too has been shown
to encourage G to match a blurred data distribution.

> Does not apply to `hinge`, which has no regression target to smooth — its
> margin already serves the purpose. The code warns rather than silently
> pretending to do something.

### Failure 2 — The discriminator memorises the dataset

**The main risk on this project.** 1687 training slices from 33 patient folders.
A PatchGAN can memorise what those specific patients' CT texture looks like.
Once it has, it is answering "have I seen this exact patch before?" rather than
"does this look like real CT". Its gradient becomes noise and G starts producing
artifacts to game a critic that measures nothing.

**Symptom:** D accuracy near 1.0 on training data while validation quality
plateaus or drifts backwards.

#### DiffAugment — `stabilizers.diffaug` (default ON)

Applies random brightness/contrast, translation and cutout to **both** the real
and the fake batch, immediately before D, using differentiable ops.

Two properties make this work, and both are essential:

1. **Both sides are augmented.** Augmenting only reals would teach G to reproduce
   the augmentation. Because both get it, the transformation cancels out of the
   objective — D's job gets harder without changing what G aims for.
2. **It is differentiable.** Gradients flow back through the augmentation into G.
   A non-differentiable augmentation would sever that path and silently turn the
   adversarial term into noise.

This is why it lives in the discriminator step and not in the dataset transform
pipeline where ordinary augmentation belongs.

> Saturation is skipped on single-channel data, where it is mathematically the
> identity. Rotation is deliberately absent — see "what is NOT augmented" below.

#### R1 gradient penalty — `stabilizers.r1` (default OFF)

Penalises the squared gradient norm of D at real samples.

The intuition is geometric: a memorising D has very sharp decision boundaries
around each individual real image — tall spikes in an otherwise flat landscape —
and a spike has a large gradient. Penalising the gradient at real points flattens
the spikes, forcing D to separate real from fake with **general rules** rather
than per-image lookups.

Cost: a second backward pass through D. Applied lazily (`every: 16`, scaled by
16), which recovers most of the benefit at a fraction of the cost. That step runs
in fp32 — a double-backward under autocast is fragile and prone to `inf` in fp16.

### Failure 3 — Everything oscillates

Even a healthy GAN wobbles: G improves, D adapts, G's advantage evaporates,
repeat. Epoch 47 can genuinely be better than 48 and worse than 49 with no trend.
"Which checkpoint do I ship?" then has no principled answer.

#### Generator EMA — `stabilizers.ema` (default ON)

Keep a second copy of G's weights trailing the live ones:

```
ema = 0.999 · ema + 0.001 · live
```

Never trained — only evaluated and checkpointed. Because it averages over ~1000
recent steps, it sits at the **centre** of the oscillation rather than at a random
point on its orbit. In practice EMA samples are visibly better than live ones,
and the validation curve becomes smooth enough to read.

Cost: one extra weight copy (~210 MB), negligible compute. Near-universal in
modern GANs.

`start_epoch: 1` avoids averaging in the random initialisation, which would
contaminate the shadow through its whole warm-up window.

### Failure 4 — Mode collapse

G finds one output that fools D and emits it regardless of input. Mostly an
*unconditional*-GAN problem: here the `λ_L1 = 100` term makes it essentially
impossible, since a collapsed G would have catastrophic L1. Worth knowing the
term; not something to defend against in this setup.

**Except in `exp5_stylegan2_vanilla`, which has no L1 at all.** That run is the
one configuration in this repository where mode collapse is a live risk, and it
is also the one whose discriminator carries a minibatch-stddev layer — the
standard defence, which lets D read the batch's output diversity directly off a
feature map.

### Failure 5 — The generator's style-to-image map is ill-conditioned

*StyleGAN2 only.* A fixed-size step in the style space W moves the image a lot in
some directions and barely at all in others, which makes the optimisation surface
awkward and training slower to settle.

#### Path-length regularization — `stabilizers.path_length` (default OFF)

Penalises the deviation of the image-space Jacobian norm from a running mean, so
a step in W produces a consistent change wherever you take it. It needs a **double
backward through G**, so it runs lazily (every 4 G steps, scaled by 4) and in fp32
as its own optimisation step — mixing a scaled and an unscaled backward through
one GradScaler is where silent gradient corruption lives.

Two caveats specific to this project:

- It is **meaningless for the U-Net**, which has no W. Enabling it there is a
  startup error rather than a silent no-op.
- The paper samples `w` from the mapping network's Gaussian prior. There is no
  prior in a translation model — every `w` is a real patient's encoding — so the
  estimate comes from the batch in flight and is noisier than the published
  version. `exp6_stylegan2_fitted` turns it off for that reason; the property it
  buys (smooth latent interpolation) is one this project never uses.

### Summary

| stabiliser | fixes | cost | default |
|---|---|---|---|
| spectral norm on D | D wins too hard | ~free | **on** |
| generator EMA | oscillation, checkpoint choice | ~free | **on** |
| DiffAugment | D memorises small dataset | ~5% | **on** |
| label smoothing | D over-confidence | free | **on** |
| TTUR | D wins too hard | free | off |
| R1 penalty | D memorises | ~30% of D | off |
| path-length reg | ill-conditioned W→image map | ~15% of G | off |

**A note on R1's γ, which is easy to get wrong.** `base.yaml` ships `gamma: 10.0`,
which is StyleGAN2's value for FFHQ at **1024 px**. The published heuristic is
`γ = 0.0002 · N² / M` for resolution N and batch M — at 256 px that is 3.3 at
batch 4 and 1.6 at batch 8. Carrying the 1024 px number across a 16× change in
pixel count gives a discriminator so smoothed it stops discriminating, which
presents as `G_GAN` flatlining and reads as "the GAN isn't earning its keep" —
corrupting exactly the comparison `exp0` exists to make.

**Which stabilisers the StyleGAN2 runs use.** They switch off spectral norm and
label smoothing and turn R1 on. That is not preference: equalized learning rate
plus R1 *is* StyleGAN2's stability mechanism, and layering this project's usual
stabilisers on top would constrain D twice over. DiffAugment stays on in both —
the original paper used no augmentation, but it had 70k images against this
dataset's 1687, so running unaugmented would reproduce a known failure rather
than reproduce the paper.

---

## Part 4 — Schedule and warm-up

### Learning rate: constant, then linear decay to zero

Adam(2e-4, β₁=0.5, β₂=0.999), constant for the first half of training, then
linear decay to zero over the second half.

β₁ = 0.5 rather than the usual 0.9: high momentum makes GAN training lurch,
because the "correct" gradient direction changes as the opponent updates.

**The decay half is not optional garnish.** It is where fine detail settles. A
run stopped at the start of the decay phase reliably produces worse samples than
one allowed to finish — so a run cut short by a Kaggle session limit should be
**resumed**, not called done.

> This is also why you must never lower `train.n_epochs` to make a run "fit" a
> session. The schedule is defined against the total, so shortening it moves the
> decay onset and changes the learning rate of epochs you already ran. Keep
> `n_epochs` at the target and let the session die; `--resume auto` handles it.

### GAN warm-up — `train.gan_warmup_epochs` (default 5)

Train with L1 (and NCE) only for the first 5 epochs; D is not updated and the
adversarial term is not applied. This lets G produce something anatomically sane
before D starts critiquing it. Cheap insurance against early collapse, when a
randomly-initialised G produces noise and a discriminator can trivially reach
100% accuracy — which is Failure 1 on epoch 1.

**It requires a reconstruction term to warm up on.** With `λ_L1 = 0` *and*
`λ_NCE = 0` — `exp5_stylegan2_vanilla`, and `exp0` for the opposite reason — a
warm-up epoch would leave the generator with no loss at all, and the backward
pass would fail on a scalar that never entered the graph. `LossPlan` refuses that
combination at config time rather than letting it crash five layers down, so a
purely adversarial run must set `gan_warmup_epochs: 0`.

---

## Part 5 — What is deliberately NOT augmented

Only horizontal flip, and small translations inside DiffAugment. **No rotation,
no scaling, no elastic deformation.**

The reason is specific to this dataset. From
`qc_app/registration_service.py:26-37`: the median CT/MRI frame mismatch is
**5.2°**, and **61 of 120 series exceed 5°**. The applied registration is a
whole-pixel in-plane translation, one per series, rejected outright for 39 of 120.

The pairs are therefore *already* imperfectly aligned. Adding synthetic rotation
would deepen the exact problem the PatchNCE term was added to compensate for. If
more augmentation is ever needed, fix the registration first — the gain from
correcting a measured 5.2° is larger than anything augmentation would buy.

Flips are applied identically to both modalities, and the training crop uses one
origin for both, for the same reason: cropping them independently would add a
second, larger misalignment on top of the existing one.

---

## Part 6 — Practical settings

### Batch size

| device | batch @ 256² | notes |
|---|---|---|
| Kaggle P100 (16 GB) | 8 | the configured default |
| Kaggle T4 (16 GB) | 8 | slower; keep AMP on |
| Kaggle T4 ×2 | 8 | single-GPU code; the second card is idle |
| CPU (local) | 2 @ 64² | smoke tests only |

At batch 8, 1687 training slices give ~210 iterations/epoch; 200 epochs is
~42k iterations, a few hours on a P100.

InstanceNorm makes results largely batch-size-independent, so dropping to 4 for
memory is safe.

### Mixed precision — `runtime.amp` (default on, GPU only)

Roughly 2× throughput on modern GPUs. If you see `nan`, disable it to confirm
the cause, then set `train.grad_clip=1.0` and re-enable.

### Workers — `runtime.num_workers`

Kaggle GPU instances give 2 vCPUs, so `2`. Locally on Windows, use `0` if
dataloader workers misbehave.

---

## See also

- `gan_evaluation_guide.md` — how to tell whether any of this is working
- `loss_function_guide.md` — the three loss terms and their λs
- `kaggle_workflow.md` — packaging, training, resuming
