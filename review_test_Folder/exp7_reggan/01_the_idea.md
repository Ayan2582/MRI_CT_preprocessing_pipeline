# 01 — What RegGAN is for

No repo code in this document. The goal is that by the end of it you could have invented
RegGAN yourself, and could also predict, before looking, roughly how it breaks.

---

## 1. The problem it solves

Paired image translation trains on a loss like

```
L(θ) = | G_θ(x) − y |₁
```

where `x` is the MRI slice and `y` is the CT slice of the same patient at the same
position. Every term in that sum is a comparison of **pixel (i, j) of the prediction
against pixel (i, j) of the target**.

That is an assumption, and on your data it is not exactly true. Your own QC record says
so: 738 slices were nudged individually, and `configs/base.yaml` records that manual QC
corrected *translation* on these pairs but could not correct **in-plane rotation**. So
some residual misalignment remains, unmeasured.

Now think about what that residual does to the loss. Suppose the CT is rotated by half a
degree relative to the MRI. At the skull vault, half a degree over a 100 mm radius is
about 0.9 mm — roughly one pixel, since 1 px = 1 mm here. So the generator is being told:

> at this pixel, produce bone (because the CT has bone there) —

while the MRI at that same pixel shows soft tissue, because the anatomy sits one pixel
over. The generator cannot satisfy that. It is being charged for an error that belongs to
the *registration*, not to the *synthesis*.

**What does a network do when asked to predict something unpredictable?** It predicts the
conditional mean. If bone is at pixel (i,j) in 60% of training examples and soft tissue in
40%, the L1-optimal output is the conditional *median*, and the L2-optimal is the mean —
either way, something in between. Across the whole boundary you get a blurred edge. So
misalignment does not just add noise to the loss; it systematically **blurs bone
boundaries**, which are the most clinically important structures in a synthetic CT.

This is the entire motivation. A model can look like it is under-fitting when in fact it
is correctly fitting a target that is itself slightly wrong.

## 2. The idea

Stop assuming the correspondence is exact. **Learn it.**

Introduce a second network `R` that outputs a dense displacement field `φ` — for every
pixel, a small (dx, dy) saying where that pixel should move. Then take the loss *after*
applying that displacement:

```
L(θ, ψ) = | warp( G_θ(x), φ_ψ ) − y |₁
```

Read that carefully. The generator produces an image; `R` deforms it; and only *then* is
it compared to the CT. If the true misalignment is a half-degree rotation, `R` can learn
exactly that rotation, the warped prediction lines up with the CT, and the generator is no
longer punished for a geometric error it did not make.

This is RegGAN (Kong et al., NeurIPS 2021), and the framing is elegant: instead of
assuming the labels are correct, model the **noise in the labels** — here, geometric
noise — and marginalise it out.

There is a second prize. `φ` is now an *estimate of the residual misalignment*. Take its
mean magnitude and you have a number, in millimetres, for a quantity that was previously
unmeasured. That is why this repo's ladder describes exp7 as "the only experiment here
that produces a measurement rather than just a score."

## 3. The obvious objection, and the standard answer

You should immediately be uneasy. If `R` can deform the prediction into the target, what
stops it from deforming *anything* into the target?

Take it to the limit. A completely unconstrained displacement field can move any pixel to
any location. Given an arbitrary generator output containing roughly the right histogram
of intensities, a sufficiently wild `φ` can rearrange those pixels into something close to
the CT. The loss goes to nearly zero and the generator has learned nothing. **The joint
minimisation over `(θ, ψ)` is under-determined** — many `(G, R)` pairs achieve the same
low loss, and most of them have a useless `G`.

The standard defence is a **smoothness penalty** on the field:

```
L_total = λ_corr · | warp(G(x), φ) − y |₁  +  λ_smooth · S(φ)
```

Real anatomical misalignment is smooth — a rotation, a translation, a gentle
non-rigid drift. Pixel-shuffling is not. Penalise roughness and you should keep the
useful deformations and forbid the cheating ones.

**Hold on to this, because it is where exp7 dies.** Write down what "roughness" means. The
natural choice, and the one everyone uses, is the magnitude of the field's spatial
derivative:

```
S(φ) = mean( |∂φ/∂x|² ) + mean( |∂φ/∂y|² )
```

Now ask: what is `S` for a **constant** field — every pixel displaced by the same (dx, dy)?

Its derivative is zero everywhere. `S = 0`. **A global translation of any size whatsoever
is completely free**, no matter how large `λ_smooth` is. And a rigid rotation by angle θ is
linear in position, so its derivative is a constant ≈ θ and its penalty is O(θ²) — cheap.

The smoothness term does not restrain the *size* of the deformation at all. It restrains
only its *raggedness*. That is not a flaw in exp7's implementation; it is a property of
the objective as universally written, and it is the door the failure walks through.

## 4. The third thing to notice

There is one more design decision, and it is the one people skip.

**In what frame do you evaluate the model?**

During training, the loss is computed on `warp(G(x), φ)`. But `R` needs `y` — the real CT
— as an input, because it is estimating a *relative* alignment between two images. At
inference time on a new patient there is no CT. That is the whole point; you are
synthesising it.

So `R` cannot exist at deployment. Whatever you ship, and whatever you validate, must be
the bare `G(x)`.

Which means:

```
trained on:    | warp(G(x), φ) − y |     <- in R's frame
evaluated on:  |      G(x)      − y |     <- in the pixel grid
```

**These are two different functions.** They agree only when `φ ≈ 0`. Every unit of
displacement `R` learns is a reduction in the training loss and, simultaneously, an
increase in the validation error — the generator has been trained to produce an image
that is correct *once shifted*, and is then scored un-shifted.

This is not a bug in RegGAN. It is a real and manageable tension, and the standard way to
manage it is to keep an ordinary un-warped L1 term in the objective alongside the warped
one. That term is the tether: it is the only thing that says the un-warped output also has
to be right, which is the thing you will actually be graded on.

**exp7 set the weight of that term to zero.**

---

## What to carry into 02

Three things, all of which follow from the objective alone, before looking at any result:

1. The joint minimisation over `(G, R)` is under-determined; the smoothness penalty is the
   only thing constraining `R`.
2. That penalty restrains the field's *roughness*, not its *magnitude*. Constant offsets
   are free.
3. Training and validation measure in different coordinate frames. An un-warped
   reconstruction term is what reconciles them, and exp7 does not have one.

Now go and look at what the run did.
