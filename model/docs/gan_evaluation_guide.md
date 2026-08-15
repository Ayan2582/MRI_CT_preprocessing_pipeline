# How do I know my GAN is improving?

The question this document answers: **epoch 47 just finished. Is it better than
epoch 46? Is this run better than the last one? How would I know?**

For an ordinary supervised model these are easy — the validation loss went down
or it didn't. For a GAN they are genuinely hard, and the naive answers are wrong
in ways that will waste weeks. This is the guide to getting them right on *this*
dataset.

---

## 1. The GAN losses do not tell you whether the model is improving

This is the first thing to internalise, and the most counter-intuitive.

In normal training, loss is a distance to a fixed target, so lower is better.
In a GAN there is no fixed target. The generator's loss is scored by a
discriminator that is *itself changing every step*. So:

> **`G_GAN` going down can mean the generator improved, or it can mean the
> discriminator got worse. The number alone cannot distinguish these.**

The same applies in reverse to `D_total`. A GAN at healthy equilibrium has both
losses roughly **flat** — the two players are matched and neither is pulling
ahead. Loss curves that plunge are usually a sign that one side has collapsed,
not that the model is working.

Concretely, on your runs:

| curve | what it does NOT mean |
|---|---|
| `G_GAN` falling | images are getting better |
| `G_GAN` rising | images are getting worse |
| `D_total` → 0 | the discriminator is doing well (it means it's about to stop teaching) |
| both flat | training has stalled (this is usually the *healthy* state) |

**`G_L1` is the exception.** It is a real distance to a real target and it does
mean what you expect. But it is measured on *training* data, so it tells you
about fit, not generalisation.

---

## 2. The one number that ranks epochs

```
val/mae_norm     — mean absolute error, normalised units, EMA generator,
                   validation split.  Lower is better.
```

This is what `best.pt` is selected on. Three deliberate choices are baked in.

### Why validation, not training

You have 1687 training slices from 33 patient folders. A 54M-parameter generator
can memorise that. Training loss will keep dropping long after generalisation
has stopped improving.

### Why the EMA generator

GAN training doesn't converge to a point, it **orbits** one. G improves, D
adapts, G's advantage evaporates, repeat. The practical consequence is that the
live generator's validation score bounces from epoch to epoch with no trend, and
"best epoch" becomes a lottery among noise.

The EMA copy averages the weights over roughly the last 1000 steps, so it sits
near the *centre* of the orbit instead of at a random point on it. Its curve is
smooth enough to actually read, and its samples are usually visibly better than
the live model's — for free. This is why `eval.use_ema: true` is the default.

### Why *normalised* units and not Hounsfield units

**This is the trap specific to your dataset.** The CT arrays were windowed per
body region before normalisation:

| region | HU window | 1.0 normalised unit = |
|---|---|---|
| brain | 0 … 80 | **80 HU** |
| abdomen | −160 … 240 | **400 HU** |
| MSK / spine | −200 … 300 | **500 HU** |

So the *same* prediction error is worth 80 HU on a brain slice and 500 HU on a
spine slice. A pooled "MAE in HU" is therefore dominated by whichever regions
have wide windows, and it moves when the *composition of your validation set*
changes rather than when your model does.

You can see this directly in the metrics output — an identical normalised error
of 0.33 reports as:

```
mae_hu    overall 95.73    brain=26.51    musculoskeletal=165.0
```

Those two numbers describe the same model quality. Pooling them produces a
number that means nothing.

**So:** `mae_norm` ranks epochs. `mae_hu` is reported **per region** and
macro-averaged, and is what you quote when you want a clinically interpretable
figure — always with the region attached.

---

## 3. Reading the GAN health curves

These don't rank epochs. They diagnose *what to change* when the ranking metric
stops improving. Plot `D_acc_real` and `D_acc_fake` — the fraction of real and
fake patches the discriminator classifies correctly.

### Healthy: D accuracy ~0.6–0.85, both losses roughly flat

```
D_acc_real  ~0.75  ────────────────
D_acc_fake  ~0.75  ────────────────
G_GAN              ~flat, noisy
val/mae_norm       slowly decreasing
```

D is winning slightly and consistently, which is what you want — it stays ahead
enough to keep teaching, without running away. **Do nothing.**

### Failure A: the discriminator overpowers the generator

```
D_acc_real  → 1.0   ╱▔▔▔▔▔▔▔▔▔▔▔▔
D_acc_fake  → 1.0   ╱▔▔▔▔▔▔▔▔▔▔▔▔
D_total     → 0.0   ╲____________
G_GAN       climbing, then flat
samples     stop getting sharper
```

D has become a perfect classifier. It is telling G "wrong" with total confidence
but no longer telling it *in which direction* — the gradient has saturated. G
falls back on the L1 term alone and you have an expensive regression model.

**Fixes, in order of preference:**
1. Enable TTUR — handicap the stronger player:
   `--set stabilizers.ttur.enabled=true stabilizers.ttur.lr_d=1e-4`
2. Lower `loss.lambda_gan` to 0.5.
3. Check `stabilizers.spectral_norm_d` is on (it is by default — this failure
   is much more likely with it off).

### Failure B: the discriminator collapses

```
D_acc_real  → 0.5   (coin flip)
D_acc_fake  → 0.5
G_GAN       → 0
samples     get worse / develop repeating texture
```

D has stopped distinguishing anything, so G's adversarial term is pure noise and
G is free to drift toward whatever artifacts happen to fool a broken critic.

**Fixes:** raise `model.discriminator.ndf` to 96, or lower `loss.lambda_gan`, or
reduce the DiffAugment policy — over-aggressive augmentation can make D's job
impossible.

### Failure C: the discriminator memorises the training set

```
D_acc_real  → 1.0 on TRAIN
val/mae_norm  plateaus or worsens
G_GAN       looks normal
samples     develop odd high-frequency texture
```

This is **the most likely failure on this dataset**, because 1687 slices from 33
folders is a small training set for a discriminator with real capacity. Once D
has memorised, it is answering "have I seen this patch before?" rather than
"does this look like CT", and its gradient stops carrying information about
realism.

**Fixes:**
1. Confirm `stabilizers.diffaug.enabled: true` (default on, and it is on
   precisely for this).
2. Enable R1: `--set stabilizers.r1.enabled=true stabilizers.r1.gamma=10`.
3. Reduce `model.discriminator.n_layers` to 2 (a smaller receptive field, less
   capacity to memorise).

### Failure D: NaN

Loss becomes `nan` and never recovers. Almost always mixed precision plus an
exploding gradient.

**Fixes:** `--set runtime.amp=false` to confirm the cause, then
`--set train.grad_clip=1.0` and re-enable AMP.

---

## 4. The blur-versus-hallucination trade-off

This is the judgement call at the heart of the project, and no single metric
captures it.

**High `lambda_l1` → blurry but honest.** L1's optimal response to uncertainty is
to predict the conditional mean, which looks like a blur. A blurry synthetic CT
is *visibly* wrong, which means nobody is misled by it.

**Low `lambda_l1` → sharp but inventive.** The adversarial term rewards output
that *looks* like real CT. A GAN will happily synthesise anatomically plausible
bone that is not in the patient, because plausible bone is exactly what fools the
discriminator. A sharp synthetic CT with fabricated structure looks *more*
trustworthy than the blurry one and is far more dangerous.

**Global metrics hide this.** Bone is a small fraction of the pixels in most
slices, so a model can invent or erase it while barely moving MAE or SSIM.

That is why these exist:

```
dice_bone           overlap of the >150 HU mask.  Higher is better.
mae_band_bone       MAE in HU restricted to true-bone pixels.
mae_band_soft       ... to true-soft-tissue pixels.
mae_band_air        ... to true-air pixels.
```

**The diagnostic pattern to watch for:**

> `mae_norm` improves, images look sharper, **and `dice_bone` drops.**
>
> That is not a better model. That is a model trading real bone accuracy for
> average-case smoothness and cosmetic sharpness. Do not ship it.

### One important caveat: these metrics skip brain slices

The brain window tops out at **80 HU**. Bone is 300–2000 HU. On a brain slice
every bone voxel is clipped to exactly 1.0 and is indistinguishable from any
other bright tissue — bone is *not present in the data* as a measurable value
there. A bone Dice over brain slices would score agreement about a saturated
constant, which is trivially near-perfect and completely uninformative.

So bone and band metrics are computed on **non-brain slices only**, and the
output states the count:

```
n_samples=230  bone/band metrics computed on 159 slices
(regions excluded: brain — its HU window saturates bone)
```

If you ever need bone metrics on brain cases, the fix is upstream in the
preprocessing pipeline — re-export brain CT with a bone window — not here.

---

## 5. Comparing runs, not just epochs

Epoch-to-epoch is the easy comparison. Run-to-run is where conclusions come from,
and it has stricter requirements.

**Requirements for two runs to be comparable at all:**

1. **The same split.** `model/data/splits.json` is committed and shared for
   exactly this reason. Regenerating it between runs silently invalidates every
   comparison — the runs would be scored on different patients.
2. **The same selection rule.** Compare `best.pt` against `best.pt`, both chosen
   on `val/mae_norm`, both from the EMA generator.
3. **The same seed** unless you are deliberately measuring seed variance.

Then:

```bash
python model/evaluation/evaluate.py --compare model/runs/exp* --split val
```

which prints one table with the best value per column marked.

**Always include `exp0_l1_only`.** It is the no-adversary regression floor and it
answers the question that is otherwise easy to never ask: *is the GAN earning its
keep at all?* If `exp1_pix2pix` doesn't beat `exp0` on anything except sharpness,
you are paying compute and hallucination risk for a cosmetic change.

Expected shape of the results — worth predicting before you look, so you notice
when reality disagrees:

| run | `mae_norm` | sharpness | `dice_bone` |
|---|---|---|---|
| exp0 L1-only | **best** | worst (blurry) | moderate |
| exp1 pix2pix | slightly worse | better | ? |
| exp2 +NCE | similar | better | ? |
| exp3 NCE-heavy | ? | better | ? |
| exp4 NCE-max | worst | best | **watch this** |

Yes — the L1-only baseline usually wins on MAE. **That is expected and is not an
argument for shipping it.** MAE rewards the conditional mean, and the conditional
mean is blurry. This is precisely why MAE alone must not choose your final model:
use it to rank epochs *within* a run, and use the full table plus your eyes to
choose *between* runs.

---

## 6. Look at the images. Systematically.

Metrics do not catch everything — checkerboard artifacts from transposed
convolutions, a systematic intensity shift, plausible-but-wrong anatomy.

`logging.sample_every` writes a panel of **fixed** validation slices, one set per
body region, as `runs/<name>/samples/epoch_XXXX.png`. Four columns: MRI, real CT,
synthetic CT, and the absolute error map on a **fixed** colour scale.

Both "fixed" choices matter:

- **Fixed slices**, because comparing epoch 40's random slices to epoch 45's
  random slices tells you about the slices, not the model. The same panel every
  time makes progress legible and makes two runs comparable image-for-image.
- **Fixed error scale** (0 to 0.5), because an autoscaled error map looks equally
  red at every epoch and hides the improvement it is supposed to show.

**What to look for, in order:**

1. **Anatomy in the right place.** Overlay-compare the synthetic and real CT.
   Structure that has moved is a registration problem, not a model problem.
2. **Bone edges.** Sharp and continuous, or smeared? Present where the real CT
   has it, absent where it doesn't?
3. **The error map.** Error concentrated at tissue boundaries is normal — some
   of it is genuine model error and some is residual alignment error that
   survived QC. Error in the *middle* of uniform
   regions means the model is getting intensity wrong, which is worse.
4. **Repeating texture.** Regular grid patterns mean checkerboard artifacts;
   identical texture across different patients means D has collapsed.

---

## 7. A practical checklist

**Every epoch — glance at:**
- [ ] `val/mae_norm` trending down (on EMA)
- [ ] `D_acc_real` / `D_acc_fake` in 0.6–0.85

**Every ~10 epochs — look at:**
- [ ] the sample panel: sharper than 10 epochs ago?
- [ ] `dice_bone` — flat or rising, not falling
- [ ] per-region `mae_hu` — is one region regressing while others improve?

**End of run:**
- [ ] `best.pt` epoch is not the very last one (if it is, train longer)
- [ ] LR decay phase completed (the last ~50% of epochs — this is where detail
      settles; a run cut off at the start of decay is not finished)
- [ ] compare against `exp0` and `exp1`
- [ ] **only now**, once and once only, score the test split

**Before believing any number:**
- [ ] same `splits.json` across every run being compared
- [ ] metrics from the EMA generator
- [ ] region-aware — never a pooled HU figure
- [ ] `n_bone_eligible` matches expectation (brain excluded)

---

## 8. Quick reference

| symptom | likely cause | first thing to try |
|---|---|---|
| `mae_norm` flat from epoch 1 | LR too low, or warm-up too long | `train.lr`, `train.gan_warmup_epochs` |
| `mae_norm` improves, images blurry | λ_L1 dominating | lower `loss.lambda_l1` |
| images sharp, `dice_bone` falling | hallucination | raise `loss.lambda_l1` |
| `D_acc` → 1.0, samples stall | D overpowering | enable TTUR |
| `D_acc` → 0.5, samples worsen | D collapsed | raise `ndf`, lower λ_GAN |
| train good, val bad | D memorising | DiffAugment, R1 |
| val metrics bounce wildly | not using EMA | `eval.use_ema=true` |
| `nan` | AMP + exploding gradient | `grad_clip=1.0` |
| two runs disagree inexplicably | different split | check `splits.json` |

---

## See also

- `training_log_reference.md` — decodes every field in the training log

- `loss_function_guide.md` — what each term does and what each λ changes
- `training_strategies.md` — the objective variants and all six stabilisers
- `kaggle_workflow.md` — packaging, training, resuming
