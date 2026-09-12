# 01 — What each term buys, and what each one costs

No repo code here. Two derivations: why an adversary is needed at all, and why it invents
things.

---

## 1. Why L1 alone is not enough

Established in [exp4_nce_max/01 §1](../exp4_nce_max/01_the_idea.md): the L1-optimal output
at a location is the **conditional median** of the target distribution given the input.
Wherever the network is uncertain, it returns the middle. Across an edge whose position is
uncertain, the middle of "bone" and "soft tissue" is a value that is neither, and the edge
comes out as a ramp.

The result is an image that is *correct on average and wrong everywhere*. It has the right
low frequencies and almost none of the high ones, because high-frequency detail is exactly
what averaging destroys.

For a synthetic CT this is not cosmetic. Bone boundaries are edges, and the electron
density on either side of one differs by a factor of two.

## 2. Why a discriminator fixes it, and what it actually asks

Add a second network `D` trained to separate real CTs from generated ones, and add a term
that rewards `G` for fooling it.

The key property: **`D` is not an average.** It is a classifier, so its gradient does not
push `G` toward the middle of the distribution — it pushes `G` toward the *support* of the
distribution. It says "real CTs do not look like this", and a smoothly-averaged image is
one of the things real CTs do not look like. So the adversarial term supplies exactly the
high-frequency structure that L1 cannot.

That is the pix2pix division of labour, and it is a good one:

```
λ_L1 · L₁          "put the structures in the right place"
λ_GAN · L_adv      "and make them look like they belong in a CT"
λ_NCE · L_NCE      "and don't mix up which input region became which output region"
```

Now write down what the adversarial term does **not** say.

`D`'s job is to answer *"is this a plausible CT?"* — a question about the **marginal
distribution** of CT images. In pix2pix the discriminator is conditional, so it sees
`cat[MRI, CT]` and can also ask "is this a plausible CT *of that MRI*". That helps. But it
is still a plausibility judgement, and plausibility is a much weaker requirement than
correctness.

Here is the consequence, and it is the whole of §4 below:

> **A generator can reduce the adversarial loss by adding structure that is
> distributionally typical but locally wrong.**

Real CTs contain bright, granular, high-contrast speckle — trabecular bone, calcification,
vessel wall, contrast. An image containing some of that scores better with `D` than a
smooth one. `D` cannot tell whether *this particular* speckle belongs at *this particular*
location, only whether the image as a whole has the right statistics.

Adding plausible detail in the wrong place is called **hallucination**, and it is not a
malfunction of the adversarial term. It is the term working exactly as specified.

## 3. Why the L1 term is supposed to prevent that

The defence is the weight. At `λ_L1: 100` against `λ_GAN: 1`, roughly 99% of the gradient
magnitude comes from L1, which is anchored to the actual target. `D` is a corrective on top
of a heavily-pinned prediction: it can adjust texture, but it cannot move structures far,
because moving them costs L1.

So the design intent is: **L1 decides what is there, `D` decides how it looks.**

That works when L1 is telling the truth. Two situations break it, and exp2 hits both:

- **Where L1 is uncertain**, its gradient toward any particular value is weak, and `D`'s
  preference for "some bright speckle somewhere" is comparatively free to act. Uncertainty
  is highest where training data is thinnest.
- **Where L1 has been driven to near zero on the training set** — that is, where the model
  has memorised — L1 stops constraining anything at inference on new patients, and `D`'s
  texture prior is all that is left.

## 4. Why overfitting is close to inevitable on this dataset

Now the second derivation, and it is arithmetic.

The generator is a U-Net with `ngf: 64, num_downs: 8`: **54.4 million parameters**. The
discriminator adds 2.8 M.

The training set is 1687 slices. Ask the question that matters: **how many independent
examples is that?**

Not 1687. Slice spacing here is 1 mm, and consecutive slices through the same anatomy are
nearly the same image — the same patient, the same scanner, the same noise characteristics,
the same body habitus, structures displaced by a millimetre. Statistically they are
repeated measurements, not independent draws.

The unit of independence in medical imaging is the **patient**. Your training split has
**32 subjects**.

So the honest ratio is 54.4 million parameters against something on the order of 32
independent examples — with, generously, a few hundred effectively-distinct views among
them. A model with that ratio can memorise its training set, and given 200 epochs it will.

This is why the split in this repo is by subject (`data/manifest.py:78-95`, grouping by PA
prefix so `PA32_..._ankle` and `PA32_..._knee` cannot straddle splits). That design is
correct and it is what makes the validation number meaningful. But splitting correctly
does not *prevent* overfitting — it only makes it **visible**.

So the prediction, before looking at any curve:

1. `train/G_L1` will fall a long way.
2. `val/mae_norm` will improve for a while, then turn around.
3. The turn will come **early**, because the effective sample size is small.
4. After the turn, the model's outputs will be increasingly governed by its priors — the
   adversarial texture prior above all — rather than by the input.
5. The damage will be worst where training data is thinnest.

## 5. What "thinnest" means here

| region | training slices | share |
|---|---|---|
| abdomen | 747 | 44.3% |
| brain | 532 | 31.5% |
| musculoskeletal | 381 | 22.6% |
| **spine** | **27** | **1.6%** |

Spine is 27 slices, from a handful of subjects. Prediction 5 says spine should be the
region where the texture prior wins most decisively, and where hallucination should be most
visible.

---

## What to check in 02

1. `train/G_L1` falls substantially — memorisation.
2. `val/mae_norm` turns around **early**.
3. Bone-bright speckle appears in soft tissue where no bone exists — hallucination.
4. Spine is the worst region, and visibly the most invented.
5. `dice_bone` looks *good* anyway, because Dice counts invented bone as a success wherever
   it lands on real bone.
