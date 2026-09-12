# 01 — What PatchNCE is, from scratch

No repo code here. By the end you should be able to derive the failure before seeing it.

---

## 1. The problem: L1 blurs, and you cannot simply turn it off

Train a translation network on `|G(x) − y|₁` alone and the output is soft. This is not a
capacity problem or an optimisation problem; it is what the loss *asks for*.

Here is the argument, and it is worth doing properly because the whole document depends
on it.

Fix a pixel location. Across the training set, the target value at that location, given
this input, is not a single number — it is a distribution. The MRI does not fully
determine the CT: the same soft-tissue appearance can sit over bone in one patient and
over marrow in another, registration is imperfect, and the network has finite capacity to
tell those cases apart. Call that conditional distribution `p(y | x)`.

The network outputs one number `v`. It is scored by `E[ |v − y| ]`. Minimise:

```
d/dv  E[|v − y|]  =  P(y < v) − P(y > v)  =  0
```

so the optimum is where those probabilities are equal — **the conditional median.**

(For L2 the same calculation gives the conditional *mean*. Both are "average-like". Keep
the distinction; §3 of [03_the_mechanism.md](03_the_mechanism.md) turns on it.)

So an L1-trained network, wherever it is uncertain, outputs the middle of the
distribution. Where the true answer is bimodal — bone at 800 HU or fat at −100 HU — the
median is somewhere in between, which is a value that occurs in **neither** case. That is
the blur. It is not a failure to converge; it is the converged answer.

The classical fix is a discriminator: L1 supplies *where things are*, the adversary
supplies *what the texture should look like*, because "plausible texture" is exactly what
an averaged prediction lacks. That is pix2pix, and it is `exp1`.

## 2. The other fix: constrain correspondence instead of pixels

Here is a different idea. What if you could tell the network

> whatever this input patch turns into, it must remain **recognisably that patch** and
> not some other patch

without ever saying what value it should have?

That is a weaker and more robust request than L1's. It permits the network to change the
appearance completely — which is the whole task, MRI to CT — while forbidding it from
mixing up locations. And crucially, it does not average, because it is not a regression at
all.

This is **contrastive learning**, and PatchNCE (from CUT, Park et al. 2020) is its
image-translation form.

## 3. Deriving the loss

Take a location *i*. You want a *query* built from the output at *i* to be matched with
the *key* built from the input at *i*, and not with keys from other locations.

Turn "matched with" into something differentiable. Give each patch a feature vector, and
measure agreement by inner product. Now you have a classification problem: given the query
`q` and a pile of candidate keys `{k₀, k₁, …, k_N}` of which `k₀` is the right one, pick
`k₀`. The natural loss is cross-entropy over a softmax of the similarities:

```
                     exp( q·k₀ / τ )
L = − log  ─────────────────────────────────
            Σⱼ exp( q·kⱼ / τ )
```

with `τ` a temperature. This is **InfoNCE**. `k₀` is the *positive* — the same location in
the input. The `kⱼ` for `j > 0` are the *negatives* — other locations, drawn from the same
image, so the network cannot win by learning "this is the brain slice."

One more standard step, and it is the one that matters. Raw inner products are unbounded,
which makes the softmax numerically unstable and lets the network cheat by inflating
vector norms. So every feature is **L2-normalised** first:

```
q ← q / ‖q‖     ,     k ← k / ‖k‖
```

after which `q·k = cos(angle between them)`, bounded in [−1, 1].

**Stop and look at what you have just built.** Every quantity in the loss is a cosine
between unit vectors. The loss cares only about *angles*, and the softmax cares only about
which angle is smallest. It is a **ranking objective**: is *i* closer to *i* than to *j*?

## 4. What a ranking objective cannot express

Here is the consequence, and it is the whole story.

Let `Φ` be the map from image patch to unit feature vector. Suppose you have a generator
`G` that satisfies the loss well. Now compose it with any invertible remapping `m` of
intensities — say, halve all contrast and shift everything toward grey — and ask what
happens to the loss for `m ∘ G`.

If the ranking of similarities is preserved, the loss is **unchanged**. Not
"approximately"; identically. The softmax sees the same ordering, assigns the same
probabilities, returns the same number.

So:

> **A CT with bone rendered black satisfies PatchNCE exactly as well as a CT with bone
> rendered white**, provided bone stays distinguishable from the tissue around it.

PatchNCE has no opinion whatsoever about absolute intensity. It cannot have one — you
normalised it away when you divided by `‖q‖`, deliberately, for good reasons, in step
three of the derivation.

**This is not a flaw.** It is precisely the property that makes the term useful. It is
what lets you apply it across two modalities whose intensities have nothing in common,
and it is why it does not blur. You wanted a term that constrains structure without
constraining values, and you got exactly that.

But it means something specific about your objective as a whole:

> Every claim about *what a Hounsfield unit should be* must come from somewhere else.

In this repo, "somewhere else" is exactly two places: the L1 term, and — weakly and
indirectly — the conditional discriminator. That is all.

## 5. The trade you are making when you set the λs

Now the λs are not two knobs. They are one:

```
L = λ_GAN · L_adv  +  λ_L1 · L_1  +  λ_NCE · L_NCE
     └ texture ┘      └ values ┘     └ structure ┘
```

Turning `λ_NCE` up and `λ_L1` down is not "trading a bit of accuracy for a bit of
structure". It is **transferring weight from the only term that carries absolute
intensity to a term that is provably indifferent to it.**

The ladder frames exp3 and exp4 as "lean on NCE" and "where does hallucination start?".
From the derivation, that framing predicts the wrong failure. Weakening the term that
pins values does not make a model invent structures that are not there — it makes a model
that declines to commit to any value it is not sure about. The expected failure is not
hallucination. It is **regression to the middle**, everywhere, and worst wherever the
target distribution is most spread out.

Which brings us to the last piece.

## 6. Why bone is the pixel that breaks first

Two facts about bone, and they compound.

**It is a minority.** In a CT slice, cortical bone is a thin shell — a few percent of the
pixels inside the body. Whatever the conditional distribution `p(y | x)` looks like at a
given location, bone is rarely the majority of it.

**It is at the extreme.** Bone is the *brightest* thing in the image, far from the middle
of the distribution. So it is exactly the value that any average-like estimator discards
first.

Put those together with §1. The L1-optimal output is the conditional **median** — and the
median is more brutal than the mean here, because it is not a weighted average at all. If
bone accounts for less than half the conditional mass at a location, the median does not
move part-way toward bone. It sits entirely in the soft-tissue mode and ignores bone
completely.

So: **the single hardest thing for a weak reconstruction loss to produce is bright bone**,
and it is the thing a synthetic CT exists to get right.

---

## What to predict before reading 02

From the derivation alone, with `λ_L1` cut from 100 to 10:

1. The **anatomy will be fine.** NCE constrains correspondence, and it is at full strength.
2. The **intensities will collapse toward the middle**, most visibly wherever the target
   distribution is widest.
3. **Bone will vanish first** and most completely, being both a minority and an extreme.
4. The **error maps will concentrate on bone** — the skull, cortical rims, vertebrae.
5. `train/G_L1` **will not fall much**, because at weight 10 it is not the term steering
   the optimisation.
6. Global averages — MAE, SSIM — will degrade only moderately, because bone is a small
   fraction of pixels. The damage will be far larger in any metric that looks *at* bone.

Now look.
