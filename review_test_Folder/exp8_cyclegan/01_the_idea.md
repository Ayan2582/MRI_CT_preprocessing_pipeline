# 01 — Unpaired translation, from first principles

No repo code here. The full mechanism is in the top-level
[02](../02_the_lambda_arithmetic.md)–[05](../05_the_background_artifacts.md); this document
is the derivation those assume.

---

## 1. The problem: an adversarial loss alone determines almost nothing

You have a pile of MRIs and a pile of CTs from different patients. No correspondences.
Train a generator `G` to make the first look like the second, judged by a discriminator
`D`.

Write down what that objective actually says. `D` is trained to separate real CTs from
generated ones, so the generator's term rewards making `D(G(x))` look real. That is a
statement about the **marginal distribution** of `G`'s outputs: the *set* of images it
produces should resemble the *set* of real CTs.

Now count the solutions.

- A generator that **ignores its input entirely** and emits one memorised, perfectly
  convincing CT slice satisfies it. Every output is a real CT.
- A generator that maps every MRI to a **random** CT satisfies it.
- A generator that maps MRIs to correct CTs satisfies it — and scores no better than the
  first two.

**Nothing in the adversarial objective connects the output to *this particular* input.**
There are infinitely many `G` achieving the optimum and almost all are useless. In the
paired setting, L1 supplies that connection. Without pairs you need something else.

## 2. The fix: make the translation reversible

Train a second generator `F` going the other way and require the round trip to return:

```
L_cycle  =  | F(G(mri)) − mri |  +  | G(F(ct)) − ct |
```

**Why this constrains anything.** If `G` discarded its input and emitted a memorised CT,
`F` would have no way to recover *which* MRI it came from — the information is gone. The
round trip can only close if `G(mri)` retains enough information to reconstruct `mri`.

So cycle consistency buys **injectivity**: the mapping must preserve information. That is a
real constraint and it rules out every degenerate solution in §1.

## 3. What it does not buy, and this is the crux

Injectivity is not correctness.

Consider any invertible encoding at all. Suppose `G` learns to hide the MRI inside its
output — in imperceptible high-frequency structure, in an unwatched corner of the image, in
the low-order bits — and `F` learns to read it back. The cycle closes perfectly. `L_cycle`
goes to zero. And the *visible* image can be anything.

This is not hypothetical; it is a documented CycleGAN failure mode, and
`model/docs/learn/11_cyclegan.md` §2 names it in this repo's own words.

So the objective has this structure:

```
L_adv    "the output must look like some CT"          — a distributional constraint
L_cycle  "the output must contain the input"          — an information constraint
        ————————————————————————————————————————
         neither says "the output must BE this patient's CT"
```

**Correctness is not in the objective anywhere.** It is hoped for as the *easiest* way to
satisfy both constraints simultaneously. Whether it actually is the easiest way depends
entirely on the data and the weights — which is where exp8 comes apart.

## 4. The identity term, and the trap inside it

Reference CycleGAN adds a third term:

```
L_idt  =  | G(ct) − ct |  +  | F(mri) − mri |
```

"A generator handed an image already in its target domain should leave it alone." The
motivation is tone: without it, nothing stops `G` applying a global brightness or contrast
shift that `D` tolerates and the cycle undoes.

Now do the arithmetic that matters. Substitute `G = F = I` — both generators are the
identity function:

- `L_cycle = |I(I(mri)) − mri| = 0`. **Exactly zero.**
- `L_idt = |I(ct) − ct| = 0`. **Exactly zero.**
- `L_adv` is the only term that objects.

**The identity map is an exact global minimum of every term except the adversarial one.**
It is reachable from anywhere, by a smooth downhill path, and it costs nothing to find.

So the weights are not a tuning detail. They decide whether the identity map is a trap the
model falls into or a curiosity it passes by. exp8 used `λ_cycle: 10`, `λ_identity: 0.5`
against `λ_GAN: 1` — and because the identity weight is *multiplicative*
(`λ_cycle × λ_identity = 5`), the split is **15 against 1**. The full arithmetic is in
[02_the_lambda_arithmetic.md](../02_the_lambda_arithmetic.md).

## 5. The assumption nobody writes down

One more thing, and it is the deepest problem here.

CycleGAN's justification assumes the two domains are **two renderings of the same content
distribution**. Horses and zebras: same poses, same fields, same lighting; only the stripes
differ. Under that assumption, matching `G`'s output marginal to `p(CT)` is *nearly* the
same as translating each image correctly, because there is a natural pairing between the
distributions even though you do not have it explicitly.

Break the assumption and the argument collapses. If your MRI pool is 50% abdomen and your
CT pool is 42% brain, then matching the marginal is **not** the same as translating
correctly — the cheapest way to match a brain-heavy target distribution is to make
everything look more like a brain CT, regardless of what went in.

exp8's disjoint patient split produced exactly that mismatch. It is measured in
[04_the_split_is_lopsided.md](../04_the_split_is_lopsided.md).

## 6. And one more: is the target even a function?

A deterministic network computes a function. So ask whether the mapping it is being asked
to learn *is* one.

In this project, CT is normalised with a **per-region HU window** — brain 0–80, abdomen
−160–240, musculoskeletal and spine −200–300. So the same normalised value means a
different tissue depending on the region, and the correct output for a given MRI depends on
which window applies.

In exp8 the generator receives an MRI and nothing else. `λ_L1` is 0, and the discriminator
is forced to be unconditional, so neither carries region information. **The same input can
require outputs on scales differing 6.25×, and the generator has no way to tell the cases
apart.** That is a one-to-many relation, not a function, and no deterministic network can
represent it. Derived in full in
[03_the_undefined_mapping.md](../03_the_undefined_mapping.md).

---

## What to predict before reading 02

1. Early epochs will show `G ≈ identity` — synth CT that looks like the input MRI.
2. `L_cycle` and `L_idt` will fall a long way, because both are minimised by that identity.
3. Validation will **not** improve alongside them, because neither term measures
   correctness.
4. When the adversarial term eventually pulls the model off the identity, it will land on a
   *plausible* CT, not a *correct* one.
5. Bone will be poor throughout — nothing in the objective specifies Hounsfield values.
6. Spine will be worst, having the fewest slices in a pool that is already halved.
