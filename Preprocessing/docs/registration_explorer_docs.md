# The Registration Explorer — v3, v6, and why the MRI tilts

*Companion to `registration_docs.md`. Reference implementation: `working_regis.py`.*

This document does three things:

1. Teaches the explorer's pipeline from first principles — no assumed background.
2. Sets out what changed between artifact **v3** and **v6**, and what is *not* recoverable.
3. Diagnoses the MRI tilt in v6 and gives an ordered plan to fix it.

---

## Part 0 — The basics, properly

Skip this if you already know what a moving image and a similarity metric are.

### 0.1 What "registration" actually means

You have two photographs of the same person taken by two different cameras on two
different days. Registration is finding the geometric operation — slide it left, turn it
a bit, stretch it — that makes the second photo line up with the first.

Two words you need, and they never change roles:

| Term | Meaning | In our case |
|---|---|---|
| **Fixed** | the image that stays put; the reference frame | **CT** |
| **Moving** | the image we transform until it matches | **MRI** |

The CT never moves. Ever. Everything the pipeline does is a question about where to put
the MRI.

### 0.2 Why we can't just subtract the two images

The obvious idea is: try lots of positions, and keep the one where `CT − MRI` is smallest.
This fails badly, and understanding *why* explains every other design choice below.

CT pixel values are **Hounsfield Units** — a calibrated physical scale. Air is −1000,
water is 0, bone is +1000, on every scanner ever built. MRI pixel values are **arbitrary**:
the same muscle might read 600 in one sequence and 1800 in another.

Worse, the *ordering* is inverted:

| Tissue | CT (HU) | CT looks | MRI (T1) | MRI looks |
|---|---|---|---|---|
| Air | −1000 | black | ~0 | black |
| Fat | −100 | dark grey | **~1200** | **bright** |
| Muscle | +50 | mid grey | ~600 | mid grey |
| Cortical bone | **+1000** | **white** | **~50** | **near-black** |

In CT, bone is much brighter than fat. In MRI, bone is much *darker* than fat. So if you
minimise the difference between pixel values, the optimiser's best move is to slide bone
**off** bone — because putting bright-on-bright means putting CT bone onto MRI fat.

> **Subtraction doesn't just give a bad answer here. It points in the wrong direction.**

### 0.3 What is stable: the relationship

What *is* reliable is this: everywhere the CT says 1000, the MRI says about 50. The
numbers don't match, but the **relationship** is consistent.

So the metric must ask *"is there a consistent relationship between these two images?"*,
not *"do the values match?"*

### 0.4 The joint histogram — the heart of everything

Take the two images, and build a 2D table. Cell `(a, b)` counts how many pixel positions
have CT value `a` and MRI value `b`.

```
        MRI value →
        0    1    2    3
CT   0 [820   4    1    0 ]     ← air in CT is air in MRI: one tight cell
 ↓   1 [  2  310  15    3 ]
     2 [  0   22 405   11 ]
     3 [  1    5   9  260 ]     ← bone in CT is dark in MRI: again tight
```

**Aligned** → each CT tissue maps to a small, predictable cluster of MRI values, because
it *is* the same tissue. The table is **tight** — mass concentrated in few cells.

**Misaligned** → CT bone lands on random MRI tissue, and every CT row smears across every
MRI column. The table becomes a **diffuse cloud**.

So: *tight table = good alignment*. We need a number that measures tightness.

### 0.5 Entropy, in one paragraph

Entropy measures how spread-out a probability distribution is.

```
H(p) = − Σ pᵢ log pᵢ
```

All the mass in one cell → `H = 0` (perfectly predictable). Mass spread evenly over
everything → `H` is at its maximum (maximally unpredictable). **Low entropy = tight.**

Read it as *"how surprised am I, on average, by the next value?"*

### 0.6 Mutual information, and why we use the *normalized* kind

From the joint table you get three entropies for free:

- `H(A)` — sum each row, take entropy → spread of the CT alone
- `H(B)` — sum each column → spread of the MRI alone
- `H(A,B)` — entropy of the whole table → spread of the pair

**Mutual information** is `MI = H(A) + H(B) − H(A,B)`. In words: how much less surprised
am I by the pair than by the two separately? High MI = knowing the CT tells you a lot
about the MRI = aligned.

The problem: MI is **unbounded** and grows with `H(A)` and `H(B)`. So MI goes up when
there's simply *more image in the frame*, regardless of alignment. Compare two different
canvas sizes with MI and you're measuring the canvas, not the alignment.

**Normalized mutual information (Studholme)** divides that dependence out:

```
NMI = ( H(A) + H(B) ) / H(A,B)          bounded in [1, 2]
```

`1` = the images are statistically independent (know nothing about each other).
`2` = identical partitions. **Higher is better.** Real numbers here live around 1.05–1.25.

> ⚠️ **Keep this in your pocket — it matters in Part 3.** NMI removes the dependence on
> *how much entropy* is in the frame. It does **not** remove the dependence on *which
> pixels* are in the frame. That loophole is the tilt bug.

---

## Part 1 — The pipeline, stage by stage

This is v6. Every stage below is verified against the published artifact source and
against `working_regis.py`, which is a line-for-line Python mirror (same constants, same
RNG magnitudes, same pyramid).

### Stage 01 — Resample both to isotropic millimetres

**The problem.** A CT pixel might be 0.49 mm across; an MRI pixel 0.875 mm. Both are
"one pixel", but they mean different physical sizes. Comparing them directly is comparing
inches to centimetres.

**The fix.** Resample both so one pixel = exactly 1.0 mm on both axes, in both modalities.

```js
function toIso(img, mm) {
  var W = Math.max(8, Math.round(img.w * sx / mm));
  var H = Math.max(8, Math.round(img.h * sy / mm));
  // bilinear resample onto the new grid
}
```

After this, **a 50 mm structure spans 50 pixels in the CT and 50 pixels in the MRI.**

> **This is load-bearing.** Because both images are now on the same physical scale, there
> is *no genuine zoom difference left* for the affine to recover. Any large scale the
> optimiser reports afterwards is therefore a cheat, not a correction. That fact is what
> makes the scale gate (Stage 06) meaningful at all.

**Then placement.** A single 2D slice carries no shared world origin, so the MRI is
dropped onto the CT's grid **centred**, padded with `NaN` for "no data":

```js
function placeOn(src, W, H) {
  for (i...) out[i] = NaN;                       // everything starts as "no data"
  var ox = Math.floor((W - src.w) / 2);
  var oy = Math.floor((H - src.h) / 2);
  // copy src into the middle
}
```

> ⚠️ **Known weakness, stated in the code itself.** The real pipeline
> (`registration_demo_sweep_v3.py`) computes the true offset in closed form from the DICOM
> `ImagePositionPatient` header — exact, free, no optimisation. The explorer *cannot*,
> because it only has one slice, so it centres and makes the optimiser find the offset
> from scratch. This is why recovered translations here are larger than they should be —
> and, as Part 3 argues, it's a contributing cause of the tilt.

### Stage 02 — Baseline: the number to beat

Score NMI with **no transform at all**. Everything afterwards must beat this, or we ship
the unregistered image. A pipeline that can't tell you "doing nothing was better" is a
pipeline that ships damage.

### Stages 03 & 04 — Rigid and affine multi-start

**Rigid** = 3 numbers: shift-x, shift-y, rotation. It can slide and turn. It **cannot**
change size or shape.

```js
if (kind === "rigid") {
  var c = Math.cos(p[2]), s = Math.sin(p[2]);
  return { M: [c, -s, s, c], t: [p[0], p[1]] };   // a true rotation matrix
}
```

**Affine** = 6 numbers: shift-x, shift-y, and all four entries of a 2×2 matrix. It can
slide, turn, stretch, squash **and shear** (turn a rectangle into a parallelogram).

```js
return { M: [p[2], p[3], p[4], p[5]], t: [p[0], p[1]] };   // anything goes
```

> **Remember this contrast.** Rigid's rotation is one explicit, readable parameter.
> Affine's rotation is *smeared across all four matrix entries* and never named. Part 3
> turns on exactly that.

**Why multi-start.** The optimiser is a hill-climber; it finds the nearest peak, not the
tallest. Run it three times from three different random starting points and keep the best.
The **spread** across those three is a free estimate of how much the optimiser is guessing.

```js
var p = kind === "rigid"
  ? [rnd()*10, rnd()*10, rnd()*0.10]                                   // ±5mm, ±0.05 rad
  : [rnd()*10, rnd()*10, 1+rnd()*0.04, rnd()*0.03,
                          rnd()*0.03, 1+rnd()*0.04];                   // ±5mm, ±2%, ±0.015
```

**The pyramid.** Each run works coarse-to-fine over three levels — shrink ×4, ×2, ×1, with
gaussian blur σ = 2, 1, 0 mm. The coarse level has no fine detail to get trapped by, so it
finds the broad alignment; finer levels refine it.

```js
function pyramidLevel(img, shrinkFactor, sigmaMm) {
  return shrink(blur(img, sigmaMm / img.mm), shrinkFactor);   // smooth THEN decimate
}
```

Order matters: decimating a 250 px image to 63 px already aliases, and no amount of
blurring *afterwards* recovers frequencies that got folded.

**The optimiser** is Nelder-Mead — a derivative-free simplex method. It keeps `n+1` points,
throws away the worst, and reflects it through the centre of the rest. 80 iterations per
level.

### Stage 05 — One run, traced

Pure instrumentation. Prints every iteration: level, NMI, parameters, which simplex move
was taken. No effect on results.

### Stage 06 — The scale gate

**The problem it solves.** Because Stage 01 already put both images on the same mm grid,
there is no real zoom left to find. But the optimiser doesn't know that. The MRI's
rectangular **border** is a huge, straight, high-contrast feature, and so is the canvas
border. Stretching the MRI until its border coincides with the frame border raises NMI —
and aligns nothing.

Measured on this dataset:

```
knee/sagittal   MRI box / canvas = 0.779   recovered scales 0.810–0.840
spine/coronal   MRI box / canvas = 0.827   recovered scales 1.177–1.206   (1/0.827 = 1.209)
```

The recovered scales sit on the canvas ratio or its reciprocal to within a few percent.

**Two tests, not one band:**

```js
function scaleVerdict(scale, cr) {
  if (Math.abs(scale - 1) > SCALE_TOL) return { ok:false, why:"outside_sanity_bound" };
  if (cr > 0.05 && cr < 0.95) {                       // only if there IS a border gap
    var sus = [cr, 1/cr];
    for (i...) if (Math.abs(scale - sus[i]) <= CANVAS_FIT_TOL)
      return { ok:false, why:"canvas_fit" };
  }
  return { ok:true, why:"" };
}
```

- **TEST 1** — outer sanity bound, `SCALE_TOL = 0.60`. Generous on purpose.
- **TEST 2** — the canvas-fit veto, `CANVAS_FIT_TOL = 0.04`. Reject scales that coincide
  with the canvas ratio or its reciprocal.

The `cr > 0.05 && cr < 0.95` guard matters: if the MRI already fills the canvas there is
no border gap to close, and a scale near 1.0 is the *correct* answer, not a suspicious one.

**And how is "scale" measured?**

```js
function matScale(M) {
  var E=(a+d)/2, F=(a-d)/2, G=(c+b)/2, H=(c-b)/2;
  var Q=Math.sqrt(E*E+H*H), R=Math.sqrt(F*F+G*G);
  return ((Q+R) + Math.abs(Q-R)) / 2;      // = (σ₁ + σ₂)/2, the MEAN SINGULAR VALUE
}
```

> ⚠️ **Circle this line. It is the bug.** Part 3 explains why.

### Stage 07 — Selection

Best rigid vs best **admissible** affine. Note the asymmetry: rigid is filtered with
`gated=false`, affine with `gated=true`.

```js
var bR = bestOf(rigid, false), bA = bestOf(affine, true);
...
else if (bA.nmi > bR.nmi + MIN_TRANSFORM_MARGIN) { chosen = bA; }
else { chosen = bR; why = "affine did not clear the margin — keep the simpler model"; }
```

Affine must beat rigid by `MIN_TRANSFORM_MARGIN = 0.005`, otherwise keep the simpler model.
The comment in the code says *"rigid: exempt — it cannot scale."* True. **But rigid can
still rotate freely, and nothing checks that either.**

### Stage 08 — Classification

```js
if (cov < 0.25)                     return "sparse_overlap";   // cause…
if (best < baseline - MIN_GAIN)     return "regressed";        // …before symptom
if (best <= baseline + MIN_GAIN)    return "marginal";
                                    return "improved";
```

`MIN_GAIN = 0.010` is a **dead band** around the baseline. The key idea: *"did not improve"*
and *"made it worse"* are different events. `marginal` is a legitimate outcome, not a
failure. Only `regressed` is a failure.

### Stage 09 — What ships

```js
if (chosen && chosen.nmi > base.nmi)  ship the registered result
else                                  ship the UNREGISTERED image
```

Never ship something worse than doing nothing.

---

## Part 2 — v3 versus v6

### 2.1 What is verifiable

**Two facts I can state with certainty:**

**A. The lab gained a stage.** In v3 "WHAT SHIPS" is stage **08**; in v6 it is stage **09**.
The current published source has nine stages:

```
01 Resample   02 Baseline   03 Rigid   04 Affine   05 One run traced
06 Scale gate   07 Selection   08 Classification   09 What ships
```

**B. The rendering changed.** v3's caption reads *"CT in amber, MRI in cyan"* and its
CT/MRI panels are rendered **tinted**. v6's reads *"Scans render greyscale, as they are
meant to be read"* and those panels are **greyscale**, with the amber/cyan reserved for
the two overlay panels.

> **The rendering change is cosmetic and is NOT the cause of the tilt.** You can see the
> MRI's own rectangular frame rotated in the v6 panel, with black triangular wedges at the
> corners. That is real geometry — the `NaN` fill outside a rotated raster. No colour map
> can produce it.

### 2.2 What is NOT recoverable — read this before trusting any v3 claim

**I could not retrieve artifact v3's source code.** The Artifact tool exposes no
version-history read, and fetching the artifact URL returns the *current* version only.
Everything I know about v3 comes from your screenshot.

So the honest position is:

| Claim about v3 | Status |
|---|---|
| WHAT SHIPS was stage 08 (8 stages, not 9) | **verified** — visible in the screenshot |
| CT/MRI panels rendered tinted, not greyscale | **verified** — visible |
| The overlay is better aligned than v6's | **your observation**, and I'll take it |
| *Which* stage was added between v3 and v6 | **unknown** |
| Whether v3's `matScale`, gate, seeds or metric differed | **unknown** |
| Whether v3 shipped rigid where v6 ships affine | **unknown** — v3's stat row is cropped |

**If you want Part 2 completed properly, paste v3's HTML** (or tell me the version number
in the artifact's version picker and I'll ask you to export it). Until then, treat every
v3 internal as unestablished.

### 2.3 The one hypothesis worth testing

v6's screenshot shows `TRANSFORM = affine`, `baseline 1.1201 → 1.2411`, `Δ +0.1210`. That
is a very large gain — roughly double the biggest gain in the entire 33-slice sweep.

**A suspiciously large NMI gain accompanied by a visible geometric distortion is the
signature of a metric being gamed, not of a better alignment.** The most likely story is
that v3 shipped **rigid** on this slice (which cannot shear, and whose rotation is one
bounded parameter) and v6 ships **affine** (which can do both, unchecked). That is testable
the moment you can see v3's stat row — but it is a hypothesis, not a finding.

---

## Part 3 — Why the MRI tilts

Four causes, independent, all real. The first two are the ones to fix.

### Cause 1 — The gate is blind to rotation. Mathematically blind.

`matScale` returns the **mean singular value**, `(σ₁ + σ₂)/2`.

Take a pure rotation by angle θ:

```
M = [ cos θ   −sin θ ]
    [ sin θ    cos θ ]

E = (a+d)/2 = cos θ      H = (c−b)/2 = sin θ      F = 0      G = 0
Q = √(cos²θ + sin²θ) = 1                          R = 0
σ₁ = Q+R = 1        σ₂ = |Q−R| = 1
matScale = (1 + 1)/2 = 1.0000        ← exactly 1, for ANY θ
```

> **A rotation of one degree and a rotation of ninety degrees both report `scale = 1.0000`.**
> The gate passes them identically. There is no threshold you can set on this number that
> will ever catch a rotation, because rotation does not change singular values. That is
> the definition of a rotation.

**The affine has 6 free parameters and the gate constrains a 1-dimensional summary of
them.** Rotation lives entirely in the null space of that summary.

### Cause 2 — The gate is nearly blind to shear too

Shear is worse, because it *looks* like it should be caught. Take a horizontal shear:

```
M = [ 1  k ]
    [ 0  1 ]          det = 1, so it preserves area

matScale = √(1 + k²/4)
```

| shear `k` | shear angle | `matScale` reports | gate verdict | `σ₁/σ₂` would report |
|---|---|---|---|---|
| 0.14 | 8.0° | 1.0024 | passes | **1.150** |
| 0.30 | 16.7° | **1.0112** | passes | **1.348** |
| 0.50 | 26.6° | 1.0308 | passes | **1.640** |
| 2.50 | 68.2° | 1.6008 | *finally* fails TEST 1 | 8.127 |

A 27° shear — which visibly turns the anatomy into a parallelogram — reports a scale of
1.03. You would need a **68° shear** before `SCALE_TOL = 0.60` reacts.

**Why:** a volume-preserving shear has `σ₁·σ₂ = 1`. Stretching one axis and squashing the
other by the same factor leaves the *mean* almost unchanged. The mean singular value is
precisely the wrong summary statistic for detecting shear.

The number that *does* detect it is the **ratio** `σ₁/σ₂` (anisotropy) — the last column.
It rises monotonically and steeply where `matScale` is flat.

> **Every figure in both tables above was computed and verified**, not estimated.
> `matScale` was also confirmed equal to numpy's mean singular value to within 6.7 × 10⁻¹⁶
> over 2000 random 2×2 matrices, so "mean singular value" is an exact description of what
> the gate measures, not an approximation.
>
> Note where `σ₁/σ₂ = 1.150` lands: at `k = 0.14`, an 8° shear. That is the empirical
> justification for `ANISOTROPY_TOL = 1.15` in Fix 2 — it puts the cut at roughly 8° of
> shear, which is about as much as patient repositioning can honestly explain.

### Cause 3 — The cost function's domain moves with the parameters

This is the subtle one, and it's the deepest.

NMI is computed over pixels valid in **both** images:

```js
var a = fx[i]; if (a !== a) continue;    // skip if fixed is NaN
var b = mv[i]; if (b !== b) continue;    // skip if moving is NaN
```

The MRI is placed with `NaN` outside its rectangle (Stage 01). **When you rotate the MRI,
the corners swing outside the frame and become `NaN`.** So rotating doesn't just move the
anatomy — it *changes which pixels are scored*.

> Recall the warning in §0.6. NMI removes dependence on how much *entropy* is in the frame.
> It does nothing about which *pixels* are in the frame. Comparing NMI at θ = 0° against
> NMI at θ = 8° is comparing two different datasets.

And here is the perverse incentive: the pixels a rotation clips first are the **corners** —
which are the parts of the MRI least likely to have a matching CT structure, i.e. the
pixels contributing most to the joint histogram's smear. **Throwing away your worst-fitting
pixels raises NMI.** The optimiser has discovered that it can improve its score by
discarding evidence.

That is very plausibly where a `Δ +0.1210` comes from.

### Cause 4 — The coverage floor is far too loose to stop it

The only thing standing against Cause 3:

```js
if (!(r.nmi === r.nmi) || r.coverage < 0.05) return 10;
```

Coverage must fall below **5%** before the cost function objects. The MRI can shed 90% of
its valid pixels and be scored as though nothing happened.

### Cause 5 (contributing) — Centring makes the optimiser travel too far

`placeOn` centres the MRI instead of using `ImagePositionPatient`. So Nelder-Mead starts
tens of millimetres from the true answer, wandering through a landscape full of
border-alignment local optima before it ever reaches the anatomical one. A rotated
border-fit is one of the peaks it can hit on the way.

**This does not apply to `registration_demo_sweep_v3.py`,** which derives the translation
in closed form from the headers. It is an explorer-only weakness — but it makes the
explorer *more* prone to the tilt than the real pipeline, which is worth knowing before
you conclude the pipeline is broken.

---

## Part 4 — The fix, in order

### Fix 1 — Decompose the transform properly *(do this first; everything else needs it)*

Right now the code reduces a 6-parameter affine to a single number. Report all four
physically meaningful quantities instead. **They all come from the `E, F, G, H` that
`matScale` already computes** — this is nearly free:

```js
function decompose(M) {
  var a = M[0], b = M[1], c = M[2], d = M[3];
  var E = (a + d) / 2, F = (a - d) / 2, G = (c + b) / 2, H = (c - b) / 2;
  var Q = Math.sqrt(E * E + H * H), R = Math.sqrt(F * F + G * G);
  var s1 = Q + R, s2 = Math.abs(Q - R);
  return {
    scale:      (s1 + s2) / 2,          // mean singular value — as today
    rotation:   Math.atan2(H, E),       // radians; polar-decomposition rotation
    anisotropy: s2 > 1e-9 ? s1 / s2 : Infinity,   // 1.0 = no shear/stretch
    sigma:      [s1, s2]
  };
}
```

`Math.atan2(H, E)` is the closed form for the rotation in the polar decomposition
`M = R·S` — the rotation "closest" to `M`. For a pure rotation it returns θ exactly.

**Sanity-check it before trusting it:** feed in a known 10° rotation and confirm
`rotation → 0.1745`, `scale → 1.0000`, `anisotropy → 1.0000`.

### Fix 2 — Gate rotation and anisotropy, not just scale

```js
var ROT_TOL_DEG    = 12;    // in-plane rotation cap, degrees
var ANISOTROPY_TOL = 1.15;  // σ₁/σ₂ ceiling

function transformVerdict(M, cr) {
  var D = decompose(M);

  if (Math.abs(D.scale - 1) > SCALE_TOL)
    return { ok:false, why:"outside_sanity_bound" };

  if (Math.abs(D.rotation) * 180/Math.PI > ROT_TOL_DEG)
    return { ok:false, why:"rotation_" + (D.rotation*180/Math.PI).toFixed(1) + "deg" };

  if (D.anisotropy > ANISOTROPY_TOL)
    return { ok:false, why:"shear_aniso_" + D.anisotropy.toFixed(3) };

  if (cr > 0.05 && cr < 0.95) {
    var sus = [cr, 1/cr];
    for (var i = 0; i < sus.length; i++)
      if (Math.abs(D.scale - sus[i]) <= CANVAS_FIT_TOL)
        return { ok:false, why:"canvas_fit" };
  }
  return { ok:true, why:"" };
}
```

**Justifying `ROT_TOL_DEG = 12`.** Both scans are of the same patient, positioned by a
radiographer, roughly supine, in the same anatomical convention. In-plane rotation between
CT and MRI of the same region is a repositioning difference — a few degrees. Twelve is
generous. If your data genuinely needs more, *raise it deliberately and write down why*;
don't leave it unbounded because unbounded was the default.

> **Apply this to rigid as well.** Today rigid is called with `gated=false` on the grounds
> that "it cannot scale". True — but it rotates freely, and an unconstrained rigid rotation
> is exactly as wrong as an unconstrained affine one. Change `bestOf(rigid, false)` to
> `bestOf(rigid, true)`; the scale and anisotropy tests are automatically no-ops for a true
> rotation matrix, so only the rotation cap will ever bite.

### Fix 3 — Freeze the scoring mask *(the real fix for Cause 3)*

Compute the valid-pixel mask **once**, at the identity transform, and score every candidate
on that same fixed set of pixels. A transform that pushes anatomy out of the mask then
scores `NaN` there and is *penalised*, instead of quietly being rewarded for discarding
its worst pixels.

```js
// once, before optimising:
var scoreMask = new Uint8Array(W*H);
for (i...) scoreMask[i] = (F.data[i]===F.data[i] && M0.data[i]===M0.data[i]) ? 1 : 0;

// inside nmiScore: iterate only where scoreMask[i], and treat warped NaN
// as a real miss (assign it a dedicated bin) rather than skipping it.
```

This makes the cost function's **domain constant**, which is what turns NMI back into a
fair comparison across candidates. It is the single most important change here.

### Fix 4 — Tighten the coverage floor

If Fix 3 is too invasive for now, this is the cheap 80% version:

```js
var covFloor = 0.90 * baseCoverage;      // computed once, at identity
...
if (!(r.nmi === r.nmi) || r.coverage < covFloor) return 10;
```

Relative to the baseline, not an absolute 5%. Losing more than 10% of your valid pixels
should cost you the run.

### Fix 5 — Add a noise floor to the affine-vs-rigid margin

v6 uses a flat `MIN_TRANSFORM_MARGIN = 0.005` (artifact line 3187). The real pipeline
(`registration_demo_sweep_v3.py:463-471`) uses an **adaptive** floor — the larger of the
constant and the affine's own seed spread:

```js
var affineNoise = spread(affine.map(r => r.nmi));
var noiseFloor  = Math.max(affineNoise || 0, MIN_TRANSFORM_MARGIN);
if (bA.nmi > bR.nmi + noiseFloor) { chosen = bA; }
```

The reasoning (from that file's header): if three affine seeds disagree by 0.04, then a
0.006 win over rigid is noise, and promoting the more complex model on noise is exactly how
you end up shipping a tilt. The explorer already *computes* the spread and prints it — it
just doesn't use it.

### Fix 6 *(structural, optional)* — Constrain the affine's parameterisation

The deepest fix is not to gate a bad parameterisation but to use a better one. Instead of
6 free matrix entries, parameterise the affine as `[tx, ty, θ, sx, sy, shear]` and build
the matrix from them. Then the optimiser physically cannot express a 40° rotation if you
bound θ, and Nelder-Mead is searching a space where every axis means something.

This is a bigger change and it alters the seeds and the search geometry, so results will
shift. Do Fixes 1–5 first and measure.

---

## Part 5 — How to know the fix worked

Fixes that only make the picture look better are how you get a *different* wrong answer.
Check all four:

1. **Unit-test the decomposition.** Known 10° rotation → `rotation = 0.1745`,
   `scale = 1.0000`, `anisotropy = 1.0000`. Known `k = 0.3` shear → `anisotropy = 1.348`.
   If these don't hold, nothing downstream means anything.

2. **Re-run this slice.** Expect the affine to be **rejected** (`rotation_*deg` or
   `shear_aniso_*`), rigid to be selected, and Δ NMI to **fall** from +0.1210 to something
   modest. **A smaller number is the success condition here**, because the large one was
   bought with a distortion.

3. **Re-run the full 33-slice sweep** (`python registration_demo_sweep_v3.py`) after
   porting Fixes 1, 2 and 5 to `registration_demo_sweep_v3.py` and `working_regis.py`.
   The current baseline to beat is on record:

   ```
   improved=26  marginal=7  regressed=0
   shipped: rigid=19  affine=13  unregistered=1
   affine seeds rejected by the scale gate: 19/99
   ```

   Expect `affine` ships to **drop** and rejections to **rise**. Watch for `regressed`
   appearing — if it does, the gate is now too tight and you've thrown away real corrections.

4. **Eyeball the overlays.** Amber/cyan fringing should be a *uniform* halo where the two
   disagree. A one-sided fringe — cyan along one edge, amber along the opposite — means
   residual rotation, and it is visible long before any number moves.

---

## Appendix — Constants, one table

| Constant | Value | Job | Where |
|---|---|---|---|
| `SCALE_TOL` | 0.60 | outer sanity bound on \|scale − 1\| | gate TEST 1 |
| `CANVAS_FIT_TOL` | 0.04 | how close to canvas ratio counts as a cheat | gate TEST 2 |
| `MIN_GAIN` | 0.010 | dead band around baseline | classification |
| `MIN_TRANSFORM_MARGIN` | 0.005 | affine must beat rigid by this | selection |
| `MIN_MRI_COVERAGE` | 0.25 | below this the slice is starved | classification |
| `NMI_BINS` | 32 (explorer) / 64 (`registration_demo.py`) | joint histogram resolution | metric |
| pyramid shrink | 4, 2, 1 | coarse-to-fine levels | `registerOnce` |
| pyramid σ (mm) | 2, 1, 0 | blur per level | `registerOnce` |
| Nelder-Mead iters | 80 per level | optimiser budget | `registerOnce` |
| **`ROT_TOL_DEG`** | **12 — proposed** | **rotation cap** | **Fix 2** |
| **`ANISOTROPY_TOL`** | **1.15 — proposed** | **shear cap** | **Fix 2** |

**Files this document describes**

| File | Role |
|---|---|
| artifact `1f78eab8` "Registration, Instrumented" | the explorer, v6 |
| `Preprocessing/working_regis.py` | line-for-line Python mirror of v6 |
| `Preprocessing/registration_demo_sweep_v3.py` | the real 3D pipeline — different program, see below |
| `Preprocessing/docs/registration_docs.md` | the theory this document is a companion to |

> **Do not confuse the explorer with the pipeline.** The explorer works on one 2D slice,
> centres the MRI, and optimises NMI with Nelder-Mead. `registration_demo_sweep_v3.py`
> works on 3D volumes, derives translation in closed form from DICOM headers, applies N4
> bias correction, and optimises Mattes MI with SimpleITK's gradient descent, using NMI only
> to *select* among candidates. Their NMI numbers are not comparable — different bin counts,
> different clipping, different pixel populations. **Only the verdicts are comparable.**
