# Registration, from first principles

**The method actually running in this repository** — `registration_idea.py`, driven by
`image_processing.estimate_volume_translation`, reviewed through `qc_app`.

This is a teaching document. Each section introduces one idea, says why it is
there, and gives an example you can run. The last section puts all of them
together on one real series. Every number here was computed from this
repository's own code and data — nothing is illustrative-only, and you can
re-derive all of it.

---

## Table of contents

- [0. The problem, and the shape of the answer](#0-the-problem-and-the-shape-of-the-answer)
- [1. The two images do not share a ruler](#1-the-two-images-do-not-share-a-ruler)
- [2. The two images do not share a language](#2-the-two-images-do-not-share-a-language)
- [3. Entropy: how surprising is an image?](#3-entropy-how-surprising-is-an-image)
- [4. Joint entropy and mutual information](#4-joint-entropy-and-mutual-information)
- [5. Why *normalised* mutual information](#5-why-normalised-mutual-information)
- [6. The fixed window, and the cheat it prevents](#6-the-fixed-window-and-the-cheat-it-prevents)
- [7. Searching: coarse-to-fine](#7-searching-coarse-to-fine)
- [8. From one slice to a whole stack](#8-from-one-slice-to-a-whole-stack)
- [9. The four gates](#9-the-four-gates)
- [10. Why we refuse to use world coordinates](#10-why-we-refuse-to-use-world-coordinates)
- [11. What surrounds the registration](#11-what-surrounds-the-registration)
- [12. Final worked example, end to end](#12-final-worked-example-end-to-end)
- [13. Parameter reference](#13-parameter-reference)

---

## 0. The problem, and the shape of the answer

We have, for one patient, a CT series and an MRI series of the same anatomy.
We want them **pixel-aligned**, so that a network trained to turn MRI into CT
sees, at every pixel, the same piece of tissue in both.

The whole method is two sentences:

> 1. Resample the CT slice and the MRI slice so that one pixel is exactly one
>    millimetre in both.
> 2. Slide the MRI over the CT, one whole pixel at a time, and keep the
>    position with the best normalised mutual information.

That is all of `registration_idea.py`. No optimiser, no gradient descent, no
random restarts.

Two properties fall out of that choice for free, and they are the reason it was
chosen:

- **A whole-pixel slide cannot rotate, scale, or shear.** Those failure modes
  are not defended against — they are *impossible to express*. A method that
  can only translate can only ever be wrong about translation.
- **There are no random numbers anywhere.** Run it twice on the same data and
  you get the same answer, to the bit. That matters when a human has approved
  a result and you need it to still be that result tomorrow.

The answer we are looking for is therefore just two integers: `dx` and `dy`,
in pixels — which, after step 1, are also millimetres.

---

## 1. The two images do not share a ruler

A CT pixel and an MRI pixel are not the same size. Straight from this dataset
(`PA33_Reshma/ST0/SE2`):

```
CT   512 x 512 pixels at 0.491 mm  ->  251 x 251 mm of patient
MRI  256 x 256 pixels at 0.781 mm  ->  200 x 200 mm of patient
```

Overlay those arrays index-for-index and you are comparing a 0.49 mm patch
against a 0.78 mm patch. Nothing downstream can recover from that.

**The fix** is `to_1mm`: bilinear resampling so one pixel spans exactly one
millimetre on both axes.

```python
# registration_idea.py
def to_1mm(a, spacing):
    sy, sx = spacing
    h, w = a.shape
    H, W = max(1, int(round(h * sy))), max(1, int(round(w * sx)))
    ...   # bilinear sample onto the new grid
```

### Example 1 — run it

```python
import numpy as np, registration_idea as ri
print(ri.to_1mm(np.zeros((512, 512)), (0.491, 0.491)).shape)   # (251, 251)
print(ri.to_1mm(np.zeros((256, 256)), (0.781, 0.781)).shape)   # (200, 200)
```

Both are now in millimetres. A 50 mm structure spans 50 pixels in each. The
arrays are still *different sizes* — 251² and 200² — and that is fine and
expected: they cover different fields of view. Section 6 explains how one is
laid over the other.

> **Why bilinear and not nearest-neighbour?** We are shrinking a fine grid onto
> a coarser one. Nearest-neighbour would discard three of every four CT samples
> and alias the result. Bilinear averages them, which is the honest thing to do
> when going from 0.49 mm to 1 mm.

> **Why 1 mm and not 0.5 mm?** Because the search is in whole pixels, the pixel
> size *is* the precision of the answer. 1 mm is finer than the through-plane
> spacing (5–7 mm here) and finer than the differences we are trying to detect.
> Halving it would quadruple the search cost for precision the data does not
> support.

---

## 2. The two images do not share a language

Here is the thing that makes CT↔MRI hard, and it is worth sitting with.

In CT, brightness means **radiodensity**, on an absolute calibrated scale
(Hounsfield Units). Bone is bright, fat is dark, and that is true in every CT
ever taken.

In MRI, brightness means **whatever the pulse sequence was designed to
emphasise**, on no absolute scale at all. In a T2 sequence, fluid is bright and
cortical bone is *black* — the exact opposite of CT.

So the naive approach — "line them up so that bright meets bright" — is not
merely inaccurate here, it is **backwards**. Sum-of-squared-differences,
correlation, and every other intensity-matching metric are unusable across
modalities.

What *is* true is weaker and sufficient:

> Wherever the two images are aligned, knowing the CT value tells you something
> about the MRI value — even if the relationship is inverted, non-linear, or
> different for each tissue.

"Tells you something about" is exactly what **mutual information** measures.
Sections 3–5 build it up.

---

## 3. Entropy: how surprising is an image?

Entropy measures how unpredictable a set of values is. Low entropy = boring and
predictable. High entropy = varied.

```
H(A) = - Σ  p(a) · log p(a)
```

where `p(a)` is the fraction of pixels having value `a`.

### Example 2 — three tiny images

| Image | Histogram | H (nats) | Reading |
|---|---|---|---|
| `[5, 5, 5, 5]` | one bin at 1.0 | **0.0000** | perfectly predictable — no information |
| `[0, 0, 9, 9]` | two bins at 0.5 | **0.6931** | one binary question's worth (= ln 2) |
| `[0, 3, 6, 9]` | four bins at 0.25 | **1.3863** | two binary questions' worth (= ln 4) |

```python
import numpy as np, registration_idea as ri
for a in ([5,5,5,5], [0,0,9,9], [0,3,6,9]):
    v, c = np.unique(a, return_counts=True)
    print(a, "H =", round(ri.entropy(c / c.sum()), 4))
```

Note the first row: **a flat image has zero entropy**. This is not a curiosity —
it is why `make_scorer` returns `None` on a blank slice. There is no
information there to align with, and saying so is more honest than returning a
number.

---

## 4. Joint entropy and mutual information

Now do the same over *pairs* of values, one from each image at the same pixel:
`H(A, B)`, the entropy of the joint histogram.

The key insight:

> When two images are **aligned**, each CT value tends to co-occur with a
> narrow range of MRI values. The joint histogram is concentrated into a few
> cells, so `H(A,B)` is **low**.
>
> When they are **misaligned**, every CT value gets smeared across many MRI
> values. The joint histogram spreads out, and `H(A,B)` is **high**.

Mutual information is how much lower the joint entropy is than it would be if
the two were unrelated:

```
MI(A, B) = H(A) + H(B) - H(A, B)
```

Crucially, MI **never assumes bright means bright**. It only asks whether
knowing one value predicts the other. Inverted contrast scores just as highly
as matched contrast — which is exactly what CT↔MRI needs.

### Example 3 — aligned vs shifted, with inverted contrast

A 4×4 CT with a dark left half and a bright right half, and an MRI with the
same boundary but **inverted** contrast:

```
CT              MRI
0 0 1 1         9 9 2 2
0 0 1 1         9 9 2 2
0 0 1 1         9 9 2 2
0 0 1 1         9 9 2 2
```

```python
import numpy as np, registration_idea as ri
ct  = np.array([[0,0,1,1]]*4, float)
mri = np.array([[9,9,2,2]]*4, float)

def nmi_of(a, b, bins=2):
    ai = ri.bin_image(a, a.min(), a.max()+1e-9, bins)
    bi = ri.bin_image(b, b.min(), b.max()+1e-9, bins)
    return ri.nmi(ai, bi, bins)

print(nmi_of(ct, mri))                      # 2.0   perfectly aligned
print(nmi_of(ct, np.roll(mri, 1, axis=1)))  # 1.0   shifted by one pixel
```

**2.0 for the inverted-contrast alignment.** The metric does not care that dark
meets bright; it cares that the mapping is *consistent*. Roll the MRI one pixel
and the boundary no longer coincides — the score collapses to 1.0, which on
this scale means "these two tell you nothing about each other".

---

## 5. Why *normalised* mutual information

Raw MI has a defect that matters here. `MI = H(A) + H(B) − H(A,B)` grows when
`H(A)` and `H(B)` grow. Change *how much of the image you are scoring* and you
change MI, without changing the alignment at all.

That gives a search a way to cheat: shift so that awkward regions leave the
frame, and the score improves for reasons that have nothing to do with anatomy.

The fix used here is the ratio form:

```
NMI(A, B) = ( H(A) + H(B) ) / H(A, B)
```

Read it as: **1.0 means independent, 2.0 means identical.** A ratio is far less
sensitive to the absolute entropies than a difference is.

In this repository's real data, NMI for a well-aligned pair sits around
**1.05 – 1.20**, and what we care about is the *gain* over doing nothing —
typically **+0.01 to +0.15**.

---

## 6. The fixed window, and the cheat it prevents

Normalising the metric is not enough on its own. Two more rules close the hole
completely.

**Rule 1 — every candidate is scored on the same pixels.** The scoring window
is the *entire CT frame*, and it does not move when the MRI moves.

**Rule 2 — MRI that falls outside the frame is not skipped.** It goes into its
own histogram bin (`bins + 1`, the "no MRI here" bin).

Together these mean a shift is **charged** for whatever it pushes out of view,
instead of being quietly rewarded for it.

### Example 4 — watch the coverage change

```python
import numpy as np, registration_idea as ri
ct  = np.random.default_rng(0).random((60, 60))
mri = ct * 0.8 + 0.1                       # perfectly correlated, same size

for dx in (0, 25):
    vals, cov = ri.sample_window(mri, ct.shape, 0, dx)
    print(f"dx={dx:2d}: MRI covers {cov*100:5.1f}% of the frame, "
          f"{np.isnan(vals).sum():4d} px in the 'no MRI here' bin")
```

```
dx= 0: MRI covers 100.0% of the frame,    0 px in the 'no MRI here' bin
dx=25: MRI covers  58.3% of the frame, 1500 px in the 'no MRI here' bin
```

Those 1500 pixels are still scored. They land in a bin that correlates with
nothing, which *raises* the joint entropy and *lowers* NMI. The shift pays for
its own emptiness.

> An earlier version inset the scoring window by the search range "to be safe",
> and ended up scoring only 31% of the CT on average. That is all cost and no
> benefit — the out-of-frame problem was already handled by giving it a bin.

This is also how **differently sized arrays** are compared. `sample_window`
centres the MRI on the CT frame, so a 200² MRI over a 251² CT sits with a
~26-pixel border of "no MRI here" all round. No reprojection needed.

---

## 7. Searching: coarse-to-fine

We need the best `(dy, dx)` in a square of ±40 mm. Trying every whole-pixel
position is 81 × 81 = **6561** evaluations per slice. Workable but wasteful,
and at ±90 mm it becomes 32761 and costs ~32 s per slice.

So the search runs in two passes:

1. **Coarse** — sweep the square at a stride of 4 mm.
2. **Fine** — re-search every whole pixel around the best **5** coarse
   positions (a ±4 mm box each).

### Example 5 — the cost

| Range | Exhaustive | Coarse + fine | Saving |
|---|---|---|---|
| ±40 mm | 6561 | 441 + 405 = **846** | 7.8× |
| ±90 mm | 32761 | 2209 + 405 = **2614** | 12.5× |

**What this gives up, stated plainly:** a strided sweep can step over a peak
narrower than the stride, so "best position found" is no longer identical to
"best position there is". Two things make that unlikely rather than merely
hoped-for:

- anatomy at 1 mm is many pixels wide, so the score varies smoothly across a
  4 mm step — there are no one-pixel spikes to fall between;
- refining the best **five** positions, not just the winner, means a true peak
  that was only runner-up on the coarse pass is still found.

Set `COARSE = 1` to recover the exhaustive search and check for yourself.

Three positions are **always** evaluated whatever the stride: `(0, 0)` — so the
result can never be worse than not moving — and the range limits, so that
hitting the edge stays detectable (this becomes Gate 0 in section 9).

---

## 8. From one slice to a whole stack

Everything so far registers **one slice**. A series has 18–24. What now?

The tempting answer — register every slice independently — is wrong, and this
repository has the measurement to prove it. From `pipeline_config.py`:

> The best per-slice shift across one 18-slice shoulder axial stack goes
> `(+18,+11) → (−57,−24) → (−66,−39)` mm — an **85 mm swing** in `dx`.

Apply those per-slice and the MRI **shears through z**: anatomy that was
continuous down the stack comes out as a staircase. You would be inventing
slice-to-slice motion that was never in the data.

So: **one shift per series.**

```
1. Pick 5 probe slices, at 10/30/50/70/90% through the stack.
2. Register each probe independently.
3. Take the MEDIAN of their answers.
4. Put that median through four gates (section 9).
5. If it passes, apply it to EVERY slice.
```

**Why 10–90% and not 0–100%?** The first and last slices of a stack are the
emptiest and least able to measure anything. Spending probes on them mostly
buys back `None`.

**Why the median and not the mean?** One probe landing on a false peak shifts a
mean; it barely moves a median.

For an 18-slice series the probes are indices `[2, 5, 8, 12, 15]`; for 24
slices, `[2, 7, 12, 16, 21]`. Those are the slices that decide the answer — a
useful thing to know when you crop or erase to improve a registration.

---

## 9. The four gates

The median is a *proposal*. Four checks decide whether it is trustworthy. Fail
any one and **no shift is applied at all** — on the reasoning that leaving the
MRI where the DICOM put it beats moving it somewhere wrong.

### Gate 0 — discard censored probes

A probe whose best shift landed *exactly* on the boundary of the search square
did not find a peak; it hit a **wall**. The true optimum is somewhere further
out that the search could not see. That is a censored observation, not a
measurement, and it is thrown away before it can vote.

**This must happen first**, and the reason is subtle and worth internalising:
several probes pinned against the *same* wall all report the *same* number.
They would sail through the agreement check below with a spread of **zero** —
unanimous censorship reading as unanimous evidence.

> Real case: `PA8_Gunjan/ST0/SE2` at ±40 mm had **5 of 5 probes on the wall**,
> hence 0 usable, hence no shift. Re-run at ±90 mm it resolved cleanly to
> `dx=+24, dy=−54` with NMI **+0.1490** — the strongest gain in the dataset.
> The true offset was ~59 mm, always outside the original window.

### Gate 1 — enough probes survived  (`REG_MIN_PROBES = 2`)

One probe is an unverifiable estimate being imposed on a whole volume. With two
there is at least something to disagree.

### Gate 2 — the probes agree  (`REG_MAX_SPREAD_MM = 20 mm`)

If the surviving probes disagree by more than 20 mm on either axis, no single
translation describes this pair, and forcing one would make most slices worse.

> Real case: `PA15_SumanLata1/ST0/SE1` at ±90 mm proposes `dy = −54 mm` but the
> probes spread **21 mm** across. One millimetre over the limit, so it is
> declined. (Independently, a human reviewer hand-nudged that same series by
> −48…−58 mm — the algorithm and the eye agree on the offset; only the
> consistency check refuses it.)

### Gate 3 — the chosen shift actually helps  (`REG_MIN_GAIN = 0.010`)

The median is **re-scored on every probe** against that probe's own do-nothing
baseline, and applied only if it raises NMI by more than 0.010 on average.

This is the most important gate, and the reason is precise: gates 1 and 2 ask
whether the *per-slice searches* were consistent. Gate 3 measures **the shift
that is actually about to be applied**. A median can be a position that no
probe ever proposed — this is what stops such a position being used on the
strength of votes cast for other ones.

### What the gates do on this dataset

Across 113 registered series:

| Outcome | Count | Meaning |
|---|---|---|
| **Shift applied** | **76** | passed all four |
| Gate 2 — probes disagree | 20 | no single translation fits |
| Gate 3 — gain too small | 16 | **already aligned; nothing to fix** |
| Gate 1 — too few probes | 1 | mostly empty slices |

Read that table carefully: **"rejected" does not mean "bad".** The 16 gate-3
failures are pairs that were already aligned, where the best available shift
improved NMI by ~0.003. Only the 21 failing gates 1–2 are genuinely unresolved.

---

## 10. Why we refuse to use world coordinates

Every DICOM slice carries `ImagePositionPatient` and `ImageOrientationPatient`
— where the scanner says that slice sits in the room. It is tempting to use
them: project the MRI onto the CT's 3D grid and let the geometry do the work.

**We deliberately do not.** Only `PixelSpacing` is read. Here is the evidence
from this dataset that settled it.

**Measurement 1 — the frames rarely agree.** Comparing the CT and MRI direction
cosines for all 120 series: only **7** agree to within 0.5°; **61** differ by
more than 5°; the median mismatch is **5.2°**.

**Measurement 2 — using them destroys data either way.** On
`PA10_Suman/ST0/SE0` (8.6° mismatch):

| Approach | Result |
|---|---|
| Overwrite the MRI's direction with the CT's | **3 of 18 slices completely blank** |
| Honour the MRI's true direction | oblique reformat → a thin diagonal sliver |
| **Ignore world coordinates, pair in 2D** | **0 blank, 96.4% of CT anatomy covered** |

An 8° rotation over a 280 mm field of view swings the far end by ~39 mm — quite
enough to push end slices out of the other volume entirely.

**Measurement 3 — the slice ordering is not reliable either.** SimpleITK sorts
a series by `ImagePositionPatient`. On **24 of 120** series that order disagrees
with the file numbering, or between the two modalities — including four where
one side is fully **reversed**, pairing the top of one stack with the bottom of
the other.

So slices are paired by **image number** (`IM7` ↔ `IM7`), which is the
dataset's own convention and what a human sees in a DICOM browser.

> **The general lesson.** Metadata is a claim, not a measurement. When a claim
> is checkable against the pixels, check it. Here the pixels say the geometry
> tags are unreliable, so the method was built not to depend on them.

---

## 11. What surrounds the registration

Registration is one step in a longer chain. Two neighbours matter enough to
mention.

### N4 bias field correction (MRI only, *before* registration)

MRI has smooth brightness shading from receive-coil sensitivity — the same
tissue appears brighter near the coil. Left in, it blurs the joint histogram:
one tissue occupies several intensity bins depending on *where* it sits, which
weakens exactly the statistical dependence NMI relies on.

N4 is fitted to the **whole 3D volume**, not slice by slice. A coil's
sensitivity profile is one smooth function over the whole bore; it does not
restart at every slice boundary. Fitting per slice hands each slice its own
free brightness scaling and manufactures slice-to-slice steps that were never
in the data.

### Intensity normalisation (*after* registration)

- **CT** → clip to a region-specific HU window, rescale to `[0,1]`. Brain uses
  `0…80 HU` (brain-tissue contrast lives in a tiny HU range); abdomen
  `−160…240`; musculoskeletal and spine `−200…300`.
- **MRI** → no absolute scale exists, so bounds come from the **0.5 and 99.5
  percentiles of non-zero voxels** of that volume. Non-zero matters: the black
  background dominates an MRI and would drag the percentiles down.

Percentiles are computed **before** the shift is applied, and that ordering is
safe: shifting moves pixels around but does not change their values, and the
only new value it introduces is the 0.0 fill, which the non-zero rule already
excludes.

---

## 12. Final worked example, end to end

**`PA33_Reshma/ST0/SE2`** — an axial head pair, 18 slices. Every number below
is what the pipeline actually recorded.

### Step 1 — load, and pair the slices

```
CT  : 18 files, IM0..IM17, axial, SeriesNumber 350
MRI : 18 files, IM0..IM17, axial, SeriesNumber 18, "t2_tse_tra_FIL_1"
```

Paired by image number: `IM0↔IM0 … IM17↔IM17` (§10). Both are axial — this
patient originally had `SE0` and `SE1` swapped between modalities, so the CT
was coronal where the MRI was sagittal. No translation can fix a plane
mismatch; the folders had to be corrected first. **Always confirm the two
series are the same plane before trusting any registration result.**

### Step 2 — N4 on the MRI volume (§11)

3D fit, anisotropic control-point mesh chosen from this series' field of view.

### Step 3 — both to 1 mm per pixel (§1)

```
CT   512x512 @ 0.49 mm  ->  251 x 251 px
MRI  256x256 @ 0.78 mm  ->  200 x 200 px
```

Different sizes, same ruler. The MRI will sit centred in the CT frame with a
~26 px border of "no MRI here" (§6).

### Step 4 — intensity bounds (§11)

```
CT window     : 0 .. 80 HU          (brain profile)
MRI percentiles: 9.0 .. 746.4       (0.5th/99.5th of non-zero voxels)
```

### Step 5 — probe and search (§7, §8)

Probes at 10/30/50/70/90% of 18 slices → indices **[2, 5, 8, 12, 15]**. Each is
searched over ±40 mm, coarse stride 4 then fine around the best 5 — **846
positions** instead of 6561.

### Step 6 — the gates (§9)

```
Gate 0  edge-censored probes ........ 0 discarded
Gate 1  usable probes ............... 5 of 5          (need >= 2)   PASS
Gate 2  probe spread ................ 3 mm / 0 mm     (limit 20)    PASS
Gate 3  mean NMI gain ............... +0.0881         (need > 0.010) PASS
```

### Step 7 — apply

```
shift +1 mm across, -20 mm down, agreed by 5 probe slices
to within 3 mm, NMI +0.0881
```

Applied to **all 18 slices** — the same whole-pixel slide on each, so no shear
is introduced through z.

### Step 8 — verify

```
CT anatomy covered by MRI : mean 92.5%,  worst slice 89.0%
Blank slices              : 0 of 18
Total runtime             : 3.8 s
```

### Reading the result

- **NMI gain +0.0881** is strong. Compare: `PA0_Ranjeet/SE0` +0.0630 (good),
  and a typical gate-3 rejection sits at +0.003.
- **Spread 3 mm / 0 mm** — five independent slices, spread across the stack,
  agreed to within 3 mm. That is the strongest evidence available that a single
  translation genuinely describes this pair.
- **dy = −20 mm, dx = +1 mm.** Almost pure vertical offset — consistent with
  patient positioning differing between the two scans, which is precisely the
  error a translation-only method is designed for.
- **Worst-slice coverage 89%** means even the end slices overlap well. When
  this number drops (`PA33_Reshma/SE1` has a worst slice at 21%), those specific
  slices are candidates for rejection rather than adjustment — the two stacks
  simply do not cover the same anatomy there, and no registration fixes that.

---

## 13. Parameter reference

All of these live in `Preprocessing/pipeline_config.py`. There is one copy; the
QC application reads them from there rather than keeping its own.

| Parameter | Value | Meaning | §|
|---|---|---|---|
| `TARGET_SPACING_MM` | 1.0 | pixel size after resampling; also the precision of the answer | 1 |
| `REG_BINS` | 32 | histogram bins per image (+1 for "no MRI here") | 4, 6 |
| `REG_SEARCH_MM` | 40.0 | search half-width; cost is **quadratic** in it | 7 |
| `REG_COARSE_MM` | 4.0 | stride of the first sweep; 1 = exhaustive | 7 |
| `REG_KEEP` | 5 | coarse positions given a fine search | 7 |
| `REG_N_PROBES` | 5 | slices sampled, at 10/30/50/70/90% | 8 |
| `REG_MIN_PROBES` | 2 | fewest usable probes before any shift | 9 |
| `REG_MAX_SPREAD_MM` | 20.0 | how far probes may disagree | 9 |
| `REG_MIN_GAIN` | 0.010 | NMI the chosen shift must buy | 9 |

### Entry points

```python
register(ct, mri, ...)    # best shift for one slice pair, or None
make_scorer(ct, mri)      # the NMI-of-a-shift function alone, for re-scoring
apply_shift(...)          # move an image, 0.0 outside the frame
```

Nothing raises on bad input, and nothing prints unless asked. A slice that
cannot be measured returns `None` — because this runs over thousands of slices
unattended, and the unmeasurable ones are near-empty end-of-stack slices the
pipeline deliberately keeps, not a sign anything is wrong.

### Glossary

| Term | Meaning |
|---|---|
| **Entropy** | how unpredictable a set of values is; 0 = all identical |
| **Joint histogram** | 2D table counting how often CT value *a* co-occurs with MRI value *b* |
| **MI** | `H(A) + H(B) − H(A,B)`; how much one image predicts the other |
| **NMI** | `(H(A) + H(B)) / H(A,B)`; 1 = independent, 2 = identical |
| **Probe slice** | one of 5 slices whose independent answers decide the stack's shift |
| **Spread** | how far the probes' answers disagree, in mm |
| **Gain** | NMI improvement of the chosen shift over doing nothing |
| **Censored probe** | one whose best shift hit the search boundary — a wall, not a peak |

---

## Exercises

1. **Break the metric.** Take an aligned CT/MRI pair and invert the MRI
   (`mri.max() - mri`). Does NMI change? Should it? Why does this rule out
   sum-of-squared-differences for cross-modality work?
2. **Prove the stride is safe.** Run one probe slice with `coarse=4` and again
   with `coarse=1`. Do they agree? Try `coarse=8` and `coarse=16` — where does
   it break, and does the peak's width explain it?
3. **Earn a rejection.** Take a series that passes and shrink
   `REG_MAX_SPREAD_MM` until it fails. How close to the limit was it really?
4. **Make the frame lie.** Score a fixed shift, then re-score it with the
   window inset by the search range (the old bug from §6). Which is higher, and
   what does that tell you about metrics that skip out-of-frame pixels?
5. **Find the cost of precision.** How many evaluations would a 0.5 mm grid
   need at ±40 mm? Is the extra precision meaningful when the through-plane
   spacing is 5–7 mm?
