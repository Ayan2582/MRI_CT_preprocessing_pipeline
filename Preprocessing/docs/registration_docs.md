# 📖 Registration Explained: from first principles to `registration_demo_sweep_v3.py`

This document teaches the MRI→CT registration procedure used in this project. It assumes you
can read Python and have seen a medical image before, and **nothing else**. Every term is
defined the first time it appears, every design decision is justified, and every number quoted
comes from an actual run you can reproduce.

It is deliberately long. Registration is a subject where the code is short and the reasoning is
not, and almost every bug this project hit came from a plausible-looking line of code resting on
a wrong idea. So the ideas come first.

> **Every concept here is introduced with a worked example.** Look for the
> **"### Example —"** headings. Where an idea is abstract (entropy, mutual information,
> coordinate frames, off-by-one errors) the example uses deliberately tiny numbers you can check
> by hand — a 4-pixel image, a 4×4 joint histogram, a number line from 0 to 100. Where it is
> concrete (the scale gate, the noise floor, the failure classes) the example uses **real
> measured values from this dataset**, so you can find the same rows in `sweep_v3_summary.csv`.
> If a section ever feels like pure assertion, scroll down — the example is directly beneath it.

**Companion documents**
- [`dicom_geometry_docs.md`](dicom_geometry_docs.md) — **read this first if §2 below moves too
  fast.** It expands index space vs world space into a full treatment, and explains what a series
  (`SE0`) and a DICOM image (`IM0`) actually are, using real headers from this dataset.
- `registration_recommendations.md` — the terse, actionable version. Read that when you know this.
- `image_processing_docs.md`, `normalization_docs.md` — line-by-line docs for the helper modules.

---

## Table of contents

1. [What we are actually trying to do](#1-what-we-are-actually-trying-to-do)
2. [A medical image is not a picture](#2-a-medical-image-is-not-a-picture)
3. [Why CT and MRI are hard to align](#3-why-ct-and-mri-are-hard-to-align)
4. [The measuring stick: mutual information, and why we normalise it](#4-the-measuring-stick-mutual-information-and-why-we-normalise-it)
5. [The transform zoo: what each model is allowed to do](#5-the-transform-zoo-what-each-model-is-allowed-to-do)
6. [How the optimiser actually works](#6-how-the-optimiser-actually-works)
7. [The pipeline, step by step](#7-the-pipeline-step-by-step)
8. [`v3` in detail: cropping as a fallback](#8-v3-in-detail-cropping-as-a-fallback)
9. [Case study: the coordinate-frame bug](#9-case-study-the-coordinate-frame-bug)
10. [Reading the outputs](#10-reading-the-outputs)
11. [How to be appropriately sceptical](#11-how-to-be-appropriately-sceptical)
12. [Glossary](#12-glossary)
13. [Open questions](#13-open-questions)

---

## 1. What we are actually trying to do

The end goal of this repository is to train a model that converts MRI images into CT-like images.
To train such a model you need **pairs**: for one physical location in one patient's body, an MRI
picture *and* a CT picture of that same location, so the model can learn "this MRI texture
corresponds to that CT texture."

The problem is that the CT and the MRI were taken on different machines, on different days, with
the patient lying differently, with different fields of view and different pixel sizes. If you
naively stack slice 5 of the CT on top of slice 5 of the MRI, you are not looking at the same
anatomy. You might be 87mm off. Training on those pairs teaches the model nonsense — worse than
nonsense, because it teaches it confidently.

**Registration** is the process of finding the geometric transformation that puts the two images
into correspondence, so that a pixel at position `p` in the CT and a pixel at position `p` in the
transformed MRI describe the same piece of the patient.

> **The single most important sentence in this document:** registration produces a *number* that
> says how well things line up, and that number is not the same thing as the images actually
> lining up. Most of the engineering below exists to stop us from believing a good number.

---

## 2. A medical image is not a picture

This is the foundation. Skipping it is why the worst bug in this project happened.

> 📚 This section is a summary. For the full treatment — what a series and a DICOM image are,
> which tags define each, and the index→world formula worked out against real headers from this
> dataset — see [`dicom_geometry_docs.md`](dicom_geometry_docs.md).

### 2.1 Index space vs world space

A PNG is a grid of pixels and nothing more. Pixel `(10, 20)` means "column 10, row 20" and that
is the whole story.

A medical image carries, in addition to the pixel grid, a description of **where in the room
that grid sits**. SimpleITK stores four things:

| Property | Meaning | Example |
|---|---|---|
| **Size** | how many voxels along each axis | `(512, 512, 18)` |
| **Spacing** | physical distance between neighbouring voxels, in mm | `(0.48, 0.48, 5.0)` |
| **Origin** | world coordinates of voxel `(0,0,0)`, in mm | `(53.3, -112.7, 40.1)` |
| **Direction** | a 3×3 rotation matrix: which way the voxel axes point in the world | `(1,0,0, 0,1,0, 0,0,1)` |

Together these define a mapping from **index space** (voxel counts, `i, j, k`) to **world space**
(millimetres in the scanner's coordinate system, `x, y, z`):

```
world_point = Origin + Direction · (Spacing ⊙ index)
```

### Example — one axis, small numbers

Forget matrices for a second. Take a 1D image, 5 voxels long, `Spacing = 2 mm`,
`Origin = 10 mm`:

```
index:    0     1     2     3     4
          |     |     |     |     |
world:   10    12    14    16    18      (mm)
```

`world = 10 + 2 × index`. So voxel **3** sits at **16 mm**. That is the whole idea. The 3D
formula is the same thing three times, with `Direction` added to handle the case where the image
axes are tilted relative to the room.

**Now the point of it.** Suppose a second image of the same patient has `Spacing = 5 mm`,
`Origin = 4 mm`:

```
image A   index:   0     1     2     3     4
          world:  10    12    14    16    18

image B   index:   0     1     2     3     4
          world:   4     9    14    19    24
```

Index 2 in A and index 2 in B **both land at 14 mm** — the same physical place — purely by
coincidence. Index 3 in A is at 16 mm, but index 3 in B is at 19 mm: 3 mm apart, different
anatomy. **Comparing images by index number is meaningless; comparing them by world position is
the only thing that works.**

SimpleITK gives you this directly:

```python
p = image.TransformContinuousIndexToPhysicalPoint((10, 20, 3))   # index -> mm
i = image.TransformPhysicalPointToContinuousIndex((53.3, -112.7, 40.1))  # mm -> index
```

The world coordinate system is anchored to the scanner, and conventionally to the patient
(roughly: `x` = left-right, `y` = front-back, `z` = head-toe). **Two images of the same patient
taken on two different machines share this world space.** That shared world space is the only
reason alignment is possible at all — it is the common language.

### 2.2 Why this matters immediately

Because two images share world space, you can ask geometric questions before doing any
registration:

- *Where does this MRI actually sit relative to this CT?* → compare their origins and extents.
- *Do they even cover the same anatomy?* → intersect their bounding boxes in world space.
- *How far apart are their centres?* → subtract the centre points.

All three questions are answered by arithmetic on the metadata, with no image processing at all,
in microseconds. This project answers them in `registration_demo_fov.py` and in v3's
`overlap_fracs`.

### 2.3 The bounding box trick

To compare two volumes' extents you cannot just compare origins, because the `Direction` matrix
may rotate one of them. The robust move is:

1. Take all **8 corners** of the voxel grid in index space.
2. Push each through `TransformContinuousIndexToPhysicalPoint`.
3. Take the min and max over those 8 world points, per axis.

That gives an **axis-aligned bounding box (AABB)** in world space, which is orientation-agnostic
and therefore comparable between the two images. This is exactly `corners_world` in v3:

```python
def corners_world(img):
    size = img.GetSize()
    idxs = [(i, j, k) for i in (0, size[0]-1) for j in (0, size[1]-1) for k in (0, size[2]-1)]
    return np.array([img.TransformContinuousIndexToPhysicalPoint(c) for c in idxs])
```

The rest of this section answers the two questions that trick always raises: **why 8, and not
fewer, and not more?**

#### Why the naïve shortcut is wrong

A volume is a box in *index* space — `[0..W-1] × [0..H-1] × [0..D-1]` — which is trivially easy
to describe. But nobody cares about index space. What matters is where that box sits in
millimetres, and the index→world map is affine:

```
world = Origin + Direction · (Spacing ⊙ index)
```

`Direction` is the DICOM direction cosine matrix. For an oblique acquisition it is a rotation, so
the box, once in world space, is a **tilted parallelepiped**. It has no simple min/max formula.

The tempting shortcut is: *"a box has a min corner and a max corner, so transform 2 points."*
That is wrong the moment there is any rotation, because **rotation shuffles which corner is
extreme along which axis**.

### Example — the 2-corner shortcut, on an oblique slab

Take an MRI slab, 100×100×20 voxels, 1 mm spacing, `Origin = (0,0,0)`, acquired at **30° oblique**
about `z` — the same situation as the spine case in this dataset. Ignore `z`; it just comes along
for the ride. With `cos 30° = 0.866`, `sin 30° = 0.5`:

```
x_world = 0.866·i − 0.5·j
y_world = 0.5·i  + 0.866·j
```

Transform the two "obvious" corners:

| index corner | world (mm)      |
|--------------|-----------------|
| (0, 0)       | (0.0, 0.0)      |
| (99, 99)     | (36.2, 135.2)   |

→ shortcut's conclusion: `x ∈ [0, 36.2]`.

Now transform the other two, the ones the shortcut skipped:

| index corner | world (mm)        |
|--------------|-------------------|
| (99, 0)      | (**85.7**, 49.5)  |
| (0, 99)      | (**−49.5**, 85.7) |

→ truth: `x ∈ [−49.5, 85.7]`.

The shortcut says the volume is **36 mm** wide in `x`. It is actually **135 mm** wide, and it
starts 49.5 mm to the *left* of where the shortcut claims it begins. The shortcut captured 27% of
the real extent — and the wrong 27%.

Look at *which* corner won: index-corner `(99, 0)` is not a "max" corner in index space at all,
yet it owns max-`x` in world space. With no rotation and positive spacing, `(0,0,0)` and
`(W-1,H-1,D-1)` do happen to be the extremes — but the moment the direction matrix is oblique, or
has a flip (negative spacing, LPS handedness), that stops being true. And you cannot tell which
subset is safe without inspecting the matrix. **8 corners is the cheap answer that never needs the
check.**

This is not a hypothetical failure. It is the shape of the bug in
[§9](#9-case-study-the-coordinate-frame-bug): an ROI computed from the wrong geometry kept 4 of
18 slices.

**Why exactly 8:** a 3D box is the product of three intervals, each with two endpoints — `2³ = 8`.
In 2D it would be 4, in 4D 16.

#### Why more than 8 buys you nothing

Not "8 is enough in practice" — 8 is enough *provably*. Every voxel in the box is a **convex
combination** of the 8 corners:

```
i = Σ λ_c · c        with λ_c ≥ 0,  Σ λ_c = 1
```

An affine map distributes over that sum:

```
T(i) = Σ λ_c · T(c)
```

So each world coordinate of *any* interior point is a weighted average of the 8 corners'
coordinates — and a weighted average can never exceed the max or fall below the min of the things
it averages. Concretely, the centre of the slab above, at index `(49.5, 49.5)`, lands at world
`(18.1, 67.6)`: comfortably inside `x ∈ [−49.5, 85.7]`.

Sampling edge midpoints, face centres, or a 10×10×10 lattice adds 1000 transforms and **cannot
move the bounding box by one micron.**

#### Where 8 stops being enough

Two caveats that matter for this pipeline specifically:

> ⚠️ **The argument requires an affine transform.** `corners_in_ct_frame` pushes corners through
> `transform.GetInverse()`, which is sound for `Euler3DTransform` and `AffineTransform` because
> both preserve convex hulls. If a BSpline or otherwise deformable stage is ever added, the
> 8-corner hull is **no longer a bound** — a bulge in the middle of the displacement field can
> push content outside the box, and the ROI will silently clip it.

> ⚠️ **The AABB over-estimates for oblique volumes.** The 30° slab above genuinely occupies a
> tilted 100×100 mm square, but its axis-aligned box is 135×135 mm — roughly 1.8× the area, all of
> it empty. That is the safe direction for a crop (you never lose content), but it means
> `overlap_fracs` is *optimistic* for obliquely-related volumes: a reported 90% axis overlap can
> be considerably less real tissue overlap. See [§7.0](#70-qc-0--field-of-view-overlap).

One minor detail: the code uses indices `0` and `size-1`, which are voxel **centres**, so the
extent is short by half a voxel at each face. For an ROI that gets `floor`/`ceil`'d anyway this
washes out — but the true physical footprint runs from continuous index `-0.5` to `size-0.5`.

---

## 3. Why CT and MRI are hard to align

Four separate difficulties, which is why this is not a two-line problem.

### 3.1 The intensities are unrelated

CT measures X-ray attenuation, in **Hounsfield Units (HU)** — a real, calibrated, absolute
physical scale. Air is −1000, water is 0, bone is +1000 and above. A given tissue has the same HU
in every CT scanner in the world.

MRI measures the radio-frequency response of hydrogen nuclei. Its numbers are **arbitrary**.
There is no unit. The same tissue in the same patient can be 300 in one sequence and 1800 in
another. Two MRIs from the same machine an hour apart can differ in overall brightness.

And critically, the *ordering* is different.

### Example — the same four tissues, in both modalities

| Tissue | CT (HU) | Looks like | MRI (T1, arbitrary units) | Looks like |
|---|---|---|---|---|
| Air | −1000 | black | ~0 | black |
| Fat | −100 | dark grey | **~1200** | **bright** |
| Muscle | +50 | mid grey | ~600 | mid grey |
| Cortical bone | **+1000** | **white** | **~50** | **near-black** |

Look at bone and fat. In CT, bone (1000) is far **brighter** than fat (−100). In MRI, bone (50)
is far **darker** than fat (1200). The ordering is inverted.

So if you tried to align these by minimising the difference between pixel values, the optimiser's
best move would be to slide bone **off** bone — because matching bright-to-bright would put CT
bone on top of MRI fat. Subtracting one image from the other is not merely inaccurate here; it is
actively pointing the wrong way.

What *is* stable is the **relationship**: everywhere CT says 1000, MRI says about 50. That
consistency is what mutual information detects, and it does not care that the numbers are
inverted, on different scales, or in different units.

**Consequence:** you cannot align these images by asking "do the pixel values match?" Subtracting
one from the other is meaningless. You need a similarity measure that asks a subtler question —
"is there a *consistent relationship* between the values?" — which is what mutual information
does (§4).

### 3.2 The patient moved

Between the CT and the MRI the patient got up, walked to another room and lay down again,
possibly on a differently shaped table, possibly days later. So the anatomy is in a different
place and at a different angle. That is the thing registration must recover.

### 3.3 The fields of view differ

The CT might cover the whole knee; the MRI might cover a tighter box around the joint. They
overlap, but neither contains the other. Any region of the output where only one modality has
data is worse than useless — it looks like anatomy but it is fill value.

### 3.4 The sampling is wildly anisotropic

Look at a typical series in this dataset:

```
CT  spacing (0.48, 0.48, 5.0) mm  size (512, 512, 18)
```

In-plane, voxels are half a millimetre. Through the slice direction, they are **five
millimetres** — ten times coarser — and there are only 18 of them.

This asymmetry drives one of the biggest design decisions in the pipeline. In-plane you have a
rich, finely sampled signal that an optimiser can work with. Through-plane you have 18 samples at
5mm; you can barely tell anything. Trying to fit a full 3D rotation to that is fitting noise.

> **Design consequence:** we correct translation in 3D (which is closed-form, no optimisation) and
> handle rotation only **in 2D per slice**, where it is well constrained by the fine in-plane
> sampling. See §7.4.

---

## 4. The measuring stick: mutual information, and why we normalise it

To register images you need a **similarity metric**: a function that takes two overlapping
images and returns a number saying how well they correspond. The optimiser then hunts for the
transform that maximises it.

### 4.1 Entropy, with numbers

Entropy measures uncertainty. Take an image, build a histogram of its intensities, turn the
counts into probabilities `p₁…pₙ`, then

```
H = −Σ pᵢ · log(pᵢ)
```

### Example — three 4-pixel images

| Image | Pixels | Probabilities | H | Meaning |
|---|---|---|---|---|
| A | `10, 10, 10, 10` | `{1.0}` | **0.0000** | every pixel identical — zero uncertainty |
| B | `10, 10, 20, 20` | `{0.5, 0.5}` | **0.6931** | a coin flip = `ln 2` |
| C | `10, 20, 30, 40` | `{0.25 ×4}` | **1.3863** | 4 equal options = `ln 4` |

Work image B by hand to see there is no magic in it:

```
H = −(0.5 × ln 0.5)  −  (0.5 × ln 0.5)
  = −(0.5 × −0.6931) − (0.5 × −0.6931)
  = 0.3466 + 0.3466
  = 0.6931          ✓  = ln 2
```

**The reading:** entropy is "how surprised am I by the next pixel." Image A never surprises you
(H = 0). Image C surprises you as much as possible for 4 values (H = ln 4). More spread across
more bins → higher entropy.

> Entropy here is in **nats**, not bits, because we use natural log. Bits would use log₂. It
> makes no difference to anything — it is a constant factor that cancels in the NMI ratio.

### 4.2 Joint entropy and the key idea

Now take **two** registered images and build a **2D histogram**: bin `(a, b)` counts the pixels
where image 1 has intensity `a` *and* image 2 has intensity `b`. That is the **joint histogram**,
and it is the heart of everything.

Here is the intuition that makes mutual information click:

> When two images are **correctly aligned**, the joint histogram is *tight* — each tissue in the
> CT maps to a small, predictable cluster of MRI values, because it is the same tissue. When
> they are **misaligned**, bone in the CT lands on random MRI tissue, and the joint histogram
> smears into a diffuse cloud.

### Example — a 4×4 toy pair, aligned then shifted

Four tissues, laid out in quadrants. CT labels them `0,1,2,3`; the MRI gives the same four
tissues completely different values (`90, 50, 70, 10`) — note the ordering is scrambled, exactly
like real CT vs MRI:

```
   CT                     MRI
   0 0 1 1                90 90 50 50
   0 0 1 1                90 90 50 50
   2 2 3 3                70 70 10 10
   2 2 3 3                70 70 10 10
```

**Aligned.** Build the joint histogram — count how often each (CT value, MRI value) pair occurs:

```
              MRI:   10    50    70    90
        CT 0:         0     0     0     4
        CT 1:         0     4     0     0
        CT 2:         0     0     4     0
        CT 3:         4     0     0     0
```

**Exactly one occupied cell per row.** 4 of 16 cells used. Knowing the CT value tells you the MRI
value with certainty.

```
MI = 1.3863        NMI = 2.0000   ← the maximum
```

**Now shift the MRI one column to the right** — same images, misaligned by one pixel:

```
              MRI:   10    50    70    90
        CT 0:         0     2     0     2
        CT 1:         0     2     0     2
        CT 2:         2     0     2     0
        CT 3:         2     0     2     0
```

**Eight occupied cells, two per row.** The mapping has become ambiguous: CT value 0 now
corresponds to MRI 50 *or* 90, half the time each.

```
MI = 0.6931        NMI = 1.3333
```

**That is the entire mechanism.** Alignment concentrates the joint histogram; misalignment
smears it. Nothing here needed the CT and MRI values to match — only to correspond *consistently*.

A tight joint histogram has *low* joint entropy. So we want to minimise `H(A, B)`. Mutual
information packages that as a quantity to maximise:

```
MI(A, B) = H(A) + H(B) − H(A, B)
```

Read it as: "how much less uncertain am I about B once I know A." Zero when the images are
statistically independent; large when knowing one tells you a lot about the other.

Check it against the aligned example above. Each image has 4 equally common values, so
`H(A) = H(B) = ln 4 = 1.3863`. The joint histogram also has 4 equally occupied cells, so
`H(A,B) = ln 4 = 1.3863` too. Therefore:

```
MI  = 1.3863 + 1.3863 − 1.3863 = 1.3863
NMI = (1.3863 + 1.3863) / 1.3863 = 2.0000
```

Crucially, MI never needs the intensities to *match*. It only needs the relationship to be
consistent. Bone-bright-in-CT can map to bone-dark-in-MRI perfectly happily. **That is why MI is
the standard metric for multi-modality registration.**

### 4.3 Why raw MI was not good enough here

Look again at the formula: `MI = H(A) + H(B) − H(A,B)`. The marginal entropies `H(A)` and `H(B)`
are in there positively. Those depend on **how much image you are looking at**, not on how well
it is aligned. Put more anatomy in the frame and MI rises even if nothing moved.

For most of registration that does not matter — you compare alignments of the same fixed frame,
so the frame contribution is constant. But this project's central question is:

> *"Is the result better on the full CT canvas, or on a smaller cropped canvas?"*

Two different frames. Raw MI cannot answer it. It systematically favours the bigger frame.

### Example — same anatomy, shrinking frame (real data, shoulder/axial z=9)

Nothing is moved here. The alignment is **identical** in every row; the only change is how much
border is included in the frame:

| Canvas | Size | raw MI | NMI |
|---|---|---|---|
| full canvas | 311×311 | **0.3048** | 1.0848 |
| trim 40 px border | 231×231 | 0.2855 | 1.0665 |
| trim 80 px border | 151×151 | 0.2045 | 1.0460 |
| trim 110 px border | 91×91 | **0.1022** | 1.0348 |

**Raw MI falls to 34% of its value.** The alignment never changed — only the frame did. If you
compared a full-canvas MI of 0.30 against a cropped-canvas MI of 0.10 you would conclude the crop
was a disaster, when in fact it is the same registration.

**NMI falls to 95% of its value** over the same range. Not perfectly invariant, but stable enough
that a genuine improvement is visible above the frame effect instead of drowned by it.

That table is the entire reason this project switched metrics.

### 4.4 Normalized mutual information (NMI)

The fix is Studholme's normalized mutual information: divide out the frame dependence instead of
subtracting it.

```
NMI(A, B) = ( H(A) + H(B) ) / H(A, B)
```

Properties that make it the right choice here:

| | |
|---|---|
| **Bounded** | `NMI ∈ [1, 2]` |
| `NMI = 1` | the images are statistically independent — knowing one tells you nothing |
| `NMI = 2` | the images induce identical partitions — perfect correspondence |
| **Comparable across frame sizes** | the first-order dependence on how much image is present divides out |

This is implemented as `nmi_score` in `registration_demo.py`:

```python
def nmi_score(fixed_slice, moving_slice, bins=NMI_BINS):
    fixed  = sitk.GetArrayFromImage(fixed_slice ).astype(np.float64).ravel()
    moving = sitk.GetArrayFromImage(moving_slice).astype(np.float64).ravel()

    # Clip (do not discard) at the 0.5/99.5 percentiles, so a handful of extreme
    # voxels - CT metal, MRI spikes - cannot collapse every real tissue value
    # into one or two bins. Every pixel still contributes; it just saturates.
    f_lo, f_hi = np.percentile(fixed,  NMI_CLIP_PERCENTILES)
    m_lo, m_hi = np.percentile(moving, NMI_CLIP_PERCENTILES)
    if f_hi <= f_lo or m_hi <= m_lo:
        return float("nan")          # a constant slice has no entropy to work with

    joint, _, _ = np.histogram2d(np.clip(fixed,  f_lo, f_hi),
                                 np.clip(moving, m_lo, m_hi),
                                 bins=bins, range=[[f_lo, f_hi], [m_lo, m_hi]])
    p = joint / joint.sum()

    def entropy(probs):
        nz = probs[probs > 0]        # 0·log0 = 0, but log(0) is -inf: filter first
        return float(-np.sum(nz * np.log(nz)))

    h_joint = entropy(p)
    if h_joint <= 0:
        return float("nan")
    return (entropy(p.sum(axis=1)) + entropy(p.sum(axis=0))) / h_joint
```

Three details worth understanding rather than skimming:

- **`p.sum(axis=1)` and `p.sum(axis=0)` are the marginals.** Summing the joint histogram along
  one axis collapses it back to the single-image histogram. That is why we only build one 2D
  histogram and get all three entropies from it.
- **`nz = probs[probs > 0]`** — `0·log(0)` is defined as 0 in information theory, but numerically
  `log(0) = −inf` and `0 · −inf = nan`. Filtering empty bins is not optional.
- **NaN, not 0, for a constant slice.** If a slice is entirely fill value, there is no answer,
  and returning `0.0` would be a lie that reads as "measured, and bad." NaN reads as "not
  measurable," which is the truth. All comparison helpers are written NaN-safe so a NaN can never
  win a comparison.

**Sanity check** (run it yourself, it takes a second):

```
identical images     → 2.000
independent noise    → 1.059     (~1.0 plus finite-sample bias)
noisy copy           → 1.153
constant moving image→ nan
```

### 4.5 The awkward compromise you need to know about

SimpleITK's `ImageRegistrationMethod` offers exactly six metrics, and **NMI is not one of them**:

```
SetMetricAsANTSNeighborhoodCorrelation, SetMetricAsCorrelation, SetMetricAsDemons,
SetMetricAsJointHistogramMutualInformation, SetMetricAsMattesMutualInformation,
SetMetricAsMeanSquares
```

So the gradient-descent optimiser inside registration still minimises **Mattes MI**. We cannot
change that without writing our own optimiser.

What we *can* control is which candidate we keep. So the arrangement is:

> **Mattes MI proposes candidate alignments. NMI decides which one is kept.**

Because we run several independent attempts anyway (§6.3), this works well: the optimiser
generates a handful of plausible alignments using whatever metric it has, and the selection —
the decision that actually determines the output — is made on the metric we trust.

---

## 5. The transform zoo: what each model is allowed to do

A transform is a rule for moving points. The more freedom you give it, the more it can correct —
and the more ways it can cheat.

| Transform | Degrees of freedom (2D) | Can do | Cannot do |
|---|---|---|---|
| **Translation** | 2 | slide | rotate, resize |
| **Euler2D / rigid** | 3 | slide + rotate | resize, shear |
| **Similarity2D** | 4 | slide + rotate + *uniform* resize | shear, non-uniform scale |
| **Affine** | 6 | slide, rotate, scale each axis, shear | bend |
| **BSpline / deformable** | hundreds | bend locally | — |

### Example — every transform applied to the same point

Take one point, `p = (100, 50)`. Each transform gets the same translation `(+10, −5)` and, where
it can, a 10° rotation:

| Transform | Settings | `p` maps to |
|---|---|---|
| Translation | shift `(+10, −5)` | `(110.00, 45.00)` |
| Rigid | rot 10°, shift `(+10, −5)` | `(99.80, 61.61)` |
| Similarity | rot 10°, **scale 1.2**, shift | `(117.76, 74.93)` |
| Affine | matrix `[[1.2, 0.3], [0.0, 0.9]]`, shift | `(145.00, 40.00)` |

Notice how the point travels further as you add freedom. Translation moves it 11 units. Affine
moves it 46 — and that extra reach is exactly what makes affine both more capable and more
dangerous. Every extra degree of freedom is another way to raise the score without improving the
anatomy.

**Why the rigid result has a *smaller* x than translation:** rotating `(100, 50)` by 10° about
the origin swings it upward and slightly left, to `(89.80, 66.61)`, and only then does the shift
`(+10, −5)` apply. Rotation acts on the point's distance from the rotation centre, so the further
a pixel is from that centre, the more a small angle moves it.

### 5.1 The rule of thumb

**Use the least powerful transform that can express the difference you actually expect.**

Between a CT and an MRI of the same patient, the real difference is: the patient lay down
differently. That is a **rigid** difference — rotation and translation. Bones do not change size
between Tuesday and Thursday.

So why offer affine at all? Because real acquisitions have small non-rigid discrepancies:
gradient non-linearity in the MRI, slightly different limb flexion, soft-tissue deformation
against a different table. A little scale and shear can absorb those.

### 5.2 How extra freedom becomes cheating

Here is the failure mode, and it is beautiful and horrible.

Give affine the freedom to resize, and it discovers that it can raise the similarity score not by
aligning anatomy, but by **stretching the moving image until its border matches the frame
border**. Frames are big high-contrast features. Matching them scores well. The anatomy inside
ends up wrong.

We caught this red-handed on the spine series. Unconstrained affine recovered a scale of
**0.825**, while the MRI/CT field-of-view ratio was **0.824** — agreement to 0.3%. And it did so
*reproducibly*: all five random seeds landed between 0.787 and 0.847.

That reproducibility is the trap. Every naive quality check says this is a great result: high
score, low variance across seeds, converged cleanly. It is completely wrong. A patient's spine
does not shrink 17% between two scans.

### 5.3 The scale gate

So we constrain affine explicitly. Extract the scale from the fitted matrix, and reject anything
that resized meaningfully:

```python
def affine_scale(transform):
    matrix = np.array(transform.GetMatrix()).reshape(2, 2)
    # Singular values are the scale factors along the transform's own principal
    # axes. Using them (rather than reading matrix entries) is robust to the fact
    # that rotation and shear are mixed into the same 2x2 block.
    return float(np.mean(np.linalg.svd(matrix, compute_uv=False)))

def scale_verdict(scale, canvas_ratio):
    if abs(scale - 1.0) > SCALE_TOL:              # TEST 1: outer sanity bound (0.60)
        return False, "outside_sanity_bound"
    if veto_applies(canvas_ratio):                # TEST 2: the canvas-fit veto
        for suspect in (canvas_ratio, 1.0 / canvas_ratio):
            if abs(scale - suspect) <= CANVAS_FIT_TOL:      # 0.04
                return False, "canvas_fit"
    return True, ""
```

**Why two tests instead of one band.** The gate was originally a single tight band,
`|scale − 1| ≤ 0.05`, and that turned out to be wrong in both directions. It rejected any real
scale difference — and it would have waved through a canvas-fit that happened to land near 1.0.

The key realisation is that **there is no legitimate zoom for affine to correct in this
pipeline.** By the time 2D registration runs, `resample_mri_to_ct_grid_v2` has already put the
MRI on the CT's grid in world millimetres: both slices are 1 mm/pixel, so a 50 mm structure spans
50 pixels in each. A field-of-view difference does not survive as magnification — it survives as
*the MRI filling less of the canvas*. So a large recovered scale is not a correction, and it is
identifiable because it lands on the **canvas ratio**: the fraction of the frame the MRI's
bounding box spans.

### Example — the veto, on two real slices

| Slice | MRI box ÷ canvas | 1 ÷ ratio | Recovered scales | Verdict |
|---|---|---|---|---|
| knee/sagittal z=17 | **0.779** | 1.283 | 0.810, 0.827, 0.840 | ❌ all sit on the ratio |
| spine/coronal z=5 | **0.827** | **1.209** | 1.177, 1.191, 1.206 | ❌ all sit on 1/ratio |
| shoulder/coronal mid | 0.886 | 1.129 | **1.009**, **1.009**, 1.102 | ✅ two near unity, one vetoed |

The first two rows are the optimiser stretching the MRI until its border meets the frame border —
and note that **NMI rises when it does this** (1.131 → 1.182 on the knee slice), which is exactly
why the score cannot be the arbiter. The third row is what a healthy affine looks like: a scale
near 1.0, nowhere near the canvas ratio.

> ⚠️ **The veto must switch itself off when the MRI fills the frame.** If the MRI spans the whole
> canvas the ratio is ~1.0, there is no border gap to close, and a scale near 1.0 is the *correct*
> answer rather than a suspicious one. `veto_applies()` requires `|ratio − 1| > 2 × CANVAS_FIT_TOL`
> so the vetoed interval can never touch unity. This is not hypothetical — knee/axial has 100%
> MRI coverage, and an unguarded veto would have rejected every good answer on it.

**Why singular values?** A 2×2 affine block mixes rotation, scale and shear together; you cannot
read the scale off the diagonal. The **singular value decomposition** factors any matrix into
*rotate → scale along axes → rotate*, and the singular values are exactly those axis scales.
Their mean is a clean, rotation-invariant answer to "how much did this transform resize things?"

### Example — why reading the diagonal fails

| Matrix | Diagonal | Singular values | True scale |
|---|---|---|---|
| pure rotation 30° | `[0.866, 0.866]` | `[1.000, 1.000]` | **1.000** |
| uniform scale 1.2 | `[1.200, 1.200]` | `[1.200, 1.200]` | **1.200** |
| rotation 30° + scale 1.2 | `[1.039, 1.039]` | `[1.200, 1.200]` | **1.200** |
| pure shear | `[1.000, 1.000]` | `[1.161, 0.861]` | 1.011 |
| a real rejected spine seed | `[1.202, 1.199]` | `[1.203, 1.199]` | **1.201** |

**Look at row 1.** A pure 30° rotation — no resizing whatsoever — has a diagonal of `0.866`. Read
the diagonal and you would report a 13% shrink and reject a perfectly good rigid rotation. The
singular values correctly say `1.000`.

**Now row 3.** Rotation *and* a genuine 1.2× scale gives a diagonal of `1.039`. Read the diagonal
and you would report a 4% scale — inside a 5% tolerance — and wave through a 20% resize.

So the diagonal is wrong in both directions: it invents scale that is not there and hides scale
that is. Only the SVD separates rotation from scale.

### 5.4 Where the gate must live — a real bug worth internalising

This is subtle and we got it wrong for a full round of results.

We run several attempts with different random seeds and keep the best (§6.3). The question is
**when** the gate is applied.

```python
# ✗ WRONG - what the code did originally
best = max(all_seeds, key=score)      # pick the highest scorer, unconstrained
if not scale_gate_ok(best):
    reject_the_entire_affine()        # throw away ALL of it

# ✓ RIGHT - gate as an admissibility predicate during selection
best = max((s for s in all_seeds if scale_gate_ok(s)), key=score)
```

The wrong version throws away a perfectly good in-tolerance alignment whenever an
out-of-tolerance one happens to outscore it — and it discards the whole transform, not just that
seed. The right version returns *the best admissible candidate*.

In SimpleITK terms this is the `accept_fn` parameter of
`run_2d_registration_multistart_detailed`:

```python
affine = demo.run_2d_registration_multistart_detailed(
    ct_slice, mri_slice, "affine", n_starts=N_STARTS, accept_fn=scale_gate_ok)
```

**How much did it matter?** On 33 test slices, gating during selection recovered a usable affine
on **10 slices** that post-hoc gating would have discarded outright. But — and this is the honest
part — under the old ±0.05 band **12 slices still lost every affine seed**, and 52 of 99 seeds
were out of tolerance. So moving the gate was necessary but not sufficient; the band itself was
the wrong test. Replacing it with the canvas-fit veto (§5.3) brought rejections down to
**18 of 99**, with only 3 slices losing every seed, and tripled how often affine ships (4 → 12
slices).

### 5.5 The fallback ladder

Given all that, a canvas produces its result by walking down a ladder:

```
best admissible affine   →   best rigid   →   unregistered baseline
```

**The last rung is not optional.** Without it, a slice can ship a result measurably worse than
doing nothing at all. That is not hypothetical: on knee/sagittal `last`, registration produced
1.081 while simply leaving the MRI where the volume alignment put it scored **1.127**. The code
shipped the worse one, and then used it as the comparison point for the next decision, which
corrupted that too.

```python
# Never ship something worse than doing nothing.
if best_kind != "unregistered" and not is_better(best_nmi, nmi_baseline):
    shipped_kind, shipped_img, shipped_nmi = "unregistered", mri_slice, nmi_baseline
```

On the 33-slice test set this rung fires on **3 slices**.

---

## 6. How the optimiser actually works

`sitk.ImageRegistrationMethod` has four moving parts. Understanding them explains every weird
behaviour you will see.

```python
registration_method = sitk.ImageRegistrationMethod()

# (1) METRIC - the score being optimised
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

# (2) SAMPLING - which pixels the metric looks at
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.2, seed=seed)

# (3) OPTIMISER - how the transform parameters are updated
registration_method.SetOptimizerAsGradientDescent(
    learningRate=0.1, numberOfIterations=num_iters,
    convergenceMinimumValue=1e-6, convergenceWindowSize=10)
registration_method.SetOptimizerScalesFromPhysicalShift()

# (4) MULTI-RESOLUTION PYRAMID - coarse-to-fine
registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
```

### 6.1 The pieces

**(1) Metric.** Mattes MI with 50 histogram bins. Note SimpleITK reports it as a **cost to
minimise** (more negative = better match), which is why `mattes_mi_score` negates it.

**(2) Sampling.** Evaluating the metric on every pixel every iteration is slow, so it samples 20%
at random. This is the source of all non-determinism — see §6.2.

**(3) Optimiser.** Plain gradient descent on the transform parameters.
`SetOptimizerScalesFromPhysicalShift` deserves a note: the parameters have wildly different
units — an angle in radians and a translation in millimetres are not comparable, and a step size
sensible for one is catastrophic for the other. This call automatically rescales the parameter
space so that a step means roughly the same physical displacement in every direction. Without it
gradient descent behaves erratically.

**(4) Pyramid.** The image is registered at ¼ resolution first (heavily smoothed), then ½, then
full. The coarse level has no fine detail to get trapped by, so it finds the broad alignment;
finer levels refine it. This is the standard defence against local optima, and it is why
registration can recover large displacements at all.

### Example — what `[4, 2, 1]` does to a real 311×311 slice

| Level | Shrink | Smoothing | Working size | What it can see |
|---|---|---|---|---|
| 1 | ÷4 | σ = 2 mm | **77 × 77** | blobs — "the shoulder is roughly *there*" |
| 2 | ÷2 | σ = 1 mm | **155 × 155** | major structures — bone outlines |
| 3 | ÷1 | σ = 0 | **311 × 311** | full detail — texture, edges |

**Why start blurry?** Imagine aligning two combs by sliding one over the other. At full detail
there is a "good" match every time a tooth lines up with a tooth — hundreds of local optima, and
gradient descent stops at whichever one it happens to be nearest. Blur the combs until the teeth
disappear and only one broad match remains: the correct one. Find that first, then sharpen and
refine.

This is also, incidentally, exactly why **spine fails** (§11): lumbar vertebrae are a comb whose
teeth are 30 mm apart, so even the blurred level has repeating matches.

### 6.2 Non-determinism, and why the seed is not enough

Here is a genuinely surprising empirical result from this project:

> Identical code, identical slice, identical `seed=42`, three separate process runs.
> MI came out **0.131**, **0.132**, and **0.289**.

The random *sampling* is seeded, but the metric is evaluated **multi-threaded**, and thread
scheduling changes the order in which partial sums are accumulated. Floating-point addition is
not associative, so the accumulated value differs in the last bits — and gradient descent
amplifies that into a different local optimum.

A partial fix is one line, at import time, before anything else:

```python
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
```

> ⚠️ **Correction — single-threading is not sufficient.** An earlier version of this document
> claimed the same seed then gives the same answer every time. Re-measured directly, it does not.
> With threads confirmed at 1 and `seed=0` held fixed, eight repeat calls on one slice gave:
>
> | Slice | NMI range at fixed seed | Recovered affine scale range |
> |---|---|---|
> | spine/coronal | 0.041 (rigid), 0.008 (affine) | **0.088** (1.124–1.212) |
> | knee/sagittal | 0.082 (rigid), 0.053 (affine) | **0.344** (0.894–1.238) |
>
> The residual randomness is in the **metric sampler**, not the thread count. Switching
> `SetMetricSamplingStrategy` to `NONE` (evaluate every pixel, no subsampling) makes it *exactly*
> reproducible — NMI range 0.00000, scale range 0.0000 over six runs — at roughly 1.5–2× the
> runtime (0.3–0.8 s per slice). `REGULAR` is **not** deterministic either (scale range 0.71).
>
> This matters far beyond bookkeeping: `MIN_GAIN` is 0.010 and the affine-vs-rigid margin is of
> the same order, so under `RANDOM` sampling those thresholds sit *below* the noise they are
> meant to discriminate against. And the recovered scale — the quantity the whole scale gate
> reasons about — wobbles by up to ±17% between identical runs.

> 🎓 **The lesson that generalises.** The reproducible answer (0.125) was *worse* than the lucky
> one (0.289). Determinism and quality are separate problems. Making a result reproducible does
> not make it good — it makes it *knowable*, which is the prerequisite for improving it.

### 6.3 Multi-start

One gradient-descent run lands in whatever local optimum its random pixel subset leads it to. So
we run several, deterministically, with different seeds, and keep the best:

```python
for seed in range(n_starts):
    resampled, transform = run_2d_registration(fixed, moving, transform_type, num_iters, seed)
    ...
```

Each attempt is individually reproducible (single-threaded, fixed seed); varying the seed is now
a *deliberate* exploration of the optimisation landscape rather than uncontrolled noise.

**The spread across seeds is as valuable as the best score.** It is a direct, free estimate of
how much the optimiser is guessing:

```python
def score_spread(scores):
    valid = [s for s in scores if s is not None and not np.isnan(s)]
    if len(valid) < 2:
        return None          # NOT 0.0 - see below
    return max(valid) - min(valid)
```

> ⚠️ **Returning `0.0` for a lone survivor would be actively misleading.** A range of zero reads
> as "every seed agreed" when it actually means "every seed but one crashed" — the opposite
> conclusion, and precisely the case you most want flagged. `None` is the honest answer. This was
> a real bug.

### 6.4 Using the spread as a noise floor

If rigid scores 1.102 and affine scores 1.108, is affine better? Only if 0.006 is larger than the
uncertainty in those numbers. And we *have* that uncertainty — it is the seed spread.

### Example — two slices, opposite verdicts

Both from the current run, both cases where affine outscored rigid. Only one of them counts.

**Slice A — brain/sagittal, first.** Affine is reproducible:

```
rigid  best 1.1388   spread 0.0045
affine best 1.1441   spread 0.0012      ← seeds agree to three decimals

gap         = 1.1441 − 1.1388 = 0.0053
noise floor = max(affine spread, MIN_TRANSFORM_MARGIN)
            = max(0.0012, 0.005)       = 0.0050

0.0053 > 0.0050  →  AFFINE WINS
```

Barely — and correctly. Affine's own seeds vary by only 0.0012, so a 0.0053 lead is four times
its wobble. The `MIN_TRANSFORM_MARGIN = 0.005` floor is what it actually had to clear here,
because the measured spread was smaller than the floor.

**Slice B — knee/coronal, middle.** Affine is not reproducible:

```
rigid  best 1.1026   spread 0.0597
affine best 1.1236   spread None       ← only one seed survived the gate

gap         = 1.1236 − 1.1026 = 0.0210
noise floor = spread over ALL affine seeds (admissible or not) = 0.0836

0.0210 < 0.0836  →  RIGID KEPT
```

Affine appears to win by 0.021, but its own seeds scatter across 0.084 — four times that lead.
Re-run it and rigid might well come out ahead. **That is not a result, it is a coin flip**, and
the margin exists to say so.

Slice B also shows the `spread = None` fallback working: fewer than two seeds survived the scale
gate, so there is no admissible-only spread, and the code falls back to the spread over *all*
affine seeds rather than to rigid's.

```python
# The affine's OWN reproducibility is the relevant uncertainty. Using
# max(rigid_spread, affine_spread) let an unstable rigid raise the bar for
# affine - the worse rigid behaved, the harder it was to replace.
affine_noise = affine["spread"]                       # admissible seeds
if affine_noise is None:
    affine_noise = demo.score_spread(affine["scores"])  # all seeds
if affine_noise is None:
    affine_noise = rigid["spread"]
noise_floor = max(affine_noise or 0.0, MIN_TRANSFORM_MARGIN)

if is_better(affine["best_score"], rigid["best_score"], noise_floor):
    ... choose affine
```

On the 33-slice test set, **18 of 33 slices had an affine-vs-rigid gap smaller than the seed
spread on that same slice.** Without this margin, more than half of the transform choices were
being made by coin flip.

With the margin computed from the affine's own spread (and the scale gate redesigned, §5.3), the
current run ships **rigid on 20 slices, affine on 12, unregistered on 1**. Under the earlier
`max()` margin and the tight ±0.05 gate it was 26 / 4 / 3 — the change roughly tripled how often
affine is used, without loosening what counts as evidence.

---

## 7. The pipeline, step by step

```
 load DICOM series
        │
        ├─► [QC 0]  field-of-view overlap report        (metadata arithmetic only)
        │
        ├─► [1] N4 bias correction              (MRI only)
        ├─► [2] in-plane resample to 1mm        (CT)
        │
        ├─► [3] volume alignment: 3D GEOMETRY translation
        │
        ├─► [4] per-slice 2D registration on the FULL canvas
        │        multi-start → gate during selection → affine → rigid → unregistered
        │
        ├─► [5] IF that slice REGRESSED: retry on the FOV-intersection crop,
        │        choose between crop / full / unregistered on the common region
        │
        └─► [6] record QC columns → normalise → export
```

### 7.0 QC 0 — Field-of-view overlap

Before touching a pixel, compare the two volumes' world-space bounding boxes and report the
per-axis overlap as a fraction of the smaller field of view.

```python
def overlap_fracs(ct_image, mri_corners):
    ct_c = corners_world(ct_image)
    ct_lo,  ct_hi  = ct_c.min(0), ct_c.max(0)
    mri_lo, mri_hi = mri_corners.min(0), mri_corners.max(0)
    fracs = []
    for i in range(3):
        overlap = max(0.0, min(ct_hi[i], mri_hi[i]) - max(ct_lo[i], mri_lo[i]))
        smaller = min(ct_hi[i] - ct_lo[i], mri_hi[i] - mri_lo[i])
        fracs.append(overlap / smaller if smaller > 0 else 0.0)
    return fracs
```

**Compute it twice — before and after alignment.** These answer different questions and
conflating them caused the worst bug in this project (§9):

| Number | Frame | Question it answers |
|---|---|---|
| `fov_overlap_raw` | scanner coordinates, as acquired | How differently were the two scans positioned? |
| `fov_overlap_aligned` | after step 3's translation | How much shared canvas does registration actually have? |

A low **raw** overlap means *look at this pair*, not *discard most of it* — step 3 may close the
gap completely. On knee/sagittal, raw overlap is **12.5%** and aligned overlap is **100%**.

### 7.1 N4 bias field correction (MRI only)

MRI images have a smooth, low-frequency brightness gradient across them — one corner brighter
than another — caused by receiver-coil sensitivity, not anatomy. It is called the **bias field**.
It is poison for intensity-based registration, because it makes the same tissue take different
values in different parts of the image, smearing the joint histogram.

`apply_n4_bias_correction` (in `image_processing.py`) fits a smooth multiplicative field and
divides it out. Implementation notes worth knowing:

- It runs **slice by slice in 2D**, not on the 3D volume, which suits this data's thick,
  non-uniformly spaced slices.
- It uses **Otsu thresholding** to build a tissue mask, so the fit is not dragged around by air.
- It **shrinks by 4×** before fitting. The bias field is smooth by definition, so it is
  perfectly well estimated at low resolution — and the fit is ~16× faster. The estimated field is
  then expanded back to full size and applied to the full-resolution slice.
- It restores `Origin`/`Direction` explicitly before `JoinSeries`, because SimpleITK refuses to
  stack slices whose physical metadata disagrees in the last decimal place.

### Example — what a bias field does to the joint histogram

Suppose muscle should read 600 everywhere. The coil makes the left of the image 30% brighter and
the right 20% darker:

```
                    left edge    centre    right edge
  true value           600         600         600
  bias multiplier      1.30        1.00        0.80
  what you measure     780         600         480
```

Now recall §4.2: mutual information works by mapping **one CT value to one MRI value**. But this
MRI reports muscle as 780, 600 *and* 480 depending on where it sits. So the joint histogram row
for "CT muscle" spreads across three columns instead of one — the same smearing that
misalignment causes.

The registration then sees a blurred histogram and cannot tell whether it is blurred because the
images are misaligned (fixable by moving) or because the brightness drifts (not fixable by
moving). N4 removes the second cause so the metric only ever responds to the first.

CT gets no equivalent step. HU are already calibrated and absolute — bone is +1000 in the corner
of the image just as much as in the centre.

### 7.2 In-plane resampling (CT)

`resample_inplane` puts the CT on an isotropic 1mm in-plane grid.

```python
new_sx = new_sy = float(target_spacing)     # 1.0 mm
new_sz = orig_sp[2]                         # through-plane spacing UNTOUCHED
```

> 🎓 **Why Z is deliberately left alone.** Resampling along Z would interpolate *between* slices
> that are 5mm apart, inventing anatomy that was never measured. Interpolating in-plane between
> 0.48mm samples is a mild, honest operation; interpolating across 5mm gaps is hallucination.
> The pipeline's rule is: never fabricate through-plane detail.

### 7.3 Volume alignment — and the bug it replaced

This is step 3, and it is where the two volumes are brought into gross correspondence in 3D.

#### What the production code does, and why it is wrong

```python
# image_processing.py :: resample_mri_to_ct_grid - DO NOT USE AS-IS
mri_aligned = mri_image                            # (a)
mri_aligned.SetDirection(ct_image.GetDirection())  # (b)
resampler.SetTransform(sitk.Transform())           # (c)
```

Three problems:

- **(a) is an alias, not a copy.** `mri_aligned = mri_image` binds a second name to the *same*
  object. The `SetDirection` on the next line therefore mutates the caller's image in place.
  Anything else holding that MRI silently gets a corrupted one. The demo scripts defend against
  this by handing it a disposable copy — `img_proc.resample_mri_to_ct_grid(sitk.Image(mri_corrected), …)` —
  but `pipeline_core.py:92` passes `mri_corrected` directly, so in production the N4-corrected
  volume really is mutated in place.
- **(b) destroys real information.** The MRI's `Direction` matrix is a measurement of how the
  patient was actually oriented. Overwriting it with the CT's does not *align* anything — it just
  asserts a falsehood and loses the data you needed.
- **(c) is the fatal one.** An identity transform means **translation is never corrected at
  all**. The MRI is resampled wherever its raw `Origin` puts it. The function's entire effect is
  to fiddle with rotation metadata.

The measured damage: translations of **−84mm (knee)** and **−56mm (shoulder)** left completely
uncorrected, and **7 of 33** test slices came out as a completely blank MRI.

#### The replacement

```python
def resample_mri_to_ct_grid_v2(mri_image, ct_image, default_pixel_value=0.0):
    mri_f = sitk.Cast(mri_image, sitk.sitkFloat32)
    ct_f  = sitk.Cast(ct_image,  sitk.sitkFloat32)

    initial_transform = sitk.CenteredTransformInitializer(
        ct_f, mri_f, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetTransform(initial_transform)
    return resampler.Execute(mri_image), initial_transform.GetTranslation()
```

`CenteredTransformInitializer` with the `GEOMETRY` mode computes the translation that maps the
CT's bounding-box centre onto the MRI's bounding-box centre. It is pure arithmetic on the
metadata — **no optimisation, no randomness, nothing for multi-start to fix.** The MRI's real
`Direction` is left untouched.

### Example — the whole calculation, on one axis of real data

Brain patient `PA0_Ranjeet`, world X axis, straight from the DICOM headers:

```
CT  spans  X ∈ [−137.6, 138.0]      centre = (−137.6 + 138.0) / 2 =   0.2 mm
MRI spans  X ∈ [ −94.6, 101.8]      centre = ( −94.6 + 101.8) / 2 =   3.6 mm

translation = MRI centre − CT centre = 3.6 − 0.2 = +3.4 mm
```

That is it. One subtraction per axis, giving `(3.4, 9.8, 8.1) mm` in 3D. The pipeline logs
`(+3.6, +10.1, +7.9)` for this pair — the small difference is because it runs the calculation
*after* resampling the CT to 1 mm in-plane, which nudges the CT grid slightly.

Compare with the knee, where the same one-line calculation gives **−86.9 mm** on X. The old code
left that entirely uncorrected, which is why 7 of 33 slices came out blank: the MRI was resampled
87 mm away from where the CT was looking.

Rotation is deliberately left at identity, for the anisotropy reason from §3.4: 9–21 slices at
5–9mm cannot constrain a 3D rotation. Rotation is recovered per-slice in 2D at step 4.

**Results:** blank-MRI slices went **7/33 → 0/33**, and rigid registration beat the old baseline
on **33/33** slices.

**Known limitation, stated honestly:** GEOMETRY aligns bounding-box *centres*, not anatomy. If
the two volumes cover different physical extents, their centres are not the same anatomical
point. Where the old hack happened to be nearly right already (brain/axial — a centred, symmetric
structure) this can be slightly worse.

### 7.4 Per-slice 2D registration

Now each slice is registered in 2D, where the fine in-plane sampling makes rotation well
constrained. Per canvas:

1. Run rigid multi-start (3 seeds).
2. Run affine multi-start (3 seeds) **with the scale gate as `accept_fn`**.
3. Prefer affine only if it beats rigid by more than the seed-spread noise floor.
4. If the winner does not beat the unregistered baseline, ship the baseline instead.

That is `register_canvas` in v3, and it returns **two** different results deliberately:

```python
best_*      # the best REGISTERED result - what the failure diagnosis is about
shipped_*   # what actually gets used - the baseline whenever registration lost
```

> 🎓 **Why they must be separate.** A *shipped* result is never worse than the baseline, by
> construction. If you diagnose failure using the shipped number, "registration made this worse"
> becomes impossible to detect — the very condition you want to catch is defined away. Diagnose
> on what registration *achieved*; ship what is *safest*.

### 7.5 Normalisation for display and export

Not part of registration, but it is what the panels show.

- **CT** → clip to a region-specific HU window, then rescale to `[0, 1]`. Brain uses `0…80` HU
  (narrow, because brain tissue contrast lives in a tiny HU range); musculoskeletal and spine use
  `−200…300`.
- **MRI** → no absolute scale exists, so bounds are computed *per volume* from the **0.5 and 99.5
  percentiles of non-zero voxels**, then each slice is clipped and rescaled. Non-zero matters: the
  black background dominates an MRI and would drag the percentiles down.

> All MRI panels in a figure use the **same** `[p1, p99]` window, so brightness differences you
> see between panels come from the processing step, not from re-normalising each panel
> separately.

---

## 8. `v3` in detail: cropping as a fallback

Now the actual subject of `registration_demo_sweep_v3.py`.

### 8.1 The idea, and why it is tempting

If the MRI covers only part of the CT's field of view, most of the CT canvas is territory the MRI
can never fill. The optimiser is spending its effort on a frame where much of the content is
fill value. So: crop the output grid to the **world-space intersection** of the two fields of
view, and register on that.

Reasonable. And when tested by applying it to every slice, the average effect was **nothing**:
mean MI change +0.012 (rigid) and −0.013 (affine) across 33 slices.

But the average hid the real structure. Per series it was **strongly bimodal**:

- It **rescued** knee/sagittal, where the MRI covers a sliver of the CT canvas.
- It **clearly harmed** well-matched pairs — spine/coronal affine 0.63 → 0.48, shoulder/axial
  about −0.09 on both transforms — by trimming away context the optimiser was using.

> 🎓 **The methodological lesson.** Averaging over two populations that respond in opposite
> directions answers nothing. The useful question is not *"does cropping help?"* but *"which
> slices belong to which population, and can I identify them in advance?"*

### 8.2 The design: full canvas first, crop only on failure

```
register on the FULL canvas
        │
        ├── classify the outcome
        │
        ├── improved / marginal ──────────────► done, ship it. No crop.
        │
        └── failed ──► is a cropped counterpart available for this slice?
                           │
                           ├── no  ──► record why, ship the full-canvas result
                           │
                           └── yes ──► register on the CROP
                                          │
                                          re-score EVERYTHING on the common region
                                          │
                                          three-way choice: crop / full / unregistered
```

### 8.3 The failure taxonomy

```python
def classify(attempt, coverage):
    baseline, best = attempt["nmi_baseline"], attempt["best_nmi"]
    if attempt["all_seeds_failed"]:       return "all_seeds_failed",  True
    if attempt["nothing_admissible"]:     return "nothing_admissible", True
    if best is None or np.isnan(best):    return "unevaluable_nmi",   True
    if coverage < MIN_MRI_COVERAGE:       return f"sparse_overlap_{coverage:.2f}", True
    if baseline is None or np.isnan(baseline):
        return "improved", False
    if best < baseline - MIN_GAIN:        return "regressed", True
    if best <= baseline + MIN_GAIN:       return "marginal",  False
    return "improved", False
```

| Outcome | Meaning | Fallback? |
|---|---|---|
| `all_seeds_failed` | every multi-start attempt crashed | yes |
| `nothing_admissible` | seeds ran, none survived the gate on either transform | yes |
| `unevaluable_nmi` | the winner cannot be scored (constant/blank slice) | yes |
| `sparse_overlap_<f>` | MRI fills less than 25% of the canvas | yes |
| `regressed` | result is **worse** than the unregistered baseline | yes |
| `marginal` | within ±`MIN_GAIN` of baseline — registration had nothing to add | **no** |
| `improved` | beats baseline by more than `MIN_GAIN` | no |

### Example — the three common outcomes, with `MIN_GAIN = 0.010`

Everything is decided by comparing `best` against `baseline`. Draw the number line once and the
whole taxonomy falls out:

```
        regressed          marginal          improved
   ←───────────────┤──────────┼──────────├───────────────→
                baseline   baseline   baseline
                 −0.010                +0.010
```

Three real slices:

| Slice | baseline | best registered | difference | Outcome | Why |
|---|---|---|---|---|---|
| brain/axial middle | 1.0940 | 1.1664 | **+0.0724** | `improved` | well clear of +0.010 |
| brain/coronal middle | 1.1042 | 1.1081 | **+0.0039** | `marginal` | helped, but under the margin |
| knee/sagittal last † | 1.1273 | 1.0811 | **−0.0462** | `regressed` | *worse* than doing nothing |

† The first two are from the current run. The `regressed` row is from the run **before** the
scale gate was redesigned — the current run has **zero** regressed slices, so there is no live
example to show. That is the intended outcome, not a gap in the table: with a gate that admits
usable affines, no slice ends up worse than doing nothing.

**The distinction that matters** is between rows 2 and 3. Both "failed to clear +0.010", and the
earlier version of this code treated them identically and sent both to the crop fallback. But
they are opposite situations:

- Row 2 improved the slice slightly. Step 3's volume alignment had already done the work; there
  was little left for 2D registration to add. **Nothing is wrong.** Cropping cannot help, because
  there is no problem to fix.
- Row 3 made the slice measurably worse. Something genuinely failed. **A different canvas might
  help**, so the fallback is warranted — and meanwhile the unregistered baseline ships, because
  1.1273 beats 1.0811.

Lumping those together sent 14 of 33 slices into a fallback that 12 of them did not need.

Two things in this function are load-bearing and were both wrong in an earlier version.

#### `marginal` is not a failure

The earlier code had a single criterion — anything that failed to *beat* the baseline by the
margin was "no_gain" and triggered the fallback. That routed **14 of 33 slices** into cropping.

When we actually looked at those 14: **twelve of them had improved**, just by less than 0.010.
Only two were genuinely worse.

So the fallback was firing on "registration was unnecessary because step 3 had already done the
job" — and cropping is not a remedy for that. Correcting the definition dropped the fallback rate
from 14/33 to **2/33**.

> 🎓 **The lesson.** "Did not improve" and "made it worse" are completely different events, and a
> single threshold cannot express both. If your criterion is `not (a > b + margin)`, ask yourself
> whether you have accidentally merged "roughly equal" with "much worse."

#### Causes are tested before symptoms

`sparse_overlap` is checked *before* `regressed`. A slice starved of MRI data will probably also
regress — but "there was barely any MRI here" explains the slice, while "the score went down"
merely describes it. Order your checks so the recorded reason is the one a human can act on.

### 8.4 The three-way decision

When the fallback runs, we have three candidate results. First, they must be made comparable:

```python
# Re-score the full-canvas result over the cropped region. NMI depends on which
# pixels it sees, so a full-canvas score and a cropped-canvas score are not
# comparable numbers even for identical anatomy.
nmi_full_common = demo.nmi_score(ct_crop, to_common(full["shipped_img"]))
nmi_base_common = crop["nmi_baseline"]
crop_ship       = crop["shipped_nmi"]
```

`to_common` crops a full-canvas 2D slice down to the intersection box, so all three numbers
describe the same physical region. Then:

```python
if is_better(crop_ship, nmi_full_common, MIN_GAIN) and is_better(crop_ship, nmi_base_common, MIN_GAIN):
    keep the CROP
elif is_better(nmi_base_common, nmi_full_common, MIN_GAIN):
    keep NOTHING - the unregistered baseline beats both
else:
    keep the FULL result            # recorded as crop_did_not_help
```

**Why three-way and not two-way.** The earlier code compared the crop only against the full
result. That lets a crop *worse than doing nothing* win, as long as the full result was worse
still. Requiring it to beat both closes that hole.

### Example — two fallbacks, opposite verdicts

Both from the run *before* the scale gate was fixed, since the current run has no fallbacks at
all. All three numbers in each block are scored on the **same** cropped region:

**Case 1 — knee/coronal first. The crop wins.**

```
unregistered baseline   1.0934
full canvas result      1.0934      ← identical, because the full canvas had
                                      already dropped to the unregistered fallback
crop result             1.1243

crop beats full  by 0.0309  > 0.010  ✅
crop beats base  by 0.0309  > 0.010  ✅
                                    →  KEEP CROP
```

**Case 2 — knee/sagittal last. The crop loses.**

```
unregistered baseline   1.1095
full canvas result      1.1095
crop result             1.1158

crop beats full  by 0.0063  < 0.010  ❌
                                    →  KEEP FULL  (crop_did_not_help)
```

Same series, same patient, both slices "failed" on the full canvas, and the two go opposite ways.
Case 2 is the important one: 1.1158 *is* higher than 1.1095, so a naive `>` comparison would have
switched to a smaller canvas for a gain of 0.006 — well inside the noise. `MIN_GAIN` is what
stops that.

> 🎓 **Notice what makes Case 1 legitimate.** The crop does not merely beat the full-canvas
> result; it beats *doing nothing*. Had the crop scored 1.10 here it would have beaten the full
> canvas (1.0934) while still being worse than the unregistered baseline — winning a race against
> a loser. The three-way test catches exactly that.

### 8.5 The thresholds

```python
MIN_GAIN         = 0.010   # NMI a result must win by to count as better
MIN_MRI_COVERAGE = 0.25    # MRI must fill this fraction of the canvas
SCALE_TOL        = 0.60    # outer sanity bound on |scale - 1|
CANVAS_FIT_TOL   = 0.04    # veto scales sitting on the canvas ratio (0 disables)
MIN_TRANSFORM_MARGIN = 0.005   # floor on the margin affine must beat rigid by
N_STARTS         = 3       # multi-start attempts per transform
```

All four are **starting points chosen from 4 patients**, not settled constants. Treat them as
hypotheses.

### 8.6 What the corrected run actually shows

33 slices, 11 series, 4 patients:

| | |
|---|---|
| `improved` | **18 / 33** |
| `marginal` | **13 / 33** |
| `regressed` | **2 / 33** |
| Crop attempted | **2 / 33** |
| Crop kept | **1 / 33** |
| Shipped: rigid / affine / unregistered | **26 / 4 / 3** |
| Mean gain of shipped result over baseline | **+0.021 NMI** (floor exactly 0.000, by construction) |

The single kept crop is knee/coronal `first`: 1.093 → 1.124 on the common region.

**The headline is how little cropping turns out to be needed** once the frame bug and the failure
definition are fixed — from 14 attempts and 4 keeps, down to 2 and 1.

> 🎓 **And one deleted result, which is the most instructive part.** The previous run's biggest
> success story was knee/sagittal `last`: "cropping rescued this slice, 1.091 → 1.197!" It did
> not. The 1.091 it was being compared against was a full-canvas result that *should never have
> shipped* — it was worse than that slice's own unregistered baseline of 1.127. Once the
> unregistered rung exists, the full canvas ships 1.110, the crop manages 1.116, and the
> difference no longer clears the margin. **The impressive gain was mostly the badness of its
> comparison point.** Whenever you see a large improvement, check what it improved *on*.

---

## 9. Case study: the coordinate-frame bug

This deserves its own section because it is the most valuable thing in this document. It was
invisible, it produced plausible output, it corrupted a headline conclusion, and the fix is three
lines.

### 9.1 The setup

Two functions, both correct-looking:

```python
# Compute the box where the two volumes overlap
roi = compute_intersection_roi(ct_image, mri_image)      # uses RAW geometry of both

# Fill that box with MRI
resampler.SetReferenceImage(ct_cropped)
resampler.SetTransform(initial_transform)                # applies the GEOMETRY translation
resampled = resampler.Execute(mri_image)
```

The box is computed from where the volumes sit **as acquired**. The box is then filled with MRI
that has been **moved** by the alignment translation — the very correction whose entire purpose is
to make the volumes overlap.

The box and its contents are in different coordinate frames.

### 9.2 Why it is invisible

A crop that is too small looks exactly like a crop that is working. That is the whole trap. You
asked for a smaller canvas, you got a smaller canvas, the slice count went down, the images look
like anatomy. Nothing throws, nothing warns, no NaN appears.

### Example — the bug in one dimension

Strip it down to a number line. CT covers 0–100 mm. MRI covers 60–160 mm — offset by 60 mm.

```
CT       [0 ─────────────────── 100]
MRI                    [60 ─────────────────── 160]
                        └── raw overlap: 60–100 ──┘
```

**Step 1 aligns the centres.** CT centre = 50, MRI centre = 110, so the translation is −60. After
it, the MRI effectively covers **0–100** — exactly on top of the CT:

```
CT       [0 ─────────────────── 100]
MRI      [0 ─────────────────── 100]     ← after the −60 shift
         └──── true overlap: 0–100 ─────┘
```

**The bug:** compute the overlap box from the *first* picture (60–100), then fill it with MRI
from the *second* (0–100). You keep 40% of the canvas and throw away the 0–60 region — which is
full of perfectly good, correctly aligned MRI.

**The fix:** shift the MRI box by the same −60 before intersecting. `[60,160] − 60 = [0,100]`,
intersected with `[0,100]`, gives the whole thing.

That is precisely what happened on knee/sagittal, with −86.9 mm instead of −60 and slices 14–17
kept instead of 0–17.

### 9.3 The arithmetic, worked out

Knee/sagittal, world X axis, all values in mm:

```
CT extent              [  53.3, 145.4 ]      width  92.1
MRI extent (as acquired)  [ -39.9,  64.8 ]      width 104.7

CT centre  = (53.3 + 145.4) / 2 =  99.35
MRI centre = (-39.9 + 64.8) / 2 =  12.45

GEOMETRY translation  t = 12.45 - 99.35 = -86.9 mm      ← matches the logged value exactly
```

**What the buggy code computed** (raw frame):

```
intersection = [ max(53.3, -39.9), min(145.4, 64.8) ] = [ 53.3, 64.8 ]   → 11.5 mm wide
overlap fraction = 11.5 / 92.1 = 12.5%                 ← the "12.5% overlap" figure
```

**What is actually true after alignment.** The resampler's transform maps *output* points to
*moving* points: output point `p` samples the MRI at `T(p)`. So `p` sees real MRI exactly when
`T(p)` lies inside the MRI — that is, when `p` lies inside `T⁻¹(MRI extent)`. With identity
rotation, `T⁻¹(p) = p − t`:

```
aligned MRI extent = [ -39.9 - (-86.9), 64.8 - (-86.9) ] = [ 47.0, 151.7 ]

intersection = [ max(53.3, 47.0), min(145.4, 151.7) ] = [ 53.3, 145.4 ]   → the ENTIRE CT
overlap fraction = 100%
```

### 9.4 The damage

| | Buggy (raw frame) | Correct (aligned frame) |
|---|---|---|
| Crop keeps CT slices | **14 – 17** (4 slices) | **0 – 17** (18 slices) |
| Slices with real MRI content | 0 – 17 | 0 – 17 |
| **Slices discarded that hold real MRI** | **14 of 18** | 0 |

And it propagated into the written conclusions. `registration_recommendations.md` stated that
knee/sagittal "yields 4 usable slices out of 18." That was never a fact about the data — it was
an artifact of this bug, and it was on its way to becoming a reason to exclude the series.

There was even a corroborating clue that went unread: `mri_coverage` reported a flat **60.7%** on
every one of the 18 slices, including the 14 the crop was discarding. A slice genuinely outside
the shared field of view resamples to all-fill and reads **0%**. A uniform non-zero coverage
across every slice was the data saying "the MRI reaches all of these," and it disagreed with the
crop for a whole round of results.

### 9.5 The fix

Push the MRI's corners through the inverse of the alignment transform before intersecting:

```python
def corners_in_ct_frame(mri_image, transform):
    inv = transform.GetInverse()
    return np.array([inv.TransformPoint(tuple(float(x) for x in c))
                     for c in corners_world(mri_image)])
```

And — just as important — change the signature so the mistake cannot recur:

```python
def compute_intersection_roi(ct_image, mri_corners):
    """
    `mri_corners` must already be in the CT frame (see corners_in_ct_frame).
    Taking corners rather than an image is deliberate, so the caller cannot
    accidentally pass geometry from the wrong frame again.
    """
```

> 🎓 **The generalisable lesson.** Any time you compute a *region* in one step and *fill* it in
> another, verify they are in the same coordinate frame. Better: make the function signature
> refuse the ambiguous input. A function that takes `mri_image` can be handed either frame and
> cannot tell; a function that takes `mri_corners` forces the caller to have thought about it.

### 9.6 Two more geometry traps in the same function

**Off-by-one in the region size.** `sitk.RegionOfInterest` takes a voxel **count**, but an
inclusive index range `[start, stop]` contains `stop − start + 1` voxels:

```python
roi_size = stop - start + 1     # forget the +1 and every axis silently loses a voxel
```

### Example — count the fingers

Indices 3 to 7 inclusive:

```
index:  0  1  2  3  4  5  6  7  8  9
                 └──────────┘
                 3  4  5  6  7   ← that is FIVE voxels
```

`7 − 3 = 4`, but there are **5**. The count is `7 − 3 + 1 = 5`.

On a 512×512×18 volume the error costs one voxel per axis — `511×511×17` — which is a 0.2%
shrink. Invisible in an image, and it looks exactly like a small genuine crop. Same failure
signature as the frame bug: **wrong in the direction you were already expecting**.

**Fit the transform against the full CT, never the cropped one.** This is the code you had
selected in the editor:

```python
resampler.SetReferenceImage(ct_cropped)   # ← the ONLY line that differs from v2
resampler.SetTransform(initial_transform) # ← fitted against the FULL ct_image
```

Cropping chooses *which part of physical space to keep*. It must not move the MRI. But
`CenteredTransformInitializer` aligns bounding-box **centres** — so fitting it against
`ct_cropped` would re-centre the MRI on the *crop's* centre instead of the CT's. A different, and
wrong, spatial relationship: roughly **40mm** of drift in testing. That was the original reason
cropping appeared to destroy registration quality.

Because v2 and v3 now apply an identical transform and differ only in output canvas, the script
**asserts it at runtime**:

```python
drift = np.abs(np.array(v3_t) - np.array(v2_t)).max()
if drift > 1e-6:
    print(f"    ! WARNING: v2/v3 transforms disagree by {drift:.3f}mm - they should be identical.")
```

> 🎓 If two code paths are supposed to agree, make the program check it every run. A cheap
> assertion beats a careful comment.

---

## 10. Reading the outputs

### 10.1 The figures

`registration_demo_output/sweep_v3/<region>_<patient>_<orientation>_<position>_v3.png`

The filename includes the patient because two patients in the same region and orientation would
otherwise silently overwrite each other — a latent bug in an earlier version.

Panels are **fusion overlays**: CT in greyscale, MRI in a `hot` colourmap at 45% alpha. Where
they align, structures coincide; where they do not, you see the MRI's orange edges sitting beside
the CT's grey ones. This is the single most valuable QC artifact in the project.

- **Two panels** — the full canvas succeeded. `baseline` vs the shipped result.
- **Three panels** — the fallback ran. `baseline` / `full` / `crop`, all on the common region,
  with `<- kept` marking the winner.

### 10.2 The CSV

`sweep_v3_summary.csv`, 33 rows × 49 columns. The ones to read first:

| Column | Why you care |
|---|---|
| `full_outcome` | one of the seven classes from §8.3 |
| `nmi_baseline_full` / `nmi_full_best` / `nmi_full_shipped` | three different numbers: what doing nothing achieves, what registration achieved, what was used |
| `full_affine_scales` | **per seed**, not just the winner — without this the gate is unauditable |
| `full_affine_rejected` | how many seeds the gate refused |
| `rigid_full_spread` / `affine_full_spread` | the noise floor. `None`, not `0.0`, when fewer than two admissible seeds survive |
| `full_noise_floor` | the margin the affine-vs-rigid choice had to clear |
| `crop_attempted` / `crop_used` | different questions — a slice can fall back and still keep the full canvas |
| `crop_skipped_reason` / `crop_rejected_reason` | "why we didn't try" vs "why we tried and discarded". One column cannot carry both |
| `final_source` / `final_kind` / `final_scoring_region` | which run it came from, which transform, **and which region the score refers to** |
| `fov_overlap_raw` / `fov_overlap_aligned` | the two QC-0 numbers |

> ⚠️ **`final_nmi` is meaningless without `final_scoring_region`.** A slice that never needed the
> fallback is scored on the full canvas; one that did is scored on the common region. Those are
> different denominators. Never aggregate `final_nmi` across rows without grouping by region
> first. (An earlier version omitted this column, which made the whole column silently
> un-aggregatable.)

### Example — reading one real row end to end

`shoulder / PA6_Vijay / coronal / middle`. Here is the row, in the order the decisions were
actually made:

```
fov_overlap_raw       0.8923     ← the two scans were positioned similarly (89% overlap)
fov_overlap_aligned   1.0000     ← after step 3, they overlap completely
mri_coverage_full     0.6519     ← MRI fills 65% of the canvas, well above the 25% floor
nmi_baseline_full     1.0747     ← doing nothing scores this
```

Registration runs. Rigid first, then affine with the gate:

```
full_canvas_ratio     0.8859     ← MRI's bounding box spans 89% of the canvas
                                   so a frame-fitting affine would land near
                                   0.886 or 1/0.886 = 1.129
full_affine_scales    1.102 | 1.009 | 1.009
full_affine_rejected  1
```

**Check the gate by hand.** The veto is active because `|0.886 − 1| = 0.114` exceeds
`2 × CANVAS_FIT_TOL = 0.08`:

| Seed | Scale | Distance to 0.886 | Distance to 1.129 | Verdict |
|---|---|---|---|---|
| 0 | 1.102 | 0.216 | **0.027** | ❌ rejected — within 0.04 of the inverse canvas ratio |
| 1 | 1.009 | 0.123 | 0.120 | ✅ admissible |
| 2 | 1.009 | 0.123 | 0.120 | ✅ admissible |

One rejection, matching `full_affine_rejected = 1`. Seed 0 wanted to stretch the MRI onto the
frame; seeds 1 and 2 found a near-unity scale, which is what a correct answer looks like.

Now the transform choice:

```
nmi_rigid_full        1.0812
nmi_affine_full       1.1234
affine_full_spread    0.0031     ← affine's seeds agree closely
full_noise_floor      0.0050     ← max(0.0031, MIN_TRANSFORM_MARGIN) = 0.005

gap = 1.1234 − 1.0812 = 0.0422   →  0.0422 > 0.0050  →  affine wins
full_best_kind        affine
```

Then the safety check and the classification:

```
nmi_full_shipped      1.1234     ← beats the 1.0747 baseline, so it ships as-is
                                   (no drop to "unregistered")
1.1234 − 1.0747 = +0.0487        →  well over MIN_GAIN
full_outcome          improved
```

And therefore no fallback:

```
crop_attempted        False
crop_skipped_reason   full_canvas_ok_improved
final_source          full
final_kind            affine
final_nmi             1.1234
final_scoring_region  full       ← so this number is on the full canvas
```

**Every intermediate value is in the row.** You never have to take the outcome on trust: the
per-seed scales, the canvas ratio, the noise floor and both baselines are all recorded, so any
decision the script made can be re-derived with a calculator.

### 10.3 Reading the console log

```
      affine multi-start: ['1.126*', '1.181*', '1.182*']  best=1.1273 (seed=None)
             spread=n/a  failed=0/3  rejected=3  -> UNREGISTERED (nothing admissible)
```

- `*` = that seed was **rejected by the gate**. It ran and scored; it just was not admissible.
- `FAIL` = that seed **crashed** — a different thing entirely, and counted separately.
- `seed=None` + `-> UNREGISTERED` = nothing admissible survived, so the unregistered baseline is
  the result.
- `spread=n/a` = fewer than two admissible seeds, so no spread can be computed. Not zero.

One error message you will see often, and it is benign:

```
ITK ERROR: MattesMutualInformationImageToImageMetricv4: All samples map outside
moving image buffer. The images do not sufficiently overlap.
```

A single seed wandered the moving image entirely off the fixed image. Multi-start catches it,
that seed is recorded as `FAIL`, and the others continue.

---

## 11. How to be appropriately sceptical

The recurring theme of this whole investigation: **a higher score is not proof of better
alignment.** MI and NMI compare intensity *distributions*. Neither has any concept of *which*
anatomical structure it matched. Normalising fixed a comparability problem, not a correctness
one.

Five checks that actually work:

**1. Sweep the parameter and look at the shape of the landscape.** Translate the moving image
across a range and plot the score. A sharp peak means the metric can localise the answer. A flat
plateau means it cannot, and any "optimum" inside that plateau is arbitrary. This costs seconds
and should be routine.

**2. Sweep every degree of freedom, not one.** A sharp peak in one axis says nothing about the
others.

**3. Compare the recovered scale against the FOV ratio.** If they match, the optimiser is fitting
the canvas, not the anatomy (§5.2).

**4. Compare the signal against the optimiser's own noise.** If the difference between two
candidates is smaller than the seed-to-seed spread, you have not measured anything (§6.4).

**5. Look at the images.** The spine failure survived three rounds of numeric reporting and was
found by eye.

### The spine, as a worked example of all five

Lumbar vertebrae are **near-periodic** — one looks much like the next. So MI cannot localise
position along the spine axis: shift the MRI by one vertebra and the score barely changes.

Measured: the difference between the baseline position and the metric's supposed "best" was
**0.013**, against an optimiser noise floor of **0.047**. Signal-to-noise **0.28**. There is a
**~78mm** window in which every position scores within noise.

Anchoring the metric on the sacrum sharpens that axis (ambiguity 45mm → 20mm) but **widens**
anterior-posterior ambiguity (10mm → 20mm), and leaves left-right — which is *through-plane* for a
sagittal series, so 2D registration cannot correct it at all — indeterminate.

> **The scale gate can prove a spine result wrong. Nothing available proves one right.** With
> only 4 spine patients in the dataset, exclusion is the cheap and honest call.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **AABB** | Axis-aligned bounding box: the smallest axis-parallel box containing a volume, in world space |
| **Affine** | Transform allowing translation, rotation, per-axis scale and shear (6 DOF in 2D) |
| **Bias field** | Smooth multiplicative brightness gradient across an MRI, from coil sensitivity, not anatomy |
| **Canvas** | The output pixel grid a result is resampled onto — full CT grid, or the cropped intersection |
| **Common region** | The intersection box, used to score full-canvas and cropped results comparably |
| **DOF** | Degrees of freedom: how many independent numbers a transform has |
| **Entropy `H`** | `−Σ p·log p`. Uncertainty in a distribution |
| **Fixed image** | The image held still. Here, always the CT |
| **Fusion overlay** | CT greyscale + MRI colourmap at partial alpha, for visual alignment QC |
| **GEOMETRY initializer** | Alignment mode that maps bounding-box centre to bounding-box centre |
| **HU** | Hounsfield Units. Calibrated absolute CT scale: air −1000, water 0, bone +1000 |
| **Joint histogram** | 2D histogram counting co-occurrences of intensity pairs between two images |
| **Mattes MI** | ITK's efficient MI implementation, using Parzen windows on a sampled joint histogram |
| **MI** | Mutual information: `H(A) + H(B) − H(A,B)`. Unbounded |
| **Moving image** | The image being transformed. Here, always the MRI |
| **Multi-start** | Running several seeded attempts and keeping the best admissible one |
| **N4** | Bias field correction algorithm (`N4BiasFieldCorrectionImageFilter`) |
| **NMI** | Normalized MI: `(H(A) + H(B)) / H(A,B)`. Bounded in `[1,2]` |
| **Noise floor** | The seed-to-seed spread; the minimum difference that counts as real |
| **Pyramid** | Coarse-to-fine multi-resolution scheme for escaping local optima |
| **Rigid / Euler2D** | Translation + rotation only (3 DOF in 2D) |
| **ROI** | Region of interest: `(start_index, size_in_voxels)` |
| **Canvas ratio** | Fraction of the frame the MRI's bounding box spans; the value a frame-fitting affine converges on |
| **Scale gate** | Two-part admissibility rule for affines: an outer bound `\|scale − 1\| ≤ SCALE_TOL`, plus a veto on scales sitting within `CANVAS_FIT_TOL` of the canvas ratio or its reciprocal |
| **SVD** | Singular value decomposition; used to extract scale from a mixed rotation/scale/shear matrix |
| **World space** | Physical millimetre coordinates shared by all images of one patient |

---

## 13. Open questions

Stated as questions, because none of them is settled.

**Why does affine want to resize on this data — mostly answered.** Under the old ±0.05 gate,
52 of 99 affine seeds were rejected and 12 of 33 slices had no admissible affine at all. The
answer turned out to be that the recovered scales were sitting on the **canvas ratio**: the
optimiser was fitting the frame border, not the anatomy. With the canvas-fit veto in place,
rejections fall to **18 of 99** and only 3 slices lose every seed. **Still open:** whether the
residual rejections are also frame-fitting, or something real.

**Is registration reproducible enough to gate on scale at all?** This is now the biggest open
item, and the answer is currently *no*. At a fixed seed, the recovered affine scale varies by
**up to 0.344** between identical runs (§6.2). Any scale threshold — 0.05 or 0.60 — is therefore
gating on a quantity that wobbles more than the threshold itself. **Next step:** switch
`SetMetricSamplingStrategy` to `NONE`, which is exactly reproducible and costs ~1.5–2× runtime.
In spot checks it also recovers scales of **0.990** (knee) and **0.975** (shoulder) — near unity,
which is further evidence that the large scales were an artifact rather than a real geometric
difference.

**Is the noise floor calibrated correctly?** Rigid ships on 20 of 33 slices and affine on 12. The
margin now uses the affine's own reproducibility rather than `max(rigid, affine)`, which fixed a
case where an unstable rigid was raising the bar for a stable affine. But with only 3 seeds the
spread estimate is itself noisy. **Next step:** re-run with more seeds.

**Do the untested criteria ever fire?** In the current run **nothing regresses**, so the crop
fallback never triggers at all — `regressed`, `all_seeds_failed`, `nothing_admissible`,
`unevaluable_nmi` and `sparse_overlap` are all untriggered. Minimum MRI coverage across all 33
slices was **27.8%**, just above the 25% threshold — close, but it has never crossed. The
fallback machinery is intact but currently dormant, which means it is also **untested against
live data** in its current form.

**Why did brain/axial get slightly worse under GEOMETRY centring?** Baseline MI fell 0.451 →
0.348 there, and it made no difference downstream (both converged to ~0.61). The likely cause is
a real rotation that a translation-only fix cannot absorb.

**Does any of this generalise?** The evidence is **4 patients, 11 series pairs, 33 slices**. That
is enough to find failure modes and fix clear bugs. It is nowhere near enough to claim these
thresholds hold for the other ~41 patients. Every constant in §8.5 is a hypothesis awaiting a
larger sample.

---

## Appendix: the scripts

| Script | What it does |
|---|---|
| `registration_demo.py` | Per-region raw→registered walkthrough. Holds `resample_mri_to_ct_grid_v2`, `nmi_score`, multi-start, `score_spread` |
| `registration_demo_sweep.py` | 33 slice-pairs across every orientation, first/middle/last |
| `registration_demo_sweep_v3.py` | **This document's subject.** Full-canvas registration with the intersection crop as a per-slice fallback |
| `registration_demo_fov.py` | Standalone field-of-view overlap diagnostic |
| `registration_demo_spine_fix.py` | Scale gate and sacrum anchor experiments on spine |
| `registration_demo_spine_axes.py` | Multi-axis landscape sweep showing what the anchor does and does not fix |

**None of these are part of the production pipeline.** They are diagnostics and reference
implementations. Production (`pipeline_core.py:92`) still calls the broken
`resample_mri_to_ct_grid` described in §7.3, followed by `register_2d_rigid` at line 130 —
single-shot, multi-threaded, no multi-start, no scale gate, no fallback ladder. **Porting the v2
alignment and the v3 selection logic into `image_processing.py` is outstanding work, not a
completed change.** Nothing in this document has shipped yet.

To reproduce every number in this document:

```bash
cd Preprocessing
python registration_demo_sweep_v3.py
```

A few minutes on a laptop. Output lands in `registration_demo_output/sweep_v3/` — 33 PNGs and
one CSV.
