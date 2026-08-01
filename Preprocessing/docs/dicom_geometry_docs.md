# 📖 DICOM Anatomy: Series vs Image, Index Space vs World Space

This document expands §2 of [`registration_docs.md`](registration_docs.md) into a full treatment,
and answers four questions that everything else in the pipeline rests on:

1. What is a **series** (`SE0`), and what defines it?
2. What is a single **DICOM image** (`IM0`), and what defines it?
3. What is the difference between them, and which properties live where?
4. What are **index space** and **world space**, and why does the distinction keep causing bugs?

**Every number in this document was read from your own dataset** — patient `PA0_Ranjeet`, study
`ST0`, series `SE0`, in both CT and MRI. You can re-run every snippet and get the same values.
Nothing here is a generic textbook example.

---

## Table of contents

1. [The hierarchy: Patient → Study → Series → Image](#1-the-hierarchy-patient--study--series--image)
2. [What a single DICOM image is](#2-what-a-single-dicom-image-is)
3. [What a series is](#3-what-a-series-is)
4. [Shared vs varying: the defining table](#4-shared-vs-varying-the-defining-table)
5. [Index space](#5-index-space)
6. [World space](#6-world-space)
7. [The bridge: the index→world formula, worked out](#7-the-bridge-the-indexworld-formula-worked-out)
8. [The Direction matrix, decoded](#8-the-direction-matrix-decoded)
9. [How 18 files become one volume](#9-how-18-files-become-one-volume)
10. [Six traps in your data](#10-six-traps-in-your-data)
11. [Frame of Reference: why registration is necessary at all](#11-frame-of-reference-why-registration-is-necessary-at-all)
12. [Inspect your own data](#12-inspect-your-own-data)
13. [Tag cheat sheet](#13-tag-cheat-sheet)

---

## 1. The hierarchy: Patient → Study → Series → Image

DICOM organises everything into four nested levels. Your folder layout mirrors them exactly:

```
Rawdata_dicom/
└── CT/                          ← modality (your own convention, not DICOM's)
    └── PA0_Ranjeet/             ← PATIENT
        └── ST0/                 ← STUDY   ("one visit to the hospital")
            ├── SE0/             ← SERIES  ("one scan sequence")   ← a 3D VOLUME
            │   ├── IM0          ← IMAGE / INSTANCE ("one slice")  ← a 2D SLICE
            │   ├── IM1
            │   ├── …
            │   └── IM17
            ├── SE1/
            └── SE2/
```

| Level | Your folder | Means | Analogy |
|---|---|---|---|
| **Patient** | `PA0_Ranjeet` | one human being | the author |
| **Study** | `ST0` | one imaging session / visit | one book |
| **Series** | `SE0` | one acquisition with one set of settings | one chapter |
| **Image (Instance)** | `IM0` | one 2D slice, one file | one page |

The two levels that matter day to day are the bottom two, and the sentence to memorise is:

> **A series is a 3D volume. An image is one 2D slice of it. The folder `SE0` is the volume; the
> file `IM0` is a slice.**

### Example — a loaf of bread

| Bread | DICOM | Your data |
|---|---|---|
| the whole loaf | the **series** | `SE0/` |
| one slice of bread | one **image** | `IM0` |
| the bag it came in | the **study** | `ST0/` |
| the baker | the **patient** | `PA0_Ranjeet/` |

Every slice of bread is the same width, same height, cut with the same knife — those facts belong
to the **loaf**, not to any one slice. But each slice sits at a different position along the
loaf, and *that* fact belongs to the slice.

DICOM works exactly this way, with one quirk: **there is no "loaf" file.** The loaf's properties
(slice width, knife type) are stamped onto *every single slice*, over and over, so that any slice
found on its own still knows what loaf it came from. That redundancy is §4.

Everything in this pipeline loads a *series* and works with the resulting 3D volume. The
individual `IM*` files are never opened one at a time by the pipeline — `sitk.ImageSeriesReader`
swallows the whole folder at once.

> ⚠️ **The folder names are yours, not DICOM's.** `SE0`, `IM0`, `ST0` are just how this dataset
> was exported to disk. DICOM itself does not care about filenames at all — a file could be
> called `banana.dcm` and still be slice 5 of series 3. The **true** identity of every level is
> stored in tags *inside* the files (§3.1). The folder structure is a convenience that happens to
> agree with the tags here; never rely on it without checking.

---

## 2. What a single DICOM image is

Open `CT/PA0_Ranjeet/ST0/SE0/IM0`. It contains two things:

```
┌─────────────────────────────────────┐
│  HEADER  — hundreds of tagged       │   ← "who, what, where, how"
│            metadata fields          │
├─────────────────────────────────────┤
│  PIXEL DATA — the raw numbers       │   ← 512 × 512 = 262,144 values
└─────────────────────────────────────┘
```

A tag is addressed by a pair of hex numbers, `(group, element)` — written in SimpleITK as
`"0020|0032"`. These are a fixed international standard: `(0020,0032)` means
`ImagePositionPatient` in every DICOM file ever written.

Your `IM0` carries **81 tags**. The two dozen below are the ones that matter here; the rest are
scanner settings, dates, and institutional bookkeeping.

### 2.1 The real header of your `IM0`

These are the actual values, grouped by what they are *for*.

**Identity — who is this and where does it belong?**

| Tag | Name | Value in your `IM0` |
|---|---|---|
| `(0008,0018)` | SOPInstanceUID | `1.2.840.113619.2.80.984800865.20130.1271308533.2` |
| `(0020,000E)` | SeriesInstanceUID | `1.2.840.113619.2.80.984800865.20130.1271308533.1.4.1` |
| `(0020,000D)` | StudyInstanceUID | `1.2.840.113619.2.415.3.2831157761.732.1743565258.430` |
| `(0020,0013)` | InstanceNumber | `1` |

**Geometry — where in the patient is this slice?** *(this is the important group)*

| Tag | Name | Value in your `IM0` |
|---|---|---|
| `(0020,0032)` | ImagePositionPatient | `-119.0751953 \ -139.6587219 \ -29.52215576` |
| `(0020,0037)` | ImageOrientationPatient | `0.9947773 \ 0.0479071 \ -0.0901276 \ -0.0743887 \ 0.9448984 \ -0.3187998` |
| `(0028,0030)` | PixelSpacing | `0.4882810116 \ 0.4882810116` |
| `(0018,0050)` | SliceThickness | `2.974968195` |
| `(0020,0052)` | FrameOfReferenceUID | `1.2.840.113619.2.415.3.2831157761.732.1743565258.432.4170.1` |

**Grid — how big is the pixel array?**

| Tag | Name | Value |
|---|---|---|
| `(0028,0010)` | Rows | `512` |
| `(0028,0011)` | Columns | `512` |

**Pixel meaning — how do I turn stored numbers into physical units?**

| Tag | Name | Value |
|---|---|---|
| `(0028,1052)` | RescaleIntercept | `-1024` |
| `(0028,1053)` | RescaleSlope | `1` |
| `(0028,0100)` | BitsAllocated | `16` |
| `(0028,0103)` | PixelRepresentation | `1` (signed) |
| `(0028,0004)` | PhotometricInterpretation | `MONOCHROME2` (0 = black) |

**Acquisition — how was it taken?**

| Tag | Name | Value |
|---|---|---|
| `(0008,0060)` | Modality | `CT` |
| `(0018,0060)` | KVP | `140` |
| `(0018,1030)` | ProtocolName | `1.1 Routine Head 5mm Axial mode` |
| `(0008,0020)` | StudyDate | `20250402` |

### 2.2 What *defines* a single image

Formally, one thing:

> **`SOPInstanceUID` (0008,0018) is the globally unique identifier of one image.** No two DICOM
> images anywhere in the world share one.

Check it across three of your files:

| File | SOPInstanceUID |
|---|---|
| `IM0` | `…1271308533.2` |
| `IM1` | `…1271308538.3` |
| `IM17` | `…1271308582.19` |

All different. Each file is its own instance.

But the identifier is only *bookkeeping*. What makes a slice **meaningful** is its geometry —
specifically `ImagePositionPatient`, which says where this particular slice sits in the patient.
That is the tag that differs from slice to slice and turns a stack of files into a volume.

---

## 3. What a series is

A series is **a set of images acquired together, with the same settings, describing one 3D
volume.**

### 3.1 What defines a series

> **`SeriesInstanceUID` (0020,000E).** Every image whose `SeriesInstanceUID` matches belongs to
> the same series. That is the definition — not the folder.

In your data:

| File | SeriesInstanceUID |
|---|---|
| `IM0` | `1.2.840.113619.2.80.984800865.20130.1271308533.1.4.1` |
| `IM1` | `1.2.840.113619.2.80.984800865.20130.1271308533.1.4.1` |
| `IM17` | `1.2.840.113619.2.80.984800865.20130.1271308533.1.4.1` |

**Identical.** That is the proof that these 18 files are one volume, and it is why
`sitk.ImageSeriesReader` can safely stack them.

Supporting descriptive tags, shared by every image in the series:

| Tag | Name | Value in `SE0` |
|---|---|---|
| `(0020,0011)` | SeriesNumber | `450` |
| `(0008,103E)` | SeriesDescription | `Processed Images` |
| `(0008,0060)` | Modality | `CT` |

### 3.2 Why `SeriesDescription` deserves suspicion

Compare the two `SE0` folders for this one patient — same person, same anatomy, same orientation:

```
CT  SE0   SeriesDescription = "Processed Images"
MRI SE0   SeriesDescription = "t2_tirm_tra_dark-fluid_FIL_1"
```

The MRI's description is richly informative: `t2` (weighting), `tirm` (sequence), **`tra`**
(transverse — i.e. axial), `dark-fluid` (FLAIR-like suppression). The CT's says nothing at all.

`io_utils.get_orientation_from_desc` hunts for exactly these substrings:

```python
if any(k in d for k in ["_tra", "_ax", "axial", "transv", "tra_"]):
    return "axial"
```

So on this patient the heuristic **succeeds on the MRI** (`_tra` matches) and **fails on the CT**
(`"Processed Images"` matches nothing, and the folder name `SE0` gives no hint either, so
`discover_series` returns `unknown`).

That asymmetry — one modality self-describing, the other silent, for the *same scan of the same
patient* — is precisely why the demo scripts hard-code verified orientations rather than trusting
the metadata:

```python
{"region": "brain", "patient": "PA0_Ranjeet", "orientation": "axial", "ct_se": "SE0", "mri_se": "SE0"},
```

> 🎓 **The lesson.** DICOM tags fall into two classes: **machine-generated** (UIDs, positions,
> spacings — trustworthy) and **human-typed** (descriptions, protocol names — a hint at best). A
> heuristic over free text is not wrong to write, but it must be allowed to return `unknown`, and
> nothing downstream may assume it succeeded. The true orientation is always recoverable from
> `ImageOrientationPatient`, which is geometry rather than prose: take the slice normal (§8) and
> see which world axis it points closest to.

### 3.3 The series' "metrics"

A series has no geometry tags of its own — it has **no header**. It is a *folder*, not a file.
Its properties are **derived** by combining the headers of its images:

| Property of the series | Where it comes from |
|---|---|
| Number of slices | count of images sharing the `SeriesInstanceUID` — **18** |
| In-plane size | `Rows` × `Columns` from any image — **512 × 512** |
| In-plane spacing | `PixelSpacing` from any image — **0.488 × 0.488 mm** |
| Slice ordering | sort by `ImagePositionPatient` projected onto the slice normal |
| **Slice spacing** | **difference between consecutive `ImagePositionPatient` values** ← derived, not a tag |
| Origin | `ImagePositionPatient` of the **first** slice |
| Direction | `ImageOrientationPatient` (+ the normal, computed as their cross product) |
| World extent (FOV) | origin + size × spacing, through the direction matrix |

That fifth row is the one that bites people, and §10.1 is devoted to it.

---

## 4. Shared vs varying: the defining table

This single table is the clearest answer to "what is the difference between `SE0` and `IM0`."

| Tag | `IM0` | `IM1` | `IM17` | Same across the series? |
|---|---|---|---|---|
| `SeriesInstanceUID` | `…533.1.4.1` | `…533.1.4.1` | `…533.1.4.1` | ✅ **identical** — this is what *makes* it a series |
| `SeriesNumber` | 450 | 450 | 450 | ✅ identical |
| `Modality` | CT | CT | CT | ✅ identical |
| `PixelSpacing` | 0.4883 | 0.4883 | 0.4883 | ✅ identical |
| `Rows` / `Columns` | 512 | 512 | 512 | ✅ identical |
| `ImageOrientationPatient` | `0.99478, 0.04791, …` | `0.99478, 0.04791, …` | `0.99478, 0.04791, …` | ✅ identical (to ~7 decimals) |
| `RescaleIntercept` | −1024 | −1024 | −1024 | ✅ identical |
| `FrameOfReferenceUID` | `…4170.1` | `…4170.1` | `…4170.1` | ✅ identical |
| | | | | |
| `SOPInstanceUID` | `…533.2` | `…538.3` | `…582.19` | ❌ **unique per image** |
| `InstanceNumber` | 1 | 2 | 18 | ❌ **the slice's ordinal** |
| `ImagePositionPatient` | `(−119.075, −139.659, −29.522)` | `(−118.431, −136.672, −20.819)` | `(−110.242, −98.727, 89.736)` | ❌ **where this slice sits** |
| Pixel data | slice 1 | slice 2 | slice 18 | ❌ different anatomy |

**Read the table this way:**

- Everything in the top block describes **the acquisition** — the settings the scanner used. It
  belongs to the *series*, and it is merely *copied into* every image because DICOM files must
  each stand alone.
- Everything in the bottom block describes **this particular slice**. It belongs to the *image*.

> 🎓 **The key insight.** DICOM has no "series file." Series-level properties are **redundantly
> duplicated into every image header** so that any single file is self-describing. If you delete
> 17 of the 18 files, the survivor still knows its pixel spacing, its orientation, and which
> series it came from. It just no longer knows it had siblings.

### 4.1 Watch the geometry march

The only geometry tag that changes is `ImagePositionPatient`, and here is what it does across
your CT series:

```
IM0   (-119.075, -139.659,  -29.522)   ← the series Origin
IM1   (-118.431, -136.672,  -20.819)   ← moved (0.645, 2.987, 8.703)
…
IM17  (-110.242,  -98.727,   89.736)   ← 18th slice
```

Each step advances along the **slice normal** — the direction perpendicular to the image plane.
Verified numerically: the unit vector of `IPP[1] − IPP[0]` dotted with the computed normal gives
**1.0000**, i.e. perfectly parallel.

That is what a series *is*, geometrically: **the same 2D grid, stepped along its own normal.**

---

## 5. Index space

**Index space is "which box in the grid."** Pure integers, no units.

```python
value = array[z, y, x]        # numpy
pixel = image[x, y, z]        # SimpleITK
```

For your CT series the valid indices are `i ∈ [0, 511]`, `j ∈ [0, 511]`, `k ∈ [0, 17]`.

Index space knows nothing about the patient. Index `(100, 200, 3)` means "column 100, row 200,
slice 3" and that is the entire content of the statement. It does not tell you where in the head
that is, how big the voxel is, or whether the patient was tilted.

### 5.1 The axis-ordering trap

Two libraries, two conventions, and they are **reversed**:

| | Order | Your CT | Your MRI |
|---|---|---|---|
| `image.GetSize()` (SimpleITK) | `(x, y, z)` = `(columns, rows, slices)` | `(512, 512, 18)` | `(208, 320, 18)` |
| `sitk.GetArrayFromImage(image).shape` (numpy) | `(z, y, x)` = `(slices, rows, columns)` | `(18, 512, 512)` | `(18, 320, 208)` |

The CT hides the problem because it is square — 512 × 512 looks the same either way. **The MRI
exposes it:** `Rows = 320`, `Columns = 208`, so `GetSize()` reports `(208, 320, 18)` while the
numpy array is `(18, 320, 208)`.

> ⚠️ Any code that mixes both APIs must convert deliberately. This is why the pipeline writes
> `image[:, :, z]` to extract a slice with SimpleITK (indexing the third axis) but
> `arr[z, :, :]` after converting to numpy (indexing the first). Both mean "slice z."

---

## 6. World space

**World space is "where in the room, in millimetres."** Real physical coordinates, anchored to
the scanner and, by convention, to the patient:

```
+X → patient's LEFT
+Y → patient's POSTERIOR  (back)
+Z → patient's SUPERIOR   (head)
```

This is the **LPS** convention, and it is what DICOM uses. A world point is a triple of
millimetres like `(-76.21, -37.82, -44.01)`.

Four pieces of metadata carry an image from index space into world space:

| Property | Meaning | Your CT `SE0` |
|---|---|---|
| **Origin** | world coordinates of voxel `(0,0,0)` | `(-119.0752, -139.6587, -29.5222)` |
| **Spacing** | mm per voxel step, per axis | `(0.4883, 0.4883, 7.4351)` |
| **Direction** | 3×3 rotation: where the voxel axes point | see §8 |
| **Size** | how many voxels | `(512, 512, 18)` |

Note that **Origin is exactly `IM0`'s `ImagePositionPatient`** — `(-119.0752, -139.6587,
-29.5222)` in both. The volume's origin *is* the first slice's position. Not a coincidence: it is
the definition.

### 6.1 Why world space is the point of the whole exercise

Index space is private to one image. Two different series have unrelated index spaces — voxel
`(100, 200, 3)` in the CT and voxel `(100, 200, 3)` in the MRI are not the same place, are not the
same size, and are not even the same shape of region.

World space is **shared**. Both series describe positions in the same millimetre coordinate
system. That shared language is the only reason you can ask geometric questions across
modalities:

- *Do these two volumes overlap?* → intersect their world bounding boxes.
- *How far apart are their centres?* → subtract their world centre points.
- *Which CT slice corresponds to this MRI slice?* → compare world positions.

Every single one of those questions is **arithmetic on metadata**, answered in microseconds
without touching a pixel. The field-of-view gate (`overlap_fracs`), the volume alignment
(`CenteredTransformInitializer`), and the intersection crop (`compute_intersection_roi`) are all
just world-space box arithmetic.

Here are your two volumes' world extents, computed by transforming all 8 corners:

```
CT   X[ -137.6,  138.0]   Y[ -139.7,  149.0]   Z[ -131.6,   89.7]   centre ( 0.2,  4.7, -20.9)
MRI  X[  -94.6,  101.8]   Y[ -143.4,  172.4]   Z[ -126.0,  100.3]   centre ( 3.6, 14.5, -12.9)

centre offset (MRI − CT) = (3.4, 9.8, 8.1) mm
```

That offset is precisely what step 3 of the registration pipeline corrects. (The pipeline logs
`(+3.6, +10.1, +7.9)` for this pair — the small difference is because it computes the offset
*after* resampling the CT to 1mm in-plane, which nudges the CT grid slightly. Same quantity,
marginally different grid.)

---

## 7. The bridge: the index→world formula, worked out

```
world_point = Origin + Direction · ( Spacing ⊙ index )
```

where `⊙` is element-wise multiplication and `·` is matrix-vector multiplication.

Read it right-to-left as three physical steps:

1. `Spacing ⊙ index` — **scale**: convert voxel counts into millimetres along the image's *own*
   axes.
2. `Direction · (…)` — **rotate**: turn those image-axis millimetres into world-axis millimetres.
3. `Origin + (…)` — **translate**: shift so voxel `(0,0,0)` lands at the right place.

### Example — tiny numbers first, no matrices

Before the real 3D case, do a 2D one where `Direction` is the identity (no tilt), so step 2 does
nothing and you can see the arithmetic bare.

`Origin = (10, 20)`, `Spacing = (2, 5)`, and we want voxel `(3, 4)`:

```
step 1  scale:      Spacing ⊙ index = (2 × 3,  5 × 4)  = (6, 20)
step 2  rotate:     identity, so unchanged             = (6, 20)
step 3  translate:  Origin + (6, 20) = (10 + 6, 20 + 20) = (16, 40)

voxel (3, 4)  →  world (16, 40) mm
```

Note the axes have **different spacings** — 2 mm horizontally, 5 mm vertically. Moving one voxel
right travels 2 mm; moving one voxel down travels 5 mm. This is not a special case, it is the
normal situation: your CT is 0.488 mm in-plane and 7.4 mm between slices, a 15× difference.

**The one thing to take from this:** "one voxel" is not a distance. It is a different distance on
every axis, and a different distance in every series.

### 7.1 Verified on your data

Take CT voxel `(100, 200, 3)`:

```python
O = np.array(ct.GetOrigin())                       # (-119.0752, -139.6587, -29.5222)
S = np.array(ct.GetSpacing())                      # (   0.4883,    0.4883,   7.4351)
D = np.array(ct.GetDirection()).reshape(3, 3)

manual = O + D.dot(S * np.array((100, 200, 3)))
```

```
manual                                       = (-76.2077, -37.8210, -44.0101)
ct.TransformContinuousIndexToPhysicalPoint(…) = (-76.2077, -37.8210, -44.0101)
```

Exact agreement. The SimpleITK call is not magic; it is those three lines.

### 7.2 And back again

```python
idx = image.TransformPhysicalPointToContinuousIndex((-76.2077, -37.8210, -44.0101))
```

which is the inverse: `index = Spacing⁻¹ ⊙ ( Directionᵀ · (world − Origin) )`. It returns
**continuous** (fractional) indices, because an arbitrary world point generally falls *between*
voxels — that fractional part is exactly what interpolation resolves.

> 🎓 **Use these two functions instead of doing the arithmetic yourself.** Not because the
> arithmetic is hard, but because every hand-rolled version eventually forgets the `Direction`
> term, and that failure is silent on axis-aligned data and catastrophic on tilted data.

---

## 8. The Direction matrix, decoded

`GetDirection()` returns 9 numbers, row-major. For your CT:

```
(0.9948, -0.0744,  0.0699,
 0.0479,  0.9449,  0.3238,
-0.0901, -0.3188,  0.9435)
```

reshaped:

```
        ┌                            ┐
        │  0.9948   -0.0744   0.0699 │
   D =  │  0.0479    0.9449   0.3238 │
        │ -0.0901   -0.3188   0.9435 │
        └                            ┘
           ↑          ↑         ↑
        i-axis     j-axis    k-axis
      (row dir)  (col dir)  (normal)
```

**The columns are the unit vectors of the image's own axes, expressed in world coordinates.**

Cross-check against the raw DICOM tag. `ImageOrientationPatient` (0020,0037) stores six numbers
— the first three are the row direction, the next three the column direction:

```
row cosine  = ( 0.9948,  0.0479, -0.0901)   ← column 1 of D ✅
col cosine  = (-0.0744,  0.9449, -0.3188)   ← column 2 of D ✅
```

DICOM does **not** store the third vector. It is computed:

```python
normal = np.cross(row, col)   # = (0.0699, 0.3238, 0.9435)  ← column 3 of D ✅
```

### 8.1 What the numbers tell you about the patient

If the patient were perfectly aligned with the scanner, `D` would be the identity matrix. Yours
is close to identity but not equal to it — the diagonal is `(0.9948, 0.9449, 0.9435)` rather than
`(1, 1, 1)`.

That is a real, physical **gantry tilt / patient tilt**, and you can read it off. The largest
off-diagonal term is `−0.3188`, and `arccos(0.9435) ≈ 19.4°` — this head CT was acquired with a
substantial tilt, which is completely normal for a routine head protocol.

Compare with the MRI of the same patient:

```
CT  normal = ( 0.0699,  0.3238,  0.9435)
MRI normal = (-0.0137,  0.3332,  0.9427)
```

Similar but **not identical** — the two scans were tilted slightly differently, because the
patient was positioned twice. That angular difference is part of what registration must recover.

> ⚠️ This is why `image_processing.resample_mri_to_ct_grid` is wrong to do
> `mri_aligned.SetDirection(ct_image.GetDirection())`. That line does not *align* the MRI — it
> overwrites a genuine measurement of how the patient was lying with a false one, and destroys
> the information needed to do the alignment properly. See `registration_docs.md` §7.3.

---

## 9. How 18 files become one volume

```python
reader = sitk.ImageSeriesReader()
names  = reader.GetGDCMSeriesFileNames(series_path)   # find + SORT the files
reader.SetFileNames(names)
reader.MetaDataDictionaryArrayUpdateOn()
image = reader.Execute()                              # stack into 3D
```

What `Execute()` actually does:

1. **Reads every header.** Collects `ImagePositionPatient` from each file.
2. **Sorts by position, not by filename.** It projects each `ImagePositionPatient` onto the slice
   normal and orders by that. This is important — `GetGDCMSeriesFileNames` returns files in
   *geometric* order, so a file named `IM10` does not have to come after `IM1`.
3. **Takes Origin from the first slice.**
4. **Takes Direction from `ImageOrientationPatient` + the computed normal.**
5. **Computes Spacing[2]** from the distances between consecutive positions — see §10.1, this is
   where it gets interesting.
6. **Applies `RescaleSlope`/`RescaleIntercept`** so the output is in Hounsfield Units, not raw
   stored values.
7. **Stacks the pixel arrays** into one 3D block.

The result is a single `sitk.Image` with the four properties from §6 — and from that moment the
individual files are irrelevant. Everything downstream sees one volume.

### 9.1 The step that involves a lie

Steps 1–4 and 6–7 are lossless bookkeeping. **Step 5 is not.**

A `sitk.Image` has room for exactly **one** number in `Spacing[2]`. It assumes slices are evenly
spaced. If they are not, that assumption cannot be represented — so the reader averages, and
prints a warning you have certainly seen:

```
WARNING: In itkImageSeriesReader.hxx, line 478
Non uniform sampling or missing slices detected, maximum nonuniformity: 3.31761
```

That warning is not noise. §10.1 explains what it is telling you about your data.

---

## 10. Six traps in your data

### 10.1 `SliceThickness` is **not** the distance between slices

This is the biggest one, and your CT demonstrates it dramatically.

```
SliceThickness tag (0018,0050)  : 2.975 mm
Actual gaps between consecutive ImagePositionPatient values:
    9.22  6.84  7.64  8.44  6.84  10.00  5.72  9.56  …
    min 4.117    max 9.997    mean 7.435
SimpleITK reports Spacing[2]    : 7.4351 mm    ← the mean
```

Three different numbers, three different meanings:

| Number | Meaning |
|---|---|
| **2.975 mm** | how thick each imaged slab is — how much tissue contributed to one slice |
| **4.1 – 10.0 mm** | how far apart consecutive slices actually are — **the true geometry** |
| **7.435 mm** | SimpleITK's single-number average, which matches no individual gap |

Your CT has slices 2.975 mm thick placed 4–10 mm apart. **There are gaps of unimaged tissue
between them**, and the gaps are not even consistent.

Now compare the MRI of the same patient:

```
SliceThickness : 5.000 mm
Actual gaps    : 8.00  8.00  8.00  8.00  …   (min 8.000, max 8.000)
```

Perfectly uniform, but still a **3 mm gap** between each 5 mm slab.

**And here is the detail that settles the matter.** There *is* a tag for the number you want —
`SpacingBetweenSlices` (0018,0088). Check whether your files carry it:

| Series | `SpacingBetweenSlices` present? | Value | Measured gaps |
|---|---|---|---|
| MRI `SE0` | ✅ yes | `8` | 8.000 every time — **agrees exactly** |
| CT `SE0` | ❌ **absent** | — | 4.117 – 9.997, irregular |

The MRI declares its spacing and is telling the truth. The CT **does not provide the tag at
all** — which is legal, because the tag is optional. So on your CT there is no tag anywhere in
the file that answers "how far apart are the slices," and `SliceThickness = 2.975` is the nearest
thing that *looks* like an answer while being off by a factor of 2–3.

This is the whole trap in one table: the modality that would have been safe to read the tag from
provides it, and the modality where guessing is catastrophic does not.

> 🎓 **Why this matters far beyond bookkeeping.** This is the hard evidence behind the pipeline's
> refusal to resample along Z:
>
> ```python
> new_sz = orig_sp[2]      # through-plane spacing UNTOUCHED
> ```
>
> Interpolating between slices 4–10 mm apart would invent anatomy across gaps where nothing was
> measured, and the gap width varies unpredictably. It is also why 3D rotation estimation is
> hopeless here (`registration_docs.md` §3.4): 18 irregular samples cannot constrain it.
>
> **Always derive slice spacing from `ImagePositionPatient` differences. Never read
> `SliceThickness` and assume it is the spacing.**

### 10.2 `Rows`/`Columns` vs `GetSize()` are transposed

Covered in §5.1. Your MRI: `Rows = 320`, `Columns = 208`, `GetSize() = (208, 320, 18)`.
`GetSize()[0]` is **Columns**.

### 10.3 Raw pixel values are not Hounsfield Units

`RescaleIntercept = -1024`, `RescaleSlope = 1`, so:

```
HU = stored_value × RescaleSlope + RescaleIntercept
```

`sitk.ImageSeriesReader` applies this for you. If you ever read raw pixel data another way and
your air comes out at 0 instead of −1000, this is why.

MRI has no equivalent — its numbers are arbitrary with no physical scale, which is exactly why
`normalization.compute_mri_percentiles` derives bounds **per volume** rather than using fixed
constants.

### 10.4 `SeriesDescription` may say nothing useful

Yours says `"Processed Images"`. See §3.2.

### 10.5 `SetDirection` mutates in place

`sitk.Image` is a reference type. This:

```python
mri_aligned = mri_image           # NOT a copy - a second name for the same object
mri_aligned.SetDirection(...)     # mutates the caller's image
```

is the aliasing bug in `image_processing.resample_mri_to_ct_grid`. To actually copy:

```python
mri_copy = sitk.Image(mri_image)  # real copy
```

### 10.6 The folder is not the authority

`SE0` under CT and `SE0` under MRI are unrelated series that merely share a folder name. The only
real identity is `SeriesInstanceUID`. Two `SE0` folders being "the same scan" is a convention of
this export, not a fact you can rely on.

---

## 11. Frame of Reference: why registration is necessary at all

Here is the tag that explains the entire registration effort, and you can verify it in one line.

`FrameOfReferenceUID` (0020,0052) declares **which coordinate system these world coordinates
belong to.** Two series sharing a `FrameOfReferenceUID` are guaranteed to be in the same physical
frame — their world coordinates are directly comparable, and no registration is needed.

Your patient `PA0_Ranjeet`:

```
CT  SE0   FrameOfReferenceUID = 1.2.840.113619.2.415.3.2831157761.732.1743565258.432.4170.1
MRI SE0   FrameOfReferenceUID = 1.3.12.2.1107.5.2.6.11111.20250402160726515.0.0.0
```

**Completely different.** Even the manufacturer prefixes differ — `1.2.840.113619` is GE,
`1.3.12.2.1107` is Siemens. Two different machines, each with its own idea of where the origin is.

This is the formal, machine-readable statement of the problem:

> The CT's millimetres and the MRI's millimetres are **both** valid physical coordinates, but they
> are measured from **different reference points**. A point at world `(0, 0, 0)` in the CT frame
> is not the same place in the patient as `(0, 0, 0)` in the MRI frame.

Everything in `registration_docs.md` exists to recover the transformation between those two
frames. And the measured mismatch for this pair — a centre offset of `(3.4, 9.8, 8.1) mm` — is
small only because this is the brain series. For the knee series the same calculation gives
**−86.9 mm** on one axis.

> 🎓 **Check this tag first when debugging alignment.** If two series *do* share a
> `FrameOfReferenceUID` and still appear misaligned, your bug is in the code, not the patient. If
> they do not share one, no amount of metadata arithmetic will align them and you need actual
> registration.

---

## 12. Inspect your own data

### One tag from one image

```python
import SimpleITK as sitk

reader = sitk.ImageFileReader()
reader.SetFileName(r"...\Rawdata_dicom\CT\PA0_Ranjeet\ST0\SE0\IM0")
reader.LoadPrivateTagsOn()
reader.ReadImageInformation()          # header only - does NOT decode pixels, so it is fast

print(reader.GetMetaData("0020|0032"))  # ImagePositionPatient
```

> Tag keys are lowercase hex separated by a pipe: `"0020|0032"`. Uppercase will not match.

### Every tag in an image

```python
for k in reader.GetMetaDataKeys():
    print(k, "=", reader.GetMetaData(k))
```

### The assembled volume's geometry

```python
import io_utils
img, n = io_utils.load_dicom_series(r"...\CT\PA0_Ranjeet\ST0\SE0")
print(img.GetSize(), img.GetSpacing(), img.GetOrigin(), img.GetDirection())
```

### The true slice spacing (the number that is not a tag)

```python
import numpy as np, os

files = sorted(os.listdir(series_dir), key=lambda n: int(n[2:]))   # IM0, IM1, IM2 …
def ipp(f):
    r = sitk.ImageFileReader(); r.SetFileName(os.path.join(series_dir, f)); r.ReadImageInformation()
    return np.array([float(x) for x in r.GetMetaData("0020|0032").split("\\")])

positions = [ipp(f) for f in files]
gaps = [np.linalg.norm(positions[i+1] - positions[i]) for i in range(len(positions) - 1)]
print("min %.3f  max %.3f  mean %.3f" % (min(gaps), max(gaps), np.mean(gaps)))
```

> Note `sorted(..., key=lambda n: int(n[2:]))`. Plain alphabetical sorting gives
> `IM0, IM1, IM10, IM11, …, IM2` — which is wrong. (The pipeline itself is safe here because
> `GetGDCMSeriesFileNames` sorts geometrically, but any manual loop over filenames must handle
> it.)

---

## 13. Tag cheat sheet

**Identity**

| Tag | Name | Level | Note |
|---|---|---|---|
| `(0008,0018)` | SOPInstanceUID | image | unique per file |
| `(0020,000E)` | SeriesInstanceUID | **series** | **defines the series** |
| `(0020,000D)` | StudyInstanceUID | study | one visit |
| `(0020,0013)` | InstanceNumber | image | ordinal, 1-based |
| `(0020,0011)` | SeriesNumber | series | scanner's own numbering |

**Geometry** — the group that matters

| Tag | Name | Level | Note |
|---|---|---|---|
| `(0020,0032)` | ImagePositionPatient | **image** | world position of this slice's first voxel |
| `(0020,0037)` | ImageOrientationPatient | series | 6 numbers: row + column direction cosines |
| `(0028,0030)` | PixelSpacing | series | in-plane mm/voxel |
| `(0018,0050)` | SliceThickness | series | ⚠️ **not** slice spacing |
| `(0018,0088)` | SpacingBetweenSlices | series | the number you actually want — but **optional**: present in your MRI (`8`), **absent in your CT**. Derive it instead |
| `(0020,0052)` | FrameOfReferenceUID | series | which coordinate system |

**Grid and pixel meaning**

| Tag | Name | Note |
|---|---|---|
| `(0028,0010)` | Rows | = `GetSize()[1]` |
| `(0028,0011)` | Columns | = `GetSize()[0]` |
| `(0028,1052)` | RescaleIntercept | −1024 for your CT |
| `(0028,1053)` | RescaleSlope | 1 for your CT |
| `(0028,0004)` | PhotometricInterpretation | `MONOCHROME2` = 0 is black |

**Descriptive** — hints, not facts

| Tag | Name | Note |
|---|---|---|
| `(0008,103E)` | SeriesDescription | free text; yours says `"Processed Images"` |
| `(0018,1030)` | ProtocolName | free text |
| `(0008,0060)` | Modality | `CT` / `MR` — reliable |

---

## Summary

| Question | Answer |
|---|---|
| What is `SE0`? | A **series** — one 3D volume, one acquisition, 18 slice files |
| What defines it? | `SeriesInstanceUID` (0020,000E), identical in all 18 files |
| What is `IM0`? | One **image / instance** — a single 2D slice, header + pixel data |
| What defines it? | `SOPInstanceUID` (0008,0018), unique; positioned by `ImagePositionPatient` |
| Difference? | The series is the **stack**; the image is **one layer**. The series has no file of its own — its properties are duplicated into every image header and its slice spacing is *derived* |
| Index space? | Integer grid coordinates. `(100, 200, 3)`. No units, private to one image |
| World space? | Millimetres in the scanner/patient LPS frame. `(-76.21, -37.82, -44.01)`. **Shared between series** — which is what makes registration possible |
| Bridge? | `world = Origin + Direction · (Spacing ⊙ index)` — verified exactly on your CT above |
| Why register at all? | CT and MRI have **different `FrameOfReferenceUID`s**, so their world coordinates are not the same coordinates |

**Next:** [`registration_docs.md`](registration_docs.md) — what to do about that mismatch.
