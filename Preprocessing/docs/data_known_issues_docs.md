# Known issues in the data

*Recorded 2026-08-08. These are properties of the DICOM files, not bugs in the
registration code. Read this before concluding that a registration result is
wrong — several of the effects below look exactly like a registration failure.*

---

## Summary

| # | Issue | Affects | Status |
|---|---|---|---|
| 1 | Two different copies of the dataset on disk | everything | **resolved** — in-project copy is authoritative |
| 2 | Missing slices in 8 series, hidden by renumbering | 8 series | **accepted, not repaired** (your decision) |
| 3 | **Non-uniform CT slice spacing** | **every CT series** | **open — the important one** |

Issue 3 is much larger than 1 and 2 and was found while investigating them.

---

## Issue 1 — there were two copies of the dataset

Two complete-looking copies existed:

```
C:\Users\moham\MRI_CT_preprocessing_pipeline\Raw_data_mri_ct\Rawdata_dicom     240 series, 4626 files
C:\Users\moham\Downloads\MRI-CT preprocessing pipeline\...\Rawdata_dicom       242 series, 4666 files
```

They are **not** the same data. `pipeline_config.DATA_ROOT` pointed at the
Downloads one, so every registration result produced before 2026-08-08 was
computed on it, while the repository showed the other.

**Resolved.** `DATA_ROOT` is now derived from `__file__`, so it always resolves
to the copy inside the repository and cannot drift again:

```python
DATA_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "Raw_data_mri_ct", "Rawdata_dicom"))
```

`Raw_data_mri_ct` is in `.gitignore`, so nothing about this copy is version
controlled. If you ever restore a machine from git alone, the data has to come
from somewhere else, and this whole class of problem can happen again.

Differences between the copies, for the record:

| series | in-project | Downloads |
|---|---|---|
| CT/PA6_Vijay/SE1 | **7** | 18 |
| MRI/PA6_Vijay/SE1 | **7** | 18 |
| CT/PA13_Brajesh/SE1 | 18 | 19 |
| MRI/PA13_Brajesh/SE2 | 17 | 18 |
| CT/PA40_Kabir/SE2 | 17 | 18 |
| MRI/PA40_Kabir/SE2 | 17 | 18 |
| CT/PA32_Mandbi_ankle/SE0 | 14 | 15 |
| MRI/PA32_Mandbi_ankle/SE0 | 14 | 15 |
| CT/PA41_Anshika/SE0 | *absent* | present |
| MRI/PA41_Anshika/SE0_coronal | *absent* | present |
| MRI/PA43_Chandan/SE1_**sagittal** | present | *spelled* SE1_**saggital** |

Only `PA6_Vijay/SE1` is used by the sweeps, so only shoulder/coronal — 3 of 33
slices — is affected by the switch.

---

## Issue 2 — missing slices, with the evidence renumbered away

Eight series are missing images. What makes this worth writing down is not the
absence but the **renumbering**: the surviving files were renamed to a
contiguous `IM0…IMn`, so nothing in the filenames shows anything is gone.

`MRI/PA6_Vijay/ST0/SE1` is the clearest case. The files are `IM0`–`IM6`, which
looks like a complete 7-slice series. The DICOM headers say otherwise:

```
file        IM0  IM1  IM2  IM3    IM4  IM5    IM6
InstanceNumber 4    5    6    7      9   10     12      <- 8 and 11 are missing
step to next  5.0  5.0  5.0  10.0   5.0  10.0   -  mm
```

It is the same acquisition as the complete copy — identical
`SeriesInstanceUID`, and the 7 surviving images share `SOPInstanceUID`s with 7
of the 18 there.

**Decision: left exactly as is. The data has not been modified.**

The consequence is measurable. A DICOM volume can hold only one slice spacing,
so SimpleITK splits the difference and puts slices where they are not:

```
true spacing        5, 5, 5, 10, 5, 10 mm
SimpleITK assumes   6.667 mm, uniform
placement error     0, 1.67, 3.33, 5.00, 1.67, 3.33, 0 mm      worst: 5.00 mm
```

Measured error for all eight:

| series | files | gaps in InstanceNumber | worst placement error |
|---|---|---|---|
| CT/PA6_Vijay/SE1 | 7 | yes (missing 14) | 1.94 mm |
| MRI/PA6_Vijay/SE1 | 7 | yes (missing 8, 11) | **5.00 mm** |
| CT/PA13_Brajesh/SE1 | 18 | yes (missing 18) | **7.10 mm** |
| MRI/PA13_Brajesh/SE2 | 17 | yes (missing 17) | **5.44 mm** |
| CT/PA40_Kabir/SE2 | 17 | **no** | **26.56 mm** |
| MRI/PA40_Kabir/SE2 | 17 | no | 0.00 mm |
| CT/PA32_Mandbi_ankle/SE0 | 14 | no | 3.03 mm |
| MRI/PA32_Mandbi_ankle/SE0 | 14 | no | 0.00 mm |

Look at `CT/PA40_Kabir/SE2`: **complete numbering, no gaps, and 26.56 mm of
error.** Missing slices cannot explain that. Which leads to the real problem.

---

## Issue 3 — every CT series has non-uniform slice spacing

This is the one that matters, and it has nothing to do with either copy.

Measured across all 22 series the sweeps use — 11 CT and 11 MRI:

| series | slices | gaps? | step min…max | worst error |
|---|---|---|---|---|
| CT/PA32_Mandbi_knee/SE3 (knee axial) | 21 | no | 1.88 … 18.75 | **16.44 mm** |
| CT/PA32_Mandbi_knee/SE5 (knee coronal) | 18 | no | 2.16 … 14.87 | **14.05 mm** |
| CT/PA0_Ranjeet/SE2 (brain sagittal) | 18 | no | 2.44 … 12.70 | **13.13 mm** |
| CT/PA0_Ranjeet/SE1 (brain coronal) | 18 | yes | 4.56 … 13.65 | **7.91 mm** |
| CT/PA0_Ranjeet/SE0 (brain axial) | 18 | no | 4.12 … 10.00 | **7.36 mm** |
| CT/PA32_Mandbi_knee/SE4 (knee sagittal) | 18 | no | 2.46 … 12.41 | **7.09 mm** |
| CT/PA18_Sangeeta/SE1 (spine coronal) | 10 | no | 1.96 … 10.71 | **5.69 mm** |
| MRI/PA6_Vijay/SE1 (shoulder coronal) | 7 | yes | 5.00 … 10.00 | **5.00 mm** |
| CT/PA6_Vijay/SE0 (shoulder axial) | 18 | no | 3.75 … 7.50 | **4.89 mm** |
| CT/PA18_Sangeeta/SE0 (spine sagittal) | 9 | no | 3.55 … 8.29 | **4.59 mm** |
| CT/PA6_Vijay/SE2 (shoulder sagittal) | 18 | no | 3.05 … 6.85 | **4.31 mm** |
| CT/PA6_Vijay/SE1 (shoulder coronal) | 7 | yes | 4.73 … 9.55 | **1.94 mm** |
| **all 10 remaining MRI series** | — | no | **constant** | **0.00 mm** |

### The pattern

**Every CT series is non-uniform. Every MRI series is exactly uniform.**
Eleven out of eleven, both ways. And most of the CT series have **no gaps in
their instance numbering**, so files are not missing — the slices were simply
reconstructed at unevenly spaced positions.

The CT `SeriesDescription` is `Processed Images` throughout, which fits: these
are reformatted or exported images rather than a raw uniform reconstruction.

### Why it matters

A DICOM volume carries one slice spacing. When the real spacing varies between
1.88 mm and 18.75 mm, that single number is wrong nearly everywhere, and
SimpleITK places slices at evenly spaced positions the scanner never used.
**Slices end up as much as 16 mm from where their own headers say they are.**

What that reaches:

- **`registration_demo_sweep_v3.py` most of all.** Its whole design rests on
  world geometry — the 8-corner bounding box, the reported field-of-view
  overlap, the closed-form `GEOMETRY` translation, and the intersection crop.
  Every one of those reads a z axis that is wrong by up to 16 mm.
- **Resampling the MRI onto the CT grid** samples the MRI at those wrong
  z positions, so a "slice 9" pairing can be showing anatomy a centimetre away.
- **Slice pairing in `sweep_og.py` and `sweep_idea.py`.** Both pair CT slice `z`
  with the MRI slice at the same fractional depth, which assumes both stacks are
  evenly sampled. The MRI stacks are. The CT stacks are not.

What it does **not** reach:

- **Anything computed inside one slice.** The 2D registrations, the in-plane
  1 mm resampling, NMI, the shift search — all work on a single slice at a time
  and never consult the z axis. `registration_idea.py` results are unaffected
  except through which slices got paired.

### What it explains

The `Non uniform sampling or missing slices detected` warning that appears
throughout every sweep log is ITK reporting exactly this. It was treated as
noise. It is not noise — it is this table.

It is also a better explanation than "registration failed" for several results
already recorded: a CT slice and an MRI slice that are supposed to be the same
place but are a centimetre apart cannot be aligned by any transform, and the
optimiser will do something desperate instead. Before blaming a gate or a
metric for a bad slice, check this table first.

---

## How to check any series yourself

```python
import numpy as np, glob, os, pydicom, io_utils

ims = sorted(glob.glob(os.path.join(series_dir, "IM*")),
             key=lambda p: int(os.path.basename(p)[2:]))
hs  = [pydicom.dcmread(p, stop_before_pixels=True) for p in ims]

# real positions, from the headers
P = np.array([[float(v) for v in h.ImagePositionPatient] for h in hs])
steps = np.linalg.norm(np.diff(P, axis=0), axis=1)
print("true steps:", steps.round(2))          # all equal = healthy

# where SimpleITK actually puts them
vol, _ = io_utils.load_dicom_series(series_dir)
Q = np.array([vol.TransformIndexToPhysicalPoint((0, 0, k))
              for k in range(vol.GetSize()[2])])
if np.linalg.norm(Q[0] - P[0]) > np.linalg.norm(Q[0] - P[-1]):
    Q = Q[::-1]
print("worst placement error:", np.linalg.norm(P - Q, axis=1).max().round(2), "mm")

# and are any images missing?
inst = sorted(int(h.InstanceNumber) for h in hs)
print("gaps in InstanceNumber:", inst != list(range(min(inst), max(inst) + 1)))
```

Three healthy signs: `steps` all equal, placement error 0.00, no gaps.

---

## Suggested order of work

1. **Pair slices by world position, not by index.** Both 2D sweeps currently use
   fractional depth, which is only valid for evenly sampled stacks. Using
   `ImagePositionPatient` to find, for each CT slice, the MRI slice physically
   nearest it removes the assumption entirely and is the single highest-value
   fix on this page. It also makes issue 3 mostly harmless for those two sweeps.
2. **Decide what `sweep_v3` should do about a non-uniform CT.** Its 3D geometry
   cannot be trusted on these volumes as loaded. The options are to resample
   each CT to a genuinely uniform grid using the true per-slice positions, or to
   restrict its 3D reasoning to series that are already uniform — of which,
   among the sweep's eleven, there are none.
3. **Leave issues 1 and 2 alone.** Both are recorded, bounded, and understood.
