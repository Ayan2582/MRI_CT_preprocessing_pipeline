# 🧲 MRI Pipeline — End-to-End Walkthrough

This document traces the **complete journey of a single MRI series**, from a folder of raw `.dcm` files on disk to a stack of normalised float32 `.npy` slices ready for a GAN dataloader.

> **Slices are saved at their native post-resample size, not a fixed square.** Cropping was deliberately moved out of this pipeline and into the GAN's dataloader — see §12.

Where the existing `*_docs.md` files explain **one module at a time**, this document follows **one modality through every module**, in execution order.

For the CT counterpart, see [`ct_pipeline_docs.md`](./ct_pipeline_docs.md).

---

## 0. The One-Sentence Summary

> The MRI is loaded → its uneven scanner illumination is flattened (N4) → it is **projected onto the CT's physical grid** so it is pixel-for-pixel aligned with the CT → its intensities are rescaled to `[0, 1]` using per-volume percentiles → empty slices are dropped → each slice is saved at its native size.

The single most important idea: **the MRI never gets its own grid.** The CT defines the geometry; the MRI is resampled *into* it. This is what guarantees that `ct_..._007.npy` and `mri_..._007.npy` describe the exact same physical cube of the patient.

---

## 1. Pipeline Map (MRI path only)

```
 preprocess_2d.py : main()
        │
        ├─ io_utils.discover_series(MRI/<patient>/ST0)     ← Stage 1: Discovery
        │      └─ io_utils.load_dicom_series(SE*)          ← Stage 2: Loading
        │
        ├─ strict-name pairing loop (preprocess_2d.py:230) ← Stage 3: Pairing
        │
        └─ pipeline_core.process_orientation_pair()
                 │
                 ├─ image_processing.apply_n4_bias_correction()   ← Stage 4  (MRI ONLY)
                 ├─ image_processing.resample_mri_to_ct_grid()    ← Stage 5
                 ├─ image_processing.volume_to_slices()           ← Stage 6
                 ├─ normalization.compute_mri_percentiles()       ← Stage 7
                 ├─ image_processing.estimate_volume_translation() ← Stage 8  (optional)
                 │      └─ registration_idea.register()  × N probe slices
                 │
                 └─ per-slice loop:
                        ├─ image_processing.apply_translation()   ← Stage 8  (optional)
                        ├─ normalization.normalize_mri_slice()    ← Stage 9
                        ├─ normalization.is_background_slice()    ← Stage 10
                        │  (Stage 11 crop/pad — REMOVED, see §12)
                        └─ export_utils.save_npy() / save_preview_png()  ← Stage 12
```

---

## 2. Stage 1 — Series Discovery

**Code:** `io_utils.discover_series()` · called from `preprocess_2d.py:213`

The pipeline walks `Raw_data_mri_ct/Rawdata_dicom/MRI/<patient_id>/ST0/` and iterates its `SE*` sub-folders in sorted order. For each one:

| Check | Behaviour |
|---|---|
| Not a directory | skipped |
| DICOM read failed | skipped (warning logged) |
| `n_slices < 2` | skipped — this filters out single-frame **scout / localiser** images |

Each surviving series becomes a dictionary:

```python
{
    "path":        ".../MRI/PA0_Ranjeet/ST0/SE0",
    "image":       <SimpleITK.Image>,   # the full 3D volume, already in memory
    "n_slices":    18,
    "orientation": "axial",
    "series_desc": "t2_tse_tra",
}
```

> ⚠️ **Memory note:** `discover_series` loads **every** MRI series of a patient into RAM before any processing starts. For patients like `PA32` (6 MRI + 6 CT series) this is the peak-memory moment of the run.

### 2.1 How orientation is decided

Orientation resolution is a two-step heuristic (`io_utils.py:128–140`):

1. **Folder name first** — if the `SE*` directory name contains `axial`, `coronal`, or `sagittal`, that wins.
2. **DICOM description fallback** — otherwise `get_orientation_from_desc()` lowercases the `Series Description` (tag `0008|103E`) and pattern-matches:
   * axial ← `_tra`, `_ax`, `axial`, `transv`, `tra_`
   * coronal ← `_cor`, `cor_`
   * sagittal ← `_sag`, `sag_`
3. Anything else → `"unknown"`.

Orientation is therefore derived from *naming conventions*, not from geometry. A geometrically correct alternative — reading the DICOM direction-cosine matrix — existed as `get_orientation_from_direction()` but was never wired in, and has since been removed (see [`io_utils_docs.md`](./io_utils_docs.md) § Removed helpers). It is worth reviving if string-based detection proves unreliable.

> 🔑 **The MRI is the orientation authority.** In `preprocess_2d.py:240` the orientation of the *pair* is taken from the MRI entry (`orient = m_entry["orientation"]`); the CT's own orientation label is never consulted. If the MRI's orientation resolves to `"unknown"`, the whole pair is skipped with a warning.

---

## 3. Stage 2 — DICOM Loading

**Code:** `io_utils.load_dicom_series()`

Uses `sitk.ImageSeriesReader` with `GetGDCMSeriesFileNames()`, which sorts the individual `IM*` files into correct anatomical order using their DICOM position tags (not filename order — `IM10` must not land between `IM1` and `IM2`).

`MetaDataDictionaryArrayUpdateOn()` and `LoadPrivateTagsOn()` are enabled so the `Series Description` can be read back and stashed onto the image object as a custom `series_desc` metadata key.

**Output:** a 3D `SimpleITK.Image` whose voxel values are the **raw scanner intensities**. This matters — unlike CT Hounsfield Units, MRI intensities have **no physical unit and no cross-patient meaning**. A value of `600` in one scan and `600` in another are unrelated. Everything downstream in Stage 7/9 exists to work around this.

---

## 4. Stage 3 — CT ↔ MRI Pairing

**Code:** `preprocess_2d.py:230–248`

For each MRI series, the pipeline takes the folder basename, splits on `_`, and looks for a CT series whose basename splits to the same token:

```
MRI  SE0      →  base "SE0"
CT   SE0_1    →  base "SE0"   ✔ match
```

This "strict name match" assumes the two modalities were exported with a consistent series-numbering convention. If no CT shares the base name, that MRI series is silently dropped; if **no** MRI series in the patient finds a partner, the patient is logged as `No identically named CT and MRI folders found - skipped.`

> ℹ️ `io_utils.select_best_series()` (pick the series with the most slices per orientation) and `get_all_valid_series()` used to exist as unused helpers; both have been removed. Strict name matching is the only pairing path.

---

## 5. Stage 4 — N4 Bias Field Correction ⭐ *MRI-only*

**Code:** `image_processing.apply_n4_bias_correction()` · called from `pipeline_core.py`

This stage has **no CT equivalent** — it corrects an artefact that only exists in MR physics.

> **Changed 2026-08-08.** This stage used to run in 2D, one independent fit per
> slice. It now fits **one field to the whole 3D volume**, with a deliberately
> anisotropic control point mesh. The rest of this section describes the new
> behaviour; §5.4 records what was wrong with the old one.

### 5.1 What it fixes
MRI receive coils are not uniformly sensitive across their field of view. The result is a smooth, low-frequency intensity gradient — the same tissue appears brighter near the coil and darker far from it. This "bias field" is multiplicative:

```
observed(x) = true(x) · exp(bias(x)) · noise
```

Left uncorrected, a percentile-based normaliser (Stage 7) computes bounds that are wrong for half the image, and a GAN learns to reproduce the coil shading instead of the anatomy.

The word doing the work is **x**. It is a point in the bore, not a point in a slice. `bias` is one continuous function over the whole volume, so it has to be estimated as one.

### 5.2 How it is done here

| Parameter | Value | Reason |
|---|---|---|
| Mode | **3D, whole volume** | The bias field is a property of the coil, which does not know where the slice boundaries are. |
| Mask | `sitk.OtsuThreshold(volume, 0, 1, 200)` | Auto-separates tissue from air. Computed on the **volume**: a nearly-empty slice has no bimodal histogram, and a per-slice Otsu on one calls its own noise "tissue". |
| Control points, in-plane | derived per series, **6–20** | Targets a fixed *physical* spacing (25–35 mm), so a 180 mm knee and a 400 mm abdomen get the same field stiffness rather than the same number. |
| Control points, through-plane | **4**, fixed | The fewest a cubic spline can have — one single span across the slab. See §5.3. |
| `shrink_factor` | `4`, **in-plane only** (CLI: `--n4_shrink`) | The field is smooth, so estimating it on a 4× downsampled grid is ~16× faster with negligible loss. Auto-reduced if it would leave fewer than 2 voxels per control point. z is **never** shrunk — these stacks are 15–24 slices, and shrinking z by 4 leaves nothing to fit across. |
| Spline order | `3` (cubic) | ITK's order is global — it cannot differ per axis. All the anisotropy therefore has to come from the control point counts. |
| Fitting levels | `1` × 100 iterations | ITK doubles the mesh on every extra level, **in all axes at once**. More than one level makes it impossible to keep z coarse while refining in-plane. One level makes the configured numbers mean exactly what they say. |
| Convergence | `0.001` | Early-exit threshold. |

The correction itself:

1. Fit N4 on the **in-plane-shrunken** volume + mask.
2. Ask for the log bias field evaluated at **full resolution** — `GetLogBiasFieldAsImage(image_f32)`. The field is a smooth analytic B-spline, so this is an exact evaluation, not an upsample. Estimate cheap, apply sharp.
3. Divide: `corrected = image_f32 / sitk.Exp(log_bias)`.

If N4 throws, the exception is caught and the **whole volume is used uncorrected** with a warning. This is now all-or-nothing per volume on purpose: a stack where some slices were corrected and some were not is worse than one where none were, because the inconsistency is invisible downstream.

### 5.3 Why the mesh is anisotropic

This is the part that makes 3D safe on stacks this thin, and it is the answer to the objection that used to justify the 2D version.

In SimpleITK index order, **axis 2 is the slice axis for every DICOM series**, so "in-plane" is always axes 0 and 1 and needs no lookup. What *does* depend on the acquisition plane is which anatomical direction each axis carries:

| orientation | axis 0 | axis 1 | axis 2 (through-plane) |
|---|---|---|---|
| axial | L-R | A-P | S-I |
| coronal | L-R | S-I | A-P |
| sagittal | A-P | S-I | L-R |

In-plane targets are set per anatomical direction, so a direction gets the same stiffness whichever plane it shows up in (`pipeline_config.N4_CONTROL_POINT_SPACING_MM`):

* **L-R — 35 mm.** Body/spine coil shading across the patient is broad and roughly symmetric; least freedom needed.
* **A-P — 30 mm.** Anterior array against posterior spine coil is the strongest single gradient in most of these scans.
* **S-I — 25 mm.** Coil arrays are segmented along the bore axis, so sensitivity changes fastest head-to-foot; most freedom needed.

Through-plane gets **one span, always**. The reasoning is that these are 2D multi-slice acquisitions with 5–10 mm slices, where through-plane intensity variation is largely *not* a bias field — it is slice profile, cross-talk and per-slice excitation. Those are genuinely discontinuous between neighbouring slices, and any mesh flexible enough to follow them is flexible enough to follow anatomy. One rigid cubic span lets N4 remove a smooth head-to-foot coil falloff and express nothing sharper.

That also disposes of the old worry that "a 3D B-spline fit destabilises on an 18-slice stack". It destabilises when the z mesh has more freedom than the stack has slices. Here the z mesh has the least freedom a spline can have — 4 control points against 15–24 slices — so there is nothing to destabilise. Verified to run without error down to 2-slice volumes.

The counts are **derived per series, not fixed**, because the MRI in this dataset spans 180 mm (knee sagittal) to 400 mm (abdomen axial) of in-plane FOV. A fixed count would mean a 15 mm mesh on the knee and a 33 mm mesh on the abdomen — two completely different amounts of freedom, and the 15 mm one is well into the range where N4 starts flattening real tissue contrast. For a spline of order `p`, `ncp` control points give `ncp - p` spans, so:

```
ncp = p + round(FOV_mm / target_spacing_mm)      clamped to [6, 20]
```

`plan_n4_control_points()` does this and is importable on its own if you want to inspect the mesh for a series without running the fit. The chosen mesh is logged per series at INFO.

> ⚠️ The mm targets are reasoned from coil geometry, **not** tuned against a measured criterion on this dataset. They are the first knob to turn if N4 is visibly eating anatomy (raise them) or leaving shading behind (lower them).

### 5.4 What was wrong with the 2D version

Every slice got its own independent bias field, and a bias field includes an overall scale. So every slice was free to choose its own brightness, with nothing tying it to its neighbours.

Two consequences:

1. **It manufactured slice-to-slice steps.** Adjacent slices that agreed in the raw data could come out at different brightness, purely as an artefact of two independent fits.
2. **It could not see, let alone remove, through-plane shading.** A head-to-foot coil falloff is invisible to any single-slice fit — within one slice it is a constant, and a constant is exactly what a per-slice fit absorbs into that slice's own scale. So the one component of the field that most needs a volume to detect was the one component guaranteed to survive.

Both feed straight into Stage 7, which computes normalisation percentiles over the whole volume: a stack with manufactured brightness steps produces percentiles that fit no slice properly.

The old code also needed a workaround that no longer exists — see §5.5.

#### Measured, 12 series across 4 patients, all three orientations

The artefact in (1) is measurable directly, and separately from anatomy. For each slice `k`, the effective gain N4 applied is `g_k = mean(corrected_k) / mean(raw_k)` over tissue voxels — real anatomy cancels in that ratio. A legitimate 3D field makes `g_k` vary smoothly with `k`; independent per-slice fits make it jump. Score it as `mean|g_k − g_{k−1}| / mean(g)`:

| | old (2D per-slice) | new (3D) |
|---|---|---|
| median gain roughness | 0.0668 | **0.0081** |
| series where the other one wins | — | **0 / 12** |

An **8× reduction, on every series tested**, with the worst case (`PA11_Shivam/SE1`, sagittal) going from 0.196 to 0.020. That is the change this rewrite was for.

> ⚠️ **The second half of the measurement is not a clean win, and should not be reported as one.** Coefficient of variation of tissue intensity — a crude proxy for "how much shading is left" — comes out essentially unchanged overall (median 0.334 old vs 0.335 new), with the new version better on 5 of 12 series and worse on 7.
>
> The split is systematic and makes sense. The large-FOV abdomen series improve substantially (`PA12_Mamta/SE2` 0.350 → 0.205, `SE1` 0.273 → 0.240) — those are exactly the volumes with strong A-P shading from body array against spine coil, which no 2D fit can see. The brain series get slightly worse (`PA0_Ranjeet/SE0` 0.254 → 0.264), and on `PA11_Shivam/SE0` the 3D fit leaves CV *above* the raw volume (0.578 raw → 0.612).
>
> Two things to keep in mind before reading that as a regression. First, CV is a bad criterion on its own: 18 independent 2D fields have far more total freedom than one 3D field, and some of the variance they remove is real tissue contrast, which is a loss dressed up as an improvement. Second, `PA11_Shivam/SE0` is worth actually looking at rather than explaining away — it is the one case here where the correction demonstrably made a volume less uniform than it started.

### 5.5 Retired: the `JoinSeries` geometry trap

The 2D version took the volume apart and put it back together with `sitk.JoinSeries()`, which refuses to merge 2D images whose origin/direction differ by even a floating-point epsilon. It defended against this by capturing `base_origin` / `base_direction` at `z == 0` and force-stamping them onto every corrected slice, then copying the original 3D geometry back onto the joined volume.

None of that exists any more. The volume is never disassembled, so its geometry is never at risk. Kept here only so the removal is not mistaken for an oversight.

---

## 6. Stage 5 — Projection onto the CT Grid ⭐ *the alignment step*

**Code:** `image_processing.resample_mri_to_ct_grid()` · called from `pipeline_core.py:100`

The N4-corrected MRI is resampled with **the already-resampled CT volume as the reference image**:

```python
resampler.SetReferenceImage(ct_image)      # size, spacing, origin, direction — all inherited
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(0.0)        # out-of-FOV → black
resampler.SetTransform(sitk.Transform())   # identity: trust the DICOM coordinates
```

Consequences worth internalising:

* The MRI ends up with **exactly the CT's voxel count and slice count**, which is why `pipeline_core.py:111` can assume `len(ct_slices) == len(mri_slices)` and index them with a single loop counter.
* The MRI inherits the CT's **1.0 mm in-plane spacing** — `--target_spacing` is never applied to the MRI directly.
* The identity transform means alignment relies **entirely on the DICOM patient-coordinate system** being consistent between the two scanners. There is no intensity-driven 3D registration.
* MRI voxels that fall outside the CT's field of view are simply lost; CT regions the MRI never covered are filled with `0.0`.

### The direction-cosine override

```python
mri_aligned.SetDirection(ct_image.GetDirection())   # image_processing.py:279
```

This *asserts* that the MRI shares the CT's orientation rather than letting the resampler reconcile a small angular difference. A sub-degree tilt discrepancy, resolved across a stack only 18 slices deep, produces visible shear/staircase artefacts. Forcing the direction trades a tiny, uncorrected rotation for a geometrically clean volume.

> ⚠️ **Caveat:** this line mutates the object in place. `mri_aligned = mri_image` is a reference, not a copy — the N4-corrected volume's direction matrix is modified as a side effect. Harmless in the current flow (the volume is not reused afterwards), but worth knowing before refactoring.

---

## 7. Stage 6 — Volume → Slices

**Code:** `image_processing.volume_to_slices()` · `pipeline_core.py:107`

`sitk.GetArrayFromImage()` converts `(x, y, z)` ITK ordering into NumPy `(z, y, x)` ordering, then the list comprehension yields one `(y, x)` array per slice. From this point on the MRI is **plain NumPy**, not SimpleITK.

---

## 8. Stage 7 — Percentile Bounds ⭐ *MRI-only*

**Code:** `normalization.compute_mri_percentiles()` · `pipeline_core.py:118`

Because MRI has no absolute intensity scale, normalisation bounds must be derived **per series**, not from a fixed constant.

```python
mri_vol = sitk.GetArrayFromImage(mri_res)          # the whole post-N4, post-resample volume
p1, p99 = norm.compute_mri_percentiles(mri_vol, 0.5, 99.5)
```

Two design points:

1. **Computed on the full 3D volume, not per slice.** If each slice were normalised against its own min/max, a slice through the middle of the brain and a slice through the skull vertex would end up with incompatible contrast — the resulting stack would flicker, and a 2D GAN would learn that flicker.
2. **Zero voxels are excluded** (`volume_arr[volume_arr > 0]`). Background air dominates the histogram; including it would drag `p_low` to 0 and compress all real tissue contrast into the top of the range. Note that this also excludes the zero-fill introduced by Stage 5, which is the desired behaviour.

Fallback: if fewer than 100 non-zero voxels exist, it falls back to the full array so `np.percentile` cannot fail on an empty input.

Defaults `0.5` / `99.5` (`--mri_p_low`, `--mri_p_high`) clip the extreme tails — flow artefacts, fat-sat spikes, metal-adjacent hyperintensities — without touching real tissue.

---

## 9. Stage 8 — Optional Translation Registration

**Code:** `image_processing.estimate_volume_translation()` + `apply_translation()` · method in `registration_idea.py` · flag `--register_2d`

Off by default. Stage 5 already aligns the two modalities through their DICOM
patient coordinates; this stage exists for the pairs where those coordinates are
wrong. It measures the leftover in-plane offset and takes it out.

**The method** (`registration_idea.py`): both images are already at 1 mm per
pixel by Stage 5, so slide the MRI over the CT one whole pixel at a time and
keep the position with the best normalised mutual information. The search is
coarse-to-fine — a strided sweep, then every whole pixel around the best few.

Two properties fall out of that and neither is a tuning choice:

* **A whole-pixel slide cannot rotate, scale or shear**, so the three failure
  modes in `registration_gates_docs.md` are not gated against here — they are
  not expressible. This is the difference from the old gradient-descent
  `Euler2DTransform` approach, which could express all three and needed gates.
* **There are no random numbers**, so two runs over the same data give the same
  shift. The old approach sampled 20 % of pixels at random with no seed.

### One shift per volume, not one per slice

The shift is estimated on `REG_N_PROBES` slices spread through the stack and
then applied to **every** slice of it. Registering slices independently is the
same mistake the 3D N4 rewrite removed (§5), in a different variable: it hands
each slice its own free translation, so the MRI shears through z relative to the
CT and continuous anatomy comes out as a staircase. On this dataset the best
per-slice shift swings 85 mm across one shoulder axial stack — see
`sweep_idea_2_summary.csv`.

### How the answer is defended

Four checks, in order. Any failure means **no shift at all**, never a worse one:

| # | Check | Config |
|---|---|---|
| 0 | A probe whose best shift lands on the edge of the search square is discarded — that is a wall, not a peak, so the value is censored rather than measured | `REG_SEARCH_MM` |
| 1 | Enough probes survived check 0 | `REG_MIN_PROBES` |
| 2 | The survivors agree with each other | `REG_MAX_SPREAD_MM` |
| 3 | Their median, re-scored on every probe, actually raises NMI on average | `REG_MIN_GAIN` |

Check 0 has to come first rather than being a warning after the fact: probes
pinned against the same wall all report the same number, so they would pass
check 2 with a spread of zero and unanimous censorship would read as unanimous
evidence. When it fires, the log says so and names the fix (raise
`--reg_search_mm`).

Check 3 is the one that matters most. Checks 1 and 2 ask whether the per-slice
searches agreed; check 3 measures the shift that is actually about to be
applied, which — being a median — may be a position no probe ever proposed.

### What lands in the CSV

Every row carries `reg_applied`, `reg_dx_mm`, `reg_dy_mm`, `reg_nmi_gain` and
`reg_note`. The measured shift is recorded **even when it was rejected**, and
`reg_note` is the plain-English reason. Nobody is going to inspect 2313 overlays
by hand, so this is what keeps a bad registration findable afterwards instead of
baked silently into the `.npy`.

---

## 10. Stage 9 — Intensity Normalisation

**Code:** `normalization.normalize_mri_slice()` · `pipeline_core.py:143`

```python
s = np.clip(slice_2d, p1, p99)
return (s - p1) / (p99 - p1)          # → [0.0, 1.0]
```

Guard: if `p99 - p1 < 1e-8` the slice has no contrast, and an all-zero array is returned instead of dividing by zero.

The same `p1`/`p99` are reused for **every slice in the series**, which is what preserves inter-slice consistency.

---

## 11. Stage 10 — Background Flagging

**Code:** `normalization.is_background_slice()` · `pipeline_core.py:152–154`

```python
np.mean(arr <= 0.02) > 0.90
```

A slice is flagged if **more than 90 % of its pixels sit below normalised intensity 0.02**.

The check is applied to the CT **and** the MRI, joined with `or` — **if either side is background, the pair is flagged.** The pair is still saved: a slice near the FOV edge can carry a thin sliver of real anatomy, and a fixed 90% threshold can't tell that apart from true empty air, so nothing is discarded here. Instead the pair is tagged `is_background=True` in `metadata.csv`, counted into `n_flagged_bg`, and reported per-orientation and in the final summary — filtering, if wanted, is a dataloader-level decision.

Tunable via `--bg_thresh` and `--bg_fraction`.

---

## 12. Stage 11 — Cropping · ⛔ REMOVED

**This stage no longer runs.** `center_crop_pad()` is no longer called from `pipeline_core.py`; slices are written at whatever size Stage 5 produced.

### Why it was removed

Cropping is a *training-time* decision, not a preprocessing one. Baking a fixed 256²/384² centre crop into the `.npy` files made three things impossible without a full re-run of the pipeline (~minutes per patient, dominated by N4):

* switching to random-crop augmentation, or to a body-mask-driven crop;
* changing the input resolution of the GAN;
* recovering anatomy that a centre crop had already clipped.

Measurements across the dataset made the cost concrete: MSK frames were up to **50 % constant zero padding**, which also silently inflates any full-frame L1/PSNR, while all 7 spine series and 10 MSK CT series were being *cropped* — up to 54 mm off each edge — with nothing logged.

### What this changes for you

* `.npy` arrays now have **variable shape**, so `torch.stack` will fail on a naive batch. The `Dataset` must crop, pad, or resize to a common size in `__getitem__`, or you must supply a custom `collate_fn`.
* **CT and MRI of the same pair still share an identical shape** (Stage 5 guarantees it), so a single crop applied to both is still exactly aligned.
* `metadata.csv` gains **`height`** and **`width`** columns so the dataloader can size batches and filter out slices smaller than its crop target without opening a single `.npy`.
* `normalization.center_crop_pad()` was **deleted**, along with the `target_size` config, the `--target_size` flag, and the `crop_size` CSV column. Crop sizing is entirely the dataloader's business now — the pipeline records what it produced (`height`/`width`) and nothing about what you should do with it.

### Reference implementation for your `Dataset`

Self-contained, so the GAN project needs no import from this one:

```python
def center_crop_pad(arr, target):
    """Symmetrically pad (if smaller) then centre-crop to (target, target)."""
    ph = max(0, target - arr.shape[0])
    pw = max(0, target - arr.shape[1])
    if ph or pw:
        arr = np.pad(arr, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)),
                     mode="constant", constant_values=0.0)
    h, w = arr.shape
    sh, sw = (h - target) // 2, (w - target) // 2
    return arr[sh:sh + target, sw:sw + target]


class PairedSliceDataset(Dataset):
    def __getitem__(self, i):
        row = self.rows[i]
        ct  = center_crop_pad(np.load(row["ct_npy"]),  self.crop)
        mri = center_crop_pad(np.load(row["mri_npy"]), self.crop)
        return torch.from_numpy(mri)[None], torch.from_numpy(ct)[None]
```

Padding with `0.0` is correct: because it runs on already-normalised arrays and air clips to `0.0` under every profile, the border is indistinguishable from real air.

The sizes the pipeline used to apply — **256 px** for brain/MSK/spine, **384 px** for abdomen — are documented with their rationale in [`ct_pipeline_docs.md`](./ct_pipeline_docs.md) §4, if you want to reproduce them via `body_region`.

> ⚠️ Whatever crop you choose, **derive one offset and apply it to both modalities.** Cropping CT and MRI independently (two random offsets, or two separate body masks) destroys the pixel-level correspondence the entire dataset exists to provide.

---

## 13. Region Awareness (what the MRI inherits)

**Code:** `preprocess_2d.py:167–182` · `pipeline_config.REGION_PROFILES` / `PREFIX_TO_REGION`

Before each patient, the ID prefix (`PA0_Ranjeet` → `PA0`) is looked up to select a region profile, which then **overwrites `args` in place**:

| Region | CT window (HU) | Patients |
|---|---|---|
| `brain` | `[0, 80]` | 15 |
| `abdomen` | `[-160, 240]` | 16 |
| `musculoskeletal` | `[-200, 300]` | 10 |
| `spine` | `[-200, 300]` | 4 |
| `default` | `[-200, 300]` | unmapped IDs |

**Nothing in this table affects the MRI's pixel data** — the HU window is CT-specific, and the profiles no longer carry a `target_size` (see §12). The MRI's own normalisation is percentile-driven and therefore already region-agnostic by construction. The only region signal the MRI carries into training is the `body_region` column in `metadata.csv`.

> ⚠️ **Gotcha:** because `args.ct_win_min` and `args.ct_win_max` are reassigned per patient, the CLI flags `--ct_win_min` and `--ct_win_max` are **silently overridden** on every patient that has a region mapping. To change them, edit `REGION_PROFILES` in `pipeline_config.py`.

---

## 14. Stage 12 — Export

**Code:** `export_utils.save_npy()` / `save_preview_png()` · `pipeline_core.py:178–186`

```
<output_dir>/<patient_id>/<orientation>/mri/ct_<CTSERIES>_mri_<MRISERIES>_<idx:03d>.npy
<output_dir>/<patient_id>/<orientation>/previews/ct_<CTSERIES>_mri_<MRISERIES>_<idx:03d>_pair.png
```

The filename deliberately encodes **both** series names and the slice index so any array can be traced back to its source DICOM folder.

* `.npy` — `float32`, values in `[0, 1]`, shape **`(height, width)` — variable per series**, matching its CT partner exactly. Loadable by a PyTorch `Dataset`, but see §12: it must be cropped or padded to a common size before batching.
* `.png` — CT on the left, a 2 px grey (`180`) divider, MRI on the right, 8-bit greyscale. Purely for human QC of alignment; enabled by `SAVE_PNG = True`.

Each saved pair also appends a row to `metadata.csv`:

```
patient_id, body_region, orientation, slice_index,
ct_series, mri_series, mri_desc,
height, width,
ct_npy, mri_npy
```

`height` / `width` are the actual saved dimensions, which let the dataloader plan batching and filter undersized slices without opening any array.

`mri_desc` carries the raw `Series Description`, which is the only place the **T1 vs T2 weighting** of a series survives into the processed dataset — worth using when stratifying splits or conditioning a model.

---

## 15. Running It

```bash
cd Preprocessing

# All patients, defaults
python preprocess_2d.py --data_root ../Raw_data_mri_ct/Rawdata_dicom --output_dir ../processed_2d

# One patient, full-resolution N4 (slow but highest quality)
python preprocess_2d.py --patient PA0_Ranjeet --n4_shrink 1

# Looser background flag (fewer sparse slices get tagged is_background)
python preprocess_2d.py --bg_fraction 0.97

# Wider MRI percentile clipping
python preprocess_2d.py --mri_p_low 1.0 --mri_p_high 99.0
```

MRI-relevant flags: `--n4_shrink`, `--mri_p_low`, `--mri_p_high`, `--register_2d`, `--reg_search_mm` (§9), `--bg_thresh`, `--bg_fraction`, `--save_png`, `--skip_existing`, `--patient`.

---

## 16. Known Limitations (MRI side)

1. **`--register_2d` corrects translation only, and only when it can prove it should.** Rotation and scale errors are not expressible by the method (§9) and so are not corrected. Pairs whose probe slices disagree, or whose offset is further out than `--reg_search_mm`, are deliberately left unshifted — `reg_note` in the CSV says which, and how many were rejected is worth checking after a run.
2. **Orientation comes from strings, not geometry.** A series named `SE3` with an unhelpful description resolves to `"unknown"` and is dropped. The direction-cosine method was removed rather than wired in.
3. **T1 and T2 are mixed.** Nothing in the pipeline separates them — a percentile-normalised T1 and T2 both land in `[0, 1]` with inverted tissue contrast. Only `mri_desc` in the CSV distinguishes them.
4. **Bias correction is 2D.** Deliberate (see §5), but it means through-plane intensity drift is not corrected.
5. **PA32 covers two anatomies (knee + ankle).** Strict name matching does not know this; `dataset_context.txt` flags that it should be split into two logical patients to prevent a knee MRI pairing with an ankle CT.
6. **No in-plane rotation correction, ever.** `--register_2d` slides; it does not turn. With registration off, alignment is only as good as the DICOM patient coordinates from the two scanners.
7. **`SKIP_EXISTING = True` by default** — re-running will skip any patient whose output directory already contains sub-directories. Delete the folder to reprocess.
8. **Output arrays are no longer a uniform shape** (§12). This is intentional, but it means the dataloader — not the pipeline — is now responsible for producing batchable tensors and for not clipping off-centre anatomy.

---

## 17. Related Docs

| File | Covers |
|---|---|
| [`ct_pipeline_docs.md`](./ct_pipeline_docs.md) | The CT half of the same pipeline |
| [`io_utils_docs.md`](./io_utils_docs.md) | DICOM reading & series discovery |
| [`image_processing_docs.md`](./image_processing_docs.md) | N4, resampling, registration internals |
| [`normalization_docs.md`](./normalization_docs.md) | Percentiles, windowing, cropping, BG filter |
| [`export_utils_docs.md`](./export_utils_docs.md) | `.npy` and PNG writers |
| [`pipeline_core_docs.md`](./pipeline_core_docs.md) | The per-pair orchestrator |
| [`pipeline_config_docs.md`](./pipeline_config_docs.md) | All tunable constants |
| [`preprocess_2d_docs.md`](./preprocess_2d_docs.md) | CLI and main loop |
