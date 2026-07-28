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
        ├─ strict-name pairing loop (preprocess_2d.py:226) ← Stage 3: Pairing
        │
        └─ pipeline_core.process_orientation_pair()
                 │
                 ├─ image_processing.apply_n4_bias_correction()   ← Stage 4  (MRI ONLY)
                 ├─ image_processing.resample_mri_to_ct_grid()    ← Stage 5
                 ├─ image_processing.volume_to_slices()           ← Stage 6
                 ├─ normalization.compute_mri_percentiles()       ← Stage 7
                 │
                 └─ per-slice loop:
                        ├─ image_processing.register_2d_rigid()   ← Stage 8  (optional)
                        ├─ normalization.normalize_mri_slice()    ← Stage 9
                        ├─ normalization.is_background_slice()    ← Stage 10
                        │  (Stage 11 crop/pad — REMOVED, see §12)
                        └─ export_utils.save_npy() / save_preview_png()  ← Stage 12
```

---

## 2. Stage 1 — Series Discovery

**Code:** `io_utils.discover_series()` · called from `preprocess_2d.py:209`

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

> 🔑 **The MRI is the orientation authority.** In `preprocess_2d.py:237` the orientation of the *pair* is taken from the MRI entry (`orient = m_entry["orientation"]`); the CT's own orientation label is never consulted. If the MRI's orientation resolves to `"unknown"`, the whole pair is skipped with a warning.

---

## 3. Stage 2 — DICOM Loading

**Code:** `io_utils.load_dicom_series()`

Uses `sitk.ImageSeriesReader` with `GetGDCMSeriesFileNames()`, which sorts the individual `IM*` files into correct anatomical order using their DICOM position tags (not filename order — `IM10` must not land between `IM1` and `IM2`).

`MetaDataDictionaryArrayUpdateOn()` and `LoadPrivateTagsOn()` are enabled so the `Series Description` can be read back and stashed onto the image object as a custom `series_desc` metadata key.

**Output:** a 3D `SimpleITK.Image` whose voxel values are the **raw scanner intensities**. This matters — unlike CT Hounsfield Units, MRI intensities have **no physical unit and no cross-patient meaning**. A value of `600` in one scan and `600` in another are unrelated. Everything downstream in Stage 7/9 exists to work around this.

---

## 4. Stage 3 — CT ↔ MRI Pairing

**Code:** `preprocess_2d.py:226–244`

For each MRI series, the pipeline takes the folder basename, splits on `_`, and looks for a CT series whose basename splits to the same token:

```
MRI  SE0      →  base "SE0"
CT   SE0_1    →  base "SE0"   ✔ match
```

This "strict name match" assumes the two modalities were exported with a consistent series-numbering convention. If no CT shares the base name, that MRI series is silently dropped; if **no** MRI series in the patient finds a partner, the patient is logged as `No identically named CT and MRI folders found - skipped.`

> ℹ️ `io_utils.select_best_series()` (pick the series with the most slices per orientation) and `get_all_valid_series()` used to exist as unused helpers; both have been removed. Strict name matching is the only pairing path.

---

## 5. Stage 4 — N4 Bias Field Correction ⭐ *MRI-only*

**Code:** `image_processing.apply_n4_bias_correction()` · called from `pipeline_core.py:74`

This stage has **no CT equivalent** — it corrects an artefact that only exists in MR physics.

### What it fixes
MRI receive coils are not uniformly sensitive across their field of view. The result is a smooth, low-frequency intensity gradient — the same tissue appears brighter near the coil and darker far from it. This "bias field" is multiplicative:

```
observed(x) = true(x) · exp(bias(x)) · noise
```

Left uncorrected, a percentile-based normaliser (Stage 7) computes bounds that are wrong for half the image, and a GAN learns to reproduce the coil shading instead of the anatomy.

### How it is done here

| Parameter | Value | Reason |
|---|---|---|
| Mode | **2D, slice-by-slice** | Not 3D. The data is highly anisotropic (thin in-plane, ~4 mm slice gaps) and volumes can be as short as 18 slices — the 3D B-spline grid fit fails or destabilises on such thin stacks. |
| Mask | `sitk.OtsuThreshold(slice, 0, 1, 200)` | Auto-separates tissue from air so N4 does not waste effort modelling the background. |
| `shrink_factor` | `4` (CLI: `--n4_shrink`) | The bias field is *smooth by definition*, so it can be estimated on a 4× downsampled image at ~16× the speed with negligible loss. |
| Iterations | `[50, 50, 50, 50]` | 4-level multi-resolution pyramid. |
| Convergence | `0.001` | Early-exit threshold. |

The correction itself (`image_processing.py:70–76`):

1. Fit N4 on the **shrunken** slice + mask.
2. Ask for the log bias field evaluated at **full resolution** — `GetLogBiasFieldAsImage(slice_2d)`. This is the key trick: estimate cheap, apply sharp.
3. Divide: `corrected = slice_2d / sitk.Exp(log_bias)`.

If N4 throws (typically on a slice that is pure air, where Otsu produces a degenerate mask), the exception is caught and the **uncorrected slice is used** with a warning — one bad slice never kills a patient.

### The `JoinSeries` geometry trap

Slices are stitched back with `sitk.JoinSeries()`, which refuses to merge 2D images whose origin/direction differ by even a floating-point epsilon. The code defends against this by capturing `base_origin` / `base_direction` at `z == 0` and force-stamping them onto every corrected slice (`image_processing.py:84–85`). Afterwards the original 3D `Spacing`, `Direction`, and `Origin` are copied back onto the joined volume (`:96–98`) so the physical DICOM geometry is preserved for Stage 5.

---

## 6. Stage 5 — Projection onto the CT Grid ⭐ *the alignment step*

**Code:** `image_processing.resample_mri_to_ct_grid()` · called from `pipeline_core.py:92`

The N4-corrected MRI is resampled with **the already-resampled CT volume as the reference image**:

```python
resampler.SetReferenceImage(ct_image)      # size, spacing, origin, direction — all inherited
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(0.0)        # out-of-FOV → black
resampler.SetTransform(sitk.Transform())   # identity: trust the DICOM coordinates
```

Consequences worth internalising:

* The MRI ends up with **exactly the CT's voxel count and slice count**, which is why `pipeline_core.py:103` can assume `len(ct_slices) == len(mri_slices)` and index them with a single loop counter.
* The MRI inherits the CT's **1.0 mm in-plane spacing** — `--target_spacing` is never applied to the MRI directly.
* The identity transform means alignment relies **entirely on the DICOM patient-coordinate system** being consistent between the two scanners. There is no intensity-driven 3D registration.
* MRI voxels that fall outside the CT's field of view are simply lost; CT regions the MRI never covered are filled with `0.0`.

### The direction-cosine override

```python
mri_aligned.SetDirection(ct_image.GetDirection())   # image_processing.py:156
```

This *asserts* that the MRI shares the CT's orientation rather than letting the resampler reconcile a small angular difference. A sub-degree tilt discrepancy, resolved across a stack only 18 slices deep, produces visible shear/staircase artefacts. Forcing the direction trades a tiny, uncorrected rotation for a geometrically clean volume.

> ⚠️ **Caveat:** this line mutates the object in place. `mri_aligned = mri_image` is a reference, not a copy — the N4-corrected volume's direction matrix is modified as a side effect. Harmless in the current flow (the volume is not reused afterwards), but worth knowing before refactoring.

---

## 7. Stage 6 — Volume → Slices

**Code:** `image_processing.volume_to_slices()` · `pipeline_core.py:99`

`sitk.GetArrayFromImage()` converts `(x, y, z)` ITK ordering into NumPy `(z, y, x)` ordering, then the list comprehension yields one `(y, x)` array per slice. From this point on the MRI is **plain NumPy**, not SimpleITK.

---

## 8. Stage 7 — Percentile Bounds ⭐ *MRI-only*

**Code:** `normalization.compute_mri_percentiles()` · `pipeline_core.py:110`

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

## 9. Stage 8 — Optional 2D Rigid Registration

**Code:** `image_processing.register_2d_rigid()` · `pipeline_core.py:128–130` · flag `--register_2d`

Off by default. When enabled, each MRI slice is aligned to its CT partner:

| Component | Choice | Reason |
|---|---|---|
| Transform | `Euler2DTransform` | Rotation + translation only. No scaling/shear — anatomy must not be deformed. |
| Init | `CenteredTransformInitializer(..., GEOMETRY)` | Start from centre-of-image alignment. |
| Metric | **Mattes Mutual Information**, 50 bins | Mandatory for cross-modality. Bone is white on CT and black on MRI, so intensity-difference metrics (MSE, NCC) are meaningless here; MI matches *statistical structure* instead. |
| Sampling | 20 % random | ~5× speedup at negligible accuracy cost. |
| Optimiser | Gradient descent, lr `0.1`, 100 iters | With `SetOptimizerScalesFromPhysicalShift()` so rotation and translation steps are comparably scaled. |
| Pyramid | shrink `[4, 2, 1]`, sigmas `[2, 1, 0]` | Coarse-to-fine, to avoid local minima. |

> 🐛 **Known issue — `--register_2d` is currently broken.**
> By the time the per-slice loop runs, `ct_slice` and `mri_slice` are **NumPy arrays** (Stage 6), but `register_2d_rigid()` opens with `sitk.Cast(fixed_slice, ...)` at `image_processing.py:180`, which requires a `SimpleITK.Image`. That call sits *outside* the function's `try` block (which begins at `:224`), so it raises rather than falling back. The exception propagates up to the per-pair handler at `preprocess_2d.py:273`, which logs `Unhandled error` and abandons the entire orientation pair.
> Even if the cast were fixed, the function returns a `SimpleITK.Image`, which `normalize_mri_slice()` would then call `.astype()` on — a second failure.
> **Fixing this requires converting to/from `sitk.GetImageFromArray` / `sitk.GetArrayFromImage` around the call.** Until then, run without the flag; Stage 5 already provides DICOM-coordinate alignment.

---

## 10. Stage 9 — Intensity Normalisation

**Code:** `normalization.normalize_mri_slice()` · `pipeline_core.py:135`

```python
s = np.clip(slice_2d, p1, p99)
return (s - p1) / (p99 - p1)          # → [0.0, 1.0]
```

Guard: if `p99 - p1 < 1e-8` the slice has no contrast, and an all-zero array is returned instead of dividing by zero.

The same `p1`/`p99` are reused for **every slice in the series**, which is what preserves inter-slice consistency.

---

## 11. Stage 10 — Background Rejection

**Code:** `normalization.is_background_slice()` · `pipeline_core.py:140–143`

```python
np.mean(arr <= 0.02) > 0.90
```

A slice is discarded if **more than 90 % of its pixels sit below normalised intensity 0.02**.

The check is applied to the CT **and** the MRI, joined with `or` — **if either side is background, the pair is dropped.** This preserves the 1:1 pairing invariant that the whole dataset depends on. Discarded slices are counted into `n_skipped_bg` and reported per-orientation and in the final summary.

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

**Code:** `export_utils.save_npy()` / `save_preview_png()` · `pipeline_core.py:162–172`

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

# Looser background filter (keep more sparse slices)
python preprocess_2d.py --bg_fraction 0.97

# Wider MRI percentile clipping
python preprocess_2d.py --mri_p_low 1.0 --mri_p_high 99.0
```

MRI-relevant flags: `--n4_shrink`, `--mri_p_low`, `--mri_p_high`, `--register_2d` (see §9 caveat), `--bg_thresh`, `--bg_fraction`, `--save_png`, `--skip_existing`, `--patient`.

---

## 16. Known Limitations (MRI side)

1. **`--register_2d` raises on the NumPy/SimpleITK type boundary** — see §9.
2. **Orientation comes from strings, not geometry.** A series named `SE3` with an unhelpful description resolves to `"unknown"` and is dropped. The direction-cosine method was removed rather than wired in.
3. **T1 and T2 are mixed.** Nothing in the pipeline separates them — a percentile-normalised T1 and T2 both land in `[0, 1]` with inverted tissue contrast. Only `mri_desc` in the CSV distinguishes them.
4. **Bias correction is 2D.** Deliberate (see §5), but it means through-plane intensity drift is not corrected.
5. **PA32 covers two anatomies (knee + ankle).** Strict name matching does not know this; `dataset_context.txt` flags that it should be split into two logical patients to prevent a knee MRI pairing with an ankle CT.
6. **No in-plane rotation correction.** With registration off, alignment is only as good as the DICOM patient coordinates from the two scanners.
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
