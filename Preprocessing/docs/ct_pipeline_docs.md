# 🦴 CT Pipeline — End-to-End Walkthrough

This document traces the **complete journey of a single CT series**, from a folder of raw `.dcm` files on disk to a stack of normalised float32 `.npy` slices ready for a GAN dataloader.

> **Slices are saved at their native post-resample size, not a fixed square.** Cropping was deliberately moved out of this pipeline and into the GAN's dataloader — see §10.

Where the existing `*_docs.md` files explain **one module at a time**, this document follows **one modality through every module**, in execution order.

For the MRI counterpart, see [`mri_pipeline_docs.md`](./mri_pipeline_docs.md).

---

## 0. The One-Sentence Summary

> The CT is loaded in Hounsfield Units → resampled in-plane to exactly 1.0 mm/px → **this resampled grid becomes the reference geometry for the entire pair** → a region-specific HU window rescales it to `[0, 1]` → empty slices are dropped → each slice is saved at its native size.

The CT's role is structurally different from the MRI's. **The CT is the geometric anchor.** It is resampled first, and the MRI is then projected onto whatever grid the CT produced. Everything about the output — voxel count, slice count, spacing, origin, direction — is decided by the CT.

The CT path is also notably *shorter* than the MRI path: it skips N4 bias correction entirely, and its normalisation uses fixed physical constants rather than per-volume statistics.

---

## 1. Pipeline Map (CT path only)

```
 preprocess_2d.py : main()
        │
        ├─ io_utils.discover_series(CT/<patient>/ST0)      ← Stage 1: Discovery
        │      └─ io_utils.load_dicom_series(SE*)          ← Stage 2: Loading
        │
        ├─ region profile lookup (preprocess_2d.py:167)    ← Stage 3: Region config
        ├─ strict-name pairing loop (preprocess_2d.py:226) ← Stage 4: Pairing
        │
        └─ pipeline_core.process_orientation_pair()
                 │
                 ├─ (no N4 — MRI only)
                 ├─ image_processing.resample_inplane()           ← Stage 5  ⭐ defines the grid
                 ├─ image_processing.volume_to_slices()           ← Stage 6
                 │
                 └─ per-slice loop:
                        ├─ normalization.normalize_ct_slice()     ← Stage 7  (HU windowing)
                        ├─ normalization.is_background_slice()    ← Stage 8
                        │  (Stage 9 crop/pad — REMOVED, see §10)
                        └─ export_utils.save_npy() / save_preview_png()  ← Stage 10
```

---

## 2. Stage 1 — Series Discovery

**Code:** `io_utils.discover_series()` · called from `preprocess_2d.py:207`

The pipeline walks `Raw_data_mri_ct/Rawdata_dicom/CT/<patient_id>/ST0/` and iterates its `SE*` sub-folders in sorted order. Rules are identical to the MRI side:

| Check | Behaviour |
|---|---|
| Not a directory | skipped |
| DICOM read failed | skipped (warning logged) |
| `n_slices < 2` | skipped — filters out single-frame **scout / topogram** images |

Each surviving series becomes a dictionary:

```python
{
    "path":        ".../CT/PA0_Ranjeet/ST0/SE0",
    "image":       <SimpleITK.Image>,   # full 3D volume, in HU
    "n_slices":    18,
    "orientation": "axial",
    "series_desc": "CT HEAD W/O CONTRAST",
}
```

Note that the CT patient list also **drives the outer loop** — `preprocess_2d.py:139` does `sorted(os.listdir(ct_root))`. A patient with an MRI but no CT folder is never even considered; a patient with a CT but no MRI is counted into `patients_skipped_mri` and skipped at `:190`.

> ⚠️ **The CT's `orientation` field is computed but never used.** Pairing keys on folder names, and the orientation of the resulting pair is taken from the **MRI** entry (`preprocess_2d.py:237`). A CT whose orientation resolves to `"unknown"` is still processed perfectly well, as long as its MRI partner resolved cleanly.

---

## 3. Stage 2 — DICOM Loading & Hounsfield Units

**Code:** `io_utils.load_dicom_series()`

`sitk.ImageSeriesReader` + `GetGDCMSeriesFileNames()` sorts the `IM*` files by their DICOM position tags, not by filename (so `IM10` does not land between `IM1` and `IM2`), and reads them into one 3D volume.

The critical property of the resulting array: **CT voxel values are Hounsfield Units, and HU are absolute and physically meaningful.**

| Tissue | HU |
|---|---|
| Air | ≈ −1000 |
| Lung | −900 … −500 |
| Fat | −120 … −70 |
| Water | 0 |
| Soft tissue / muscle | +20 … +80 |
| Contrast-enhanced blood | +100 … +400 |
| Trabecular bone | +300 … +800 |
| Cortical bone | +800 … +3000 |
| Metal implants | > +3000 |

The GDCM reader applies `RescaleSlope` / `RescaleIntercept` during read, so the values in memory are already true HU — no manual rescaling is needed anywhere in this codebase.

This absoluteness is the single biggest difference from MRI, and it cascades through the rest of the pipeline:

* The CT needs **no bias-field correction** — there is no receive-coil sensitivity to flatten.
* The CT needs **no per-volume percentile estimation** — a fixed HU window means the same thing on every scanner, every patient, every day.
* The CT can be padded with a physically correct value (`−1024` HU = air) instead of an arbitrary zero.

---

## 4. Stage 3 — Region Profile Selection ⭐ *CT-specific effect*

**Code:** `preprocess_2d.py` · `pipeline_config.PREFIX_TO_REGION` / `REGION_PROFILES`

Before each patient is processed, the ID prefix (`PA0_Ranjeet` → `PA0`) is looked up in `PREFIX_TO_REGION`, and the matching profile **overwrites `args` in place**:

```python
args.ct_win_min  = profile["ct_win_min"]
args.ct_win_max  = profile["ct_win_max"]
```

The **crop size column below is historical** — `REGION_PROFILES` no longer carries `target_size`, since the pipeline does not crop (§10). It is kept in this table because the sizes and their rationale are still the right starting point for your dataloader.

| Region | Crop size *(historical)* | CT window (HU) | Patients | Rationale |
|---|---|---|---|---|
| `brain` | 256 | `[0, 80]` | 15 | Grey/white matter differ by only a few HU. A narrow window is essential — a `[-200, 300]` window would compress the entire brain into ~16 % of the output range and flatten the contrast the model has to learn. |
| `abdomen` | **384** | `[-160, 240]` | 16 | Classic soft-tissue window: keeps fat (−100), organs (+40), and contrast-enhanced vessels visible. 384 px is needed because a torso does not fit in a 256 mm FOV at 1.0 mm/px. |
| `musculoskeletal` | 256 | `[-200, 300]` | 10 | Wide enough to keep fat, muscle, cartilage, and cortical-bone margins simultaneously. |
| `spine` | 256 | `[-200, 300]` | 4 | Same as MSK — bone plus paraspinal soft tissue. |
| `default` | 256 | `[-200, 300]` | unmapped IDs | Safe general-purpose fallback. |

> ⚠️ **Gotcha:** because both values are reassigned on **every** patient iteration, the CLI flags `--ct_win_min` and `--ct_win_max` are **silently overridden** for any patient with a region mapping. Passing `--ct_win_min -1000` has no effect on `PA0`. To change windowing behaviour, edit `REGION_PROFILES` in `pipeline_config.py`.
>
> A second consequence: because `args` is mutated rather than copied, the values persist across patients. An unmapped ID falls back to `"default"`, so this happens to be harmless — but the pattern is fragile.
>
> A second consequence: because `args` is mutated rather than copied, the values persist across patients. An unmapped ID falls back to `"default"`, so this happens to be harmless — but the pattern is fragile.

---

## 5. Stage 4 — CT ↔ MRI Pairing

**Code:** `preprocess_2d.py:226–244`

Iteration is driven by the **MRI** list; for each MRI series the code searches the CT list for a basename match on the pre-underscore token:

```
MRI  SE0      →  base "SE0"
CT   SE0_1    →  base "SE0"   ✔ match
```

`next(...)` returns the **first** matching CT. If a patient has several CT series sharing a base token, the sorted-first one wins — there is no "best series" selection. (`io_utils.select_best_series()`, which would have picked the series with the most slices, was never wired in and has since been removed.)

A CT series that no MRI matches is simply never processed.

---

## 6. Stage 5 — In-Plane Resampling ⭐ *defines the pair's geometry*

**Code:** `image_processing.resample_inplane(ct_entry["image"], args.target_spacing, is_ct=True)` · `pipeline_core.py:86`

This is the CT's most consequential stage — its output grid is handed to `resample_mri_to_ct_grid()` as the reference image, so it dictates the shape of **both** modalities.

### What changes and what does not

```python
new_sx = new_sy = 1.0            # forced isotropic in-plane
new_sz = orig_sp[2]              # slice spacing left ALONE
new_nx = round(orig_nx * orig_sx / 1.0)
new_ny = round(orig_ny * orig_sy / 1.0)
new_nz = orig_nz                 # slice count unchanged
```

* **In-plane (X, Y) → exactly 1.0 mm/px.** Native CT in this dataset ranges roughly 0.18–0.84 mm/px depending on region and reconstruction FOV. Standardising removes scale as a nuisance variable, so the network does not have to learn that a knee at 0.3 mm/px and a torso at 0.8 mm/px are the same anatomy at different zoom levels. It also makes any downstream crop size interpretable in millimetres: 256 px = 256 mm.
* **Through-plane (Z) is deliberately untouched.** Slice spacing can be 3–5 mm. Interpolating along Z would synthesise anatomy that was never scanned — which is exactly the failure mode you cannot afford in a dataset whose purpose is training a generative model. The pipeline is honest about being 2D.

### Interpolation and fill value

```python
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(-1024 if is_ct else 0)   # image_processing.py:139
```

Linear interpolation is applied to HU values directly — valid because HU is a linear physical scale (unlike, say, interpolating an already-windowed 8-bit image). The `is_ct=True` branch fills any newly created border with **−1024 HU, i.e. air**, so padding is physically correct rather than an artificial "water-density" halo that a `0` fill would produce.

`SetOutputDirection` and `SetOutputOrigin` copy the source values through, and the transform is identity — this is a pure rescale, never a rotation.

---

## 7. Stage 6 — Volume → Slices

**Code:** `image_processing.volume_to_slices()` · `pipeline_core.py:98`

`sitk.GetArrayFromImage()` converts ITK `(x, y, z)` ordering to NumPy `(z, y, x)`, and the list comprehension yields one `(y, x)` array per slice.

`n_pair = len(ct_slices)` at `pipeline_core.py:103` is the loop bound for the entire per-slice stage — and it is taken from the **CT**. This is safe only because Stage 5 of the MRI path resampled the MRI onto this exact grid, guaranteeing identical slice counts.

---

## 8. Stage 7 — HU Windowing & Normalisation ⭐ *CT-only*

**Code:** `normalization.normalize_ct_slice()` · `pipeline_core.py:134`

```python
s = np.clip(slice_2d.astype(np.float32), window_min, window_max)
return (s - window_min) / float(window_max - window_min)     # → [0.0, 1.0]
```

Two things happen, and both matter:

1. **Clipping is an attention mechanism.** Raw CT spans roughly −1000 to +3000 HU. If that entire range were linearly mapped to `[0, 1]`, the soft-tissue band that actually distinguishes muscle from fat from organ (a ~100 HU span) would occupy about 2.5 % of the output range — effectively invisible to the model, and quantised into near-nothing. Clipping to a 500 HU window expands that same tissue band across a fifth of the range.

2. **Information outside the window is destroyed, permanently.** Everything above `window_max` saturates to `1.0` and everything below `window_min` to `0.0`. With the brain profile's `[0, 80]` window, **all bone and all air are flattened into two constants** — the skull becomes a uniform white silhouette with no internal structure. This is intentional for brain work, but it is a hard, irreversible choice baked into the `.npy` files. If a downstream task ever needs bone detail on brain patients, the pipeline must be re-run with a different window; it cannot be recovered from the outputs.

This is where the CT and MRI philosophies diverge most sharply: the MRI's bounds are **learned from the data** (percentiles, per volume), while the CT's are **imposed from physics and clinical convention** (fixed HU, per region).

Unlike the MRI path there is no divide-by-zero guard — none is needed, since `window_max > window_min` is guaranteed by every entry in `REGION_PROFILES`.

---

## 9. Stage 8 — Background Rejection

**Code:** `normalization.is_background_slice()` · `pipeline_core.py:140–143`

```python
np.mean(arr <= 0.02) > 0.90
```

A slice is discarded if **more than 90 % of its pixels fall below normalised intensity 0.02**. Checked on the CT and the MRI, joined with `or` — **if either side is background, the pair is dropped**, preserving the 1:1 pairing invariant.

> ⚠️ **Window-dependent behaviour on CT.** The threshold is applied *after* normalisation, so what counts as "background" shifts with the region profile:
> * Brain `[0, 80]`: air (−1000 HU) and fat (−100 HU) both clip to `0.0`, so the effective cutoff is "≤ 1.6 HU". Very aggressive — anything at or below water density reads as background.
> * MSK `[-200, 300]`: `0.02` maps back to −190 HU. Only true air is rejected; fat and lung parenchyma survive.
>
> This means the same anatomical slice could be kept under one profile and dropped under another. Tune with `--bg_thresh` / `--bg_fraction` if a region is losing too many usable slices — the per-orientation log line reports exactly how many were discarded.

---

## 10. Stage 9 — Cropping · ⛔ REMOVED

**This stage no longer runs.** `center_crop_pad()` is no longer called from `pipeline_core.py`; slices are written at whatever size Stage 5 produced. Cropping now happens in the GAN's dataloader.

### Why it was removed

Cropping is a *training-time* decision, not a preprocessing one. Baking a fixed 256²/384² centre crop into the `.npy` files meant that changing crop strategy (random-crop augmentation, body-mask crop), changing the GAN's input resolution, or recovering clipped anatomy all required a **full re-run from DICOM** — minutes per patient, dominated by N4.

The measurements below made the cost concrete, and they are the evidence for the change.

### What this changes for you

* `.npy` arrays now have **variable shape**, so `torch.stack` will fail on a naive batch. The `Dataset` must crop/pad/resize to a common size in `__getitem__`, or supply a custom `collate_fn`.
* **CT and MRI of the same pair still share an identical shape** (Stage 5 guarantees it), so one crop applied to both stays exactly aligned.
* `metadata.csv` gains **`height`** and **`width`** columns so the dataloader can size batches and filter undersized slices without opening any array.
* `normalization.center_crop_pad()` was **deleted**, along with the `target_size` config, the `--target_size` flag, and the `crop_size` CSV column. A self-contained reference implementation for your `Dataset` is in [`mri_pipeline_docs.md`](./mri_pipeline_docs.md) §12.

> ⚠️ Whatever crop you choose, **derive one offset and apply it to both modalities.** Two independent crops (two random offsets, or two separate body masks) destroy the pixel-level CT↔MRI correspondence the dataset exists to provide.

> ℹ️ **When you do pad, `0.0` is the correct fill — it reads as air.** Padding happens on already-normalised arrays, and real air (−1000 HU) is below **every** `window_min` in `REGION_PROFILES` (`0`, `−160`, `−200`), so air clips to `window_min` and normalises to exactly `0.0`. The border is numerically indistinguishable from the air already in the scan, in all four profiles. (Same on the MRI side: raw background `0 < p1`, so it clips to `p1` → `0.0`.)

### The evidence: padding vs. clipped anatomy under the old centre crop

Measured across all 124 series in `Raw_data_mri_ct/`, showing what the removed stage *was* doing:

| Modality | Region | Series | Mean padding | Max padding | Series cropped | Max clipped/side |
|---|---|---|---|---|---|---|
| CT | brain | 39 | 4.4 % | 4.6 % | 2 | 4 mm |
| CT | abdomen | 45 | 6.4 % | 46.8 % | 30 | 23 mm |
| CT | musculoskeletal | 33 | 19.6 % | 50.6 % | 10 | 28 mm |
| CT | spine | 7 | 0.0 % | 0.0 % | 7 | 54 mm |
| MRI | brain | 39 | 10.3 % | 28.9 % | 26 | 12 mm |
| MRI | musculoskeletal | 33 | 25.4 % | 50.6 % | 3 | 7 mm |

Two consequences — both now inherited by the dataloader, which is where they must be handled:

1. **Padding dominated MSK.** A 180 mm knee FOV padded into 256² leaves up to **50 % of the frame as constant zeros**. Beyond wasted compute this distorts evaluation: an L1 or PSNR over the full frame is half-measured on background that is *identical* in input and target, so the metric looks good regardless of anatomy quality. Compute losses/metrics inside a body mask, or use a tighter crop for MSK (192 fits every MSK series with far less waste).
2. **A centre crop is content-blind and silent.** All 7 spine series were cropped, up to 54 mm off each edge; 10 MSK CT series lost up to 28 mm per side. For centred anatomy that is harmless, but an off-centre knee or shoulder loses real tissue and nothing logs it. A body-mask bbox on the HU volume (threshold ≈ −500 HU → `binary_fill_holes` → largest connected component) is a cheap, deterministic fix now that you control the crop at load time.

---

## 11. Stage 10 — Export

**Code:** `export_utils.save_npy()` / `save_preview_png()` · `pipeline_core.py:162–172`

```
<output_dir>/<patient_id>/<orientation>/ct/ct_<CTSERIES>_mri_<MRISERIES>_<idx:03d>.npy
<output_dir>/<patient_id>/<orientation>/previews/ct_<CTSERIES>_mri_<MRISERIES>_<idx:03d>_pair.png
```

The filename encodes **both** series names plus the slice index, so any array traces back to its source DICOM folder. The CT and its MRI partner share an identical filename stem in sibling `ct/` and `mri/` directories — a dataloader can pair them by string substitution alone.

* `.npy` — `float32`, values in `[0, 1]`, shape **`(height, width)` — variable per series**, matching its MRI partner exactly. See §10: crop or pad to a common size before batching.
* `.png` — CT on the **left**, a 2 px grey (`180`) divider, MRI on the right, 8-bit greyscale. For human QC of alignment.

Each saved pair appends a row to `metadata.csv`:

```
patient_id, body_region, orientation, slice_index,
ct_series, mri_series, mri_desc,
height, width,
ct_npy, mri_npy
```

`height` / `width` are the actual saved dimensions.

`body_region` is re-derived from the patient prefix at `pipeline_core.py:176–177` — use it to stratify train/val/test splits, since a random split across a 45-patient, 10-region dataset will otherwise leak anatomy between folds.

> ℹ️ There is **no `ct_desc` column.** The CT `series_desc` is read and logged but never written to the CSV, so the acquisition protocol (contrast/no-contrast, bone/soft kernel) does not survive into the processed dataset. Add it to the `metadata_rows` dict and to `fieldnames` in `preprocess_2d.py:282` if you need it.

---

## 12. Running It

```bash
cd Preprocessing

# All patients, defaults
python preprocess_2d.py --data_root ../Raw_data_mri_ct/Rawdata_dicom --output_dir ../processed_2d

# One patient
python preprocess_2d.py --patient PA0_Ranjeet

# Coarser in-plane grid — halves memory and disk, keeps the same physical FOV per pixel count
python preprocess_2d.py --target_spacing 2.0

# Keep more sparse slices
python preprocess_2d.py --bg_fraction 0.97
```

CT-relevant flags: `--target_spacing` (effective), `--bg_thresh`, `--bg_fraction`, `--save_png`, `--skip_existing`, `--patient`.
**Ineffective on mapped patients:** `--ct_win_min`, `--ct_win_max` (see §4).

---

## 13. Known Limitations (CT side)

1. **HU windows are irreversible.** The brain profile's `[0, 80]` window discards every bone and air detail. Re-windowing requires re-running from DICOM.
2. **CLI window/size flags are overridden per patient.** See §4 — a documented trap, not a crash.
3. **First-match series selection.** With multiple CT series sharing a base name, the sorted-first one wins, with no slice-count or quality tie-break.
4. **CT orientation is computed but discarded.** The MRI's label governs the pair, so a mislabelled MRI mislabels the CT output folder too.
5. **Output arrays are no longer a uniform shape.** Cropping moved to the dataloader (§10), which now owns both the batching problem and the risk of content-blindly clipping off-centre anatomy.
6. **No metal-artefact handling.** Streak artefacts from implants (> +3000 HU) clip to `1.0` along with cortical bone and stay in the training data. `PA32` (ankle/knee) and other MSK cases are the likeliest to be affected.
7. **No slice-thickness harmonisation.** Deliberate (§6), but it means Z-spacing varies across the dataset — relevant if anyone later tries to reassemble the 2D outputs into pseudo-3D volumes.
8. **PA32 covers two anatomies (knee + ankle).** Strict name matching cannot tell them apart; `dataset_context.txt` recommends splitting it into two logical patients.
9. **`SKIP_EXISTING = True` by default** — re-running skips any patient whose output directory already contains sub-directories. Delete the folder to reprocess.

---

## 14. CT vs MRI — Side by Side

| Stage | CT | MRI |
|---|---|---|
| Bias correction | ✗ not needed | ✓ N4, 2D slice-by-slice |
| Resampling | `resample_inplane()` → **defines the grid** | `resample_mri_to_ct_grid()` → **inherits the grid** |
| Padding fill (resample) | `−1024` HU (air) | `0.0` |
| Intensity scale | Absolute (Hounsfield Units) | Arbitrary, scanner/sequence-dependent |
| Normalisation bounds | Fixed HU window, per **region** | Percentiles `[0.5, 99.5]`, per **volume** |
| Bounds computed from | `pipeline_config.REGION_PROFILES` | The data itself |
| Optional 2D registration | Fixed image (reference) | Moving image (see MRI doc §9 — currently broken) |
| Region profile affects | HU window only | nothing |
| Cropping | ✗ removed — deferred to dataloader | ✗ removed — deferred to dataloader |
| Saved shape | native, variable per series | identical to its CT partner |
| Description in CSV | ✗ not written | ✓ `mri_desc` |

---

## 15. Related Docs

| File | Covers |
|---|---|
| [`mri_pipeline_docs.md`](./mri_pipeline_docs.md) | The MRI half of the same pipeline |
| [`io_utils_docs.md`](./io_utils_docs.md) | DICOM reading & series discovery |
| [`image_processing_docs.md`](./image_processing_docs.md) | N4, resampling, registration internals |
| [`normalization_docs.md`](./normalization_docs.md) | Percentiles, windowing, cropping, BG filter |
| [`export_utils_docs.md`](./export_utils_docs.md) | `.npy` and PNG writers |
| [`pipeline_core_docs.md`](./pipeline_core_docs.md) | The per-pair orchestrator |
| [`pipeline_config_docs.md`](./pipeline_config_docs.md) | All tunable constants |
| [`preprocess_2d_docs.md`](./preprocess_2d_docs.md) | CLI and main loop |
