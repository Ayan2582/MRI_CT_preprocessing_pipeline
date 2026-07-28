# 📖 Output Structure: What the Pipeline Produces

This is a reference for **what lands on disk** after running `preprocess_2d.py` — the folder layout, file formats, and the `metadata.csv` schema a PyTorch `Dataset` should actually be built on.

---

## 1. Directory tree

```
<OUTPUT_DIR>/                          e.g. c:\...\processed_2d
├── pipeline.log                       # full run log (see §5)
├── metadata.csv                       # master index — one row per saved slice pair (see §3)
│
├── PA0_Ranjeet/                       # one folder per patient (patient_id)
│   ├── axial/                         # only exists if this patient had a CT+MRI pair in this plane
│   │   ├── ct/
│   │   │   ├── ct_SE0_mri_SE1_000.npy
│   │   │   ├── ct_SE0_mri_SE1_001.npy
│   │   │   └── ...
│   │   ├── mri/
│   │   │   ├── ct_SE0_mri_SE1_000.npy   ← identical filename to its CT counterpart
│   │   │   ├── ct_SE0_mri_SE1_001.npy
│   │   │   └── ...
│   │   └── previews/                  # only present if --save_png
│   │       ├── ct_SE0_mri_SE1_000_pair.png
│   │       └── ...
│   ├── coronal/                       # same layout, only if that orientation was paired
│   └── sagittal/                      # same layout, only if that orientation was paired
│
├── PA1_.../
│   └── ...
└── ...
```

**Notes:**

- **Orientation folders are conditional.** `axial` / `coronal` / `sagittal` only appear for a patient if a CT series and an MRI series with matching folder names existed for that plane. A patient with only axial scans gets only an `axial/` folder.
- **`ct/` and `mri/` filenames are identical**, just in different subfolders: `ct_{ct_series}_mri_{mri_series}_{slice_index:03d}.npy`. This is what lets a dataloader zip the two by filename rather than by index alone.
- **No fixed image size.** Each `.npy` keeps its native post-resample width/height — different series (and different patients) produce different shapes. Cropping/padding to a uniform size is intentionally deferred to the GAN's `Dataset` (see §4).

---

## 2. File formats

| File | Format | Contents |
|---|---|---|
| `ct/*.npy`, `mri/*.npy` | NumPy binary, `float32`, 2-D | Normalised intensities in **`[0.0, 1.0]`**. CT is HU-windowed (`normalize_ct_slice`), MRI is percentile-clipped (`normalize_mri_slice`). Load with `np.load(path)`. |
| `previews/*_pair.png` | 8-bit grayscale PNG | CT (left) \| 2px grey divider \| MRI (right), scaled `[0,1]→[0,255]`, for human QC of alignment only — not meant to be used as training input. |
| `metadata.csv` | CSV | One row per saved slice pair — the authoritative index (§3). |
| `pipeline.log` | Plain text | Full run log, same content that was printed to console. |

---

## 3. `metadata.csv` schema

This is what a `Dataset` should actually load from — **not** a directory walk, since only this file carries shape info and the background flag.

| Column | Type | Meaning |
|---|---|---|
| `patient_id` | str | e.g. `PA0_Ranjeet` |
| `body_region` | str | `brain` / `abdomen` / `musculoskeletal` / `spine` / `default`, from `PREFIX_TO_REGION` in `pipeline_config.py` |
| `orientation` | str | `axial` / `coronal` / `sagittal` |
| `slice_index` | int | Position within the series (matches the `_NNN` suffix in the filename) |
| `ct_series` | str | Source CT DICOM series folder name |
| `mri_series` | str | Source MRI DICOM series folder name |
| `mri_desc` | str | MRI `SeriesDescription` DICOM tag |
| `height`, `width` | int | Actual saved array shape — **varies row to row**, do not assume a constant size |
| `ct_npy`, `mri_npy` | str | Absolute paths to the two `.npy` files |
| `is_background` | bool | `True` if ≥90% of either the CT or MRI slice is near-empty (see §4). The slice is still saved — this is a tag, not a filter. |

---

## 4. Things a downstream `Dataset` must handle

1. **Variable shape.** `torch.stack` will fail on a naive batch — slices are not a uniform size. Crop or pad to a common size in `__getitem__`, applying the *identical* crop to the CT and MRI of a pair (they're pixel-aligned; cropping them independently destroys that alignment).
2. **`is_background` is informational, not a discard.** Filter, downweight, or keep these rows as your training strategy needs — the pipeline no longer makes that call for you.
3. **CT and MRI normalisation bounds differ in kind.** CT uses a fixed HU window per body region (same meaning across all patients); MRI uses per-volume percentiles (scanner/sequence-dependent, no fixed physical unit). Don't assume `0.5` means the same tissue in both modalities.

---

## 5. Console / log summary

At the end of a run, `pipeline.log` (and stdout) print:

```
=================================================================
  PIPELINE COMPLETE - SUMMARY
=================================================================
  Patients processed       : <n> / <total>
  Patients skipped (no MRI): <n>
  Total paired slices saved: <n>
  Slices flagged (bg, kept): <n>
  Breakdown by orientation :
    axial     : <n> slices
    coronal   : <n> slices
    sagittal  : <n> slices
  Metadata CSV             : <path>
  Log file                 : <path>
=================================================================
```

Use this to sanity-check a run — e.g. an unexpectedly high `Slices flagged (bg, kept)` count for a region may indicate an FOV or alignment issue worth inspecting in the `previews/` PNGs before training.

---

## 6. Related docs

| Doc | Covers |
|---|---|
| [`preprocess_2d_docs.md`](./preprocess_2d_docs.md) | The orchestrating script and CLI |
| [`pipeline_core_docs.md`](./pipeline_core_docs.md) | The per-orientation processing loop that produces these files |
| [`normalization_docs.md`](./normalization_docs.md) | Windowing, percentiles, background flagging logic |
| [`ct_pipeline_docs.md`](./ct_pipeline_docs.md) / [`mri_pipeline_docs.md`](./mri_pipeline_docs.md) | Modality-specific stage-by-stage walkthroughs |
| [`export_utils_docs.md`](./export_utils_docs.md) | `save_npy` / `save_preview_png` implementation |
