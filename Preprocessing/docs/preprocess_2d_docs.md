# 📖 Learn the Code: `preprocess_2d.py`

This file is the **Conductor** of the orchestra. It doesn't contain the complex math itself; instead, it imports `pipeline_config.py`, `io_utils.py`, and `pipeline_core.py` (which in turn calls into `image_processing.py` and `normalization.py`) and coordinates them to process the entire dataset from start to finish.

---

## 🚀 The Main Flow (`main()`)

When you run `python preprocess_2d.py`, the `main()` function is triggered. Here is the step-by-step logic it follows:

1. **Parse Arguments (`parse_args`)**: 
   It reads the command-line flags (like `--skip_existing` or `--patient PA3`). It merges these with the defaults from `pipeline_config.py`.
2. **Discover Patients**: 
   It scans the `CT/` and `MRI/` raw data folders, matching them up strictly by their patient folder names (e.g. `PA0_Ranjeet`).
3. **Discover & Pair Series (per patient)**: 
   Within each patient's CT and MRI study folders, `io_utils.discover_series` enumerates every `SE*` sub-folder. Series are then paired by **strict folder-name-prefix match** (e.g. CT series `SE0_1` pairs with MRI series `SE0` because both start with `SE0`) — logged as `"Pairing (Strict Name Match)"`. This is *not* based on slice count or orientation; a helper for picking the series with the most slices per orientation (`select_best_series`) once existed in `io_utils.py` but was never wired in and has been removed.
4. **Loop Over Patients**: 
   For every patient, it extracts the prefix (e.g., `PA3`), looks up the correct `REGION_PROFILE` (e.g., `[-200, 300] HU`), and passes each paired series into the core processing function: `process_orientation_pair`.

---

## ⚙️ The Core Logic (`process_orientation_pair()`)

This function handles the extraction of a single CT/MRI pair (like the Axial view of PA0).

### Step 1: Loading & N4 Correction
It loads the DICOM folders using SimpleITK. Because MRI receive coils are not uniformly sensitive across their field of view, it immediately passes the MRI into `img_proc.apply_n4_bias_correction` (from `image_processing.py`) to flatten the resulting shading.

This runs on the **whole 3D volume at once**, and it has to happen here, first. Every step after it works one slice at a time, and once the stack has been taken apart the through-plane half of the bias field is no longer visible to anything. `orientation` is passed through because the B-spline mesh is anisotropic — lots of control points in-plane, the bare cubic minimum through-plane — and which anatomical direction each in-plane axis carries depends on the acquisition plane. See [`mri_pipeline_docs.md` §5](./mri_pipeline_docs.md).

The relevant flag is `--n4_shrink` (default `4`). It is applied **in-plane only** — the slice axis is never downsampled, because these stacks are only 15–24 slices deep.

### Step 2: Physical Grid Matching (The Alignment)
This is where the magic happens.
- It resamples the CT to `1.0mm` isotropic resolution via `img_proc.resample_inplane`.
- It projects the MRI perfectly onto the CT's physical grid using `img_proc.resample_mri_to_ct_grid`.
- Because they are now on the exact same physical grid, we are guaranteed that Slice #15 of the CT perfectly matches Slice #15 of the MRI!

### Step 3: Slicing to 2D
It uses `img_proc.volume_to_slices` to shatter the 3D SimpleITK volumes into a list of 2D Numpy arrays.

### Step 3.5: (Optional) Measure one shift for the whole stack
If `--register_2d` was passed, `img_proc.estimate_volume_translation` slides the MRI over the CT — a whole pixel at a time, best normalised mutual information wins — on a handful of probe slices, and pools their answers into a **single** translation for the entire stack. It applies that shift only if the probes agree and it demonstrably improves NMI; otherwise the MRI is left exactly where its DICOM origin put it. One shift per stack rather than one per slice is what stops the MRI shearing through z. See `mri_pipeline_docs.md` §9.

### Step 4: Iterating over Slices (The `for` loop)
For every matching 2D slice in the CT and MRI:
1. **(Optional) Shift**: If a stack-wide shift was accepted above, `img_proc.apply_translation` moves this MRI slice by it — the same shift for every slice.
2. **Normalize**: It applies the Hounsfield Window to the CT (`norm.normalize_ct_slice`), and the Percentile Clipping to the MRI (`norm.normalize_mri_slice`), scaling both to `[0, 1]`.
3. **Flag Background**: If `90%` of the slice is black air, it's tagged `is_background=True` in `metadata.csv` — but still saved. Slices near the FOV edge can carry a thin sliver of real anatomy, so nothing is silently discarded here; filtering (if wanted) is left to the GAN's dataloader.
4. ~~**Crop & Pad**~~: **Removed.** The pipeline no longer crops. Slices are kept at their native post-resample size, and cropping is done by the GAN's dataloader instead — so you can change crop strategy or input resolution without re-running preprocessing.

### Step 5: Saving
Finally, it saves the aligned, `[0,1]` normalized 2D numpy arrays into the output folder as `.npy` files, along with their dimensions (`height`, `width`) and the `is_background` flag in `metadata.csv`.

> ⚠️ Because slices are no longer a uniform square, **`torch.stack` will fail on a naive batch**. Your `Dataset.__getitem__` must crop or pad to a common size first — see [`mri_pipeline_docs.md`](./mri_pipeline_docs.md) §12 for a drop-in implementation. Apply the *same* crop to the CT and the MRI of a pair, or you will destroy their pixel-level alignment.
