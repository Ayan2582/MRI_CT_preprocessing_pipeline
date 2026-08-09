# CT / MRI Registration QC

A local browser tool for reviewing the output of the existing preprocessing
pipeline, pair by pair, and deciding what goes into the training set.

**It does not implement registration.** Every numerical step is a call into
`Preprocessing/`. No file in `Preprocessing/` was modified to build this.

---

## Setup

```bash
pip install -r qc_app/requirements.txt
```

## Run

```bash
python -m qc_app
```

Scans the dataset on first start (~15 s for 240 series), then opens
<http://127.0.0.1:8000>. Restarting is instant — the scan is skipped once the
manifest is populated.

```bash
python -m qc_app --help                      # all options
python -m qc_app --scan_only                 # validate the dataset, print the report, exit
python -m qc_app --data_root D:/other/data   # a different dataset
python -m qc_app --rescan                    # re-walk after adding patients
python -m qc_app --reg_search_mm 90          # widen the search (cost is quadratic)
```

## Dataset configuration

The default `--data_root` is `pipeline_config.DATA_ROOT`, the repository copy.
The expected layout is what is actually on disk:

```
Rawdata_dicom/
  CT/  PA0_Ranjeet/ST0/SE0/IM0, IM1, …      (no file extension)
  MRI/ PA0_Ranjeet/ST0/SE0/IM0, IM1, …
```

Patient and series names are never hard-coded. Any folder name works, as long
as the `CT/<patient>/<study>/<series>/` shape holds on both sides.

**The input tree is opened read-only. Nothing is ever written into it.**

---

## A) How CT–MRI pairing works

Two levels, because the pipeline works at one and a reviewer works at the other.

### Series pairing

`CT/PAx/ST0/SE0` ↔ `MRI/PAx/ST0/SE0`, matched by the production rule from
`preprocess_2d.py:253` — compare `name.split('_')[0]`.

That rule is not cosmetic. Eight MRI series in this dataset are suffix-named
(`SE1_axial`, `SE0_sagittal`) against a plain `SE1`/`SE0` on the CT side, and
only the base-name comparison pairs them. A CT series claimed by two MRI series
is reported as **ambiguous** and skipped rather than handed to whichever was
enumerated first.

### Slice pairing

Index *i* of the CT stack ↔ index *i* of the MRI stack — but *i* is **not
filename order**.

`GetGDCMSeriesFileNames` sorts by `ImagePositionPatient`, the geometric order,
and that is the order `io_utils.load_dicom_series` builds its volume in. So it
is the order every array index in this tool refers to. (`IM10` sorts before
`IM2` alphabetically; neither alphabetical nor numeric filename order is
guaranteed to be the spatial order.) The source filename at each index is
recorded in the manifest, which is how the UI names the DICOM you are looking
at without ever re-deriving it from a sort.

The MRI is resampled **onto the CT grid**, so the pair count is always the CT's
slice count. A shorter MRI does not shorten the output; it produces zero-filled
regions where it does not reach.

### What the scan verifies

Missing CT or MRI for a patient · series with no counterpart · ambiguous
many-to-one matches · folders holding more than one DICOM series UID ·
unreadable series · single-slice scouts · differing CT/MRI slice counts. All
reported before processing, none fatal.

Measured on this dataset: **120 series pairs, 2313 slice pairs, 45 patients,
zero count mismatches.**

---

## B) How `registration_idea.py` is integrated

Through `image_processing.estimate_volume_translation`, exactly as the
production pipeline uses it. `registration_service.process_series` performs the
same calls in the same order as `pipeline_core.process_orientation_pair`:

| Step | Call |
|---|---|
| 1 | `io_utils.load_dicom_series` — series folder → 3D volume |
| 2 | `img_proc.apply_n4_bias_correction` — N4 on the **whole** MRI volume |
| 3 | `img_proc.resample_inplane` — CT → 1 mm in-plane, z untouched |
| 4 | `img_proc.resample_mri_to_ct_grid` — MRI → the CT's physical grid |
| 5 | `img_proc.volume_to_slices` |
| 6 | `norm.compute_mri_percentiles` — whole volume, before any shift |
| 7 | `img_proc.estimate_volume_translation` → `registration_idea.register` |
| 8 | `img_proc.apply_translation` — one shift, every slice |
| 9 | `norm.normalize_ct_slice` / `normalize_mri_slice` / `is_background_slice` |

### Why the unit is a series, not an image

`estimate_volume_translation` produces **one shift per stack, on purpose**.
`pipeline_config.py:140-153` records the measurement behind that: the best
per-slice shift across one shoulder axial stack swings **85 mm**, and applying
those per-slice would shear the MRI through z — continuous anatomy would come
out as a staircase.

Registering each slice independently would therefore have been substituting a
different algorithm for the one in the repository. So this tool **registers per
series and reviews per slice**: 120 registration jobs, 2313 review decisions.

### The four acceptance checks, surfaced in the UI

A shift is applied only if it survives all of them, and the metrics panel shows
each one:

0. probes whose best shift landed on the edge of the search square are
   discarded as censored (`hit_edge`)
1. at least `REG_MIN_PROBES` = 2 usable probes remain
2. they agree within `REG_MAX_SPREAD_MM` = 20 mm
3. the median, **re-scored on every probe**, gains more than
   `REG_MIN_GAIN` = 0.010 NMI

`reg_reason` is shown verbatim — a rejection always says why in plain English.

### One deliberate deviation

`resample_mri_to_ct_grid` takes an alias of its argument and then calls
`SetDirection` on it, mutating the caller's image in place — documented in
`docs/registration_docs.md` §7.3. The production file passes its only copy and
wears the mutation. This tool passes `sitk.Image(mri_corrected)`, a throwaway
copy, exactly as the old demo scripts did. **The production file is not
edited.**

---

## C) How the production pipeline is integrated

Outputs are written with `export_utils.save_npy` and
`export_utils.save_preview_png` — the same functions `pipeline_core` uses, so
the files are byte-for-byte what the production pipeline produces: float32
`.npy`, and the same side-by-side preview with the same divider.

`metadata.csv` carries **every column `preprocess_2d.py` writes, in its order**,
so an existing dataloader reads it unchanged:

```
patient_id, body_region, orientation, slice_index, ct_series, mri_series,
mri_desc, height, width, ct_npy, mri_npy, is_background,
reg_applied, reg_dx_mm, reg_dy_mm, reg_nmi_gain, reg_note
```

then the QC columns:

```
study, ct_file, mri_file, qc_status, qc_note,
roi_x, roi_y, roi_w, roi_h, roi_mode, reviewed_at
```

Region CT windows come from `cfg.REGION_PROFILES` via `cfg.PREFIX_TO_REGION`,
identically to production — an unknown patient prefix falls back to `default`
(−200…300 HU).

---

## Output structure

Mirrors the input hierarchy exactly. Nothing is flattened or renamed; the file
stem is the source DICOM's own name, taken from the CT side for CT and the MRI
side for MRI (they can differ — `SE0` vs `SE0_axial` — so both are recorded per
row).

```
qc_workspace/
  qc.db                                          the manifest (SQLite)
  cache/PA0_Ranjeet/ST0/SE0.npz                  registered slices, for review
  output/
    CT/PA0_Ranjeet/ST0/SE0/IM0.npy               normalised float32
    MRI/PA0_Ranjeet/ST0/SE0/IM0.npy              N4 → CT grid → shift → normalised
    previews/PA0_Ranjeet/ST0/SE0/IM0_pair.png
    registration/PA0_Ranjeet/ST0/SE0.json        the full registration record
    metadata.csv
```

Only accepted pairs are written. Rejecting a previously accepted pair **deletes
its output files**, so the training set never keeps a slice you just rejected.

---

## Cropping — and why it is valid

Draw a rectangle on the result view (`C`, then drag). It restricts **what the
NMI metric sees**, then re-runs the series.

This is a legitimate operation rather than naive array slicing for one specific
reason: by the time the ROI is applied, `resample_mri_to_ct_grid` has already
put the MRI on the CT's grid, so the two stacks share a coordinate system, a
shape, and a pixel size. **An index rectangle is therefore a physical
rectangle**, identical on both, and cropping cannot move one relative to the
other. At the default 1 mm target spacing the ROI is also directly in
millimetres, which is why the UI can label it `140 x 140 mm`.

Cropping raw DICOM arrays *before* that resample would not have this property —
their pixel sizes and origins differ.

Use it when a table, an arm or a coil dominates the frame and drags the match
away from the anatomy. Minimum 16×16 px; below that the joint histogram has too
few samples to be stable.

**Drawing a crop does not re-register.** The region is stored and the Re-run
button lights up; nothing is measured until you press it. Adjusting a rectangle
is usually not a request to throw away a registration that is already good.

By default the ROI affects **only the metric** — full-frame slices are still
exported, matching the production decision to leave cropping to the dataloader
(`pipeline_config.py:35-39`). Tick *Export cropped to ROI* to write the
sub-rectangle instead. Either way the ROI is recorded in `metadata.csv`.

---

## Manual nudge — `N`

For when registration is close but you can see it could be better. Arrow buttons,
or `Shift`+arrow keys; step size is settable in mm.

It is an **offset on top of** the measured shift, applied to the whole series
(the same reason the shift itself is per-series: per-slice offsets shear the MRI
through z). The measured value is never overwritten — `metadata.csv` keeps
`reg_dx_mm`/`reg_dy_mm` for what the algorithm found and `manual_dx_mm`/
`manual_dy_mm` for what you added, so a hand-adjusted pair stays auditable and
the total is just the sum.

**Only the relative offset exists.** The CT defines the frame, so moving the CT
by (+5, −3) and moving the MRI by (−5, +3) are the same pair. The dropdown
offers both directions for convenience and exactly one number is stored —
otherwise a pair could carry two offsets that disagree. Nudging the CT then
nudging the MRI by the same amount returns you to zero, as it should.

Applied to the cached result, so it is instant and needs no re-run. It is a
whole-pixel slide with no interpolation, so adjusting repeatedly never softens
the image. Offsets beyond ±2× the search range are refused — at that point the
answer is a wider `--reg_search_mm` and a re-run, not a nudge.

---

## Separate CT and MRI export rectangles

In **Crop** mode the target dropdown chooses what the rectangle applies to:

| Target | Effect |
|---|---|
| **both** | the shared ROI — restricts the NMI metric, and the export if *Export cropped* is on |
| **CT export only** | crops the saved CT `.npy` alone |
| **MRI export only** | crops the saved MRI `.npy` alone |

Two different rectangles are safe **because both are stated in the same 1 mm
frame** — each says which physical millimetres to keep from one shared
coordinate system, so neither moves relative to the other. This is not the same
as cropping two images by index in frames that were never reconciled.

Saved slices may then differ in size between modalities. `metadata.csv` records
`height`/`width` for the CT and `mri_height`/`mri_width` for the MRI, plus the
rectangles themselves, so a dataloader can size batches without opening files.
The side-by-side preview pads the shorter array to a common canvas; the `.npy`
files keep their true shapes.

Per-modality rectangles are **export only** — the metric still sees the full
frame, or the shared ROI if one is set. A per-modality rect takes precedence
over *Export cropped to ROI*.

---

## Erasing the bed, cradle and rails

Press `E` and paint. The brush removes the region from **both** the CT and the
MRI, in the rendered views and in the exported `.npy`.

| | |
|---|---|
| zoom | scroll — works the same in every mode, including while painting |
| pan | middle-drag or right-drag (left button is the brush) |
| brush size | `[` / `]`, `Alt`+scroll, or the slider — shown in mm (1 px = 1 mm) |
| undo | removes the last stroke |
| clear slice | drops every stroke on this slice |
| apply to series | copies this slice's strokes to all slices in the series |

The brush outline scales with the zoom, so what the circle covers is what gets
zeroed at any magnification.

Strokes are stored **per slice, as geometry** — a radius and a list of points,
a few hundred bytes — not as a bitmap. So they stay editable, survive
re-registration, and are rasterised identically on the server (round caps,
round joins) to what the red overlay shows you.

Painting takes effect in the images immediately. To fold it into the
*measurement*, press **Re-run**: the shift is estimated once per series from
probe slices, so only erasing a probe slice changes the number, and re-running
is the explicit way to ask for it.

*Apply to series* is the usual case for hardware — the patient does not move
within a series, so a rail sits in the same place on every slice.

### Why there is no automatic version

Both obvious approaches were tested on this dataset and both are unsafe:

- **2D largest-connected-component** removes the bed arc cleanly on axial
  slices — and deletes the patient's jaw, mouth and teeth on sagittal ones,
  where that region is separated from the skull by air *within the slice*.
  19.5% of one slice, silently.
- **3D largest-connected-component** keeps the jaw correctly and removes the
  thin rails, but leaves the arc: the head rests on the cradle, so they are one
  connected object in 3D. Only 0.5–2.2% removed.

A HU threshold does not separate them either — cradle and bone overlap in
density. The property that makes the arc removable in 2D is exactly what puts
anatomy at risk, so this stays a human decision.

---

## Persistent state and resume

Everything needed to resume is in `qc_workspace/qc.db`. Close the app mid-review
and reopen it: it continues where it was. Nothing is inferred from which output
files happen to exist.

- `series` — one row per series pair, holding registration state
  (`PENDING`/`PROCESSING`/`REGISTERED`/`ERROR`) and the full result
- `pairs` — one row per slice pair, holding QC state
  (`PENDING`/`ACCEPTED`/`REJECTED`/`ERROR`)
- `settings` — reviewer position, last scan report

A series left `PROCESSING` by an abrupt shutdown is returned to `PENDING` at
startup, so it is picked up again instead of sitting in a state that only looks
busy.

Re-registering a series **never overwrites a human decision.** Accepted and
rejected stay as set; only slices previously marked `ERROR` are returned to
`PENDING`.

---

## Interface

| Key | Action | | Key | Action |
|---|---|---|---|---|
| `A` | Accept | | `1`–`4` | Overlay / Fusion / Checker / Difference |
| `R` | Reject | | `B` | Toggle unregistered MRI |
| `C` | Crop region | | `F` | Fit to window |
| `E` | Erase brush | | `[` / `]` | Brush smaller / larger |
| `N` | Nudge | | `Shift`+arrows | Nudge by one step |
| `→` / `Space` | Next | | `←` | Previous |
| `Esc` | Cancel crop / erase | | `L` | Link / unlink views |
| double-click | Fit this pane | | | |

Four result views, because they fail differently:

- **Overlay** — opacity slider, blended in the browser so it is instant
- **Fusion** — CT green, MRI magenta; misalignment shows as colour fringing on
  every edge, which the eye catches faster than an offset
- **Checker** — alternating tiles; a structure crossing a tile boundary steps
  exactly where the two disagree
- **Difference** — bright where they disagree. Never zero (different
  modalities); read it as *structure*, not brightness — sharp outlines mean
  edges that do not coincide, a diffuse glow is just modality contrast

*Show unregistered* flips to the pre-shift MRI — the fastest way to confirm a
shift actually helped.

### Linked views

Each pane has its own transform, fitted to its own viewport. With **Link views**
on (the default), a pan or zoom in one pane is applied to the others as a
*delta*: they end up at the same magnification showing the same region, each
still centred in its own pane. Comparing two images is only meaningful when both
show the same region at the same magnification, which is why it defaults on.

Press `L` or untick it to move a pane on its own. **Panning a pane never changes
the registration** — it moves pixels on screen only, and the metrics are
unaffected. To actually change the alignment, use *Crop region* or *Re-run*.

**Whole series ▾** applies one decision to all slices of a series. Since the
shift is estimated per series, a shift that is wrong is wrong for the whole
stack; individual slices can still be overridden afterwards.

---

## Performance

The dataset is never loaded into memory. One background thread registers one
series at a time — peak memory is one CT volume plus one MRI volume plus their
resampled copies. Slices reach the browser from the on-disk cache through a
3-slot LRU.

A thread rather than a process pool because the expensive work (N4, resampling,
the NMI sweep) is SimpleITK and numpy releasing the GIL; a pool would add
Windows spawn cost, a second SimpleITK import per worker, and volume pickling
for very little gain.

The worker registers `--prefetch` series ahead of you, at low priority, so
paging forward does not stall. Anything you ask for directly overtakes queued
prefetch work.

Measured: **~15 s per series** (18 slices, ±40 mm search, N4 shrink 4).
120 series ≈ 30 minutes to register the whole dataset in the background while
you review.

---

## Testing procedure

```bash
python -m qc_app --scan_only        # expect: 120 series, 2313 pairs, 0 problems
```

Then start the app and check a small subset — `PA0_Ranjeet/ST0/SE0`, slices 0-2:

1. The first pair renders in all four views
2. Metrics show `applied=yes`, `−8, −7 mm`, gain `+0.0412`, 5 of 5 probes within 3 mm
3. **Accept** → files appear under `output/CT/PA0_Ranjeet/ST0/SE0/IM0.npy`
4. **Reject** → those files disappear again
5. **Crop** a region → `dy` changes from −7 to −5 mm, gain from 0.0412 to 0.0359
6. Close the app, reopen → counts and position unchanged
7. `Export CSV` → `metadata.csv` with the production column order

What was verified during the build, on the real dataset:

| Check | Result |
|---|---|
| Pairing correctness | 120/120 series, 2313/2313 pairs, 0 mismatches |
| Registration uses the existing implementation | `estimate_volume_translation` → `registration_idea.register` |
| All 8 image views render | PNG, 200 OK |
| Accept / reject / reset, with output creation and deletion | pass |
| Crop changes the measurement | dy −7 → −5 mm, gain 0.0412 → 0.0359 |
| Output paths mirror input | `CT/PZ99_NewPatient/ST0/SE0/IM0.npy` |
| Restart preserves progress | counts identical across restart |
| A failed series does not stop the queue | marked `ERROR`, queue kept running, retry recovered it |
| New patient, unseen name, suffixed MRI series | paired, registered, accepted |
| Corrupt / orphan / MRI-less series | reported, skipped, non-fatal |
| Input tree unmodified | mtimes unchanged |
| No NaN reaches a saved array | pass |

---

## Changes to existing files

**`Preprocessing/` pipeline modules: none.** `preprocess_2d.py`,
`pipeline_core.py`, `image_processing.py`, `registration_idea.py`,
`io_utils.py`, `normalization.py`, `export_utils.py` and `pipeline_config.py`
are untouched.

Separately, at your request, the superseded registration implementations were
removed: `registration_demo*.py`, `registration_og.py`, `sweep_og.py` and
`working_regis.py` — ITK rigid/affine multi-start with scale gates and crop
fallbacks, a different method from the production one. `sweep_idea.py` was kept
(it exercises `registration_idea` and reproduces the evidence cited in
`pipeline_config.py:150`) with its one constant from the deleted files inlined.
Their CSV outputs remain under `registration_demo_output/`, and the scripts are
recoverable from git history.
