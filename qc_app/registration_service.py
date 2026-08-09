"""
registration_service.py
───────────────────────
Runs the EXISTING production pipeline on one series pair and caches the result.

Nothing in this file implements registration. Every numerical step is a call
into Preprocessing/, in the same order pipeline_core.process_orientation_pair
performs them:

    io_utils.load_dicom_series          series folder -> 3D volume
    img_proc.apply_n4_bias_correction   N4 on the WHOLE MRI volume
    registration_idea.to_1mm            each slice -> 1 mm per pixel  (2D)
    norm.compute_mri_percentiles        intensity bounds, whole volume
    img_proc.estimate_volume_translation  ONE shift for the stack
    img_proc.apply_translation          that shift, on every slice
    norm.normalize_ct_slice / _mri_slice
    norm.is_background_slice

THE GEOMETRY IS 2D, AND ONLY 2D
───────────────────────────────
PixelSpacing is the only DICOM geometry read. No origins, no direction
cosines, no world coordinates - the same restriction registration_idea.py
works under, and the reason it has no geometric failure mode.

An earlier version of this file followed pipeline_core exactly and used
img_proc.resample_mri_to_ct_grid to project the MRI onto the CT's 3D grid.
That step is the one thing in the production chain that depends on the two
acquisitions sharing a world frame, and on this dataset they mostly do not:
the median CT/MRI frame mismatch is 5.2 degrees and 61 of 120 series exceed
5 degrees. Projecting through it either asserts a false direction (rotating
the slab until end slices fall outside coverage and come out blank) or
honours the true one (reformatting an oblique volume into a thin sliver).
Measured both ways, each choice helped some series and destroyed others.

Pairing slice i with slice i and sliding in 2D has neither failure. It is
sound on this dataset because CT and MRI slice counts match exactly in all
120 series (2313 files on each side, zero mismatches).

WHY THE UNIT IS A SERIES
────────────────────────
estimate_volume_translation produces one shift per stack, on purpose.
pipeline_config.py:140-153 records the measurement behind that decision: the
best per-slice shift across one shoulder axial stack swings 85 mm, and applying
those per-slice would shear the MRI through z. Registering each slice
independently here would be substituting a different algorithm for the one in
the repository, so this application registers per series and reviews per slice.

Not using resample_mri_to_ct_grid also sidesteps its in-place mutation of the
caller's image (an alias plus SetDirection, documented in
docs/registration_docs.md §7.3). Nothing here writes to a loaded volume.

The production files are not edited by any of this.
"""

import logging
import os
import time
import traceback

import numpy as np
import SimpleITK as sitk

from . import bootstrap, scanner

_pp = bootstrap.preprocessing_modules()
cfg      = _pp.cfg
io_utils = _pp.io_utils
img_proc = _pp.img_proc
norm     = _pp.norm
reg_idea = _pp.reg_idea

logger = logging.getLogger(__name__)


class SeriesProcessingError(RuntimeError):
    """Raised for a failure that belongs to one series and must not stop the queue."""


def _ordered_names(series_dir):
    """Basenames in the same order load_dicom_series builds its volume."""
    return [os.path.basename(n) for n in
            sitk.ImageSeriesReader.GetGDCMSeriesFileNames(series_dir)]


def slices_1mm_per_file(series_dir):
    """
    Read a series ONE FILE AT A TIME, resampling each slice to 1 mm per pixel.

    The fallback for series SimpleITK cannot stack into a volume. Its
    ImageSeriesReader builds a single 3D array and requires every slice to have
    the same pixel dimensions, taking the FIRST file's dimensions as the
    requirement - so one odd slice rejects the entire series.

    That constraint is SimpleITK's, not ours. This method resamples every slice
    to 1 mm per pixel independently, and after that step a differently-sized
    slice is indistinguishable from its neighbours. Measured on the one series
    in this dataset that trips it, PA16_SumanLata2/ST0/SE0:

        IM0        512x512  @ 0.445 mm  ->  228 x 228 px
        IM1..IM20 1024x1024 @ 0.223 mm  ->  228 x 228 px

    Same 228 mm field of view, same output grid. IM0 is not corrupt - it was
    reconstructed on a coarser matrix - so dropping it, or dropping the twenty
    slices that disagree with it, would both be discarding good data over a
    constraint that does not apply here.

    Returns (slices at 1 mm, basenames), in the reader's geometric order so it
    lines up with _ordered_names.
    """
    import pydicom

    names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(series_dir)
    slices, kept = [], []
    for path in names:
        try:
            d = pydicom.dcmread(path)
            a = d.pixel_array.astype(np.float64)
            # Real units, not stored values - the same rescale io_utils gets
            # from the series reader.
            a = (a * float(getattr(d, "RescaleSlope", 1) or 1)
                 + float(getattr(d, "RescaleIntercept", 0) or 0))
            sp = getattr(d, "PixelSpacing", None)
            sp = (float(sp[0]), float(sp[1])) if sp is not None else (1.0, 1.0)
            slices.append(reg_idea.to_1mm(a, sp))
            kept.append(os.path.basename(path))
        except Exception as e:
            # One unreadable file must not cost the series. Report and continue.
            logger.warning("skipping %s: %s", path, e)
    return slices, kept


def load_modality_slices(series_dir, label, want_volume=False):
    """
    A series as 1 mm slices, plus its volume when one could be built.

    `want_volume=True` for the MRI, which needs a genuine 3D volume for the N4
    fit. The CT never does - it is only ever consumed slice by slice - so it
    goes straight to whichever path works.

    Returns (slices, names, volume_or_None).
    """
    image, n = io_utils.load_dicom_series(series_dir)
    if image is not None and n >= 2:
        return slices_at_1mm(image), _ordered_names(series_dir), image

    slices, names = slices_1mm_per_file(series_dir)
    if len(slices) < 2:
        raise SeriesProcessingError(
            f"{label} series could not be read at all ({len(slices)} slices): {series_dir}")
    logger.warning(
        "%s: %s could not be stacked into a volume; read %d slices individually",
        series_dir, label, len(slices))
    return slices, names, None


def slices_at_1mm(image):
    """
    A volume's slices as 2D arrays at exactly 1 mm per pixel.

    This is registration_idea.to_1mm applied slice by slice, and it is the ONLY
    geometry this application uses. PixelSpacing and nothing else - no origin,
    no direction cosines, no world coordinates.

    That restriction is the whole point. Once both modalities are at 1 mm per
    pixel a 50 mm structure spans 50 pixels in each, so they are directly
    comparable and the only thing left to find is where one sits relative to
    the other - which is exactly what registration_idea.register searches for.

    SimpleITK spacing is (x, y, z); the numpy array is (z, y, x). So a slice's
    (row, column) spacing is (spacing[1], spacing[0]).
    """
    arr = sitk.GetArrayFromImage(image)
    sp = image.GetSpacing()
    row_col = (float(sp[1]), float(sp[0]))
    return [reg_idea.to_1mm(arr[i].astype(np.float64), row_col)
            for i in range(arr.shape[0])]


def mri_on_ct_frame(mri_slice, ct_shape, dy=0, dx=0):
    """
    One MRI slice laid over the CT's frame, via registration_idea.sample_window.

    The two stacks do not need the same pixel dimensions: sample_window centres
    the MRI on the CT frame and fills anything the MRI does not reach with 0.0.
    That is how a 384x384 MRI and a 512x512 CT become comparable without either
    being reprojected through world coordinates.
    """
    shifted, _ = reg_idea.apply_shift(
        np.asarray(mri_slice, dtype=np.float64), tuple(ct_shape), int(dy), int(dx), fill=0.0)
    return shifted


def rasterize_strokes(strokes, shape):
    """
    Turn brush strokes into a boolean mask, True where the reviewer painted.

    `strokes` is a list of {"r": radius_px, "pts": [[x, y], ...]} in image
    pixels, which at 1 mm per pixel are also millimetres. Stored as geometry
    rather than as a bitmap so a stroke stays editable, costs a few hundred
    bytes instead of a second image, and survives re-registration unchanged.

    Returns None when there is nothing painted, so callers can skip the work
    entirely on the overwhelming majority of slices.
    """
    if not strokes:
        return None

    from PIL import Image as _Image, ImageDraw as _ImageDraw

    h, w = int(shape[0]), int(shape[1])
    img = _Image.new("1", (w, h), 0)
    draw = _ImageDraw.Draw(img)
    painted = False

    for s in strokes:
        pts = [(float(p[0]), float(p[1])) for p in (s.get("pts") or [])]
        if not pts:
            continue
        r = max(0.5, float(s.get("r", 8)))
        if len(pts) > 1:
            # joint="curve" rounds the corners between segments, so a fast
            # drag does not leave notches where the polyline turns.
            draw.line(pts, fill=1, width=int(round(r * 2)), joint="curve")
        # Round caps at every vertex, which is also what makes a single click
        # paint a dot rather than nothing.
        for (x, y) in pts:
            draw.ellipse([x - r, y - r, x + r, y + r], fill=1)
        painted = True

    return np.array(img, dtype=bool) if painted else None


def thin_structure_mask(ct_slice, thresh=0.12, max_thickness=6.0, min_area=20):
    """
    Everything that is thin and NOT part of the main body - table rails, the
    head cradle, the edge of a positioning board.

    Rails are distinguished from anatomy by SHAPE, not size. A rail is a long
    narrow streak; anatomy is blobby. Thickness here is area divided by the
    longer side of the bounding box, which is a rail's width in pixels.

    Measured on this dataset:

        PA10_Suman/SE0 slice 9
          component  2: bbox 163x10,  thickness  2.80 px  -> rail
          component 15: bbox 126x18,  thickness  2.21 px  -> rail
          component 16: bbox 116x86,  thickness 42.15 px  -> anatomy

    That last one is 21% of the largest component, so a size threshold would
    have had to choose between deleting it and keeping the rails. Thickness
    separates them with a wide margin.

    The largest component is never touched, whatever its shape, so the body
    itself cannot be removed. Returns None when there is nothing to remove.
    """
    try:
        from scipy import ndimage
    except ImportError:
        return None

    bw = ndimage.binary_closing(np.asarray(ct_slice) > thresh, np.ones((3, 3)))
    lab, n = ndimage.label(bw)
    if n < 2:
        return None

    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    boxes = ndimage.find_objects(lab)

    out = np.zeros(lab.shape, dtype=bool)
    for i in range(1, n + 1):
        if i == biggest or sizes[i - 1] < min_area:
            continue
        sl = boxes[i - 1]
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if sizes[i - 1] / max(h, w, 1) < max_thickness:
            out |= (lab == i)
    # A touch of dilation so the rail's soft edge goes with it.
    return ndimage.binary_dilation(out, np.ones((3, 3))) if out.any() else None


def apply_erase(arr, mask, fill=0.0):
    """
    Blank the painted region of one slice.

    Used identically on the CT and the MRI. Erasing both keeps a pair
    consistent: if a region is excluded from the CT it must be excluded from
    its MRI too, or the network is asked to invent CT for MRI that has no
    counterpart - the reverse of the problem the erase exists to solve.

    `fill` is 0.0 everywhere, the same value the pipeline already uses for
    "no data here", so an erased region is indistinguishable downstream from
    an area the MRI never reached.
    """
    if mask is None:
        return arr
    out = np.array(arr, copy=True)
    out[mask[:out.shape[0], :out.shape[1]]] = fill
    return out


def apply_nudge(mri_slice, dy, dx):
    """
    Slide an already-registered MRI slice by a manual offset.

    Applied to the CACHED result rather than re-running registration, so a
    nudge is instant and costs nothing. It is the same whole-pixel slide
    registration_idea uses - no interpolation, so nudging does not soften the
    image however many times it is adjusted.

    Only the relative offset between the two modalities has any meaning: the
    CT defines the frame, so "move the CT by (+dx, +dy)" and "move the MRI by
    (-dx, -dy)" describe the identical pair. The UI offers both directions for
    convenience and stores exactly one number, which is what stops a pair from
    carrying two offsets that disagree.
    """
    if not dy and not dx:
        return mri_slice
    shifted, _ = reg_idea.apply_shift(
        np.asarray(mri_slice, dtype=np.float64), mri_slice.shape,
        int(dy), int(dx), fill=0.0)
    return shifted.astype(np.asarray(mri_slice).dtype, copy=False)


def crop_to_rect(arr, rect):
    """
    Cut an array down to an export rectangle given in the common 1 mm frame.

    CT and MRI may be given DIFFERENT rectangles. That is safe here, and only
    here, because both are expressed in the same frame: the rectangle says
    which physical millimetres to keep, so two different rectangles keep two
    different regions of one shared coordinate system rather than moving one
    image relative to the other. Cropping by index in two frames that were
    never reconciled is the thing this must not be confused with.

    The saved slices then differ in size between modalities, which is the
    caller's business to know about - it is recorded per row in metadata.csv.
    """
    if not rect:
        return arr
    x, y, w, h = (int(round(v)) for v in rect)
    H, W = arr.shape
    x0, y0 = max(0, min(x, W - 1)), max(0, min(y, H - 1))
    x1, y1 = max(x0 + 1, min(x + w, W)), max(y0 + 1, min(y + h, H))
    return arr[y0:y1, x0:x1]


def region_for_patient(patient_id: str) -> str:
    """The production mapping, patient folder prefix -> body region."""
    return cfg.PREFIX_TO_REGION.get(patient_id.split("_")[0], "default")


def ct_window_for_region(region: str):
    profile = cfg.REGION_PROFILES.get(region, cfg.REGION_PROFILES["default"])
    return float(profile["ct_win_min"]), float(profile["ct_win_max"])


def cache_path_for(app_cfg, series_key: str) -> str:
    return os.path.join(app_cfg.cache_dir, *series_key.split("/")) + ".npz"


def _crop(arrays, roi):
    """
    Apply an ROI rectangle to a list of same-shaped slices.

    The caller is responsible for having put both stacks on the same frame
    first (see the ROI branch in process_series, which lays the MRI onto the
    CT frame at zero shift before cropping). Once they share a frame and a
    1 mm pixel, an index rectangle IS a physical rectangle, identical on both,
    so cropping cannot move one relative to the other.

    roi is (x, y, w, h) in pixels, which at 1 mm per pixel is also millimetres.
    """
    if roi is None:
        return arrays
    x, y, w, h = (int(round(v)) for v in roi)
    out = []
    for a in arrays:
        H, W = a.shape
        x0, y0 = max(0, min(x, W - 1)), max(0, min(y, H - 1))
        x1, y1 = max(x0 + 1, min(x + w, W)), max(y0 + 1, min(y + h, H))
        out.append(a[y0:y1, x0:x1])
    return out


def process_series(series_row: dict, app_cfg, roi=None, search_mm=None,
                   progress=None, erase=None) -> dict:
    """
    Register one CT/MRI series pair and write its cache.

    Returns a dict of everything the manifest and the UI need. Raises
    SeriesProcessingError on failure so the caller can mark this one series
    ERROR and carry on with the queue.
    """
    t0 = time.time()
    key = series_row["key"]
    orientation = series_row.get("orientation") or "default"
    search_mm = float(search_mm if search_mm is not None else app_cfg.reg_search_mm)

    def step(msg):
        if progress:
            progress(msg)

    # ── load ──────────────────────────────────────────────────────────────────
    # The CT is only ever used slice by slice, so it never needs to stack into
    # a volume. If SimpleITK cannot build one, the per-file path reads it
    # anyway - see slices_1mm_per_file.
    step("loading CT series")
    ct_all, ct_names, _ct_vol = load_modality_slices(series_row["ct_path"], "CT")

    step("loading MRI series")
    mri_all, mri_names, mri_image = load_modality_slices(
        series_row["mri_path"], "MRI", want_volume=True)

    # ── N4, on the whole volume ───────────────────────────────────────────────
    # N4 fits a 3D B-spline, so it genuinely needs the volume. A series that
    # could only be read slice by slice is processed WITHOUT bias correction
    # rather than not at all - stated here and recorded in the result, because
    # an uncorrected MRI is a real difference in the output, not a detail.
    n4_applied = mri_image is not None
    if n4_applied:
        step(f"N4 bias correction ({len(mri_all)} slices, 3D fit)")
        mri_corrected = img_proc.apply_n4_bias_correction(
            mri_image, orientation=orientation, shrink_factor=app_cfg.n4_shrink)
        mri_all = slices_at_1mm(mri_corrected)
    else:
        logger.warning("%s: MRI could not be stacked into a volume - "
                       "N4 bias correction SKIPPED for this series", key)
        mri_corrected = None

    # ── to 1 mm per pixel, in 2D, exactly as registration_idea does ──────────
    #
    # NOT resample_mri_to_ct_grid. That projects the MRI through world
    # coordinates onto the CT's 3D grid, and on this dataset that step is where
    # the damage happens: the two acquisitions rarely share a frame (median
    # mismatch 5.2 degrees, 61 of 120 series above 5), so the projection either
    # asserts a false direction and rotates the slab out of coverage, or
    # honours the real one and reformats an oblique volume into slivers. Both
    # produce blank or near-blank MRI slices out of data that is perfectly
    # usable.
    #
    # registration_idea.py does not have that failure mode because it never
    # uses world coordinates. It reads PixelSpacing, puts both modalities at
    # 1 mm per pixel, and slides one over the other. Slice i of the CT pairs
    # with slice i of the MRI - which is sound here because every series in
    # this dataset has identical CT and MRI slice counts (verified: 120/120
    # series, 2313 files each side, zero mismatches).
    # Reorder into IMAGE-NUMBER order, which is how the manifest pairs them.
    # The volumes come back in ImagePositionPatient order, and on 24 of the 120
    # series here that order disagrees with the file numbering - or between the
    # two modalities, four of them fully reversed. Selecting by name is what
    # makes slice i of this list the same IMn on both sides.
    step("pairing slices by image number")
    ct_pos = {n: i for i, n in enumerate(ct_names)}
    mri_pos = {n: i for i, n in enumerate(mri_names)}

    ct_slices, mri_slices = [], []
    for c_name, m_name in scanner.pair_slice_names(ct_names, mri_names):
        ci, mi = ct_pos.get(c_name), mri_pos.get(m_name)
        if ci is None or mi is None or ci >= len(ct_all) or mi >= len(mri_all):
            continue
        ct_slices.append(ct_all[ci])
        mri_slices.append(mri_all[mi])

    n_pair = len(ct_slices)
    if n_pair == 0:
        raise SeriesProcessingError("no CT/MRI slices could be paired")
    if n_pair != len(ct_all) or n_pair != len(mri_all):
        logger.warning("%s: CT has %d slices, MRI has %d, %d paired by image number",
                       key, len(ct_all), len(mri_all), n_pair)

    # ── intensity bounds, whole volume, before any shift ─────────────────────
    # Same ordering as production: shifting moves pixels but does not change
    # their values, and the 0.0 fill it introduces is excluded along with the
    # existing background. Taken from the N4-corrected volume in its own
    # geometry, which is where the MRI's real intensities live.
    # From the N4-corrected volume in its own geometry where there is one; from
    # the 1 mm slices otherwise. The percentiles are of non-zero voxels, which
    # resampling barely moves, so the two agree closely.
    mri_vol = (sitk.GetArrayFromImage(mri_corrected).astype(np.float32)
               if mri_corrected is not None
               else np.stack([np.asarray(s_, dtype=np.float32) for s_ in mri_all]))
    p_low, p_high = norm.compute_mri_percentiles(
        mri_vol, cfg.MRI_PERCENTILE_LOW, cfg.MRI_PERCENTILE_HIGH)

    # ── the registration itself ──────────────────────────────────────────────
    step(f"estimating one in-plane shift (+/-{search_mm:.0f} mm, "
         f"{cfg.REG_N_PROBES} probes)")
    # ── erased regions, before anything is measured ──────────────────────────
    # Painted-out pixels (bed rails, head cradle, table) are removed from the
    # CT BEFORE the metric sees them. They are CT-only structures with no MRI
    # counterpart, so leaving them in makes the NMI score a correspondence that
    # cannot exist. Only slices the reviewer actually painted are touched, and
    # only the probe slices influence the shift - but every slice is erased
    # here so that what is measured and what is exported are the same pixels.
    masks = [None] * n_pair
    if erase:
        for i in range(n_pair):
            masks[i] = rasterize_strokes(erase.get(i), ct_slices[i].shape)
            if masks[i] is not None:
                ct_slices[i] = apply_erase(ct_slices[i], masks[i])
        n_painted = sum(1 for m in masks if m is not None)
        if n_painted:
            step(f"applied erase on {n_painted} slice(s)")

    # `roi` may be a single rectangle for the whole stack, or a dict keyed by
    # slice index. Per-slice is the general form: crops are a slice property,
    # and each probe is searched independently, so a probe cropped tightly and
    # its neighbour left whole are both self-consistent - every candidate shift
    # for a given probe is scored on that probe's own fixed window.
    if isinstance(roi, dict):
        roi_of = lambda i: roi.get(i)
        any_roi = bool(roi)
    else:
        roi_of = lambda i: roi
        any_roi = roi is not None

    if not any_roi:
        # The ordinary path: hand registration_idea the two stacks as they are.
        # sample_window centres the MRI on the CT frame itself, so the two do
        # not need matching pixel dimensions.
        metric_ct = ct_slices[:n_pair]
        metric_mri = mri_slices[:n_pair]
    else:
        # With an ROI the two must be cropped by the SAME physical rectangle,
        # which means putting the MRI on the CT frame first (at zero shift, the
        # do-nothing baseline) so an index rectangle means the same thing in
        # both. Both are at 1 mm per pixel, so the rectangle is also in mm.
        #
        # Caveat worth knowing: MRI that falls outside the CT frame becomes 0
        # in this step and cannot slide back into view, so a very large shift
        # is slightly under-measured under an ROI. The ROI exists to focus the
        # metric on anatomy, not to find large offsets.
        metric_ct, metric_mri = [], []
        for i in range(n_pair):
            r = roi_of(i)
            if r:
                framed = mri_on_ct_frame(mri_slices[i], ct_slices[i].shape)
                metric_ct.append(_crop([ct_slices[i]], r)[0])
                metric_mri.append(_crop([framed], r)[0])
            else:
                metric_ct.append(ct_slices[i])
                metric_mri.append(mri_slices[i])
    reg = img_proc.estimate_volume_translation(
        metric_ct, metric_mri,
        spacing_mm=app_cfg.target_spacing,
        search_mm=search_mm)

    # ── normalise and assemble what the reviewer will see ────────────────────
    step("normalising slices")
    region = series_row.get("body_region") or region_for_patient(series_row["patient"])
    ct_win_min, ct_win_max = ct_window_for_region(region)

    ct_out, mri_before, mri_after, is_bg = [], [], [], []
    for i in range(n_pair):
        ct_slice, mri_slice = ct_slices[i], mri_slices[i]

        ct_norm = norm.normalize_ct_slice(ct_slice, ct_win_min, ct_win_max)

        # Both the before and after images are laid on the CT's frame, so every
        # array the reviewer sees and every array saved has the CT's shape.
        # "Before" is the do-nothing baseline, the same (0, 0) the metric
        # scores against.
        before_norm = norm.normalize_mri_slice(
            mri_on_ct_frame(mri_slice, ct_slice.shape), p_low, p_high)

        if reg["applied"]:
            shifted = img_proc.apply_translation(
                mri_slice, ct_slice.shape, reg["dy"], reg["dx"])
            after_norm = norm.normalize_mri_slice(shifted, p_low, p_high)
        else:
            after_norm = before_norm

        # The same painted region is taken out of the MRI. Both are on the CT
        # frame here, so one mask applies to both - and a pair must not have a
        # region present in one modality and blanked in the other.
        if masks[i] is not None:
            before_norm = apply_erase(before_norm, masks[i])
            after_norm = apply_erase(after_norm, masks[i])

        ct_out.append(ct_norm.astype(np.float32))
        mri_before.append(before_norm.astype(np.float32))
        mri_after.append(after_norm.astype(np.float32))
        is_bg.append(bool(
            norm.is_background_slice(ct_norm, cfg.BG_INTENSITY_THRESH,
                                     cfg.BG_PIXEL_FRACTION) or
            norm.is_background_slice(after_norm, cfg.BG_INTENSITY_THRESH,
                                     cfg.BG_PIXEL_FRACTION)))

    # ── cache ────────────────────────────────────────────────────────────────
    step("writing cache")
    cpath = cache_path_for(app_cfg, key)
    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    tmp = cpath + ".tmp.npz"
    np.savez_compressed(
        tmp,
        ct=np.stack(ct_out),
        mri_before=np.stack(mri_before),
        mri_after=np.stack(mri_after),
        is_background=np.array(is_bg, dtype=bool),
    )
    os.replace(tmp, cpath)   # atomic: a half-written cache is never readable

    return {
        "key":          key,
        "cache_path":   cpath,
        "n_pairs":      n_pair,
        "orientation":  orientation,
        "body_region":  region,
        "ct_win_min":   ct_win_min,
        "ct_win_max":   ct_win_max,
        "mri_p_low":    float(p_low),
        "mri_p_high":   float(p_high),
        "is_background": is_bg,
        "search_mm":    search_mm,
        "roi":          ({str(k): list(v) for k, v in roi.items()} if isinstance(roi, dict)
                         else (list(roi) if roi else None)),
        "duration_s":   time.time() - t0,
        "method":       "2d_1mm",   # registration_idea: PixelSpacing only
        "n4_applied":   bool(n4_applied),
        "reg":          _clean_reg(reg),
    }


def _clean_reg(reg: dict) -> dict:
    """JSON-safe copy of the registration result (numpy scalars, NaN -> None)."""
    def num(v):
        if v is None:
            return None
        f = float(v)
        return None if np.isnan(f) else f

    return {
        "applied":      bool(reg["applied"]),
        "dx":           int(reg["dx"]),
        "dy":           int(reg["dy"]),
        "dx_mm":        num(reg["dx_mm"]),
        "dy_mm":        num(reg["dy_mm"]),
        "mean_gain":    num(reg["mean_gain"]),
        "n_probes":     int(reg["n_probes"]),
        "n_usable":     int(reg["n_usable"]),
        "spread_y_mm":  num(reg["spread_mm"][0]),
        "spread_x_mm":  num(reg["spread_mm"][1]),
        "hit_edge":     int(reg["hit_edge"]),
        "search_mm":    num(reg["search_mm"]),
        "probe_slices": [int(i) for i in reg["probe_slices"]],
        "reason":       str(reg["reason"]),
    }


def load_cache(cache_path: str):
    """The cached arrays for one series, or None if the cache is absent/stale."""
    if not cache_path or not os.path.exists(cache_path):
        return None
    try:
        with np.load(cache_path) as z:
            return {
                "ct":            z["ct"],
                "mri_before":    z["mri_before"],
                "mri_after":     z["mri_after"],
                "is_background": z["is_background"],
            }
    except Exception:
        # A corrupt cache is a recoverable condition: the series simply needs
        # re-registering, which the caller can queue.
        return None


def format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()
