"""
outputs.py
──────────
Write accepted pairs out in a layout the production pipeline can consume.

THE INPUT TREE IS READ-ONLY. Nothing in this application opens a file under
data_root for writing. Every path built here is under workspace/output, which
this tool created and owns.

LAYOUT - the input hierarchy, preserved exactly:

    output/
      CT/PA0_Ranjeet/ST0/SE0/IM0.npy          normalised float32, region HU window
      MRI/PA0_Ranjeet/ST0/SE0/IM0.npy         N4 -> CT grid -> shift -> normalised
      previews/PA0_Ranjeet/ST0/SE0/IM0_pair.png
      registration/PA0_Ranjeet/ST0/SE0.json   the whole registration record
      metadata.csv                            one row per accepted pair

Patient, study and series folders are never flattened or renamed, and the file
stem is the SOURCE DICOM's own name - the CT file's name on the CT side, the
MRI file's name on the MRI side. Those two can differ in principle, so both are
recorded per row in metadata.csv rather than assumed equal.

metadata.csv carries every column preprocess_2d.py writes, in its order, so an
existing dataloader reads it unchanged. QC columns are appended after those.
"""

import csv
import json
import os
import time

import numpy as np

from . import bootstrap, registration_service as regsvc

_pp = bootstrap.preprocessing_modules()
export = _pp.export

# preprocess_2d.py:303-309, verbatim, then this tool's own columns.
PRODUCTION_FIELDS = [
    "patient_id", "body_region", "orientation", "slice_index",
    "ct_series", "mri_series", "mri_desc",
    "height", "width",
    "ct_npy", "mri_npy", "is_background",
    "reg_applied", "reg_dx_mm", "reg_dy_mm", "reg_nmi_gain", "reg_note",
]
QC_FIELDS = [
    "study", "ct_file", "mri_file", "qc_status", "qc_note",
    "roi_x", "roi_y", "roi_w", "roi_h", "roi_mode",
    # Manual adjustments, kept in their own columns so a hand-nudged pair is
    # never mistaken for a measured one. reg_dx_mm/reg_dy_mm above stay exactly
    # what registration found; total shift is the sum of the two.
    "manual_dx_mm", "manual_dy_mm",
    "ct_rect", "mri_rect", "mri_height", "mri_width", "erased",
    "reviewed_at",
]
ALL_FIELDS = PRODUCTION_FIELDS + QC_FIELDS


def _stem(filename: str, fallback: str) -> str:
    if not filename:
        return fallback
    stem = os.path.splitext(filename)[0]
    return stem or fallback


def output_paths(app_cfg, pair_row: dict, series_row: dict) -> dict:
    patient = pair_row["patient"]
    study = pair_row["study"]
    idx = pair_row["slice_index"]
    ct_stem = _stem(pair_row.get("ct_file"), f"IM{idx}")
    mri_stem = _stem(pair_row.get("mri_file"), f"IM{idx}")

    return {
        "ct": os.path.join(app_cfg.output_dir, "CT", patient, study,
                           series_row["ct_series"], f"{ct_stem}.npy"),
        "mri": os.path.join(app_cfg.output_dir, "MRI", patient, study,
                            series_row["mri_series"], f"{mri_stem}.npy"),
        "preview": os.path.join(app_cfg.output_dir, "previews", patient, study,
                                series_row["ct_series"], f"{ct_stem}_pair.png"),
        "registration": os.path.join(app_cfg.output_dir, "registration", patient,
                                     study, f"{series_row['ct_series']}.json"),
    }


def write_pair(app_cfg, pair_row: dict, series_row: dict, cache: dict,
               roi=None, crop_export: bool = False, save_preview: bool = True) -> dict:
    """
    Write one accepted slice pair. Returns the paths written.

    Uses export_utils.save_npy and export_utils.save_preview_png so the file
    format is byte-for-byte what the production pipeline produces - float32
    .npy, and the same side-by-side preview with the same divider.

    `crop_export` writes the ROI sub-rectangle instead of the full frame. Off by
    default, because the production pipeline deliberately saves slices at their
    native post-resample size and leaves cropping to the dataloader
    (pipeline_config.py:35-39). The ROI is recorded in metadata.csv either way,
    so a dataloader can apply it without the pixels having been thrown away.
    """
    i = pair_row["slice_index"]
    ct_arr = cache["ct"][i]
    mri_arr = cache["mri_after"][i]

    # Manual nudge, on top of the measured shift. Applied before erase and
    # before cropping, because it is a change of alignment and the other two
    # are stated in the frame that alignment produces.
    ndy, ndx = int(pair_row.get("nudge_dy") or 0), int(pair_row.get("nudge_dx") or 0)
    if ndy or ndx:
        mri_arr = regsvc.apply_nudge(mri_arr, ndy, ndx)

    # Automatic thin-structure removal (table rails, cradle), if enabled for
    # this series. Applied before the brush so the two compose.
    if series_row.get("auto_erase"):
        auto = regsvc.thin_structure_mask(ct_arr)
        if auto is not None:
            ct_arr = regsvc.apply_erase(ct_arr, auto)
            mri_arr = regsvc.apply_erase(mri_arr, auto)

    # Painted-out regions are removed from what is saved, not just from what is
    # displayed. Applied to both modalities so the exported pair stays
    # consistent - a region blanked in one and present in the other would ask
    # the network to invent tissue from nothing.
    if pair_row.get("erase_json"):
        try:
            mask = regsvc.rasterize_strokes(
                json.loads(pair_row["erase_json"]), ct_arr.shape)
        except (TypeError, ValueError):
            mask = None
        if mask is not None:
            ct_arr = regsvc.apply_erase(ct_arr, mask)
            mri_arr = regsvc.apply_erase(mri_arr, mask)

    # Export rectangles. A per-modality rect wins over the shared ROI, so a
    # reviewer who set one gets exactly it; otherwise the ROI applies to both
    # when "export cropped" is on.
    ct_rect = json.loads(pair_row["ct_rect_json"]) if pair_row.get("ct_rect_json") else None
    mri_rect = json.loads(pair_row["mri_rect_json"]) if pair_row.get("mri_rect_json") else None
    if roi is None and pair_row.get("roi_json"):
        roi = json.loads(pair_row["roi_json"])
    crop_export = crop_export or (pair_row.get("roi_mode") == "export")

    if ct_rect or mri_rect:
        if ct_rect:
            ct_arr = regsvc.crop_to_rect(ct_arr, ct_rect)
        if mri_rect:
            mri_arr = regsvc.crop_to_rect(mri_arr, mri_rect)
    elif crop_export and roi:
        ct_arr, mri_arr = regsvc._crop([ct_arr, mri_arr], roi)

    # Only the three files this function actually writes. The registration
    # record belongs to the series, not the slice, and is written once by
    # write_registration_record - returning its path from here would advertise
    # a file that may not exist yet.
    all_paths = output_paths(app_cfg, pair_row, series_row)
    paths = {k: all_paths[k] for k in ("ct", "mri", "preview")}

    if not app_cfg.allow_output_overwrite:
        for p in (paths["ct"], paths["mri"]):
            if os.path.exists(p):
                raise FileExistsError(
                    f"{p} already exists and overwriting is disabled. Enable "
                    f"allow_output_overwrite, or clear the output directory.")

    export.save_npy(ct_arr, paths["ct"])
    export.save_npy(mri_arr, paths["mri"])

    if save_preview:
        # export_utils.save_preview_png hstacks the two arrays, which needs
        # equal heights. Independent per-modality rectangles can break that, so
        # pad the shorter one into a common canvas rather than either crashing
        # or silently skipping the preview. Padding is display-only; the .npy
        # files above keep their true, different shapes.
        if ct_arr.shape != mri_arr.shape:
            h = max(ct_arr.shape[0], mri_arr.shape[0])
            def _pad(a):
                out = np.zeros((h, a.shape[1]), dtype=a.dtype)
                out[:a.shape[0], :] = a
                return out
            export.save_preview_png(_pad(ct_arr), _pad(mri_arr), paths["preview"])
        else:
            export.save_preview_png(ct_arr, mri_arr, paths["preview"])
    else:
        paths["preview"] = None

    return paths


def remove_pair_outputs(app_cfg, pair_row: dict, series_row: dict) -> None:
    """
    Delete the output files for a pair that has been un-accepted.

    Only ever touches paths built by output_paths, i.e. inside the output tree.
    A rejected pair that still has files on disk from a previous accept would
    otherwise silently stay in the training set.
    """
    for key in ("ct", "mri", "preview"):
        p = output_paths(app_cfg, pair_row, series_row).get(key)
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def write_registration_record(app_cfg, series_row: dict, result: dict) -> str:
    """The full registration record for a series, next to its slices."""
    paths = {"registration": os.path.join(
        app_cfg.output_dir, "registration", series_row["patient"],
        series_row["study"], f"{series_row['ct_series']}.json")}
    os.makedirs(os.path.dirname(paths["registration"]), exist_ok=True)

    record = {
        "series_key":    series_row["key"],
        "patient":       series_row["patient"],
        "study":         series_row["study"],
        "ct_series":     series_row["ct_series"],
        "mri_series":    series_row["mri_series"],
        "ct_path":       series_row["ct_path"],
        "mri_path":      series_row["mri_path"],
        "orientation":   result.get("orientation"),
        "body_region":   result.get("body_region"),
        "n_pairs":       result.get("n_pairs"),
        "ct_window_hu":  [result.get("ct_win_min"), result.get("ct_win_max")],
        "mri_percentiles": [result.get("mri_p_low"), result.get("mri_p_high")],
        "target_spacing_mm": app_cfg.target_spacing,
        "n4_shrink":     app_cfg.n4_shrink,
        # How the two modalities were brought onto a common frame. Recorded per
        # series so an output can always be traced back to it, and so a dataset
        # cannot silently mix two geometries.
        "geometry": result.get("method", "2d_1mm"),
        "roi":           result.get("roi"),
        "registration":  result.get("reg"),
        "duration_s":    result.get("duration_s"),
        "written_at":    time.time(),
        "method": ("image_processing.estimate_volume_translation over "
                   "registration_idea.register - one whole-pixel in-plane "
                   "translation per series, applied to every slice"),
    }
    tmp = paths["registration"] + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    os.replace(tmp, paths["registration"])
    return paths["registration"]


def export_metadata_csv(app_cfg, manifest, statuses=("ACCEPTED",)) -> dict:
    """
    Rebuild metadata.csv from the manifest.

    Regenerated from the database rather than appended to as pairs are
    accepted, so a pair that was accepted, then rejected, then re-accepted
    appears exactly once and with its current decision.
    """
    placeholders = ",".join("?" for _ in statuses)
    rows = manifest.query(
        f"""SELECT p.*, s.orientation, s.series_desc, s.body_region,
                   s.reg_applied, s.reg_dx_mm, s.reg_dy_mm, s.reg_nmi_gain,
                   s.reg_reason, s.cache_path
            FROM pairs p JOIN series s ON s.key = p.series_key
            WHERE p.qc_status IN ({placeholders})
            ORDER BY p.seq""", tuple(statuses))

    out_path = os.path.join(app_cfg.output_dir, "metadata.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Slice dimensions come from the .npy actually written, not from the cache,
    # so a cropped export reports the size it really has on disk.
    written = 0
    tmp = out_path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_FIELDS)
        writer.writeheader()
        for r in rows:
            # Sizes come from the files actually written, per modality, because
            # independent export rectangles mean CT and MRI need not match.
            def _shape(path):
                if path and os.path.exists(path):
                    try:
                        a = np.load(path, mmap_mode="r")
                        return int(a.shape[0]), int(a.shape[1])
                    except Exception:
                        pass
                return "", ""

            ct_npy = r.get("output_ct") or ""
            h, w = _shape(ct_npy)
            mh, mw = _shape(r.get("output_mri"))

            roi = json.loads(r["roi_json"]) if r.get("roi_json") else None
            writer.writerow({
                "patient_id":   r["patient"],
                "body_region":  r.get("body_region") or "",
                "orientation":  r.get("orientation") or "",
                "slice_index":  r["slice_index"],
                "ct_series":    r["ct_series"],
                "mri_series":   r["mri_series"],
                "mri_desc":     r.get("series_desc") or "",
                "height":       h,
                "width":        w,
                "ct_npy":       ct_npy,
                "mri_npy":      r.get("output_mri") or "",
                "is_background": bool(r.get("is_background")),
                "reg_applied":  bool(r.get("reg_applied")),
                "reg_dx_mm":    r.get("reg_dx_mm") if r.get("reg_dx_mm") is not None else 0.0,
                "reg_dy_mm":    r.get("reg_dy_mm") if r.get("reg_dy_mm") is not None else 0.0,
                "reg_nmi_gain": r.get("reg_nmi_gain") if r.get("reg_nmi_gain") is not None else "",
                "reg_note":     r.get("reg_reason") or "",
                "study":        r["study"],
                "ct_file":      r.get("ct_file") or "",
                "mri_file":     r.get("mri_file") or "",
                "qc_status":    r["qc_status"],
                "qc_note":      r.get("note") or "",
                "roi_x":        roi[0] if roi else "",
                "roi_y":        roi[1] if roi else "",
                "roi_w":        roi[2] if roi else "",
                "roi_h":        roi[3] if roi else "",
                "roi_mode":     r.get("roi_mode") or "",
                "manual_dx_mm": r.get("nudge_dx") or 0,   # p.* - per slice
                "manual_dy_mm": r.get("nudge_dy") or 0,
                "ct_rect":      r.get("ct_rect_json") or "",
                "mri_rect":     r.get("mri_rect_json") or "",
                "mri_height":   mh,
                "mri_width":    mw,
                "erased":       bool(r.get("erase_json")),
                "reviewed_at":  r.get("reviewed_at") or "",
            })
            written += 1
    os.replace(tmp, out_path)
    return {"path": out_path, "rows": written}
