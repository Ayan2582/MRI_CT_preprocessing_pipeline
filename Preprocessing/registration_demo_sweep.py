"""
registration_demo_sweep.py
────────────────────────────
Broader validation of the v2-baseline + multi-start registration fix from
registration_demo.py: instead of one hand-picked "most tissue" slice per
patient, this sweeps THREE fixed slice positions - first, middle, last -
across every available orientation (axial, coronal, sagittal) for each of
the 4 patients, to see whether the earlier single-slice results generalize
across the volume and across orientation.

Positions are literal (index 0, n//2, n-1), not content-picked, so this
intentionally includes edge slices that may be mostly background - that's
the point: robustness across the volume, not just the best-looking slice.

Reuses every processing function from registration_demo.py unchanged. Uses
a lighter multi-start (3 seeds, not 5) given ~33 slices to cover instead of 4.

Scores are normalized MI (demo.nmi_score), matching the rest of the
investigation - bounded in [1,2] and comparable across slices, which raw
Mattes MI is not. Earlier runs of this script reported raw MI, so its
numbers are not directly comparable with those in the older CSV.
"""
import os
import csv
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import io_utils
import image_processing as img_proc
import normalization as norm
import pipeline_config as cfg
import registration_demo as demo

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "registration_demo_output", "sweep")
N_STARTS = 3

# Every confirmed CT<->MRI series pair per patient, across all available
# orientations (folder names verified via io_utils.discover_series in the
# earlier single-slice investigation). PA18_Sangeeta has no axial series
# for this patient - skipped, not silently substituted with another orientation.
ORIENTATION_CANDIDATES = [
    {"region": "brain",    "patient": "PA0_Ranjeet",      "orientation": "axial",    "ct_se": "SE0", "mri_se": "SE0"},
    {"region": "brain",    "patient": "PA0_Ranjeet",      "orientation": "coronal",  "ct_se": "SE1", "mri_se": "SE1"},
    {"region": "brain",    "patient": "PA0_Ranjeet",      "orientation": "sagittal", "ct_se": "SE2", "mri_se": "SE2"},
    {"region": "shoulder", "patient": "PA6_Vijay",        "orientation": "axial",    "ct_se": "SE0", "mri_se": "SE0"},
    {"region": "shoulder", "patient": "PA6_Vijay",        "orientation": "coronal",  "ct_se": "SE1", "mri_se": "SE1"},
    {"region": "shoulder", "patient": "PA6_Vijay",        "orientation": "sagittal", "ct_se": "SE2", "mri_se": "SE2"},
    {"region": "spine",    "patient": "PA18_Sangeeta",    "orientation": "sagittal", "ct_se": "SE0", "mri_se": "SE0"},
    {"region": "spine",    "patient": "PA18_Sangeeta",    "orientation": "coronal",  "ct_se": "SE1", "mri_se": "SE1"},
    {"region": "knee",     "patient": "PA32_Mandbi_knee", "orientation": "axial",    "ct_se": "SE3", "mri_se": "SE3"},
    {"region": "knee",     "patient": "PA32_Mandbi_knee", "orientation": "sagittal", "ct_se": "SE4", "mri_se": "SE4"},
    {"region": "knee",     "patient": "PA32_Mandbi_knee", "orientation": "coronal",  "ct_se": "SE5", "mri_se": "SE5"},
]


def prepare_volume(cand):
    """Same setup as registration_demo.prepare_candidate, but returns the
    whole-volume products without picking a single slice - the sweep picks
    three fixed positions (first/middle/last) per volume instead."""
    region, patient, orientation = cand["region"], cand["patient"], cand["orientation"]
    print(f"=== {region.upper()} : {patient} [{orientation}] ===")

    ct_path = os.path.join(cfg.DATA_ROOT, "CT", patient, "ST0", cand["ct_se"])
    mri_path = os.path.join(cfg.DATA_ROOT, "MRI", patient, "ST0", cand["mri_se"])
    ct_image, _ = io_utils.load_dicom_series(ct_path)
    mri_image, _ = io_utils.load_dicom_series(mri_path)
    if ct_image is None or mri_image is None:
        print("    ! Failed to load series, skipping.")
        return None

    mri_corrected = img_proc.apply_n4_bias_correction(mri_image, shrink_factor=4)
    ct_res = img_proc.resample_inplane(ct_image, target_spacing=cfg.TARGET_SPACING_MM, is_ct=True)
    # resample_mri_to_ct_grid mutates its input's Direction in place (plain
    # assignment, not a copy) - feed it a disposable copy so v2 still sees
    # the MRI's true, unmutated geometry.
    baseline_old_res = img_proc.resample_mri_to_ct_grid(sitk.Image(mri_corrected), ct_res, default_pixel_value=0.0)
    baseline_v2_res, v2_t = demo.resample_mri_to_ct_grid_v2(mri_corrected, ct_res, default_pixel_value=0.0)

    prefix = patient.split("_")[0]
    body_region = cfg.PREFIX_TO_REGION.get(prefix, "default")
    profile = cfg.REGION_PROFILES.get(body_region, cfg.REGION_PROFILES["default"])
    ct_win_min, ct_win_max = profile["ct_win_min"], profile["ct_win_max"]

    mri_arr_vol = sitk.GetArrayFromImage(baseline_v2_res).astype(np.float32)
    mri_p1, mri_p99 = norm.compute_mri_percentiles(mri_arr_vol, cfg.MRI_PERCENTILE_LOW, cfg.MRI_PERCENTILE_HIGH)

    n_slices = sitk.GetArrayFromImage(ct_res).shape[0]
    print(f"    v2 translation: ({v2_t[0]:+.1f},{v2_t[1]:+.1f},{v2_t[2]:+.1f})mm  n_slices={n_slices}")

    return {
        "ct_res": ct_res,
        "baseline_old_res": baseline_old_res,
        "baseline_v2_res": baseline_v2_res,
        "ct_win_min": ct_win_min, "ct_win_max": ct_win_max,
        "mri_p1": mri_p1, "mri_p99": mri_p99,
        "n_slices": n_slices,
        "v2_translation_mm": v2_t,
    }


def process_slice(prep, cand, z, position_label):
    ct_slice = prep["ct_res"][:, :, z]
    baseline_old_slice = prep["baseline_old_res"][:, :, z]
    baseline_v2_slice = prep["baseline_v2_res"][:, :, z]

    mi_old = demo.nmi_score(ct_slice, baseline_old_slice)
    mi_v2 = demo.nmi_score(ct_slice, baseline_v2_slice)

    rigid_img, rigid_t, mi_rigid, rigid_seed, rigid_scores = demo.run_2d_registration_multistart(
        ct_slice, baseline_v2_slice, "rigid", n_starts=N_STARTS
    )
    affine_img, affine_t, mi_affine, affine_seed, affine_scores = demo.run_2d_registration_multistart(
        ct_slice, baseline_v2_slice, "affine", n_starts=N_STARTS
    )

    print(f"      [{position_label:6s} z={z:3d}] NMI old={mi_old:.3f} v2={mi_v2:.3f} "
          f"rigid={mi_rigid:.3f} affine={mi_affine:.3f}")

    ct_win_min, ct_win_max = prep["ct_win_min"], prep["ct_win_max"]
    mri_p1, mri_p99 = prep["mri_p1"], prep["mri_p99"]
    ct_disp = norm.normalize_ct_slice(sitk.GetArrayFromImage(ct_slice), ct_win_min, ct_win_max)
    old_disp = norm.normalize_mri_slice(sitk.GetArrayFromImage(baseline_old_slice), mri_p1, mri_p99)
    v2_disp = norm.normalize_mri_slice(sitk.GetArrayFromImage(baseline_v2_slice), mri_p1, mri_p99)
    rigid_disp = norm.normalize_mri_slice(sitk.GetArrayFromImage(rigid_img), mri_p1, mri_p99)
    affine_disp = norm.normalize_mri_slice(sitk.GetArrayFromImage(affine_img), mri_p1, mri_p99)

    fig, axes = plt.subplots(1, 4, figsize=(10, 2.6))
    demo.fusion_panel(axes[0], ct_disp, old_disp, f"old {mi_old:.2f}")
    demo.fusion_panel(axes[1], ct_disp, v2_disp, f"v2 {mi_v2:.2f}")
    demo.fusion_panel(axes[2], ct_disp, rigid_disp, f"rigid {mi_rigid:.2f}")
    demo.fusion_panel(axes[3], ct_disp, affine_disp, f"affine {mi_affine:.2f}")
    fig.suptitle(f"{cand['region']} {cand['patient']} {cand['orientation']} {position_label} (z={z})", fontsize=8)
    fig.tight_layout()
    png_name = f"{cand['region']}_{cand['orientation']}_{position_label}.png"
    out_png = os.path.join(OUTPUT_DIR, png_name)
    fig.savefig(out_png, dpi=90)
    plt.close(fig)

    # None (not 0.0) when fewer than two seeds survived - see demo.score_spread.
    rigid_spread_val = demo.score_spread(rigid_scores)
    affine_spread_val = demo.score_spread(affine_scores)

    return {
        "region": cand["region"], "patient": cand["patient"], "orientation": cand["orientation"],
        "position": position_label, "slice_index": z, "n_slices": prep["n_slices"],
        "nmi_baseline_old": mi_old, "nmi_baseline_v2": mi_v2,
        "nmi_rigid": mi_rigid, "nmi_affine": mi_affine,
        "rigid_spread": rigid_spread_val,
        "affine_spread": affine_spread_val,
        "rigid_failed": demo.count_failed(rigid_scores),
        "affine_failed": demo.count_failed(affine_scores),
        "png": out_png,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    for cand in ORIENTATION_CANDIDATES:
        prep = prepare_volume(cand)
        if prep is None:
            continue
        n = prep["n_slices"]
        positions = [("first", 0), ("middle", n // 2), ("last", n - 1)]
        for label, z in positions:
            row = process_slice(prep, cand, z, label)
            rows.append(row)

    csv_path = os.path.join(OUTPUT_DIR, "sweep_summary.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {csv_path}")
    else:
        print("\nNo rows processed.")


if __name__ == "__main__":
    main()
