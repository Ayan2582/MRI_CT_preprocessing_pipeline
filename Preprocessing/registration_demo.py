"""
registration_demo.py
─────────────────────
Standalone test/demo (NOT part of the production pipeline): for one patient
per body region, compares three MRI-to-CT alignment strategies on a single
2D slice, and shows every step from the raw DICOM image to the final result:

  raw CT -> resampled CT (1mm)                       [CT track]
  raw MRI -> N4 bias-corrected -> baseline (resampled onto CT grid)  [MRI track]
  baseline -> rigid (multi-start) -> affine (multi-start)            [registration]

Where "baseline" is the current pipeline's approach: resample_mri_to_ct_grid()
forces the MRI's direction matrix to match the CT's, then resamples with an
identity transform.

Multi-start fix: a single gradient-descent registration run turned out to be
non-reproducible even with a fixed sampling seed (confirmed: same code, same
slice, same seed -> different MI scores across process runs, traced to
multi-threaded execution). Each rigid/affine registration here now runs
N_STARTS independent, individually-deterministic attempts (seeds 0..N-1,
single-threaded) and keeps whichever attempt scores best by Mattes MI,
instead of trusting one gradient-descent pass.

Produces one comparison PNG per patient plus a summary.csv with Mattes MI
scores, the winning seed, and the recovered transform parameters.
"""
import os
import csv
import numpy as np
import SimpleITK as sitk

# SimpleITK's Mattes MI random sampling isn't fully determined by its seed
# under multi-threading - thread scheduling still perturbs the outcome
# (confirmed empirically: same seed, same slice, 3 different MI scores
# across process runs; forcing 1 thread made it exactly reproducible).
# Single-threading here makes each multi-start attempt below trustworthy
# and repeatable; N_STARTS varying the seed on purpose is what actually
# explores the optimization landscape.
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import io_utils
import image_processing as img_proc
import normalization as norm
import pipeline_config as cfg

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "registration_demo_output")
N_STARTS = 5  # multi-start attempts per transform type

# One confirmed CT<->MRI series pair per region (folder names + orientation
# verified against io_utils.discover_series and each patient's DICOM
# BodyPartExamined tag before picking these).
CANDIDATES = [
    {"region": "brain",    "patient": "PA0_Ranjeet",      "ct_se": "SE0", "mri_se": "SE0", "orientation": "axial"},
    {"region": "shoulder", "patient": "PA6_Vijay",        "ct_se": "SE0", "mri_se": "SE0", "orientation": "axial"},
    {"region": "spine",    "patient": "PA18_Sangeeta",    "ct_se": "SE0", "mri_se": "SE0", "orientation": "sagittal"},
    {"region": "knee",     "patient": "PA32_Mandbi_knee", "ct_se": "SE3", "mri_se": "SE3", "orientation": "axial"},
]


def run_2d_registration(fixed_slice, moving_slice, transform_type="rigid", num_iters=100, seed=0):
    """
    Mirrors image_processing.register_2d_rigid's settings (Mattes MI, geometry
    init, [4,2,1] pyramid, gradient descent) but is parameterized over the
    transform model and the sampling seed, and also returns the fitted
    transform (needed to report rotation/scale/shear), which the production
    function discards.
    """
    fixed = sitk.Cast(fixed_slice, sitk.sitkFloat32)
    moving = sitk.Cast(moving_slice, sitk.sitkFloat32)

    if transform_type == "rigid":
        base_transform = sitk.Euler2DTransform()
    elif transform_type == "affine":
        base_transform = sitk.AffineTransform(2)
    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")

    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, base_transform,
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.2, seed=seed)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=0.1,
        numberOfIterations=num_iters,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # inPlace=True so Execute() returns the concrete Euler2DTransform/AffineTransform
    # directly (not wrapped in a CompositeTransform), so we can read back rotation/scale.
    registration_method.SetInitialTransform(initial_transform, inPlace=True)

    try:
        final_transform = registration_method.Execute(fixed, moving)
    except Exception as e:
        print(f"      ! {transform_type} seed={seed} failed: {e}")
        return moving_slice, None

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(final_transform)
    resampled = resampler.Execute(moving)

    return resampled, final_transform


def run_2d_registration_multistart(fixed_slice, moving_slice, transform_type="rigid", num_iters=100, n_starts=N_STARTS):
    """
    Run n_starts independent, individually-deterministic attempts (seeds
    0..n_starts-1) and keep whichever scores best by Mattes MI. A single
    gradient-descent run can land on different local optima depending on
    which random pixel subset it draws; this trades n_starts x the compute
    for a much more reliable result than trusting one shot.
    """
    best_mi, best_resampled, best_transform, best_seed = None, moving_slice, None, None
    scores = []
    for seed in range(n_starts):
        resampled, transform = run_2d_registration(fixed_slice, moving_slice, transform_type, num_iters, seed)
        if transform is None:
            scores.append(None)
            continue
        mi = mattes_mi_score(fixed_slice, resampled)
        scores.append(mi)
        if best_mi is None or mi > best_mi:
            best_mi, best_resampled, best_transform, best_seed = mi, resampled, transform, seed

    if best_mi is None:
        # every attempt failed - fall back to the untouched moving image
        best_mi = mattes_mi_score(fixed_slice, moving_slice)
        best_resampled = moving_slice

    valid_scores = [s for s in scores if s is not None]
    print(f"      {transform_type:6s} multi-start: "
          f"{['%.3f' % s if s is not None else 'FAIL' for s in scores]}  "
          f"best={best_mi:.4f} (seed={best_seed})  "
          f"spread={max(valid_scores) - min(valid_scores):.4f}" if valid_scores else "all failed")

    return best_resampled, best_transform, best_mi, best_seed, scores


def mattes_mi_score(fixed_slice, moving_slice):
    """
    Evaluate Mattes MI between two slices that already share the same
    physical grid (identity transform). SimpleITK reports this as a cost
    to MINIMIZE (more negative = better match); we negate it so higher =
    better match, matching intuition.
    """
    fixed = sitk.Cast(fixed_slice, sitk.sitkFloat32)
    moving = sitk.Cast(moving_slice, sitk.sitkFloat32)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetInitialTransform(sitk.Transform(2, sitk.sitkIdentity))
    return -reg.MetricEvaluate(fixed, moving)


def pick_best_slice(ct_arr_vol, mri_arr_vol, ct_win_min, ct_win_max, mri_p1, mri_p99):
    """
    Return the z-index with the least combined background fraction, so the
    demo doesn't land on a mostly-air slice on thin series (e.g. 9-slice spine).
    """
    best_z, best_score = 0, None
    for z in range(ct_arr_vol.shape[0]):
        ct_norm = norm.normalize_ct_slice(ct_arr_vol[z], ct_win_min, ct_win_max)
        mri_norm = norm.normalize_mri_slice(mri_arr_vol[z], mri_p1, mri_p99)
        bg_frac = (
            float(np.mean(ct_norm <= cfg.BG_INTENSITY_THRESH)) +
            float(np.mean(mri_norm <= cfg.BG_INTENSITY_THRESH))
        )
        if best_score is None or bg_frac < best_score:
            best_score, best_z = bg_frac, z
    return best_z


def gray_panel(ax, arr, title, cmap="gray", vmin=0.0, vmax=1.0):
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9.5)
    ax.axis("off")


def fusion_panel(ax, ct_norm, mri_norm, title):
    ax.imshow(ct_norm, cmap="gray", vmin=0, vmax=1)
    ax.imshow(mri_norm, cmap="hot", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9.5)
    ax.axis("off")


def prepare_candidate(cand):
    """
    Load + run the same steps the production pipeline runs (N4, in-plane
    resample, baseline resample-to-CT-grid), then pick the most tissue-rich
    slice. Returns every intermediate volume (raw CT/MRI, N4-corrected MRI,
    resampled CT, baseline MRI) plus display-normalization parameters, so
    callers can show the full raw-to-registered pipeline, not just the
    final baseline slice.
    """
    region, patient = cand["region"], cand["patient"]
    print(f"=== {region.upper()} : {patient} ({cand['orientation']}) ===")

    ct_path = os.path.join(cfg.DATA_ROOT, "CT", patient, "ST0", cand["ct_se"])
    mri_path = os.path.join(cfg.DATA_ROOT, "MRI", patient, "ST0", cand["mri_se"])

    ct_image, _ = io_utils.load_dicom_series(ct_path)
    mri_image, _ = io_utils.load_dicom_series(mri_path)
    if ct_image is None or mri_image is None:
        print("    ! Failed to load series, skipping.")
        return None

    # -- Same steps the production pipeline runs --
    mri_corrected = img_proc.apply_n4_bias_correction(mri_image, shrink_factor=4)
    ct_res = img_proc.inplane(ct_image, target_spacing=cfg.TARGET_SPACING_MM, is_ct=True)
    baseline_mri_res = img_proc.resample_mri_to_ct_grid(mri_corrected, ct_res, default_pixel_value=0.0)

    prefix = patient.split("_")[0]
    body_region = cfg.PREFIX_TO_REGION.get(prefix, "default")
    profile = cfg.REGION_PROFILES.get(body_region, cfg.REGION_PROFILES["default"])
    ct_win_min, ct_win_max = profile["ct_win_min"], profile["ct_win_max"]

    ct_arr_vol = sitk.GetArrayFromImage(ct_res)
    mri_arr_vol = sitk.GetArrayFromImage(baseline_mri_res).astype(np.float32)
    mri_p1, mri_p99 = norm.compute_mri_percentiles(mri_arr_vol, cfg.MRI_PERCENTILE_LOW, cfg.MRI_PERCENTILE_HIGH)

    z = pick_best_slice(ct_arr_vol, mri_arr_vol, ct_win_min, ct_win_max, mri_p1, mri_p99)
    print(f"    Selected slice index: {z} / {ct_arr_vol.shape[0]}")
    print(f"    CT native spacing: {tuple(round(s, 3) for s in ct_image.GetSpacing())} mm  "
          f"-> resampled: {tuple(round(s, 3) for s in ct_res.GetSpacing())} mm")

    # raw_mri_image/mri_corrected share the same grid (N4 only changes
    # intensity), so the same z indexes the same physical slice for both.
    # ct_image is Z-untouched by resample_inplane, so z also indexes the
    # same physical slice there. Indexing raw_mri_image/ct_res with the
    # SAME z assumes the paired raw series have matching slice counts,
    # which every candidate here was confirmed to have.
    return {
        "ct_image_raw": ct_image,
        "mri_image_raw": mri_image,
        "mri_corrected": mri_corrected,
        "ct_res": ct_res,
        "baseline_mri_res": baseline_mri_res,
        "ct_slice_img": ct_res[:, :, z],
        "baseline_mri_slice_img": baseline_mri_res[:, :, z],
        "ct_win_min": ct_win_min,
        "ct_win_max": ct_win_max,
        "mri_p1": mri_p1,
        "mri_p99": mri_p99,
        "z": z,
        "n_slices": ct_arr_vol.shape[0],
    }


def process_candidate(cand):
    region, patient = cand["region"], cand["patient"]
    prep = prepare_candidate(cand)
    if prep is None:
        return None

    ct_slice_img = prep["ct_slice_img"]
    baseline_mri_slice_img = prep["baseline_mri_slice_img"]
    ct_win_min, ct_win_max = prep["ct_win_min"], prep["ct_win_max"]
    mri_p1, mri_p99 = prep["mri_p1"], prep["mri_p99"]
    z = prep["z"]

    print("    Registering (multi-start, single-threaded, seeds 0..%d)..." % (N_STARTS - 1))
    rigid_img, rigid_transform, mi_rigid, rigid_seed, rigid_scores = run_2d_registration_multistart(
        ct_slice_img, baseline_mri_slice_img, "rigid"
    )
    affine_img, affine_transform, mi_affine, affine_seed, affine_scores = run_2d_registration_multistart(
        ct_slice_img, baseline_mri_slice_img, "affine"
    )

    mi_baseline = mattes_mi_score(ct_slice_img, baseline_mri_slice_img)
    print(f"    MI  baseline={mi_baseline:.4f}  rigid={mi_rigid:.4f}  affine={mi_affine:.4f}")

    # -- Transform parameters for the report --
    rigid_angle_deg = rigid_tx = rigid_ty = None
    if rigid_transform is not None:
        rigid_angle_deg = np.degrees(rigid_transform.GetAngle())
        rigid_tx, rigid_ty = rigid_transform.GetTranslation()

    affine_scale = affine_tx = affine_ty = None
    if affine_transform is not None:
        matrix = np.array(affine_transform.GetMatrix()).reshape(2, 2)
        affine_scale = np.linalg.svd(matrix, compute_uv=False)  # robust scale estimate (ignores shear/rotation coupling)
        affine_tx, affine_ty = affine_transform.GetTranslation()

    # -- Every step, raw to registered --
    raw_ct_arr = sitk.GetArrayFromImage(prep["ct_image_raw"][:, :, z])
    raw_mri_arr = sitk.GetArrayFromImage(prep["mri_image_raw"][:, :, z])
    n4_mri_arr = sitk.GetArrayFromImage(prep["mri_corrected"][:, :, z])
    ct_res_arr = sitk.GetArrayFromImage(ct_slice_img)
    baseline_arr = sitk.GetArrayFromImage(baseline_mri_slice_img)
    rigid_arr = sitk.GetArrayFromImage(rigid_img)
    affine_arr = sitk.GetArrayFromImage(affine_img)

    raw_ct_disp = norm.normalize_ct_slice(raw_ct_arr, ct_win_min, ct_win_max)
    ct_res_disp = norm.normalize_ct_slice(ct_res_arr, ct_win_min, ct_win_max)
    # Same [mri_p1, mri_p99] window for every MRI-derived panel (raw through
    # affine) so brightness differences you see across the row come from the
    # processing step itself, not from re-normalizing each panel separately.
    raw_mri_disp = norm.normalize_mri_slice(raw_mri_arr, mri_p1, mri_p99)
    n4_mri_disp = norm.normalize_mri_slice(n4_mri_arr, mri_p1, mri_p99)
    baseline_disp = norm.normalize_mri_slice(baseline_arr, mri_p1, mri_p99)
    rigid_disp = norm.normalize_mri_slice(rigid_arr, mri_p1, mri_p99)
    affine_disp = norm.normalize_mri_slice(affine_arr, mri_p1, mri_p99)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f"{region.upper()} — {patient}  [{cand['orientation']}, slice {z}]  raw → registered", fontsize=13)

    gray_panel(axes[0, 0], raw_ct_disp, "1. CT raw (native spacing)")
    gray_panel(axes[0, 1], ct_res_disp, f"2. CT resampled ({cfg.TARGET_SPACING_MM}mm in-plane)")
    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.0, 0.5,
        f"Region: {region}\nOrientation: {cand['orientation']}\nSlice: {z}\n\n"
        f"Rigid best: seed={rigid_seed}, angle={rigid_angle_deg:.2f}°, "
        f"t=({rigid_tx:.1f},{rigid_ty:.1f})px\n"
        f"Affine best: seed={affine_seed}, scale={affine_scale}, "
        f"t=({affine_tx:.1f},{affine_ty:.1f})px"
        if rigid_transform is not None and affine_transform is not None else "Registration failed",
        fontsize=9, va="center", wrap=True,
    )

    gray_panel(axes[1, 0], raw_mri_disp, "3. MRI raw (native spacing, pre-N4)")
    gray_panel(axes[1, 1], n4_mri_disp, "4. MRI after N4 bias correction")
    gray_panel(axes[1, 2], baseline_disp, "5. MRI baseline (resampled to CT grid)")

    fusion_panel(axes[2, 0], ct_res_disp, baseline_disp, f"6. Baseline fusion  MI={mi_baseline:.3f}")
    fusion_panel(axes[2, 1], ct_res_disp, rigid_disp, f"7. Rigid (best of {N_STARTS})  MI={mi_rigid:.3f}")
    fusion_panel(axes[2, 2], ct_res_disp, affine_disp, f"8. Affine (best of {N_STARTS})  MI={mi_affine:.3f}")

    fig.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, f"{region}_{patient}.png")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"    Saved {out_png}")

    def fmt_scores(scores):
        return "|".join("%.3f" % s if s is not None else "FAIL" for s in scores)

    return {
        "region": region,
        "patient": patient,
        "orientation": cand["orientation"],
        "slice_index": z,
        "mi_baseline": mi_baseline,
        "mi_rigid": mi_rigid,
        "mi_affine": mi_affine,
        "rigid_seed": rigid_seed,
        "rigid_scores": fmt_scores(rigid_scores),
        "affine_seed": affine_seed,
        "affine_scores": fmt_scores(affine_scores),
        "rigid_angle_deg": rigid_angle_deg,
        "rigid_tx": rigid_tx,
        "rigid_ty": rigid_ty,
        "affine_scale_1": affine_scale[0] if affine_scale is not None else None,
        "affine_scale_2": affine_scale[1] if affine_scale is not None else None,
        "affine_tx": affine_tx,
        "affine_ty": affine_ty,
        "png": out_png,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    for cand in CANDIDATES:
        row = process_candidate(cand)
        if row is not None:
            rows.append(row)

    csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSummary written to {csv_path}")
    else:
        print("\nNo candidates processed successfully.")


if __name__ == "__main__":
    main()
