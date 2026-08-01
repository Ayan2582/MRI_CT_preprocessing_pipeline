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
single-threaded) and keeps whichever attempt scores best, instead of
trusting one gradient-descent pass.

Scoring metric: normalized MI (nmi_score), not raw Mattes MI. The optimizer
still minimizes Mattes MI internally because ITK exposes nothing else in
that family, but every number compared or reported here is NMI - bounded
in [1,2] and far less sensitive to how much image is in the frame, which
raw MI is not. mattes_mi_score is kept for reproducing earlier results.

Produces one comparison PNG per patient plus a summary.csv with NMI scores,
the winning seed, and the recovered transform parameters.
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


def resample_mri_to_ct_grid_v2(mri_image, ct_image, default_pixel_value=0.0):
    """
    Alternative to image_processing.resample_mri_to_ct_grid. The production
    version force-overwrites the MRI's own Direction to match the CT's, then
    resamples with an identity transform - which (a) corrupts the MRI's real
    orientation metadata instead of just assuming a relationship, and more
    importantly (b) never corrects translation at all: the MRI's raw DICOM
    Origin is left untouched, which is exactly why baseline showed ~55-58px
    uncorrected offsets for shoulder/knee in this report.

    This version leaves the MRI's real Direction alone and instead computes
    an explicit transform via CenteredTransformInitializer(GEOMETRY): it
    aligns the two volumes' bounding-box centers, in 3D. Rotation is
    deliberately left at identity - a true 3D rotation estimate needs more
    through-plane sampling than this dataset's sparse, anisotropic slices
    can support (established earlier in this investigation); that stays a
    2D-per-slice job (run_2d_registration below). Translation, unlike
    rotation, is a closed-form calculation from image geometry alone - no
    optimization, no randomness, nothing for multi-start to fix here.
    """
    mri_f = sitk.Cast(mri_image, sitk.sitkFloat32)
    ct_f = sitk.Cast(ct_image, sitk.sitkFloat32)

    initial_transform = sitk.CenteredTransformInitializer(
        ct_f, mri_f, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetTransform(initial_transform)

    return resampler.Execute(mri_image), initial_transform.GetTranslation()


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


def count_failed(scores):
    """Attempts that crashed or produced an unevaluable (NaN) score."""
    return sum(1 for s in scores if s is None or np.isnan(s))


def score_spread(scores):
    """
    Spread across the multi-start attempts that actually survived, or None
    if fewer than two did.

    Returning 0.0 for a lone survivor would be actively misleading: a range
    of zero reads as "every seed agreed" when it really means "every seed
    but one crashed" - the opposite conclusion, and exactly the case worth
    flagging.
    """
    valid = [s for s in scores if s is not None and not np.isnan(s)]
    if len(valid) < 2:
        return None
    return max(valid) - min(valid)


def run_2d_registration_multistart_detailed(fixed_slice, moving_slice, transform_type="rigid",
                                            num_iters=100, n_starts=N_STARTS, score_fn=None,
                                            accept_fn=None):
    """
    Run n_starts independent, individually-deterministic attempts (seeds
    0..n_starts-1) and keep whichever scores best by NMI (score_fn, default
    nmi_score - pass mattes_mi_score to reproduce the earlier MI-selected
    results). A single gradient-descent run can land on different local
    optima depending on which random pixel subset it draws; this trades
    n_starts x the compute for a much more reliable result than trusting
    one shot.

    Note the optimizer itself still minimizes Mattes MI internally - ITK
    offers no NMI metric - so this is "propose with MI, select with NMI".

    `accept_fn(transform) -> bool` filters candidates DURING selection. A
    rejected seed is still run, scored and reported, but cannot win. This is
    the whole difference between "keep the best-scoring attempt, then check
    whether it is admissible" and "keep the best admissible attempt" - the
    first silently discards a perfectly good alignment whenever an
    inadmissible one happens to outscore it, which is exactly what the
    post-hoc affine scale gate was doing (19 of 33 affines rejected outright
    rather than falling back to the best in-tolerance seed).

    If every seed is rejected or fails, this returns the UNREGISTERED moving
    image scored as-is, with best_transform None - the documented last
    resort, and honest about the fact that nothing admissible was found.

    Returns a dict; run_2d_registration_multistart is the 5-tuple wrapper
    kept for the callers that don't need per-seed detail.
    """
    score_fn = score_fn or nmi_score
    best_score, best_transform, best_seed = None, None, None
    best_resampled = moving_slice
    scores, transforms, accepted = [], [], []

    for seed in range(n_starts):
        resampled, transform = run_2d_registration(fixed_slice, moving_slice, transform_type, num_iters, seed)
        transforms.append(transform)
        if transform is None:
            scores.append(None)
            accepted.append(False)
            continue

        ok = True if accept_fn is None else bool(accept_fn(transform))
        accepted.append(ok)
        score = score_fn(fixed_slice, resampled)
        scores.append(score)
        if not ok:
            continue
        # NaN comparisons are always False in Python, so a NaN can never
        # win here - but if it were picked first (best_score is None), every
        # later "score > best_score" against that NaN would also be False,
        # permanently locking in a degenerate result. Guard explicitly.
        if not np.isnan(score) and (best_score is None or np.isnan(best_score) or score > best_score):
            best_score, best_resampled, best_transform, best_seed = score, resampled, transform, seed

    n_rejected = sum(1 for i, t in enumerate(transforms) if t is not None and not accepted[i])
    fell_back = best_score is None
    if fell_back:
        # nothing admissible survived - score the untouched moving image
        best_score = score_fn(fixed_slice, moving_slice)
        best_resampled = moving_slice

    spread = score_spread([s for s, a in zip(scores, accepted) if a])
    n_failed = count_failed(scores)
    spread_str = f"{spread:.4f}" if spread is not None else "n/a"
    marks = []
    for s, a in zip(scores, accepted):
        if s is None or np.isnan(s):
            marks.append("FAIL")
        else:
            marks.append(("%.3f" % s) + ("" if a else "*"))
    print(f"      {transform_type:6s} multi-start: {marks}  "
          f"best={best_score:.4f} (seed={best_seed})  "
          f"spread={spread_str}  failed={n_failed}/{len(scores)}  rejected={n_rejected}"
          + ("  -> UNREGISTERED (nothing admissible)" if fell_back else ""))

    return {
        "best_img": best_resampled,
        "best_transform": best_transform,
        "best_score": best_score,
        "best_seed": best_seed,
        "scores": scores,
        "transforms": transforms,
        "accepted": accepted,
        "n_rejected": n_rejected,
        "n_failed": n_failed,
        # spread over ADMISSIBLE seeds only - the spread among candidates that
        # could actually have been selected is the one that bounds the noise
        # on the selection.
        "spread": spread,
        "fell_back_to_unregistered": fell_back,
    }


def run_2d_registration_multistart(fixed_slice, moving_slice, transform_type="rigid", num_iters=100,
                                   n_starts=N_STARTS, score_fn=None):
    """Backwards-compatible 5-tuple wrapper around the detailed version."""
    d = run_2d_registration_multistart_detailed(
        fixed_slice, moving_slice, transform_type, num_iters, n_starts, score_fn)
    return d["best_img"], d["best_transform"], d["best_score"], d["best_seed"], d["scores"]


NMI_BINS = 64
NMI_CLIP_PERCENTILES = (0.5, 99.5)


def nmi_score(fixed_slice, moving_slice, bins=NMI_BINS):
    """
    Studholme normalized mutual information between two slices already on
    the same physical grid:

        NMI = (H(fixed) + H(moving)) / H(fixed, moving)

    Higher = better, bounded in [1, 2] (1 = statistically independent,
    2 = identical partitions). This replaces raw Mattes MI as the scoring
    metric everywhere a number is compared or reported.

    Why NMI rather than MI. Mattes MI is unbounded and grows with the
    marginal entropies, so its value depends on how much image is in the
    frame as much as on how well the two images line up. That made the
    numbers in this investigation hard to compare across regions, and
    outright misleading across canvas sizes - the exact comparison the
    conditional-crop logic in registration_demo_sweep_v3.py has to make.
    Dividing by the joint entropy removes that first-order dependence.

    Intensities are clipped to their 0.5/99.5 percentiles before binning
    so a handful of extreme voxels (CT metal, MRI spikes) can't collapse
    every real tissue value into one or two bins. Clipping, not discarding:
    every pixel still contributes, it just saturates.

    Returns NaN when either slice is constant over that percentile range -
    e.g. an edge slice resampled entirely to default fill. An entropy of
    zero makes the ratio undefined, and NaN is the honest record of "no
    overlap to score" rather than a number that looks like a result.

    NOTE this is a SCORING metric only. The optimizer inside
    run_2d_registration still minimizes Mattes MI, because SimpleITK 2.5
    exposes no NMI metric to ImageRegistrationMethod (the MI-family options
    are Mattes and JointHistogram only). Multi-start turns that into a
    smaller problem than it sounds: Mattes proposes candidate alignments,
    NMI decides which one is kept.
    """
    fixed = sitk.GetArrayFromImage(fixed_slice).astype(np.float64).ravel()
    moving = sitk.GetArrayFromImage(moving_slice).astype(np.float64).ravel()
    if fixed.shape != moving.shape:
        raise ValueError(f"NMI needs matching grids, got {fixed.shape} vs {moving.shape}")

    finite = np.isfinite(fixed) & np.isfinite(moving)
    if not np.any(finite):
        return float("nan")
    fixed, moving = fixed[finite], moving[finite]

    f_lo, f_hi = np.percentile(fixed, NMI_CLIP_PERCENTILES)
    m_lo, m_hi = np.percentile(moving, NMI_CLIP_PERCENTILES)
    if f_hi <= f_lo or m_hi <= m_lo:
        return float("nan")

    joint, _, _ = np.histogram2d(
        np.clip(fixed, f_lo, f_hi), np.clip(moving, m_lo, m_hi),
        bins=bins, range=[[f_lo, f_hi], [m_lo, m_hi]],
    )
    p = joint / joint.sum()

    def entropy(probs):
        nz = probs[probs > 0]
        return float(-np.sum(nz * np.log(nz)))

    h_joint = entropy(p)
    if h_joint <= 0:
        return float("nan")
    return (entropy(p.sum(axis=1)) + entropy(p.sum(axis=0))) / h_joint


def mattes_mi_score(fixed_slice, moving_slice):
    """
    Evaluate Mattes MI between two slices that already share the same
    physical grid (identity transform). SimpleITK reports this as a cost
    to MINIMIZE (more negative = better match); we negate it so higher =
    better match, matching intuition.

    Superseded by nmi_score for reporting and for multi-start selection;
    kept so earlier MI-based results stay reproducible and comparable.

    Returns NaN if evaluation fails - which happens whenever one of the
    slices is perfectly constant (e.g. an edge slice fully outside the
    other image's real content after a large uncorrected offset, resampled
    to all-default-fill). Mattes MI's internal histogram can't form a bin
    width from zero intensity variance and ITK raises rather than reporting
    "no overlap"; NaN is the honest way to record that here instead of
    crashing the caller.
    """
    fixed = sitk.Cast(fixed_slice, sitk.sitkFloat32)
    moving = sitk.Cast(moving_slice, sitk.sitkFloat32)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetInitialTransform(sitk.Transform(2, sitk.sitkIdentity))
    try:
        return -reg.MetricEvaluate(fixed, moving)
    except Exception as e:
        print(f"      ! MI evaluation failed (likely a blank/constant slice): {e}")
        return float("nan")


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
    ct_res = img_proc.resample_inplane(ct_image, target_spacing=cfg.TARGET_SPACING_MM, is_ct=True)
    # resample_mri_to_ct_grid does `mri_aligned = mri_image` then mutates its
    # Direction - a plain assignment, not a copy, so it silently overwrites
    # mri_corrected's real Direction in place. Feed it a disposable copy so
    # v2 (and the N4 display panel below) still see the MRI's true geometry.
    baseline_old_mri_res = img_proc.resample_mri_to_ct_grid(sitk.Image(mri_corrected), ct_res, default_pixel_value=0.0)
    baseline_mri_res, v2_translation_mm = resample_mri_to_ct_grid_v2(mri_corrected, ct_res, default_pixel_value=0.0)
    print(f"    v2 baseline: GEOMETRY translation = "
          f"({v2_translation_mm[0]:+.1f}, {v2_translation_mm[1]:+.1f}, {v2_translation_mm[2]:+.1f}) mm  "
          f"(old baseline left this at 0,0,0)")

    prefix = patient.split("_")[0]
    body_region = cfg.PREFIX_TO_REGION.get(prefix, "default")
    profile = cfg.REGION_PROFILES.get(body_region, cfg.REGION_PROFILES["default"])
    ct_win_min, ct_win_max = profile["ct_win_min"], profile["ct_win_max"]

    ct_arr_vol = sitk.GetArrayFromImage(ct_res)
    # Percentiles + slice selection now use the v2 baseline, since that's the
    # MRI content that actually goes on to be registered/displayed below.
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
        "baseline_old_mri_res": baseline_old_mri_res,
        "baseline_mri_res": baseline_mri_res,
        "v2_translation_mm": v2_translation_mm,
        "ct_slice_img": ct_res[:, :, z],
        "baseline_old_mri_slice_img": baseline_old_mri_res[:, :, z],
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
    baseline_old_slice_img = prep["baseline_old_mri_slice_img"]
    baseline_mri_slice_img = prep["baseline_mri_slice_img"]
    ct_win_min, ct_win_max = prep["ct_win_min"], prep["ct_win_max"]
    mri_p1, mri_p99 = prep["mri_p1"], prep["mri_p99"]
    z = prep["z"]

    # Registration now runs on top of the v2 (GEOMETRY-translation-corrected)
    # baseline, not the old direction-hack one - it starts from a better place.
    print("    Registering (multi-start, single-threaded, seeds 0..%d)..." % (N_STARTS - 1))
    rigid_img, rigid_transform, mi_rigid, rigid_seed, rigid_scores = run_2d_registration_multistart(
        ct_slice_img, baseline_mri_slice_img, "rigid"
    )
    affine_img, affine_transform, mi_affine, affine_seed, affine_scores = run_2d_registration_multistart(
        ct_slice_img, baseline_mri_slice_img, "affine"
    )

    mi_baseline_old = nmi_score(ct_slice_img, baseline_old_slice_img)
    mi_baseline = nmi_score(ct_slice_img, baseline_mri_slice_img)
    print(f"    NMI  baseline(old)={mi_baseline_old:.4f}  baseline(v2)={mi_baseline:.4f}  "
          f"rigid={mi_rigid:.4f}  affine={mi_affine:.4f}")

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
    baseline_old_arr = sitk.GetArrayFromImage(baseline_old_slice_img)
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
    baseline_old_disp = norm.normalize_mri_slice(baseline_old_arr, mri_p1, mri_p99)
    baseline_disp = norm.normalize_mri_slice(baseline_arr, mri_p1, mri_p99)
    rigid_disp = norm.normalize_mri_slice(rigid_arr, mri_p1, mri_p99)
    affine_disp = norm.normalize_mri_slice(affine_arr, mri_p1, mri_p99)

    v2t = prep["v2_translation_mm"]
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
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
    axes[1, 2].axis("off")

    fusion_panel(axes[2, 0], ct_res_disp, baseline_old_disp,
                 f"5. Baseline OLD (direction-hack)  NMI={mi_baseline_old:.3f}")
    fusion_panel(axes[2, 1], ct_res_disp, baseline_disp,
                 f"6. Baseline v2 (GEOMETRY translation)  NMI={mi_baseline:.3f}")
    axes[2, 2].axis("off")
    axes[2, 2].text(
        0.0, 0.5,
        f"v2 fix: leave MRI's real Direction alone,\n"
        f"align volume centers instead (3D, no\n"
        f"rotation search - closed-form, no seed).\n\n"
        f"Recovered translation:\n"
        f"({v2t[0]:+.1f}, {v2t[1]:+.1f}, {v2t[2]:+.1f}) mm\n\n"
        f"NMI  old→v2: {mi_baseline_old:.3f} → {mi_baseline:.3f}",
        fontsize=9, va="center", wrap=True,
    )

    fusion_panel(axes[3, 0], ct_res_disp, baseline_disp, f"7. Baseline v2 fusion  NMI={mi_baseline:.3f}")
    fusion_panel(axes[3, 1], ct_res_disp, rigid_disp, f"8. Rigid (best of {N_STARTS})  NMI={mi_rigid:.3f}")
    fusion_panel(axes[3, 2], ct_res_disp, affine_disp, f"9. Affine (best of {N_STARTS})  NMI={mi_affine:.3f}")

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
        "nmi_baseline_old": mi_baseline_old,
        "nmi_baseline": mi_baseline,
        "v2_translation_mm": "%.1f|%.1f|%.1f" % tuple(prep["v2_translation_mm"]),
        "nmi_rigid": mi_rigid,
        "nmi_affine": mi_affine,
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
