"""
registration_demo_knee_fix.py
──────────────────────────────
Follow-up experiment on the knee case from registration_demo.py, which
converged badly (rigid barely moved, affine visibly got worse while scoring
higher). Two suspected causes:

  1. CenteredTransformInitializer(..., GEOMETRY) centers the two image
     *canvases*, not the actual tissue. Since the baseline resample left the
     MRI's real content confined to a corner of its canvas (an uncorrected
     translation, not a wrong slice), GEOMETRY starts the optimizer from a
     bad guess.
  2. Most of the resampled canvas is empty background, so unmasked Mattes MI
     is dominated by background-matches-background rather than
     tissue-matches-tissue.

This tries MOMENTS initialization (intensity-weighted centroid) + an
Otsu-thresholded tissue mask on the SAME slice already picked by
registration_demo.py, and compares against the original GEOMETRY/unmasked
result on identical inputs (same fixed seed) to see whether it actually
closes the gap.
"""
import os
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import normalization as norm
import registration_demo as demo

KNEE_CANDIDATE = next(c for c in demo.CANDIDATES if c["region"] == "knee")


def to_normalized_sitk(arr, ref_img, is_ct, ct_win_min=None, ct_win_max=None, mri_p1=None, mri_p99=None):
    """Wrap a normalised [0,1] numpy slice back into an sitk.Image sharing
    ref_img's geometry, so CenteredTransformInitializer/OtsuThreshold see
    correct physical spacing/origin (needed for MOMENTS to be meaningful)."""
    if is_ct:
        norm_arr = norm.normalize_ct_slice(arr, ct_win_min, ct_win_max)
    else:
        norm_arr = norm.normalize_mri_slice(arr, mri_p1, mri_p99)
    img = sitk.GetImageFromArray(norm_arr.astype(np.float32))
    img.CopyInformation(ref_img)
    return img


def run_2d_registration_v2(fixed_raw, moving_raw, fixed_norm, moving_norm, transform_type, num_iters=200):
    """Same Mattes-MI / pyramid / gradient-descent settings as
    registration_demo.run_2d_registration, but: MOMENTS init instead of
    GEOMETRY, and an Otsu tissue mask on each image restricting the metric
    to real anatomy. Initialization + optimization run on the normalized
    [0,1] images (MOMENTS needs non-negative "mass"; raw CT HU includes
    large negative background values that would badly skew the centroid).
    The fitted transform is then re-applied to the RAW images so the
    resulting MI score is directly comparable to the original run's numbers.
    """
    fixed_n = sitk.Cast(fixed_norm, sitk.sitkFloat32)
    moving_n = sitk.Cast(moving_norm, sitk.sitkFloat32)

    base_transform = sitk.Euler2DTransform() if transform_type == "rigid" else sitk.AffineTransform(2)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_n, moving_n, base_transform,
        sitk.CenteredTransformInitializerFilter.MOMENTS
    )

    fixed_mask = sitk.OtsuThreshold(fixed_n, 0, 1, 200)
    moving_mask = sitk.OtsuThreshold(moving_n, 0, 1, 200)

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricFixedMask(fixed_mask)
    registration_method.SetMetricMovingMask(moving_mask)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.2, seed=42)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=0.1, numberOfIterations=num_iters,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(initial_transform, inPlace=True)

    try:
        final_transform = registration_method.Execute(fixed_n, moving_n)
    except Exception as e:
        print(f"    ! v2 {transform_type} registration failed: {e}")
        return moving_raw, None

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_raw)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(final_transform)
    resampled_raw = resampler.Execute(sitk.Cast(moving_raw, sitk.sitkFloat32))

    return resampled_raw, final_transform


def main():
    prep = demo.prepare_candidate(KNEE_CANDIDATE)
    ct_slice_img = prep["ct_slice_img"]
    baseline_mri_slice_img = prep["baseline_mri_slice_img"]
    ct_win_min, ct_win_max = prep["ct_win_min"], prep["ct_win_max"]
    mri_p1, mri_p99 = prep["mri_p1"], prep["mri_p99"]
    z = prep["z"]

    ct_arr = sitk.GetArrayFromImage(ct_slice_img)
    mri_arr = sitk.GetArrayFromImage(baseline_mri_slice_img)
    ct_norm_img = to_normalized_sitk(ct_arr, ct_slice_img, True, ct_win_min=ct_win_min, ct_win_max=ct_win_max)
    mri_norm_img = to_normalized_sitk(mri_arr, baseline_mri_slice_img, False, mri_p1=mri_p1, mri_p99=mri_p99)

    print("=== KNEE fix experiment: GEOMETRY/unmasked (old) vs MOMENTS/masked (new) ===")

    # -- Old settings, rerun on the exact same slice with the same fixed seed --
    rigid_old_img, rigid_old_t = demo.run_2d_registration(ct_slice_img, baseline_mri_slice_img, "rigid")
    affine_old_img, affine_old_t = demo.run_2d_registration(ct_slice_img, baseline_mri_slice_img, "affine")

    # -- New settings --
    rigid_new_img, rigid_new_t = run_2d_registration_v2(
        ct_slice_img, baseline_mri_slice_img, ct_norm_img, mri_norm_img, "rigid"
    )
    affine_new_img, affine_new_t = run_2d_registration_v2(
        ct_slice_img, baseline_mri_slice_img, ct_norm_img, mri_norm_img, "affine"
    )

    mi_baseline = demo.mattes_mi_score(ct_slice_img, baseline_mri_slice_img)
    mi_rigid_old = demo.mattes_mi_score(ct_slice_img, rigid_old_img)
    mi_affine_old = demo.mattes_mi_score(ct_slice_img, affine_old_img)
    mi_rigid_new = demo.mattes_mi_score(ct_slice_img, rigid_new_img)
    mi_affine_new = demo.mattes_mi_score(ct_slice_img, affine_new_img)

    print(f"    MI baseline          = {mi_baseline:.4f}")
    print(f"    MI rigid  (old)      = {mi_rigid_old:.4f}")
    print(f"    MI affine (old)      = {mi_affine_old:.4f}")
    print(f"    MI rigid  (v2)       = {mi_rigid_new:.4f}")
    print(f"    MI affine (v2)       = {mi_affine_new:.4f}")

    # -- Visualization: baseline | old rigid/affine | new rigid/affine --
    ct_disp = norm.normalize_ct_slice(ct_arr, ct_win_min, ct_win_max)
    baseline_disp = norm.normalize_mri_slice(mri_arr, mri_p1, mri_p99)
    rigid_old_disp = norm.normalize_mri_slice(sitk.GetArrayFromImage(rigid_old_img), mri_p1, mri_p99)
    affine_old_disp = norm.normalize_mri_slice(sitk.GetArrayFromImage(affine_old_img), mri_p1, mri_p99)
    rigid_new_disp = norm.normalize_mri_slice(sitk.GetArrayFromImage(rigid_new_img), mri_p1, mri_p99)
    affine_new_disp = norm.normalize_mri_slice(sitk.GetArrayFromImage(affine_new_img), mri_p1, mri_p99)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f"KNEE — {KNEE_CANDIDATE['patient']}  [slice {z}]  GEOMETRY/unmasked vs MOMENTS+Otsu-masked", fontsize=12)

    demo.fusion_panel(axes[0, 0], ct_disp, baseline_disp, f"Baseline  MI={mi_baseline:.3f}")
    demo.fusion_panel(axes[0, 1], ct_disp, rigid_old_disp, f"Rigid (old: GEOMETRY)  MI={mi_rigid_old:.3f}")
    demo.fusion_panel(axes[0, 2], ct_disp, affine_old_disp, f"Affine (old: GEOMETRY)  MI={mi_affine_old:.3f}")
    axes[1, 0].axis("off")
    demo.fusion_panel(axes[1, 1], ct_disp, rigid_new_disp, f"Rigid (v2: MOMENTS+mask)  MI={mi_rigid_new:.3f}")
    demo.fusion_panel(axes[1, 2], ct_disp, affine_new_disp, f"Affine (v2: MOMENTS+mask)  MI={mi_affine_new:.3f}")

    fig.tight_layout()
    out_png = os.path.join(demo.OUTPUT_DIR, "knee_PA32_Mandbi_knee_fix.png")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"    Saved {out_png}")


if __name__ == "__main__":
    main()
