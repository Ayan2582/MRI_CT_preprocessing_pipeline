"""
registration_demo_spine_fix.py
────────────────────────────────
Two candidate fixes for the spine registration failure diagnosed earlier:
Mattes MI is nearly blind to position along the lumbar spine's long axis
(signal-to-noise 0.28 - the gap between "correct" and "a whole vertebra off"
is a third of the optimizer's own run-to-run noise), and full affine exploits
that by shrinking the MRI until its extent matches the CT's (recovered scale
0.825 vs an MRI/CT FOV ratio of 0.824) rather than by aligning anatomy.

  A. SCALE-CONSTRAINED - Similarity2DTransform (isotropic scale) instead of
     full affine, plus explicit rejection of any solution whose scale strays
     more than SCALE_TOL from 1.0. The same patient's spine cannot genuinely
     change size by 17% between scans, so a large recovered scale is prima
     facie evidence of a bad fit, not a real correction.

  B. SACRUM-ANCHORED - restrict the METRIC to the inferior portion of the
     slice, which contains the sacrum and the L5-S1 junction. Lumbar
     vertebrae are near-periodic and so give MI nothing to lock onto, but
     the sacrum is a unique, non-repeating shape. The transform is still
     applied to the whole slice; only the region the metric scores is
     restricted.

The decisive test is not the final MI value - that would be validating an MI
failure with the same MI. It is the SHAPE of the MI-vs-offset landscape: if
anchoring works, sliding the MRI along the spine should produce one sharp
peak instead of the ~78mm plateau the unmasked metric shows.
"""
import os
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import normalization as norm
import registration_demo as demo

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "registration_demo_output")
N_STARTS = 5
SCALE_TOL = 0.05          # reject solutions scaling by more than +/-5%
SACRUM_FRACTION = 0.45    # inferior fraction of the slice used as the anchor


def make_inferior_mask(ct_slice, fraction=SACRUM_FRACTION):
    """
    Binary metric mask covering the inferior `fraction` of the slice - the
    region holding the sacrum and L5-S1. Rendered and saved below so the
    region can be eyeballed rather than taken on trust.
    """
    arr = np.zeros(sitk.GetArrayFromImage(ct_slice).shape, dtype=np.uint8)
    cut = int(arr.shape[0] * (1.0 - fraction))
    arr[cut:, :] = 1
    mask = sitk.GetImageFromArray(arr)
    mask.CopyInformation(ct_slice)
    return mask


def register(fixed, moving, transform_type, seed, fixed_mask=None, num_iters=100):
    """One registration attempt. transform_type: rigid | similarity | affine."""
    f = sitk.Cast(fixed, sitk.sitkFloat32)
    m = sitk.Cast(moving, sitk.sitkFloat32)

    base = {"rigid": sitk.Euler2DTransform,
            "similarity": sitk.Similarity2DTransform,
            "affine": lambda: sitk.AffineTransform(2)}[transform_type]()

    init = sitk.CenteredTransformInitializer(
        f, m, base, sitk.CenteredTransformInitializerFilter.GEOMETRY)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    if fixed_mask is not None:
        reg.SetMetricFixedMask(fixed_mask)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, seed=seed)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(learningRate=0.1, numberOfIterations=num_iters,
                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(init, inPlace=True)

    try:
        t = reg.Execute(f, m)
    except Exception as e:
        print(f"        ! {transform_type} seed={seed} failed: {e}")
        return None, None

    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(f); r.SetInterpolator(sitk.sitkLinear)
    r.SetDefaultPixelValue(0.0); r.SetTransform(t)
    return r.Execute(m), t


def recovered_scale(transform, transform_type):
    if transform_type == "rigid":
        return 1.0
    if transform_type == "similarity":
        return float(transform.GetScale())
    sv = np.linalg.svd(np.array(transform.GetMatrix()).reshape(2, 2), compute_uv=False)
    return float(np.mean(sv))


def masked_mi(fixed, moving, mask=None):
    f = sitk.Cast(fixed, sitk.sitkFloat32)
    m = sitk.Cast(moving, sitk.sitkFloat32)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    if mask is not None:
        reg.SetMetricFixedMask(mask)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetInitialTransform(sitk.Transform(2, sitk.sitkIdentity))
    try:
        return -reg.MetricEvaluate(f, m)
    except Exception:
        return float("nan")


def run_condition(name, fixed, moving, transform_type, fixed_mask, enforce_scale):
    """Multi-start, keeping the best attempt that also passes the scale gate."""
    best = None
    for seed in range(N_STARTS):
        img, t = register(fixed, moving, transform_type, seed, fixed_mask)
        if t is None:
            continue
        sc = recovered_scale(t, transform_type)
        if enforce_scale and abs(sc - 1.0) > SCALE_TOL:
            print(f"        seed={seed} REJECTED (scale {sc:.3f} outside "
                  f"1.0+/-{SCALE_TOL})")
            continue
        # score on the same region the optimiser saw, for a like-for-like pick
        score = masked_mi(fixed, img, fixed_mask)
        if best is None or score > best["score"]:
            best = dict(img=img, transform=t, scale=sc, score=score, seed=seed)
    if best is None:
        print(f"      {name}: no attempt survived")
        return None
    print(f"      {name}: seed={best['seed']}  scale={best['scale']:.3f}  "
          f"score(on metric region)={best['score']:.3f}")
    return best


def mi_landscape(fixed, moving, mask, offsets):
    out = []
    for dy in offsets:
        t = sitk.TranslationTransform(2, (0.0, float(dy)))
        r = sitk.ResampleImageFilter()
        r.SetReferenceImage(fixed); r.SetInterpolator(sitk.sitkLinear)
        r.SetDefaultPixelValue(0.0); r.SetTransform(t)
        out.append(masked_mi(fixed, r.Execute(sitk.Cast(moving, sitk.sitkFloat32)), mask))
    return np.array(out)


def main():
    cand = next(c for c in demo.CANDIDATES if c["region"] == "spine")
    prep = demo.prepare_candidate(cand)
    ct, mri = prep["ct_slice_img"], prep["baseline_mri_slice_img"]
    p1, p99 = prep["mri_p1"], prep["mri_p99"]
    wmin, wmax = prep["ct_win_min"], prep["ct_win_max"]
    sacrum = make_inferior_mask(ct)

    # ---- 1. Does anchoring sharpen the landscape? ----
    print("\n  [1] MI-vs-offset landscape, unmasked vs sacrum-anchored")
    offs = np.arange(-70, 71, 2)
    land_full = mi_landscape(ct, mri, None, offs)
    land_sac = mi_landscape(ct, mri, sacrum, offs)

    def describe(land, label):
        rng = land.max() - land.min()
        peak = offs[int(np.argmax(land))]
        # width of the region within 10% of the peak's height above the floor
        thresh = land.max() - 0.1 * rng
        width = (offs[land >= thresh].max() - offs[land >= thresh].min())
        print(f"      {label:16s} peak at {peak:+4d}mm  range={rng:.3f}  "
              f"near-peak width={width:3d}mm")
        return peak, rng, width

    peak_f, rng_f, w_f = describe(land_full, "unmasked")
    peak_s, rng_s, w_s = describe(land_sac, "sacrum-anchored")
    print(f"      -> anchoring narrows the ambiguous window {w_f}mm -> {w_s}mm "
          f"and deepens contrast {rng_f:.3f} -> {rng_s:.3f}")

    # ---- 2. The candidate fixes ----
    print("\n  [2] registration conditions")
    conds = {}
    conds["affine (unconstrained)"] = run_condition(
        "affine (unconstrained)", ct, mri, "affine", None, False)
    conds["similarity (scale-constrained)"] = run_condition(
        "similarity (scale-constrained)", ct, mri, "similarity", None, True)
    conds["similarity + sacrum anchor"] = run_condition(
        "similarity + sacrum anchor", ct, mri, "similarity", sacrum, True)

    # ---- 3. Figure ----
    ct_d = norm.normalize_ct_slice(sitk.GetArrayFromImage(ct), wmin, wmax)
    base_d = norm.normalize_mri_slice(sitk.GetArrayFromImage(mri), p1, p99)

    fig = plt.figure(figsize=(13.5, 7.6))
    ax = fig.add_subplot(2, 3, (1, 2))
    ax.plot(offs, land_full, lw=1.7, color="#8a91a0", label=f"unmasked (ambiguous over {w_f}mm)")
    ax.plot(offs, land_sac, lw=1.9, color="#20748c", label=f"sacrum-anchored (peak within {w_s}mm)")
    ax.axvline(peak_s, ls="--", lw=1, color="#a13c50")
    ax.set_xlabel("MRI shifted along the spine axis (mm)")
    ax.set_ylabel("Mattes MI (on its own metric region)")
    ax.set_title("Anchoring on the sacrum gives the metric a unique target", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=.2)

    axm = fig.add_subplot(2, 3, 3)
    axm.imshow(ct_d, cmap="gray", vmin=0, vmax=1)
    axm.imshow(sitk.GetArrayFromImage(sacrum), cmap="spring", alpha=.30)
    axm.set_title(f"anchor region (inferior {int(SACRUM_FRACTION*100)}%)", fontsize=9)
    axm.axis("off")

    panels = [("baseline (no 2D reg)", mri)]
    for k in ("affine (unconstrained)", "similarity (scale-constrained)", "similarity + sacrum anchor"):
        if conds[k] is not None:
            panels.append((k, conds[k]["img"]))

    for i, (title, img) in enumerate(panels[:3]):
        a = fig.add_subplot(2, 3, 4 + i)
        a.imshow(ct_d, cmap="gray", vmin=0, vmax=1)
        a.imshow(norm.normalize_mri_slice(sitk.GetArrayFromImage(img), p1, p99),
                 cmap="hot", alpha=.45, vmin=0, vmax=1)
        extra = ""
        if title in conds and conds[title] is not None:
            extra = f"\nscale={conds[title]['scale']:.3f}"
        a.set_title(f"{title}{extra}", fontsize=8.5)
        a.axis("off")

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "spine_fix_comparison.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n  saved {out}")

    if len(panels) > 3:
        fig2, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))
        for a, (title, img) in zip(axes, panels[2:4]):
            a.imshow(ct_d, cmap="gray", vmin=0, vmax=1)
            a.imshow(norm.normalize_mri_slice(sitk.GetArrayFromImage(img), p1, p99),
                     cmap="hot", alpha=.45, vmin=0, vmax=1)
            a.set_title(title, fontsize=8.5); a.axis("off")
        fig2.tight_layout()
        out2 = os.path.join(OUTPUT_DIR, "spine_fix_comparison_b.png")
        fig2.savefig(out2, dpi=120); plt.close(fig2)
        print(f"  saved {out2}")


if __name__ == "__main__":
    main()
