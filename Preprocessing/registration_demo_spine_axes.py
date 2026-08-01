"""
registration_demo_spine_axes.py
─────────────────────────────────
The sacrum-anchor diagnostic swept ONE degree of freedom - translation along
the spine's long axis - and showed the anchor turns a flat plateau into a
sharp peak. That is necessary but nowhere near sufficient: it says nothing
about the other axes. This script tests the two that were left unexamined.

For this sagittal series the image axes map to anatomy as:
    in-plane vertical   -> superior-inferior  (the axis already swept)
    in-plane horizontal -> anterior-posterior
    through-plane       -> LEFT-RIGHT, 5.57 mm spacing, 9 slices

So left-right is THROUGH-PLANE. No amount of 2D per-slice registration can
correct it; it is decided entirely by the volume-level centering step, and
if it is wrong the two slices show genuinely different anatomy (mid-sagittal
vertebral bodies versus paramedian pedicles and facet joints).

  TEST 1  in-plane landscape over BOTH in-plane axes at once, masked and
          unmasked - is the anchored optimum a sharp point, or a ridge that
          is only well-determined along one direction?

  TEST 2  through-plane sweep - slide the MRI along left-right in fine steps
          and score. Answers whether the slice correspondence is right, and
          whether the metric could even detect it if it were not.
"""
import os
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import normalization as norm
import registration_demo as demo
from registration_demo_spine_fix import make_inferior_mask, masked_mi

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "registration_demo_output")


def inplane_landscape(ct, mri, mask, dys, dxs):
    out = np.zeros((len(dys), len(dxs)))
    for i, dy in enumerate(dys):
        for j, dx in enumerate(dxs):
            t = sitk.TranslationTransform(2, (float(dx), float(dy)))
            r = sitk.ResampleImageFilter()
            r.SetReferenceImage(ct); r.SetInterpolator(sitk.sitkLinear)
            r.SetDefaultPixelValue(0.0); r.SetTransform(t)
            out[i, j] = masked_mi(ct, r.Execute(sitk.Cast(mri, sitk.sitkFloat32)), mask)
    return out


def main():
    cand = next(c for c in demo.CANDIDATES if c["region"] == "spine")
    prep = demo.prepare_candidate(cand)
    ct, mri = prep["ct_slice_img"], prep["baseline_mri_slice_img"]
    ct3, mri3 = prep["ct_res"], prep["mri_corrected"]
    z = prep["z"]
    p1, p99 = prep["mri_p1"], prep["mri_p99"]
    wmin, wmax = prep["ct_win_min"], prep["ct_win_max"]
    sacrum = make_inferior_mask(ct)

    # ---------- TEST 1: both in-plane axes ----------
    print("\n  [1] in-plane landscape over BOTH axes (S-I and A-P)")
    dys = np.arange(-60, 61, 5)
    dxs = np.arange(-40, 41, 5)
    land_full = inplane_landscape(ct, mri, None, dys, dxs)
    land_sac = inplane_landscape(ct, mri, sacrum, dys, dxs)

    def report(land, label):
        i, j = np.unravel_index(np.argmax(land), land.shape)
        rng = land.max() - land.min()
        thr = land.max() - 0.1 * rng
        near = land >= thr
        si_w = (dys[near.any(axis=1)].max() - dys[near.any(axis=1)].min())
        ap_w = (dxs[near.any(axis=0)].max() - dxs[near.any(axis=0)].min())
        print(f"      {label:16s} peak at S-I {dys[i]:+4d}mm, A-P {dxs[j]:+4d}mm | "
              f"near-peak spread: S-I {si_w:3d}mm, A-P {ap_w:3d}mm")
        return dys[i], dxs[j], si_w, ap_w

    fy, fx, fsi, fap = report(land_full, "unmasked")
    sy, sx, ssi, sap = report(land_sac, "sacrum-anchored")

    # ---------- TEST 2: through-plane = LEFT-RIGHT ----------
    print("\n  [2] through-plane (LEFT-RIGHT) sweep - the axis 2D registration cannot touch")
    lr_offsets = np.arange(-25, 26, 2.5)
    lr_scores_full, lr_scores_sac = [], []
    base_tx = sitk.CenteredTransformInitializer(
        sitk.Cast(ct3, sitk.sitkFloat32), sitk.Cast(mri3, sitk.sitkFloat32),
        sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    base_t = np.array(base_tx.GetTranslation())
    for d in lr_offsets:
        t = sitk.Euler3DTransform()
        t.SetCenter(base_tx.GetCenter())
        t.SetTranslation(tuple(base_t + np.array([d, 0.0, 0.0])))  # world X = L-R
        r = sitk.ResampleImageFilter()
        r.SetReferenceImage(ct3); r.SetInterpolator(sitk.sitkLinear)
        r.SetDefaultPixelValue(0.0); r.SetTransform(t)
        vol = r.Execute(sitk.Cast(mri3, sitk.sitkFloat32))
        sl = vol[:, :, z]
        lr_scores_full.append(masked_mi(ct, sl, None))
        lr_scores_sac.append(masked_mi(ct, sl, sacrum))
    lr_scores_full = np.array(lr_scores_full); lr_scores_sac = np.array(lr_scores_sac)
    pk_f = lr_offsets[int(np.argmax(lr_scores_full))]
    pk_s = lr_offsets[int(np.argmax(lr_scores_sac))]
    print(f"      unmasked        peak at L-R {pk_f:+.1f}mm  "
          f"(range {lr_scores_full.max()-lr_scores_full.min():.3f})")
    print(f"      sacrum-anchored peak at L-R {pk_s:+.1f}mm  "
          f"(range {lr_scores_sac.max()-lr_scores_sac.min():.3f})")
    print(f"      current pipeline sits at L-R 0.0mm -> "
          f"unmasked says it is off by {pk_f:+.1f}mm "
          f"({abs(pk_f)/ct3.GetSpacing()[2]:.1f} slices), "
          f"anchored says {pk_s:+.1f}mm ({abs(pk_s)/ct3.GetSpacing()[2]:.1f} slices)")

    # ---------- figure ----------
    fig = plt.figure(figsize=(13.5, 8.4))
    ext = [dxs[0], dxs[-1], dys[-1], dys[0]]
    for k, (land, ttl, py, px) in enumerate(
            [(land_full, "unmasked", fy, fx), (land_sac, "sacrum-anchored", sy, sx)]):
        a = fig.add_subplot(2, 3, 1 + k)
        im = a.imshow(land, extent=ext, aspect="auto", cmap="magma")
        a.contour(dxs, dys, land, levels=8, colors="w", linewidths=.4, alpha=.5)
        a.plot(px, py, "o", ms=8, mfc="none", mec="#5fb8d4", mew=2)
        a.set_xlabel("A-P shift (mm)"); a.set_ylabel("S-I shift (mm)")
        a.set_title(f"in-plane landscape - {ttl}", fontsize=9.5)
        fig.colorbar(im, ax=a, fraction=.046)

    a = fig.add_subplot(2, 3, 3)
    a.plot(lr_offsets, lr_scores_full, lw=1.7, color="#8a91a0", label="unmasked")
    a.plot(lr_offsets, lr_scores_sac, lw=1.9, color="#20748c", label="sacrum-anchored")
    a.axvline(0, ls="--", lw=1.2, color="#a13c50", label="where the pipeline sits")
    a.set_xlabel("LEFT-RIGHT shift (mm, through-plane)"); a.set_ylabel("Mattes MI")
    a.set_title("the axis 2D registration cannot fix", fontsize=9.5)
    a.legend(fontsize=7.5); a.grid(alpha=.2)

    ct_d = norm.normalize_ct_slice(sitk.GetArrayFromImage(ct), wmin, wmax)
    for k, d in enumerate([-10.0, 0.0, 10.0]):
        t = sitk.Euler3DTransform(); t.SetCenter(base_tx.GetCenter())
        t.SetTranslation(tuple(base_t + np.array([d, 0.0, 0.0])))
        r = sitk.ResampleImageFilter(); r.SetReferenceImage(ct3)
        r.SetInterpolator(sitk.sitkLinear); r.SetDefaultPixelValue(0.0); r.SetTransform(t)
        sl = r.Execute(sitk.Cast(mri3, sitk.sitkFloat32))[:, :, z]
        a = fig.add_subplot(2, 3, 4 + k)
        a.imshow(ct_d, cmap="gray", vmin=0, vmax=1)
        a.imshow(norm.normalize_mri_slice(sitk.GetArrayFromImage(sl), p1, p99),
                 cmap="hot", alpha=.45, vmin=0, vmax=1)
        a.set_title(f"L-R {d:+.0f}mm  MI={masked_mi(ct, sl, None):.3f}", fontsize=8.5)
        a.axis("off")

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "spine_axes_diagnostic.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"\n  saved {out}")


if __name__ == "__main__":
    main()
