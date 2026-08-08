"""
sweep_og.py — registration_og (the v3 explorer pipeline) over every candidate.

The same 11 region x orientation pairs and the same first/middle/last positions
as registration_demo_sweep_v3.py, so the two CSVs can be read side by side.

    python sweep_og.py

WHAT THIS IS AND IS NOT
───────────────────────
This is the v3 EXPLORER's pipeline run at sweep scale. It is deliberately NOT
the production pipeline, and the differences are the point:

                        sweep_og.py (v3)          registration_demo_sweep_v3.py
  dimensionality        2D slice pairs            3D volumes
  MRI placement         letterbox, centred        closed-form translation from
                                                  ImagePositionPatient (3D)
  N4 bias correction    none                      yes, shrink 4
  grid                  common 128x128            CT grid at 1 mm in-plane
  optimiser             Nelder-Mead on NMI        SimpleITK grad. descent on
                                                  Mattes MI, NMI only to select
  histogram bins        32                        64
  crop fallback         none                      8-corner FOV intersection

So the NMI numbers are NOT comparable between the two CSVs — different bin
counts, different pixel populations, different placement. What IS comparable is
the VERDICT columns: outcome, shipped transform, and whether affine beat rigid.
Read those, not the scores.

SLICE PAIRING — the honest weak point
─────────────────────────────────────
The explorer takes two 2D images and has no way to know which MRI slice
corresponds to which CT slice; a single slice carries no shared world origin.
sweep_v3 does not have this problem, because it resamples the whole MRI volume
onto the CT grid first, so slice z is well defined.

Here, CT slice z is paired with the MRI slice at the same FRACTIONAL depth:

    z_mri = round( z / (n_ct - 1) * (n_mri - 1) )

That is a defensible approximation and nothing better is available in 2D, but it
is an approximation: it assumes both stacks cover the same anatomy end to end.
Where they do not, the pair is simply wrong and no amount of registration fixes
it. Rows carry both indices so this is auditable.
"""

import os
import csv
import math

import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import io_utils
import pipeline_config as cfg
import registration_demo_sweep as sweep
import registration_og as og

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "registration_demo_output", "sweep_og")
GRID = 128
N_STARTS = 3


def _norm(a):
    """Percentile-stretch to 0..1 for display; NaN (no data) reads as black."""
    v = a[~np.isnan(a)]
    if v.size == 0:
        return np.zeros_like(a)
    lo, hi = np.percentile(v, [0.5, 99.5])
    return np.clip((np.nan_to_num(a, nan=lo) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def _overlay(ct, mri):
    """CT -> red channel, MRI -> cyan. Agreement reads pale/neutral, disagreement
    fringes amber on one side and cyan on the other. Same scheme the explorer
    uses, because a one-sided fringe is how residual ROTATION shows up long
    before any number moves."""
    rgb = np.zeros(ct.shape + (3,))
    rgb[..., 0] = _norm(ct)
    rgb[..., 1] = _norm(mri) * 0.75
    rgb[..., 2] = _norm(mri)
    return rgb


def save_figure(out, cand, label, z, z_mri, path):
    """
    Four panels plus the diagnostics that explain a bad slice.

    The two numbers to read are in the title: `rot` and `cov`. Nothing in this
    pipeline constrains rotation (the scale gate tests the mean singular value,
    which is exactly 1.0 for any rotation), and the cost function only rejects a
    candidate once coverage falls below 5%. So a large `rot` together with a
    collapse from `cov 100% -> 6%` is not a good alignment - it is the optimiser
    rotating until the pixels it cannot explain fall outside the frame.
    """
    fixed, moving, shipped = out["fixed"], out["moving"], out["shipped"]
    fig, ax = plt.subplots(1, 4, figsize=(13.2, 3.7))
    ax[0].imshow(_norm(fixed), cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("CT · fixed", fontsize=9)
    ax[1].imshow(_norm(shipped), cmap="gray", vmin=0, vmax=1)
    ax[1].set_title(f"MRI · {out['shipped_kind']}", fontsize=9)
    ax[2].imshow(_overlay(fixed, moving))
    ax[2].set_title(f"overlay · before   NMI {out['baseline_nmi']:.4f}", fontsize=9)
    ax[3].imshow(_overlay(fixed, shipped))
    ax[3].set_title(f"overlay · after   NMI {out['shipped_nmi']:.4f}", fontsize=9)
    for a in ax:
        a.axis("off")

    rot, ani = out["chosen_rotation_deg"], out["chosen_anisotropy"]
    cov0, cov1 = out["baseline_coverage"] * 100, out["chosen_coverage"] * 100
    flags = []
    if rot is not None and abs(rot) > 12:
        flags.append(f"ROTATION {rot:+.1f}deg UNGATED")
    if ani is not None and ani > 1.15:
        flags.append(f"ANISOTROPY {ani:.2f} UNGATED")
    if cov1 < 0.9 * cov0:
        flags.append(f"COVERAGE {cov0:.0f}%->{cov1:.0f}%")

    fig.suptitle(
        f"{cand['region']} {cand['patient']} {cand['orientation']} {label} "
        f"(ct_z={z}, mri_z={z_mri})   "
        f"NMI {out['baseline_nmi']:.4f} -> {out['shipped_nmi']:.4f} "
        f"({out['shipped_nmi'] - out['baseline_nmi']:+.4f})   {out['outcome']}\n"
        f"scale {out['chosen_scale']:.3f}   rot {rot:+.1f}deg   aniso {ani:.3f}   "
        f"cov {cov0:.0f}% -> {cov1:.0f}%   vetoed {out['affine_vetoed']}/{N_STARTS}"
        + (("\n!! " + "   |   ".join(flags)) if flags else ""),
        fontsize=8.5, color=("#b4442f" if flags else "#222222"))
    fig.tight_layout()
    fig.savefig(path, dpi=95)
    plt.close(fig)


def slice_and_spacing(volume, z):
    """One 2D slice as (row, col) float array, plus its (row, col) spacing in mm."""
    arr = sitk.GetArrayFromImage(volume)[z].astype(np.float64)
    sx, sy, _ = volume.GetSpacing()          # SimpleITK is (x, y, z)
    return arr, (float(sy), float(sx))       # numpy is [y, x]


def process_volume(cand, rows):
    region, patient, orientation = cand["region"], cand["patient"], cand["orientation"]
    print(f"\n=== {region.upper()} : {patient} [{orientation}] ===")

    ct_path = os.path.join(cfg.DATA_ROOT, "CT", patient, "ST0", cand["ct_se"])
    mri_path = os.path.join(cfg.DATA_ROOT, "MRI", patient, "ST0", cand["mri_se"])
    ct_vol, _ = io_utils.load_dicom_series(ct_path)
    mri_vol, _ = io_utils.load_dicom_series(mri_path)
    if ct_vol is None or mri_vol is None:
        print("    ! Failed to load series, skipping.")
        return

    n_ct, n_mri = ct_vol.GetSize()[2], mri_vol.GetSize()[2]
    print(f"    CT {ct_vol.GetSize()}  spacing {tuple(round(s, 3) for s in ct_vol.GetSpacing())}")
    print(f"    MRI {mri_vol.GetSize()}  spacing {tuple(round(s, 3) for s in mri_vol.GetSpacing())}")

    for label, z in (("first", 0), ("middle", n_ct // 2), ("last", n_ct - 1)):
        frac = z / (n_ct - 1) if n_ct > 1 else 0.0
        z_mri = int(round(frac * (n_mri - 1))) if n_mri > 1 else 0

        ct_arr, ct_sp = slice_and_spacing(ct_vol, z)
        mri_arr, mri_sp = slice_and_spacing(mri_vol, z_mri)

        out = og.run_og(ct_arr, ct_sp, mri_arr, mri_sp, grid=GRID,
                        n_starts=N_STARTS, quiet=True,
                        ct_name=f"{patient}/{cand['ct_se']}",
                        mri_name=f"{patient}/{cand['mri_se']}")

        png = os.path.join(
            OUTPUT_DIR,
            f"{region}_{patient}_{orientation}_{label}_og.png")
        save_figure(out, cand, label, z, z_mri, png)

        bR, bA, ch = out["best_rigid"], out["best_affine"], out["chosen"]
        print(f"      [{label:6s} ct_z={z:3d} mri_z={z_mri:3d}] "
              f"cov={out['baseline_coverage'] * 100:5.1f}%  ratio={out['canvas_ratio']:.3f}  "
              f"base={out['baseline_nmi']:.4f} -> {out['shipped_nmi']:.4f} "
              f"({out['shipped_nmi'] - out['baseline_nmi']:+.4f})  "
              f"{out['shipped_kind']:<13s} {out['outcome']}"
              + (f"  rot={out['chosen_rotation_deg']:+.1f}deg" if ch else ""))

        rows.append({
            "region": region, "patient": patient, "orientation": orientation,
            "position": label,
            "ct_slice": z, "mri_slice": z_mri, "n_ct_slices": n_ct, "n_mri_slices": n_mri,
            "grid": GRID, "n_starts": N_STARTS, "bins": og.NMI_BINS,
            "ct_mm_per_px": out["ct_mm_per_px"], "mri_mm_per_px": out["mri_mm_per_px"],
            # >1 means the two letterboxed grids are at different physical scales,
            # i.e. part of any recovered affine scale is a genuine correction.
            "grid_scale_mismatch": out["mri_mm_per_px"] / out["ct_mm_per_px"],
            "ct_fills": out["ct_fills"], "mri_fills": out["mri_fills"],
            "canvas_ratio": out["canvas_ratio"],
            "baseline_coverage": out["baseline_coverage"],
            "chosen_coverage": out["chosen_coverage"],
            "coverage_lost": out["baseline_coverage"] - out["chosen_coverage"],
            "png": png,
            "nmi_baseline": out["baseline_nmi"],
            "nmi_rigid_best": bR.nmi if bR else None,
            "nmi_affine_best": bA.nmi if bA else None,
            "rigid_spread": spread([r.nmi for r in out["rigid"]]),
            "affine_spread": spread([r.nmi for r in out["affine"]]),
            "affine_vetoed": out["affine_vetoed"],
            "selected_kind": ch.kind if ch else "none",
            "shipped_kind": out["shipped_kind"],
            "nmi_shipped": out["shipped_nmi"],
            "delta_nmi": out["shipped_nmi"] - out["baseline_nmi"],
            "outcome": out["outcome"], "failed": out["failed"],
            # These three are what the tilt investigation needs. The scale gate
            # tests only `scale`; rotation and anisotropy are currently ungated.
            "chosen_scale": out["chosen_scale"],
            "chosen_rotation_deg": out["chosen_rotation_deg"],
            "chosen_shear": out["chosen_shear"],
            "chosen_anisotropy": out["chosen_anisotropy"],
        })


def spread(scores):
    valid = [s for s in scores if s is not None and not math.isnan(s)]
    return (max(valid) - min(valid)) if len(valid) > 1 else None


def summarize(rows):
    n = len(rows)
    print(f"\n{'=' * 74}\nSUMMARY  ({n} slices, v3/og pipeline, NMI, no crop fallback)")

    def tally(key):
        d = {}
        for r in rows:
            k = str(r[key]).split("_0.")[0]     # collapse sparse_overlap_0.NN
            d[k] = d.get(k, 0) + 1
        return ", ".join(f"{k}={v}" for k, v in sorted(d.items()))

    print("  outcomes               : " + tally("outcome"))
    print("  transform shipped      : " + tally("shipped_kind"))
    vet = sum(int(r["affine_vetoed"]) for r in rows)
    print(f"  affine seeds vetoed by the scale gate: {vet}/{n * N_STARTS}")

    d = [r["delta_nmi"] for r in rows]
    print(f"  delta NMI              : mean {np.mean(d):+.4f}  "
          f"max {max(d):+.4f}  min {min(d):+.4f}")

    rot = [abs(r["chosen_rotation_deg"]) for r in rows if r["chosen_rotation_deg"] is not None]
    ani = [r["chosen_anisotropy"] for r in rows if r["chosen_anisotropy"] is not None]
    if rot:
        # Nothing in the pipeline constrains either of these today. Printing them
        # is how you find out whether that matters on real data.
        print(f"  |rotation| shipped     : mean {np.mean(rot):5.2f}deg  max {max(rot):5.2f}deg  "
              f"({sum(1 for r in rot if r > 12)}/{len(rot)} exceed 12deg)")
        print(f"  anisotropy shipped     : mean {np.mean(ani):.4f}  max {max(ani):.4f}  "
              f"({sum(1 for a in ani if a > 1.15)}/{len(ani)} exceed 1.15)")

    mism = [r["grid_scale_mismatch"] for r in rows]
    print(f"  grid scale mismatch    : mean {np.mean(mism):.3f}  "
          f"min {min(mism):.3f}  max {max(mism):.3f}   (1.0 would mean equal mm/px)")
    print("=" * 74)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    for cand in sweep.ORIENTATION_CANDIDATES:
        process_volume(cand, rows)

    csv_path = os.path.join(OUTPUT_DIR, "sweep_og_summary.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        summarize(rows)
        print(f"\nWrote {len(rows)} rows to {csv_path}")
    else:
        print("\nNo rows processed.")


if __name__ == "__main__":
    main()
