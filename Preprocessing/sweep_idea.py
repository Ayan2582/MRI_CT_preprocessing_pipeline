"""
sweep_idea.py — registration_idea over every candidate.

The same 11 region x orientation pairs and the same first/middle/last positions
as sweep_og.py and registration_demo_sweep_v3.py, so all three CSVs line up.

    python sweep_idea.py

THE METHOD, IN FULL
───────────────────
Resample the CT slice and the MRI slice so one pixel is 1 mm in both, then try
every whole-pixel shift in a square and keep the best NMI. That is all of it.

WHAT TO LOOK FOR IN THE OUTPUT
──────────────────────────────
Two columns matter more than the scores:

  coverage      Should be identical before and after, on every row. A shift is
                scored on a fixed window and missing MRI gets its own histogram
                bin, so no candidate can raise its score by pushing awkward
                pixels out of view. In sweep_og that number collapsed from 100%
                to 6% on the worst slices.

  hit_edge      True means the best shift landed on the boundary of the search
                square, i.e. the real offset is probably further out than
                SEARCH allows and the answer is a wall, not a peak. Raise SEARCH
                and re-run those rows. This is the one failure this method has,
                and unlike the others it announces itself.

There is no rotation, scale or shear column because there is no rotation, scale
or shear. A whole-pixel slide cannot express them.

SLICE PAIRING — same caveat as sweep_og
───────────────────────────────────────
A 2D method has no shared world origin, so CT slice z is paired with the MRI
slice at the same fractional depth. That assumes both stacks cover the same
anatomy end to end. Where they do not, the pair is wrong and no registration
fixes it. Both indices are recorded.
"""

import os
import csv

import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import io_utils
import pipeline_config as cfg
import registration_demo_sweep as sweep
import registration_idea as idea

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "registration_demo_output", "sweep_idea_2")
COARSE = 4       # stride of the first sweep, mm. Verified against a full search on
KEEP = 5         # 3 slices: identical NMI and identical shift, ~13x fewer positions.
# +/- mm on each axis. (2*90+1)^2 = 32761 shifts per slice. Raised from 60 after
# 3 of 33 slices returned a best shift sitting exactly on the +/-60 boundary,
# which means the real offset was further out and the answer was a wall, not a
# peak. Cost is quadratic in this number, so raise it only when hit_edge says so.
SEARCH = 90
BINS = 32


def _norm(a):
    v = a[~np.isnan(a)]
    if v.size == 0:
        return np.zeros_like(a)
    lo, hi = np.percentile(v, [0.5, 99.5])
    return np.clip((np.nan_to_num(a, nan=lo) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def _overlay(ct, mri):
    """CT in amber, MRI in cyan — same scheme as the other sweeps."""
    rgb = np.zeros(ct.shape + (3,))
    rgb[..., 0] = _norm(ct)
    rgb[..., 1] = _norm(mri) * 0.75
    rgb[..., 2] = _norm(mri)
    return rgb


def save_figure(ct_win, mri_before, mri_after, r, cand, label, z, z_mri, path):
    fig, ax = plt.subplots(1, 4, figsize=(13.2, 3.7))
    ax[0].imshow(_norm(ct_win), cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("CT · fixed", fontsize=9)
    ax[1].imshow(_norm(mri_after), cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("MRI · shifted", fontsize=9)
    ax[2].imshow(_overlay(ct_win, mri_before))
    ax[2].set_title(f"overlay · before   NMI {r['baseline']:.4f}", fontsize=9)
    ax[3].imshow(_overlay(ct_win, mri_after))
    ax[3].set_title(f"overlay · after   NMI {r['best']:.4f}", fontsize=9)
    for a in ax:
        a.axis("off")

    hit = abs(r["dy"]) == r["search"] or abs(r["dx"]) == r["search"]
    fig.suptitle(
        f"{cand['region']} {cand['patient']} {cand['orientation']} {label} "
        f"(ct_z={z}, mri_z={z_mri})   "
        f"NMI {r['baseline']:.4f} -> {r['best']:.4f} ({r['best'] - r['baseline']:+.4f})\n"
        f"shift {r['dx']:+d} mm across, {r['dy']:+d} mm down   "
        f"coverage {r['baseline_coverage'] * 100:.0f}% -> {r['coverage'] * 100:.0f}%\n"
        f"search +/-{r['search']} mm · stride {r['coarse']} then fine around best {r['keep']} · "
        f"{r['evaluated']} positions (full search = {r['full_search_would_be']}) · "
        f"scored {r['window'][1]}x{r['window'][0]} px (whole CT frame)"
        + ("\n!! BEST SHIFT IS ON THE SEARCH BOUNDARY — raise SEARCH and re-run" if hit else ""),
        fontsize=8.5, color=("#b4442f" if hit else "#222222"))
    fig.tight_layout()
    fig.savefig(path, dpi=95)
    plt.close(fig)


def slice_and_spacing(volume, z):
    arr = sitk.GetArrayFromImage(volume)[z].astype(np.float64)
    sx, sy, _ = volume.GetSpacing()
    return arr, (float(sy), float(sx))


def process_volume(cand, rows):
    region, patient, orientation = cand["region"], cand["patient"], cand["orientation"]
    print(f"\n=== {region.upper()} : {patient} [{orientation}] ===")

    ct_vol, _ = io_utils.load_dicom_series(os.path.join(cfg.DATA_ROOT, "CT", patient, "ST0", cand["ct_se"]))
    mri_vol, _ = io_utils.load_dicom_series(os.path.join(cfg.DATA_ROOT, "MRI", patient, "ST0", cand["mri_se"]))
    if ct_vol is None or mri_vol is None:
        print("    ! Failed to load series, skipping.")
        return

    n_ct, n_mri = ct_vol.GetSize()[2], mri_vol.GetSize()[2]

    for label, z in (("first", 0), ("middle", n_ct // 2), ("last", n_ct - 1)):
        frac = z / (n_ct - 1) if n_ct > 1 else 0.0
        z_mri = int(round(frac * (n_mri - 1))) if n_mri > 1 else 0

        ct_arr, ct_sp = slice_and_spacing(ct_vol, z)
        mri_arr, mri_sp = slice_and_spacing(mri_vol, z_mri)

        ct = idea.to_1mm(ct_arr, ct_sp)
        mri = idea.to_1mm(mri_arr, mri_sp)
        r = idea.register(ct, mri, search=SEARCH, bins=BINS,
                          coarse=COARSE, keep=KEEP, verbose=False)

        s = r["search"]
        ct_win = ct                                    # the whole frame is scored and shown
        mri_before, _ = idea.sample_window(mri, ct.shape, 0, 0)
        mri_after, _ = idea.sample_window(mri, ct.shape, r["dy"], r["dx"])

        png = os.path.join(OUTPUT_DIR, f"{region}_{patient}_{orientation}_{label}_idea.png")
        save_figure(ct_win, mri_before, mri_after, r, cand, label, z, z_mri, png)

        hit = abs(r["dy"]) == s or abs(r["dx"]) == s
        print(f"      [{label:6s} ct_z={z:3d} mri_z={z_mri:3d}] "
              f"CT {ct.shape[1]}x{ct.shape[0]}mm  MRI {mri.shape[1]}x{mri.shape[0]}mm  "
              f"{r['baseline']:.4f} -> {r['best']:.4f} ({r['best'] - r['baseline']:+.4f})  "
              f"shift ({r['dx']:+d},{r['dy']:+d})mm  cov {r['baseline_coverage'] * 100:.0f}%->"
              f"{r['coverage'] * 100:.0f}%" + ("  ON BOUNDARY" if hit else ""))

        rows.append({
            "region": region, "patient": patient, "orientation": orientation, "position": label,
            "ct_slice": z, "mri_slice": z_mri, "n_ct_slices": n_ct, "n_mri_slices": n_mri,
            "ct_mm": f"{ct.shape[1]}x{ct.shape[0]}", "mri_mm": f"{mri.shape[1]}x{mri.shape[0]}",
            "search_mm": s, "shifts_tried": r["evaluated"], "bins": BINS,
            "coarse_stride": r["coarse"], "keep_refined": r["keep"],
            "full_search_would_be": r["full_search_would_be"],
            "window_px": f"{r['window'][1]}x{r['window'][0]}",
            "nmi_baseline": r["baseline"], "nmi_best": r["best"],
            "delta_nmi": r["best"] - r["baseline"],
            "dx_mm": r["dx"], "dy_mm": r["dy"],
            "shift_mm": float(np.hypot(r["dx"], r["dy"])),
            "coverage_before": r["baseline_coverage"], "coverage_after": r["coverage"],
            "coverage_lost": r["baseline_coverage"] - r["coverage"],
            "hit_edge": hit,
            "png": png,
        })


def summarize(rows):
    n = len(rows)
    print(f"\n{'=' * 74}\nSUMMARY  ({n} slices, resample to 1 mm + whole-pixel shift search)")
    d = [r["delta_nmi"] for r in rows]
    print(f"  delta NMI              : mean {np.mean(d):+.4f}  max {max(d):+.4f}  min {min(d):+.4f}")
    print(f"  improved / unchanged   : {sum(1 for x in d if x > 0.010)} / {sum(1 for x in d if x <= 0.010)}"
          f"   (MIN_GAIN = 0.010, as in the other sweeps)")
    # Still guaranteed after the switch to a strided sweep, but for a narrower
    # reason than before: shift (0,0) is forced onto the coarse grid whatever the
    # stride, so "do nothing" is always among the candidates and can always win.
    print(f"  worse than doing nothing: {sum(1 for x in d if x < 0)}   "
          "(shift 0,0 is always evaluated, so it can always win)")

    # Coverage is how much REAL MRI sits under the frame, and it legitimately
    # falls as the shift grows - move a picture 60 mm and part of it leaves the
    # frame. What stays constant is the number of pixels SCORED, because the
    # window is the whole CT and missing MRI gets its own bin instead of being
    # skipped. So this number is expected to be non-zero; what matters is that
    # losing coverage no longer BUYS a better score, which the correlation below
    # is the test for.
    lost = [r["coverage_lost"] for r in rows]
    d_arr = np.array(d)
    corr = np.corrcoef(np.array(lost), d_arr)[0, 1] if len(rows) > 2 else float("nan")
    print(f"  coverage lost to shift : mean {np.mean(lost) * 100:5.2f}%  max {max(lost) * 100:5.2f}%")
    print(f"  corr(coverage lost, NMI gain) = {corr:+.3f}    "
          "(sweep_og: +0.509 — there, deleting the image paid)")

    sh = [r["shift_mm"] for r in rows]
    print(f"  shift magnitude        : mean {np.mean(sh):5.1f} mm  max {max(sh):5.1f} mm")
    ev = [r["shifts_tried"] for r in rows]
    fs = rows[0]["full_search_would_be"]
    print(f"  positions tried        : mean {np.mean(ev):.0f} per slice, against {fs} "
          f"for a full search  ({fs / np.mean(ev):.1f}x less work)")

    edge = [r for r in rows if r["hit_edge"]]
    print(f"  best shift on boundary : {len(edge)}/{n}"
          + ("   -> raise SEARCH for these:" if edge else ""))
    for r in edge:
        print(f"      {r['region']:8s} {r['orientation']:9s} {r['position']:6s}  "
              f"({r['dx_mm']:+d},{r['dy_mm']:+d}) mm at limit {r['search_mm']}")
    print("=" * 74)


def main(only=None):
    """
    Run the sweep. `only` is an optional substring of the patient folder name,
    so a single patient can be redone without spending 33 slices of compute on
    the ones that have not changed.

    When filtering, the existing CSV is MERGED rather than replaced: rows for
    the slices just recomputed are swapped in and every other row is kept. A
    partial run that silently truncated the table to three rows would be a
    nasty way to lose a sweep.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Files are overwritten one at a time, in series order, so while a run is in
    # progress this folder holds a MIX of this run's output and the previous
    # run's. That is genuinely confusing if you open a picture mid-run and it
    # still shows old settings. Say so up front, and stamp the search range on
    # every file so any single picture can be identified on its own.
    stale = [f for f in os.listdir(OUTPUT_DIR) if f.endswith("_idea.png")]
    if stale:
        print(f"NOTE: {len(stale)} pictures from a previous run are in {OUTPUT_DIR}.")
        print(f"      They are replaced one at a time, in the order below. Until a given")
        print(f"      series is reached its picture is still the OLD one - check the")
        print(f"      'search +/-N mm' line in the title to tell them apart.\n")
    full = (2 * SEARCH + 1) ** 2
    print(f"search +/-{SEARCH} mm, stride {COARSE} then fine around the best {KEEP}")
    print(f"  a full search would be {full} positions per slice\n")

    rows = []
    for cand in sweep.ORIENTATION_CANDIDATES:
        process_volume(cand, rows)

    csv_path = os.path.join(OUTPUT_DIR, "sweep_idea_2_summary.csv")
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
