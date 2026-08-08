"""
registration_og.py — the ORIGINAL (v3) explorer pipeline, in Python.

`working_regis.py` is the reference implementation of the CURRENT (v6) explorer.
This file is the reference implementation of **v3**, the earlier one, kept so the
two can be run against the same slice and diffed.

    python registration_og.py CT_FILE MRI_FILE [--grid 128] [--starts 3]

WHAT ACTUALLY DIFFERS BETWEEN v3 AND v6
───────────────────────────────────────
Almost nothing. Same metric, same optimiser, same gate, same constants, same
pyramid. The whole difference is **stage 01 — how the two images are put onto a
common grid** — plus one extra reporting stage that v6 added.

  v3 (this file)   LETTERBOX. Each image is scaled to fit a common square grid
                   (default 128x128) with its own physical aspect ratio
                   preserved, and padded with "invalid" outside. The two
                   modalities are NOT brought to a common millimetres-per-pixel.
                   A CT at 0.488 mm/px over 144 px and an MRI at 0.875 mm/px
                   over 144 px both become 128 px wide, so afterwards they sit
                   at 0.549 and 0.984 mm/px respectively.

  v6 (working_regis)  RESAMPLE TO 1.0 mm ISOTROPIC, then centre the MRI on the
                   CT's grid with NaN padding. Both modalities end at exactly
                   1.0 mm/px, so a 50 mm structure spans 50 px in each.

That difference has a consequence worth stating plainly, because it changes what
the scale gate MEANS:

  * Under v6 there is no genuine zoom left for affine to recover, so any large
    recovered scale is a cheat. That is the entire premise of the canvas-fit veto.
  * Under v3 the two images are still at different mm/px, so SOME of the scale
    affine recovers is a real, physical correction. The same gate is therefore
    doing a different job here, and a scale away from 1.0 is not automatically
    suspicious.

Stage numbering follows v3's own labels (8 stages). v6 inserted "One run, traced
iteration by iteration" at position 05, which is why its "What ships" is 09 and
v3's is 08. That stage is pure instrumentation and changes no result.

Everything numeric is imported from working_regis rather than copied, so the two
files cannot silently drift apart. If a number differs between v3 and v6, it is
because of `letterbox()` below and nothing else.

Dependencies: numpy, working_regis; pydicom only if you pass DICOM paths.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

import working_regis as wr
from working_regis import (
    NMI_BINS, SCALE_TOL, CANVAS_FIT_TOL, MIN_GAIN,
    MIN_TRANSFORM_MARGIN, MIN_MRI_COVERAGE,
    PYRAMID_SHRINK, CT_PRESETS,
    load_slice, nmi_score, valid_range, register_once, scale_verdict, classify,
    canvas_ratio, _sample_bilinear,
)

GRID = 128          # the common square grid, as reported in v3's footer


# ─────────────────────────── stage 01: the only real difference ───────────────────────────

@dataclass
class Boxed:
    """One image letterboxed onto the common grid."""
    data: np.ndarray        # (grid, grid) float64, NaN outside the source
    mm_per_px: float        # what one grid pixel is worth, in mm, for THIS image
    fills: float            # fraction of the grid the source occupies
    src_shape: tuple
    src_spacing: tuple
    name: str = ""
    modality: str = "?"


def letterbox(data: np.ndarray, spacing: tuple, grid: int = GRID,
              name: str = "", modality: str = "?") -> Boxed:
    """
    Fit `data` onto a `grid` x `grid` square, preserving its PHYSICAL aspect
    ratio, centred, with NaN outside.

    Physical rather than pixel aspect: for anisotropic PixelSpacing the pixel
    grid is not the shape of the anatomy, and squeezing a 0.5 x 1.5 mm raster
    into a square by pixel count would be exactly the "artificial stretch" the
    letterbox exists to avoid. For the isotropic spacings in this dataset the
    two readings coincide.

    Note what this deliberately does NOT do: it does not bring the two
    modalities to a common mm/px. Each is scaled by its own longest physical
    side. That is v3's behaviour and it is the whole difference from v6.
    """
    h, w = data.shape
    ext_y, ext_x = h * spacing[0], w * spacing[1]        # physical extent, mm
    s = grid / max(ext_x, ext_y)                         # grid px per mm
    out_w = max(1, int(round(ext_x * s)))
    out_h = max(1, int(round(ext_y * s)))

    gx, gy = np.meshgrid(np.arange(out_w), np.arange(out_h))
    xs = (gx + 0.5) * (w / out_w) - 0.5
    ys = (gy + 0.5) * (h / out_h) - 0.5
    scaled = _sample_bilinear(data, xs, ys)

    out = np.full((grid, grid), np.nan, dtype=np.float64)
    oy, ox = (grid - out_h) // 2, (grid - out_w) // 2
    sy0, sy1 = max(0, -oy), min(out_h, grid - oy)
    sx0, sx1 = max(0, -ox), min(out_w, grid - ox)
    if sy1 > sy0 and sx1 > sx0:
        out[oy + sy0:oy + sy1, ox + sx0:ox + sx1] = scaled[sy0:sy1, sx0:sx1]

    return Boxed(data=out, mm_per_px=1.0 / s, fills=(out_w * out_h) / (grid * grid),
                 src_shape=(h, w), src_spacing=tuple(spacing), name=name, modality=modality)


# ─────────────────────────── transform decomposition ───────────────────────────

def decompose(M: np.ndarray) -> dict:
    """
    Pull a 2x2 apart into the four quantities that mean something physically.

    v3's affine table reports a `shear` column that v6 dropped; this is where it
    comes from. All four fall out of the same E/F/G/H that mat_scale already
    computes, so reporting them costs nothing.

        scale       (s1 + s2) / 2      mean singular value  — what the gate tests
        rotation    atan2(H, E)        the rotation in the polar decomposition
        anisotropy  s1 / s2            1.0 = no shear/stretch; rises with shear
        shear       s1/s2 - 1          reported as v3 reports it, ~0 when clean

    Worth knowing while reading the gate: a pure rotation has singular values
    (1, 1), so `scale` is exactly 1.0 for ANY angle — the gate is blind to
    rotation by construction. `anisotropy` is the number that does see shear.
    """
    a, b, c, d = float(M[0, 0]), float(M[0, 1]), float(M[1, 0]), float(M[1, 1])
    E, F, G, H = (a + d) / 2, (a - d) / 2, (c + b) / 2, (c - b) / 2
    Q, R = math.hypot(E, H), math.hypot(F, G)
    s1, s2 = Q + R, abs(Q - R)
    return {
        "scale": (s1 + s2) / 2.0,
        "rotation": math.atan2(H, E),
        "anisotropy": (s1 / s2) if s2 > 1e-9 else float("inf"),
        "shear": (s1 / s2 - 1.0) if s2 > 1e-9 else float("inf"),
        "sigma": (s1, s2),
    }


# ─────────────────────────── the pipeline ───────────────────────────

def run_og(ct_data: np.ndarray, ct_spacing: tuple,
           mri_data: np.ndarray, mri_spacing: tuple,
           grid: int = GRID, n_starts: int = 3, bins: int = NMI_BINS,
           ct_clip: Optional[tuple] = None, quiet: bool = False,
           ct_name: str = "CT", mri_name: str = "MRI") -> dict:
    """
    v3's eight stages. Returns a dict; prints the derivation unless `quiet`.
    """
    rule = "-" * 74

    def head(n, title, badge=""):
        if not quiet:
            print(f"\n{rule}\n{n:>2}. {title}" + (f"   [{badge}]" if badge else "") + f"\n{rule}")

    def say(*a):
        if not quiet:
            print(*a)

    # ---- 01. prepare both images ------------------------------------------
    F = letterbox(ct_data, ct_spacing, grid, ct_name, "CT")
    M = letterbox(mri_data, mri_spacing, grid, mri_name, "MRI")
    fixed, moving = F.data, M.data
    ratio = canvas_ratio(moving)

    head(1, "PREPARE BOTH IMAGES", f"canvas ratio {ratio:.3f}")
    say("  Each image is letterboxed onto a common square grid — aspect ratio")
    say("  preserved, so no artificial stretch is introduced — and areas outside")
    say("  the source are marked invalid so they never enter the metric.\n")
    say(f"  {'Source':<6} {'Grid':>9} {'Fills':>7} {'Range':>18} {'Spacing mm':>11} {'-> mm/px':>9}")
    for tag, B in (("CT", F), ("MRI", M)):
        v = B.data[~np.isnan(B.data)]
        rng = f"{v.min():.0f} … {v.max():.0f}" if v.size else "empty"
        say(f"  {tag:<6} {B.src_shape[1]}x{B.src_shape[0]:<4} -> {grid}x{grid}"
            f" {B.fills * 100:6.1f}% {rng:>18} {B.src_spacing[1]:>11.3f} {B.mm_per_px:>9.3f}")
    if abs(F.mm_per_px - M.mm_per_px) > 1e-6:
        say(f"\n  NOTE: the two grids are NOT at the same mm/px "
            f"({F.mm_per_px:.3f} vs {M.mm_per_px:.3f}, ratio {M.mm_per_px / F.mm_per_px:.3f}).")
        say( "  Some of the scale affine recovers below is therefore a REAL correction,")
        say( "  not a cheat. This is what v6 changed by resampling both to 1.0 mm.")

    # ---- intensity ranges for the metric ----------------------------------
    if ct_clip is not None:
        ct_window = tuple(float(x) for x in ct_clip)
    else:
        ct_window = valid_range(fixed)
    mri_window = valid_range(moving, nonzero=True)
    ranges = (ct_window, mri_window)

    # ---- 02. baseline ------------------------------------------------------
    base = nmi_score(fixed, moving, bins, ranges=ranges)
    head(2, "BASELINE — THE NUMBER TO BEAT", f"NMI {base.nmi:.4f}")
    say(f"  valid overlap  = {base.valid} of {fixed.size} px  ({base.coverage * 100:.1f}%)")
    say(f"  occupied bins  = {base.used_bins} of {bins * bins}\n")
    say(f"  H(CT)          = {base.h_fixed:.6f}")
    say(f"  H(MRI)         = {base.h_moving:.6f}")
    say(f"  H(joint)       = {base.h_joint:.6f}\n")
    say(f"  MI             = {base.h_fixed:.4f} + {base.h_moving:.4f} - "
        f"{base.h_joint:.4f} = {base.mi:.6f}")
    say(f"  NMI            = ( {base.h_fixed:.4f} + {base.h_moving:.4f} ) / "
        f"{base.h_joint:.4f} = {base.nmi:.6f}")

    # ---- 03 / 04. multi-start ---------------------------------------------
    results = {}
    for step_no, kind in ((3, "rigid"), (4, "affine")):
        rows = []
        for seed in range(n_starts):
            # mm=1.0: the grid is the unit here, so translations come out in
            # PIXELS, which is how v3 reports them. v6 reports millimetres
            # because its grid is 1 mm/px by construction.
            rows.append(register_once(fixed, moving, kind, seed, bins=bins,
                                      ranges=ranges, report_ranges=ranges, mm=1.0))
        valid = [r.nmi for r in rows if not math.isnan(r.nmi)]
        spread = (max(valid) - min(valid)) if len(valid) > 1 else None
        head(step_no, f"{kind.upper()} MULTI-START  ({3 if kind == 'rigid' else 6} degrees of freedom)",
             f"best {max(valid):.4f} · spread {spread:.4f}" if valid and spread is not None else "")
        if kind == "rigid":
            say(f"  {'Start':<8}{'tx px':>9}{'ty px':>9}{'θ°':>9}{'NMI':>9}{'evals':>7}   Gate")
        else:
            say(f"  {'Start':<8}{'tx px':>9}{'ty px':>9}{'scale':>9}{'shear':>9}"
                f"{'NMI':>9}{'evals':>7}   Gate")
        for r in rows:
            D = decompose(r.matrix)
            ok, why = scale_verdict(r.scale, ratio)
            gate = "admissible" if (kind == "rigid" or ok) else why
            if kind == "rigid":
                say(f"  seed {r.seed:<3}{r.params[0]:>9.2f}{r.params[1]:>9.2f}"
                    f"{math.degrees(r.params[2]):>9.2f}{r.nmi:>9.4f}{r.evals:>7}   {gate}")
            else:
                say(f"  seed {r.seed:<3}{r.params[0]:>9.2f}{r.params[1]:>9.2f}"
                    f"{D['scale']:>9.4f}{D['shear']:>9.4f}{r.nmi:>9.4f}{r.evals:>7}   {gate}")
        results[kind] = rows

    # ---- 05. the scale gate ------------------------------------------------
    vetoed = sum(1 for r in results["affine"] if not scale_verdict(r.scale, ratio)[0])
    head(5, "THE SCALE GATE", f"{vetoed} of {n_starts} affine vetoed")
    say(f"  canvas ratio          = {ratio:.4f}   (MRI content box ÷ frame)")
    say(f"  suspect scales        = {ratio:.4f}  and  {1 / ratio:.4f}\n")
    say(f"  TEST 1  reject if |scale − 1| > {SCALE_TOL:.2f}")
    say(f"  TEST 2  reject if |scale − suspect| ≤ {CANVAS_FIT_TOL:.2f}  (canvas-fit veto)")
    say(f"  {'Candidate':<16}{'scale':>9}{'|s−1|':>9}{'|s−r|':>9}{'|s−1/r|':>10}   Verdict")
    for r in results["affine"]:
        ok, why = scale_verdict(r.scale, ratio)
        say(f"  affine seed {r.seed:<4}{r.scale:>9.4f}{abs(r.scale - 1):>9.4f}"
            f"{abs(r.scale - ratio):>9.4f}{abs(r.scale - 1 / ratio):>10.4f}   "
            f"{'admissible' if ok else why.upper()}")
    say(f"  {'rigid (all)':<16}{1.0:>9.4f}{0.0:>9.4f}{'—':>9}{'—':>10}   exempt · cannot scale")

    # ---- 06. selection -----------------------------------------------------
    def best_of(rows, gated):
        ok = [r for r in rows if not math.isnan(r.nmi)
              and (not gated or scale_verdict(r.scale, ratio)[0])]
        return max(ok, key=lambda r: r.nmi) if ok else None

    bR, bA = best_of(results["rigid"], False), best_of(results["affine"], True)
    if bR and bA:
        cleared = bA.nmi > bR.nmi + MIN_TRANSFORM_MARGIN
        chosen = bA if cleared else bR
        why = ("affine clears rigid by more than the margin" if cleared
               else "affine did not clear the margin — keep the simpler model")
    else:
        chosen = bA or bR
        why = "only one candidate survived"

    head(6, "SELECTION", chosen.kind if chosen else "none")
    say(f"  best rigid             = " + (f"{bR.nmi:.6f}   (seed {bR.seed})" if bR else "none"))
    say(f"  best admissible affine = "
        + (f"{bA.nmi:.6f}   (seed {bA.seed}, scale {bA.scale:.4f})" if bA else "none"))
    if bR and bA:
        say(f"\n  affine − rigid         = {bA.nmi - bR.nmi:+.6f}")
        say(f"  margin required        = {MIN_TRANSFORM_MARGIN:.3f}   "
            f"{'cleared' if bA.nmi > bR.nmi + MIN_TRANSFORM_MARGIN else 'NOT cleared'}")
    say(f"\n  selected               = {chosen.kind if chosen else 'none'}   {why}")

    # ---- 07. classification ------------------------------------------------
    best_nmi = chosen.nmi if chosen else float("nan")
    cov = chosen.coverage if chosen else base.coverage
    outcome, failed = classify(best_nmi, base.nmi, cov)
    head(7, "CLASSIFICATION", outcome)
    say(f"  baseline               = {base.nmi:.6f}")
    say(f"  best registered        = {best_nmi:.6f}")
    say(f"  difference             = {best_nmi - base.nmi:+.6f}")
    say(f"  MIN_GAIN               = {MIN_GAIN:.3f}")
    say(f"  MRI coverage           = {cov * 100:.1f}%   (fails below {MIN_MRI_COVERAGE * 100:.0f}%)")
    say(f"\n  outcome                = {outcome}   "
        f"{'FAILURE — would trigger the crop fallback' if failed else 'no fallback needed'}")

    # ---- 08. what ships ----------------------------------------------------
    if chosen and chosen.nmi > base.nmi:
        ship_kind, ship_nmi, ship_img = chosen.kind, chosen.nmi, chosen.warped
    else:
        ship_kind, ship_nmi, ship_img = "unregistered", base.nmi, moving
    head(8, "WHAT SHIPS", ship_kind)
    say(f"  Baseline {base.nmi:.4f}   Shipped {ship_nmi:.4f}   "
        f"Δ NMI {ship_nmi - base.nmi:+.4f}   Transform {ship_kind}   Outcome {outcome}")
    if ship_kind == "unregistered":
        say("  nothing beat doing nothing — a legitimate outcome, not a bug")
    say(f"\n  Grid {grid}x{grid}, {n_starts} starts per transform, {bins} histogram bins, "
        f"pyramid {list(PYRAMID_SHRINK)}.")

    D = decompose(chosen.matrix) if chosen else None
    return {
        "fixed": fixed, "moving": moving, "shipped": ship_img,
        "ct_mm_per_px": F.mm_per_px, "mri_mm_per_px": M.mm_per_px,
        "ct_fills": F.fills, "mri_fills": M.fills,
        "baseline_nmi": base.nmi, "baseline_coverage": base.coverage,
        # Coverage AFTER the chosen transform. Read it against baseline_coverage:
        # a big drop means the transform improved its score by pushing pixels out
        # of the frame, i.e. by discarding the evidence it could not explain.
        "chosen_coverage": cov,
        "shipped_nmi": ship_nmi, "shipped_kind": ship_kind,
        "outcome": outcome, "failed": failed, "canvas_ratio": ratio,
        "rigid": results["rigid"], "affine": results["affine"], "chosen": chosen,
        "best_rigid": bR, "best_affine": bA,
        "affine_vetoed": vetoed,
        "chosen_scale": D["scale"] if D else None,
        "chosen_rotation_deg": math.degrees(D["rotation"]) if D else None,
        "chosen_shear": D["shear"] if D else None,
        "chosen_anisotropy": D["anisotropy"] if D else None,
    }


def run_pipeline(ct_path: str, mri_path: str, grid: int = GRID, n_starts: int = 3,
                 ct_clip: Optional[tuple] = None) -> dict:
    """CLI entry: two DICOM/raster files."""
    ct, mri = load_slice(ct_path), load_slice(mri_path)
    return run_og(ct.data, ct.spacing, mri.data, mri.spacing, grid=grid,
                  n_starts=n_starts, ct_clip=ct_clip,
                  ct_name=ct.name, mri_name=mri.name)


# ─────────────────────────── cli ───────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="v3 (original) 2D CT/MRI registration, with the derivation printed.")
    ap.add_argument("ct", help="fixed image — DICOM file or raster")
    ap.add_argument("mri", help="moving image — DICOM file or raster")
    ap.add_argument("--grid", type=int, default=GRID, help="common square grid (default 128)")
    ap.add_argument("--starts", type=int, default=3, help="multi-start count (default 3)")
    ap.add_argument("--ct-clip", nargs=2, type=float, metavar=("LO", "HI"))
    ap.add_argument("--ct-window", choices=sorted(CT_PRESETS))
    a = ap.parse_args(argv)

    ct_clip = a.ct_clip
    if a.ct_window:
        ct_clip = CT_PRESETS[a.ct_window]
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    for p in (a.ct, a.mri):
        if not os.path.exists(p):
            print(f"no such file: {p}", file=sys.stderr)
            return 2
    run_pipeline(a.ct, a.mri, grid=a.grid, n_starts=a.starts, ct_clip=ct_clip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
