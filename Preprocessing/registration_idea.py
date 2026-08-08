"""
registration_idea.py — the simplest thing that could work.

    1. resample the CT and the MRI so one pixel is exactly 1 mm in both
    2. slide the MRI over the CT and keep the best NMI

That is the entire method. No optimiser, no random starts, no gates, no
classification, no fallback.

Two properties fall out of that for free:

  * A whole-pixel slide cannot rotate, scale or shear. The three failure modes
    in docs/registration_gates_docs.md are not gated against here — they are
    impossible to express.
  * There are no random numbers anywhere, so running it twice gives the same
    answer.

THE SEARCH IS COARSE-TO-FINE, NOT EXHAUSTIVE
────────────────────────────────────────────
An earlier version tried every whole-pixel position, which at +/-90 mm is 32761
of them and took ~32 s a slice. It now sweeps the range at a stride of COARSE
first, keeps the best KEEP positions, and re-searches every whole pixel around
each of those. Roughly 13x less work for the same answer on this dataset.

This does give something up and it should be said plainly: **a strided sweep can
step over a peak narrower than the stride**, so "best position found" is no
longer quite the same statement as "best position there is". Two things keep
that unlikely rather than merely hoped for:

  * anatomy at 1 mm is many pixels wide, so the score varies smoothly over a
    4 mm step - there are no one-pixel spikes to fall between;
  * refining the best KEEP positions rather than only the winner means the true
    peak still gets found when it was merely runner-up on the coarse pass.

Set COARSE = 1 to get the old exhaustive behaviour back and check.

Also, "do nothing" (shift 0,0) and the four range limits are always evaluated,
whatever the stride - the first so the result can never be worse than not
moving, the others so hitting the edge of the range is still detectable.

Every candidate is scored on the SAME fixed window of pixels, and MRI that falls
outside the frame goes into its own histogram bin instead of being skipped. So a
shift cannot improve its score by pushing awkward pixels out of view - the
"crossing out the questions you got wrong" problem from the gates document.

    python registration_idea.py CT_FILE MRI_FILE [--range 40] [--bins 32]

Dependencies: numpy, and pydicom for the command line only — it is imported
inside read_dicom, so importing this module as a library does not need it.

USED BY THE PIPELINE
────────────────────
This is the registration the production pipeline runs under --register_2d. It
gets there through image_processing.estimate_volume_translation, which calls
`register` on a handful of probe slices and turns their answers into ONE shift
for the whole volume; see that function for why a per-slice shift is the wrong
shape of answer for a stack. The three entry points it uses are:

    register(ct, mri, ...)   find the best shift, or None if unmeasurable
    make_scorer(ct, mri)     the NMI-of-a-shift function on its own, for
                             re-scoring a shift that was chosen elsewhere
    apply_shift(...)         actually move an image, with 0 outside the frame

Nothing here raises on bad input and nothing here prints unless asked. A slice
that cannot be measured returns None. That matters because the pipeline runs
this over thousands of slices with nobody watching, and the unmeasurable ones
are near-empty end-of-stack slices it deliberately keeps rather than a sign
anything is wrong.
"""

import argparse
import os
import sys

import numpy as np

BINS = 32
RANGE = 40          # search +/- this many mm (= pixels) on each axis
COARSE = 4          # stride of the first sweep, in mm. 1 = try every position
KEEP = 5            # how many coarse positions get a fine search around them


# ─────────────────────────── load ───────────────────────────

def read_dicom(path):
    """Pixel array in real units, plus (row, col) spacing in mm."""
    import pydicom
    d = pydicom.dcmread(path)
    a = d.pixel_array.astype(np.float64)
    a = a * float(getattr(d, "RescaleSlope", 1) or 1) + float(getattr(d, "RescaleIntercept", 0) or 0)
    sp = getattr(d, "PixelSpacing", None)
    sp = (float(sp[0]), float(sp[1])) if sp else (1.0, 1.0)
    return a, sp


# ─────────────────────────── step 1: 1 mm per pixel ───────────────────────────

def to_1mm(a, spacing):
    """
    Resample so one pixel is one millimetre on both axes.

    This is the whole idea. Afterwards a 50 mm structure spans 50 pixels in the
    CT and 50 pixels in the MRI, so the two images are directly comparable and
    the only thing left to find is where one sits relative to the other.
    """
    sy, sx = spacing
    h, w = a.shape
    H, W = max(1, int(round(h * sy))), max(1, int(round(w * sx)))

    ys = (np.arange(H) + 0.5) / sy - 0.5
    xs = (np.arange(W) + 0.5) / sx - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    fy = (ys - np.floor(ys))[:, None]
    fx = (xs - np.floor(xs))[None, :]

    return (a[np.ix_(y0, x0)] * (1 - fy) * (1 - fx) + a[np.ix_(y0, x1)] * (1 - fy) * fx +
            a[np.ix_(y1, x0)] * fy * (1 - fx) + a[np.ix_(y1, x1)] * fy * fx)


# ─────────────────────────── the metric ───────────────────────────

def bin_image(a, lo, hi, bins):
    """Map values to 0..bins-1. NaN becomes `bins`, a bin of its own."""
    idx = np.floor((a - lo) / (hi - lo) * bins)
    idx = np.clip(np.nan_to_num(idx, nan=bins), 0, bins - 1).astype(np.int64)
    return np.where(np.isnan(a), bins, idx)


def entropy(p):
    nz = p[p > 0]
    return float(-np.sum(nz * np.log(nz)))


def nmi(a_idx, b_idx, nb):
    """NMI from two already-binned images of the same shape."""
    j = np.bincount(a_idx.ravel() * nb + b_idx.ravel(), minlength=nb * nb)
    p = j.reshape(nb, nb) / j.sum()
    hj = entropy(p)
    if hj <= 0:
        return float("nan")
    return (entropy(p.sum(1)) + entropy(p.sum(0))) / hj


# ─────────────────────────── step 2: slide it ───────────────────────────

def sample_window(mri, ct_shape, dy, dx):
    """
    The MRI laid over the whole CT frame, shifted by (dy, dx). NaN where the MRI
    does not reach. Shift (0, 0) means the two images are centred on each other.

    The window is the FULL CT. It does not depend on the shift, which is the
    only property the metric needs: every candidate is scored on the same set of
    pixels. An earlier version inset the window by the search range "to be safe"
    and ended up scoring 31% of the CT on average - all cost, no benefit, since
    MRI that falls outside is already handled by giving it its own histogram bin
    rather than dropping it.
    """
    ch, cw = ct_shape
    mh, mw = mri.shape
    oy, ox = (mh - ch) // 2, (mw - cw) // 2
    ys = np.arange(ch)[:, None] + oy + dy
    xs = np.arange(cw)[None, :] + ox + dx
    ok = (ys >= 0) & (ys < mh) & (xs >= 0) & (xs < mw)
    return np.where(ok, mri[np.clip(ys, 0, mh - 1), np.clip(xs, 0, mw - 1)], np.nan), float(np.mean(ok))


def apply_shift(mri, ct_shape, dy, dx, fill=0.0):
    """
    The MRI as `sample_window` lays it down, with the outside-the-frame NaN
    replaced by `fill`. This is the function anything OUTSIDE the metric should
    use to actually move an image.

    NaN is the right value inside the metric — it is what earns missing MRI its
    own histogram bin — and the wrong value everywhere else. np.clip leaves NaN
    untouched, so a NaN that escapes from here travels through intensity
    normalisation into a saved .npy, and `arr <= threshold` is False for NaN, so
    a background check would count empty frame as anatomy. 0.0 is the same fill
    the pipeline's MRI resampling already uses for MRI that does not reach.
    """
    vals, cov = sample_window(mri, ct_shape, dy, dx)
    return np.where(np.isnan(vals), fill, vals), cov


def make_scorer(ct, mri, bins=BINS):
    """
    Build the "NMI of this shift" function for one CT/MRI pair, or return None
    if either image has no contrast to measure with.

    Returning None rather than raising is deliberate, and it is what lets this
    module run unattended. The slices that arrive here with a flat histogram are
    not corrupt files — they are the near-empty slices at the ends of a stack,
    which the pipeline keeps on purpose because a sliver of real anatomy can sit
    in one. "No shift can be measured on this slice" is an ordinary answer for
    them, and the caller is in a far better position than this function to
    decide what to do about it.

    Everything the metric needs that does not depend on the shift — the CT
    binning, both intensity ranges — is computed once here and closed over, so
    scoring a candidate is only the sampling and the joint histogram.
    """
    c_lo, c_hi = np.percentile(ct, [0.5, 99.5])
    m_lo, m_hi = np.percentile(mri[mri > 0] if (mri > 0).any() else mri, [0.5, 99.5])
    if c_hi <= c_lo or m_hi <= m_lo:
        return None

    nb = bins + 1                       # +1 for the "no MRI here" bin
    ct_idx = bin_image(ct, c_lo, c_hi, bins)

    def score(dy, dx):
        vals, cov = sample_window(mri, ct.shape, dy, dx)
        return nmi(ct_idx, bin_image(vals, m_lo, m_hi, bins), nb), cov

    return score


def register(ct, mri, search=RANGE, bins=BINS, coarse=COARSE, keep=KEEP, verbose=True):
    """
    Find the best whole-pixel shift in the square [-search, +search].

    Two passes: sweep the square at a stride of `coarse`, then re-search every
    whole pixel around the best `keep` positions from that sweep. Both images
    must already be at 1 mm per pixel. Pass coarse=1 for a full search.

    Returns None when there is nothing to measure — either image flat, or every
    candidate degenerate. See make_scorer for why that is a return value and not
    an exception.
    """
    # Every candidate is scored on the whole CT frame - the same pixels, every
    # time. MRI that has been shifted off the frame is not skipped; it lands in
    # a histogram bin of its own, so a shift is charged for what it moves out of
    # view instead of being quietly rewarded for it.
    win = ct
    wh, ww = win.shape

    score = make_scorer(ct, mri, bins)
    if score is None:
        return None

    seen = {}

    def once(dy, dx):
        """Score a position, remembering it so the two passes never overlap."""
        if (dy, dx) not in seen:
            seen[(dy, dx)] = score(dy, dx)
        return seen[(dy, dx)]

    # Offsets for the coarse sweep, built outwards from 0 so that "do nothing"
    # is always one of them whatever the stride, and with the two range limits
    # forced in so that landing on the edge stays detectable.
    offs = sorted(set(list(range(0, search + 1, coarse)) +
                      list(range(0, -search - 1, -coarse)) + [-search, search]))

    for dy in offs:
        for dx in offs:
            once(dy, dx)
    n_coarse = len(seen)

    # Refine around the best few, not just the best one: if the true peak was
    # runner-up on the coarse grid it is still reachable. The window is +/-coarse,
    # which is exactly the gap the stride left unexamined.
    top = sorted((v[0], k) for k, v in seen.items()
                 if v[0] is not None and not np.isnan(v[0]))[-keep:]
    for _, (cy, cx) in top:
        for dy in range(cy - coarse, cy + coarse + 1):
            for dx in range(cx - coarse, cx + coarse + 1):
                if -search <= dy <= search and -search <= dx <= search:
                    once(dy, dx)

    # A candidate scores NaN when its joint histogram collapses to a single
    # cell, which happens on slices with almost nothing in them. If "do nothing"
    # itself is NaN there is no baseline to improve on, and if EVERY candidate is
    # NaN there is no answer at all - both mean this slice cannot be measured,
    # which is the same "no result" that make_scorer reports.
    base, base_cov = seen[(0, 0)]
    usable = [(k, v) for k, v in seen.items() if not np.isnan(v[0])]
    if np.isnan(base) or not usable:
        return None

    (best_dy, best_dx), (best, best_cov) = max(usable, key=lambda kv: kv[1][0])

    if verbose:
        print(f"    coarse sweep (stride {coarse}): {n_coarse} positions")
        print(f"    fine search around the best {keep}: {len(seen) - n_coarse} more")
        print(f"    {len(seen)} total, versus {(2 * search + 1) ** 2} for a full search")

    return {
        "baseline": base, "baseline_coverage": base_cov,
        "best": best, "dy": best_dy, "dx": best_dx, "coverage": best_cov,
        "evaluated": len(seen), "window": (wh, ww), "search": search,
        "coarse": coarse, "keep": keep,
        "full_search_would_be": (2 * search + 1) ** 2,
    }


# ─────────────────────────── run ───────────────────────────

def run(ct_path, mri_path, search=RANGE, bins=BINS):
    ct_raw, ct_sp = read_dicom(ct_path)
    mri_raw, mri_sp = read_dicom(mri_path)

    print(f"  CT   {ct_raw.shape[1]}x{ct_raw.shape[0]} px at {ct_sp[1]:.3f} mm  "
          f"= {ct_raw.shape[1] * ct_sp[1]:.0f}x{ct_raw.shape[0] * ct_sp[0]:.0f} mm")
    print(f"  MRI  {mri_raw.shape[1]}x{mri_raw.shape[0]} px at {mri_sp[1]:.3f} mm  "
          f"= {mri_raw.shape[1] * mri_sp[1]:.0f}x{mri_raw.shape[0] * mri_sp[0]:.0f} mm")

    ct = to_1mm(ct_raw, ct_sp)
    mri = to_1mm(mri_raw, mri_sp)
    print(f"\n  after resampling to 1 mm/px:  CT {ct.shape[1]}x{ct.shape[0]}   "
          f"MRI {mri.shape[1]}x{mri.shape[0]}   (1 px = 1 mm in both)\n")

    r = register(ct, mri, search, bins)
    if r is None:
        print("\n  one of these slices has no contrast to measure - no shift found")
        return None

    print(f"\n  scored window          = {r['window'][1]}x{r['window'][0]} px, identical for every shift")
    print(f"  shifts evaluated       = {r['evaluated']}  (+/-{r['search']} mm on each axis)")
    print(f"  baseline NMI (no shift)= {r['baseline']:.6f}   MRI covers {r['baseline_coverage'] * 100:.1f}%")
    print(f"  best NMI               = {r['best']:.6f}   MRI covers {r['coverage'] * 100:.1f}%")
    print(f"  shift found            = {r['dx']:+d} mm across, {r['dy']:+d} mm down")
    print(f"  change                 = {r['best'] - r['baseline']:+.6f}")
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description="Resample both to 1 mm, then slide one over the other.")
    ap.add_argument("ct")
    ap.add_argument("mri")
    ap.add_argument("--range", type=int, default=RANGE, help="search +/- this many mm (default 40)")
    ap.add_argument("--bins", type=int, default=BINS, help="histogram bins (default 32)")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    for p in (a.ct, a.mri):
        if not os.path.exists(p):
            print(f"no such file: {p}", file=sys.stderr)
            return 2
    run(a.ct, a.mri, a.range, a.bins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
