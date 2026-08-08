"""
working_regis.py — the registration logic from the interactive explorer, in Python.

This is the reference implementation of what "Instrument 09" in
registration-explorer.html actually does, extracted so it can be run, diffed and
tested against the SimpleITK pipeline. Every stage prints the arithmetic it used,
so a run of this file is a readable derivation rather than a single number.

    python working_regis.py CT_FILE MRI_FILE [--mm 1.0] [--starts 3] [--trace]

    python working_regis.py \
        ../Raw_data_mri_ct/Rawdata_dicom/CT/PA0_Ranjeet/ST0/SE0/IM9 \
        ../Raw_data_mri_ct/Rawdata_dicom/MRI/PA0_Ranjeet/ST0/SE0/IM9 --trace

The pipeline, in order:

    1. resample both slices to isotropic millimetres     -> in-plane match
    2. baseline NMI with the identity transform          -> the number to beat
    3. rigid multi-start   (3 DOF: tx, ty, theta)
    4. affine multi-start  (6 DOF: tx, ty, a, b, c, d)
    5. the scale gate      (sanity bound + canvas-fit veto)
    6. selection           (affine must clear MIN_TRANSFORM_MARGIN)
    7. classification      (improved / marginal / regressed vs MIN_GAIN)

HOW THIS DIFFERS FROM registration_demo_sweep_v3.py
---------------------------------------------------
This is deliberately a *reimplementation of the logic*, not a wrapper around it:

  * It optimises NMI directly with Nelder-Mead. The real pipeline lets
    SimpleITK descend Mattes MI and uses NMI only to choose among candidates,
    because SimpleITK does not expose NMI as a metric.
  * It works on two 2D slices. No N4 bias correction, no through-plane geometry.
  * A single slice carries no shared world origin, so the MRI is placed on the
    CT grid CENTRED rather than by ImagePositionPatient. The real pipeline
    computes that offset in closed form from the headers first, and its absence
    is why the recovered translations here are larger than they should be.

So the numbers here will not reproduce sweep_v3_summary.csv. What they do
reproduce is the reasoning: the same metric, the same gate, the same taxonomy.

Dependencies: numpy, and pydicom only if you pass DICOM paths.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

# ─────────────────────────── constants ───────────────────────────
# Mirrored from registration_demo_sweep_v3.py so the verdicts line up.

SCALE_TOL = 0.60             # outer sanity bound on |scale - 1|
CANVAS_FIT_TOL = 0.04        # how close to the canvas ratio counts as a canvas fit
MIN_GAIN = 0.010             # dead band around the baseline
MIN_TRANSFORM_MARGIN = 0.005 # affine must beat rigid by this to be promoted
MIN_MRI_COVERAGE = 0.25      # below this the slice is starved, not misregistered

NMI_BINS = 32
NMI_CLIP_PERCENTILES = (0.5, 99.5)

PYRAMID_SHRINK = (4, 2, 1)
PYRAMID_SIGMA = (2.0, 1.0, 0.0)
NM_ITERS = 80


# ─────────────────────────── loading ───────────────────────────

@dataclass
class Slice:
    """One 2D slice plus the geometry needed to put it in millimetres."""
    data: np.ndarray          # float64, shape (h, w); NaN marks "no data"
    spacing: tuple            # (row_mm, col_mm) == DICOM PixelSpacing
    modality: str = "?"
    name: str = ""
    window: Optional[tuple] = None   # (centre, width) for display only

    @property
    def h(self) -> int:
        return self.data.shape[0]

    @property
    def w(self) -> int:
        return self.data.shape[1]

    @property
    def extent_mm(self) -> tuple:
        return (self.w * self.spacing[1], self.h * self.spacing[0])


def load_slice(path: str) -> Slice:
    """DICOM if pydicom can read it, otherwise any raster PIL can open."""
    try:
        import pydicom

        d = pydicom.dcmread(path)
        a = d.pixel_array.astype(np.float64)
        # Raw stored values are not Hounsfield Units. Applying the rescale is
        # what makes CT comparable between series and machines.
        slope = float(getattr(d, "RescaleSlope", 1) or 1)
        intercept = float(getattr(d, "RescaleIntercept", 0) or 0)
        a = a * slope + intercept

        spacing = getattr(d, "PixelSpacing", None)
        spacing = (float(spacing[0]), float(spacing[1])) if spacing else (1.0, 1.0)

        def first(v):
            if v is None:
                return None
            try:
                return float(v[0])
            except (TypeError, IndexError):
                return float(v)

        wc, ww = first(getattr(d, "WindowCenter", None)), first(getattr(d, "WindowWidth", None))
        return Slice(
            data=a,
            spacing=spacing,
            modality=str(getattr(d, "Modality", "?")),
            name=os.path.basename(path),
            window=(wc, ww) if wc is not None and ww else None,
        )
    except Exception:
        pass

    from PIL import Image  # optional, only for the raster fallback

    im = Image.open(path).convert("L")
    return Slice(
        data=np.asarray(im, dtype=np.float64),
        spacing=(1.0, 1.0),
        modality="?",
        name=os.path.basename(path),
    )


# ─────────────────────────── geometry ───────────────────────────

def _sample_bilinear(data: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear sample at float coordinates. Outside the source -> NaN."""
    h, w = data.shape
    out = np.full(xs.shape, np.nan, dtype=np.float64)

    inside = (xs >= -0.5) & (ys >= -0.5) & (xs <= w - 0.5) & (ys <= h - 0.5)
    if not inside.any():
        return out

    xi, yi = xs[inside], ys[inside]
    x0 = np.floor(xi).astype(np.int64)
    y0 = np.floor(yi).astype(np.int64)
    fx, fy = xi - x0, yi - y0
    x1, y1 = x0 + 1, y0 + 1

    x0 = np.clip(x0, 0, w - 1); x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1); y1 = np.clip(y1, 0, h - 1)

    out[inside] = (
        data[y0, x0] * (1 - fx) * (1 - fy)
        + data[y0, x1] * fx * (1 - fy)
        + data[y1, x0] * (1 - fx) * fy
        + data[y1, x1] * fx * fy
    )
    return out


def to_isotropic(sl: Slice, mm: float) -> Slice:
    """
    Resample so one pixel is `mm` millimetres on BOTH axes.

    This is the step that makes the two modalities comparable. A CT at
    0.49 mm/px and an MRI at 0.875 mm/px are not the same grid; afterwards a
    50 mm structure spans 50/mm pixels in each. The practical consequence is
    that no genuine zoom is left for affine to recover, which is precisely
    what makes the scale gate in `scale_verdict` meaningful: any large
    recovered scale is now a cheat, not a correction.
    """
    sy, sx = sl.spacing
    out_w = max(8, int(round(sl.w * sx / mm)))
    out_h = max(8, int(round(sl.h * sy / mm)))

    gx, gy = np.meshgrid(np.arange(out_w), np.arange(out_h))
    xs = (gx + 0.5) * mm / sx - 0.5
    ys = (gy + 0.5) * mm / sy - 0.5

    return Slice(
        data=_sample_bilinear(sl.data, xs, ys),
        spacing=(mm, mm),
        modality=sl.modality,
        name=sl.name,
        window=sl.window,
    )


def place_on(src: Slice, out_h: int, out_w: int) -> Slice:
    """
    Drop an already-isotropic slice onto the fixed image's grid, centred.

    A single slice carries no shared world origin, so centring is the only
    defensible placement here; recovering the true offset is the optimiser's
    job. The real pipeline instead computes the offset in closed form from
    ImagePositionPatient, which is both exact and free.
    """
    out = np.full((out_h, out_w), np.nan, dtype=np.float64)
    oy = (out_h - src.h) // 2
    ox = (out_w - src.w) // 2

    sy0, sy1 = max(0, -oy), min(src.h, out_h - oy)
    sx0, sx1 = max(0, -ox), min(src.w, out_w - ox)
    if sy1 > sy0 and sx1 > sx0:
        out[oy + sy0:oy + sy1, ox + sx0:ox + sx1] = src.data[sy0:sy1, sx0:sx1]

    return Slice(out, src.spacing, src.modality, src.name, src.window)


def _box_sum(x: np.ndarray, r: int, axis: int) -> np.ndarray:
    """
    Box sum of radius r along one axis, via cumulative sums.

    Cumsum rather than np.roll because roll WRAPS: the opposite edge of the
    image bleeds into the margin, which puts scalp next to background and
    invents gradients the optimiser then chases.
    """
    x = np.moveaxis(x, axis, 0)
    n = x.shape[0]
    pad = np.zeros((1,) + x.shape[1:], dtype=np.float64)
    cs = np.concatenate([pad, np.cumsum(x, axis=0)], axis=0)
    lo = np.clip(np.arange(n) - r, 0, n)
    hi = np.clip(np.arange(n) + r + 1, 0, n)
    return np.moveaxis(cs[hi] - cs[lo], 0, axis)


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """
    NaN-aware separable box blur, two passes per axis, as a cheap gaussian.

    A box of radius r has variance r(r+1)/3, so two passes along an axis give
    2*r(r+1)/3. Setting that equal to sigma^2 and solving for r gives

        r = (-1 + sqrt(1 + 6*sigma^2)) / 2

    which is exact at sigma = 2 (r = 2), the coarsest level this pipeline uses.
    Using r = sigma under-blurs by ~18%; using r = 1.22*sigma over-blurs by
    ~22%, because that approximates the variance as r^2/3 and drops the r term.

    The original NaN mask is restored at the end: smoothing must not grow the
    valid region, or the metric starts scoring pixels that have no data.
    """
    if sigma <= 0:
        return a
    r = max(1, int(round((-1.0 + math.sqrt(1.0 + 6.0 * sigma * sigma)) / 2.0)))

    mask = ~np.isnan(a)
    num = np.where(mask, a, 0.0).astype(np.float64)
    den = mask.astype(np.float64)
    for _ in range(2):
        for axis in (0, 1):
            num = _box_sum(num, r, axis)
            den = _box_sum(den, r, axis)

    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return np.where(mask, out, np.nan)


def _shrink(a: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return a
    h, w = a.shape
    H, W = max(8, int(round(h / factor))), max(8, int(round(w / factor)))
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    xs = (gx + 0.5) * w / W - 0.5
    ys = (gy + 0.5) * h / H - 0.5
    return _sample_bilinear(a, xs, ys)


def pyramid_level(a: np.ndarray, shrink_factor: int,
                  sigma_mm: float, mm: float) -> np.ndarray:
    """
    Smooth THEN decimate.

    Doing it the other way round cannot work: bilinear decimation of a 250 px
    image down to 63 px already aliases, and no amount of blurring afterwards
    recovers the frequencies that got folded. Sigmas are in millimetres, as in
    the pipeline's SmoothingSigmasAreSpecifiedInPhysicalUnitsOn().
    """
    return _shrink(_blur(a, sigma_mm / mm), shrink_factor)


# ─────────────────────────── transforms ───────────────────────────

def params_to_mat(p: Sequence[float], kind: str):
    """rigid: [tx, ty, theta] · affine: [tx, ty, a, b, c, d] with M = [[a,b],[c,d]]"""
    if kind == "rigid":
        c, s = math.cos(p[2]), math.sin(p[2])
        return np.array([[c, -s], [s, c]]), np.array([p[0], p[1]])
    return np.array([[p[2], p[3]], [p[4], p[5]]]), np.array([p[0], p[1]])


def warp(moving: np.ndarray, M: np.ndarray, t: np.ndarray,
         out_h: int, out_w: int) -> np.ndarray:
    """
    Resampler convention, as in the pipeline: an OUTPUT point p samples the
    MOVING image at T(p). So output coordinates are pushed through the
    transform, not the other way round.
    """
    cy, cx = (out_h - 1) / 2.0, (out_w - 1) / 2.0
    gx, gy = np.meshgrid(np.arange(out_w), np.arange(out_h))
    dx, dy = gx - cx, gy - cy
    xs = M[0, 0] * dx + M[0, 1] * dy + t[0] + cx
    ys = M[1, 0] * dx + M[1, 1] * dy + t[1] + cy
    return _sample_bilinear(moving, xs, ys)


def mat_scale(M: np.ndarray) -> float:
    """
    Mean singular value of the 2x2 block — the scale the gate reasons about.

    Using singular values rather than reading matrix entries is what makes this
    robust to rotation and shear being mixed into the same block.
    """
    return float(np.mean(np.linalg.svd(M, compute_uv=False)))


# ─────────────────────────── the metric ───────────────────────────

def _entropy(p: np.ndarray) -> float:
    nz = p[p > 0]          # 0*log0 == 0, but log(0) is -inf: filter first
    return float(-np.sum(nz * np.log(nz)))


@dataclass
class NMIResult:
    nmi: float
    mi: float = float("nan")
    h_fixed: float = float("nan")
    h_moving: float = float("nan")
    h_joint: float = float("nan")
    coverage: float = 0.0
    valid: int = 0
    used_bins: int = 0
    joint: Optional[np.ndarray] = None


def nmi_score(fixed: np.ndarray, moving: np.ndarray,
              bins: int = NMI_BINS,
              ranges: Optional[tuple] = None,
              want_joint: bool = False) -> NMIResult:
    """
    Studholme's normalized mutual information over pixels valid in BOTH images.

        NMI = ( H(A) + H(B) ) / H(A,B)        in [1, 2]

    `ranges` fixes the histogram bin edges. Passing them makes successive
    evaluations directly comparable and skips two sorts per call, which is what
    makes the optimiser inner loop fast; omitting them reproduces nmi_score()
    from registration_demo.py exactly, recomputing the 0.5/99.5 percentile clip
    on every call. The pipeline reports the recomputed version and optimises
    the fixed-range one; the gap between them is ~1e-3.
    """
    ok = ~(np.isnan(fixed) | np.isnan(moving))
    n_total = fixed.size
    valid = int(ok.sum())
    if valid < 32:
        return NMIResult(nmi=float("nan"), coverage=valid / n_total, valid=valid)

    f, m = fixed[ok], moving[ok]

    if ranges is None:
        f_lo, f_hi = np.percentile(f, NMI_CLIP_PERCENTILES)
        m_lo, m_hi = np.percentile(m, NMI_CLIP_PERCENTILES)
    else:
        (f_lo, f_hi), (m_lo, m_hi) = ranges

    if f_hi <= f_lo or m_hi <= m_lo:
        # a constant slice has no entropy to work with — NaN, not 0.0, because
        # 0.0 would read as "measured, and bad" rather than "not measurable"
        return NMIResult(nmi=float("nan"), coverage=valid / n_total, valid=valid)

    # Clip rather than discard, so a handful of extreme voxels (CT metal, MRI
    # spikes) cannot collapse every real tissue value into one or two bins.
    joint, _, _ = np.histogram2d(
        np.clip(f, f_lo, f_hi), np.clip(m, m_lo, m_hi),
        bins=bins, range=[[f_lo, f_hi], [m_lo, m_hi]],
    )
    p = joint / joint.sum()

    h_joint = _entropy(p)
    if h_joint <= 0:
        return NMIResult(nmi=float("nan"), coverage=valid / n_total, valid=valid)

    # Summing the joint histogram along one axis collapses it back to the
    # single-image histogram — one 2D histogram yields all three entropies.
    h_a = _entropy(p.sum(axis=1))
    h_b = _entropy(p.sum(axis=0))

    return NMIResult(
        nmi=(h_a + h_b) / h_joint,
        mi=h_a + h_b - h_joint,
        h_fixed=h_a, h_moving=h_b, h_joint=h_joint,
        coverage=valid / n_total, valid=valid,
        used_bins=int((p > 0).sum()),
        joint=p if want_joint else None,
    )


def valid_range(a: np.ndarray, lo: float = 0.5, hi: float = 99.5,
                nonzero: bool = False) -> tuple:
    """
    Percentile bounds over the valid pixels.

    `nonzero` is what MRI needs: the black background dominates an MRI and
    would drag the percentiles down, so bounds are computed from the non-zero
    voxels only. CT does not want this — air at -1000 HU is real signal.
    """
    v = a[~np.isnan(a)]
    if nonzero:
        pos = v[v > 0]
        if pos.size >= 8:
            v = pos
    if v.size < 4:
        return (0.0, 1.0)
    r = tuple(np.percentile(v, [lo, hi]))
    return r if r[1] > r[0] else (r[0], r[0] + 1.0)


# ─────────────────────────── optimiser ───────────────────────────

@dataclass
class TraceRow:
    level: int
    shrink: int
    grid: str
    iteration: int
    nmi: float
    params: np.ndarray
    op: str
    evals: int


def nelder_mead(f: Callable[[np.ndarray], float],
                x0: np.ndarray, step: np.ndarray,
                max_iter: int = NM_ITERS, tol: float = 1e-6,
                on_step: Optional[Callable] = None):
    """Plain Nelder-Mead. Minimises, so callers pass -NMI."""
    n = len(x0)
    simplex = [np.array(x0, dtype=np.float64)]
    for i in range(n):
        p = np.array(x0, dtype=np.float64)
        p[i] += step[i]
        simplex.append(p)
    fv = [f(p) for p in simplex]
    evals = n + 1
    op = "init"

    for it in range(max_iter):
        order = np.argsort(fv)
        simplex = [simplex[i] for i in order]
        fv = [fv[i] for i in order]

        if on_step is not None:
            on_step(it, fv[0], simplex[0], op, evals)
        if abs(fv[-1] - fv[0]) < tol:
            break

        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]

        def comb(k):
            return centroid + k * (centroid - worst)

        xr = comb(1.0); fr = f(xr); evals += 1
        if fr < fv[0]:
            xe = comb(2.0); fe = f(xe); evals += 1
            if fe < fr:
                simplex[-1], fv[-1], op = xe, fe, "expand"
            else:
                simplex[-1], fv[-1], op = xr, fr, "reflect"
        elif fr < fv[-2]:
            simplex[-1], fv[-1], op = xr, fr, "reflect"
        else:
            xc = comb(-0.5); fc = f(xc); evals += 1
            if fc < fv[-1]:
                simplex[-1], fv[-1], op = xc, fc, "contract"
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fv[i] = f(simplex[i]); evals += 1
                op = "shrink"

    best = int(np.argmin(fv))
    return simplex[best], fv[best], evals


@dataclass
class RunResult:
    kind: str
    seed: int
    params: np.ndarray
    matrix: np.ndarray
    translation: np.ndarray
    scale: float
    nmi: float
    coverage: float
    evals: int
    warped: np.ndarray = field(repr=False, default=None)
    trace: list = field(repr=False, default_factory=list)


def register_once(fixed: np.ndarray, moving: np.ndarray, kind: str, seed: int,
                  bins: int = NMI_BINS, ranges: Optional[tuple] = None,
                  trace: bool = False,
                  report_ranges: Optional[tuple] = None,
                  mm: float = 1.0) -> RunResult:
    """
    One multi-resolution registration run.

    The pyramid is the standard defence against local optima: the coarse level
    has no fine detail to get trapped by, so it finds the broad alignment, and
    finer levels refine it. It is why registration can recover large
    displacements at all — and why the spine series fails, because lumbar
    vertebrae are a comb whose teeth survive the blur.
    """
    rng = np.random.RandomState(seed)
    if kind == "rigid":
        # translations are in millimetres, because the grid is mm-isotropic
        p = np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-0.05, 0.05)])
        step_base = np.array([4.0, 4.0, 0.06])
    else:
        p = np.array([rng.uniform(-5, 5), rng.uniform(-5, 5),
                      1 + rng.uniform(-0.02, 0.02), rng.uniform(-0.015, 0.015),
                      rng.uniform(-0.015, 0.015), 1 + rng.uniform(-0.02, 0.02)])
        step_base = np.array([4.0, 4.0, 0.05, 0.04, 0.04, 0.05])

    rows: list = []
    total_evals = 0

    for level, (shrink_f, sigma) in enumerate(zip(PYRAMID_SHRINK, PYRAMID_SIGMA)):
        f_lvl = pyramid_level(fixed, shrink_f, sigma, mm)
        m_lvl = pyramid_level(moving, shrink_f, sigma, mm)
        H, W = f_lvl.shape
        ratio = W / fixed.shape[1]

        p_lvl = p.copy()
        p_lvl[0] *= ratio
        p_lvl[1] *= ratio
        step = step_base.copy()
        step[0] = 4.0 * ratio + 1
        step[1] = 4.0 * ratio + 1

        def cost(q, _f=f_lvl, _m=m_lvl, _H=H, _W=W):
            M, t = params_to_mat(q, kind)
            r = nmi_score(_f, warp(_m, M, t, _H, _W), bins, ranges)
            if math.isnan(r.nmi) or r.coverage < 0.05:
                return 10.0
            return -r.nmi

        base_evals = total_evals
        on_step = None
        if trace:
            def on_step(it, fval, x, op, ev, _l=level, _s=shrink_f,
                        _g=f"{W}x{H}", _r=ratio, _b=base_evals):
                xr = np.array(x, dtype=np.float64).copy()
                xr[0] /= _r; xr[1] /= _r
                rows.append(TraceRow(_l, _s, _g, it, -fval, xr, op, _b + ev))

        p_lvl, _, evals = nelder_mead(cost, p_lvl, step, on_step=on_step)
        total_evals += evals
        p = p_lvl.copy()
        p[0] /= ratio
        p[1] /= ratio

    M, t = params_to_mat(p, kind)
    warped = warp(moving, M, t, *fixed.shape)
    final = nmi_score(fixed, warped, bins, ranges=report_ranges)

    return RunResult(
        kind=kind, seed=seed, params=p, matrix=M, translation=t,
        scale=mat_scale(M), nmi=final.nmi, coverage=final.coverage,
        evals=total_evals, warped=warped, trace=rows,
    )


# ─────────────────────────── gate + taxonomy ───────────────────────────

def canvas_ratio(img: np.ndarray) -> float:
    """
    Fraction of the frame the moving image's content bounding box spans.

    This is the number a canvas-fitting affine lands on, which is what makes
    that particular cheat identifiable.
    """
    v = img[~np.isnan(img)]
    if v.size == 0:
        return 1.0
    lo, hi = np.percentile(v, [2, 98])
    thr = lo + (hi - lo) * 0.08
    mask = (~np.isnan(img)) & (img > thr)
    if not mask.any():
        return 1.0
    ys, xs = np.where(mask)
    h, w = img.shape
    return float(((xs.max() - xs.min() + 1) / w + (ys.max() - ys.min() + 1) / h) / 2)


def scale_verdict(scale: float, ratio: float) -> tuple:
    """
    Two tests, not one band.

    A single tight band was wrong in both directions: it rejected any real
    scale difference, and would have waved through a canvas fit that happened
    to land near 1.0. TEST 1 is the outer sanity bound; TEST 2 is the veto that
    names the specific failure — affine stretching the moving image until its
    border matches the frame border, which scores well and aligns nothing.
    """
    if math.isnan(scale):
        return False, "unevaluable"
    if abs(scale - 1.0) > SCALE_TOL:
        return False, "outside_sanity_bound"
    if 0.05 < ratio < 0.95:
        for suspect in (ratio, 1.0 / ratio):
            if abs(scale - suspect) <= CANVAS_FIT_TOL:
                return False, "canvas_fit"
    return True, ""


def classify(best: float, baseline: float, coverage: float) -> tuple:
    """
    Causes are tested before symptoms.

    A slice starved of MRI data will probably also regress, but "there was
    barely any MRI here" explains the slice while "the score went down" merely
    describes it. And `marginal` is NOT a failure: "did not improve" and "made
    it worse" are different events, and a single threshold cannot express both.
    """
    if best is None or math.isnan(best):
        return "unevaluable_nmi", True
    if coverage < MIN_MRI_COVERAGE:
        return f"sparse_overlap_{coverage:.2f}", True
    if baseline is None or math.isnan(baseline):
        return "improved", False
    if best < baseline - MIN_GAIN:
        return "regressed", True
    if best <= baseline + MIN_GAIN:
        return "marginal", False
    return "improved", False


# ─────────────────────────── the pipeline ───────────────────────────

# Named CT windows, in absolute Hounsfield Units. HU are calibrated, so these
# mean the same thing on every scan ever taken — which is what makes a fixed
# window legitimate for CT and impossible for MRI.
CT_PRESETS = {
    "soft":  (-160.0, 240.0),
    "brain": (-15.0, 85.0),
    "bone":  (-450.0, 1050.0),
    "lung":  (-1400.0, 200.0),
}


def run_pipeline(ct_path: str, mri_path: str, mm: float = 1.0,
                 n_starts: int = 3, show_trace: bool = False,
                 ct_clip: Optional[tuple] = None,
                 mri_pct: tuple = NMI_CLIP_PERCENTILES,
                 clip_metric: bool = True) -> dict:
    rule = "-" * 74

    def head(n, title):
        print(f"\n{rule}\n{n:>2}. {title}\n{rule}")

    # ---- 1. resample -------------------------------------------------
    head(1, f"RESAMPLE BOTH TO {mm} mm ISOTROPIC")
    ct_raw, mri_raw = load_slice(ct_path), load_slice(mri_path)
    for tag, s in (("CT ", ct_raw), ("MRI", mri_raw)):
        print(f"  {tag} {s.name:<28} {s.w:>4}x{s.h:<4} px  "
              f"{s.spacing[1]:.4f} x {s.spacing[0]:.4f} mm/px  "
              f"FOV {s.extent_mm[0]:.0f} x {s.extent_mm[1]:.0f} mm")

    fixed_s = to_isotropic(ct_raw, mm)
    mov_iso = to_isotropic(mri_raw, mm)
    moving_s = place_on(mov_iso, fixed_s.h, fixed_s.w)
    fixed, moving = fixed_s.data, moving_s.data

    print(f"\n  CT  -> {fixed_s.w}x{fixed_s.h} px at {mm} mm   (the output grid)")
    print(f"  MRI -> {mov_iso.w}x{mov_iso.h} px at {mm} mm, centred on the CT grid")
    print(f"  a 50 mm structure now spans {50/mm:.0f} px in BOTH images")

    ratio = canvas_ratio(moving)
    print(f"  canvas ratio = {ratio:.4f}")

    # ---- 1b. intensity clipping --------------------------------------
    # If a CT window is given it becomes the joint histogram's range. Values
    # outside are folded into the end bins rather than discarded — clip, don't
    # discard — so extreme voxels saturate instead of collapsing every real
    # tissue value into one bin. Narrowing onto soft tissue spends all the bins
    # there, at the cost of making bone and air indistinguishable.
    if ct_clip is None:
        if ct_raw.window:
            wc, ww = ct_raw.window
            ct_window = (wc - ww / 2.0, wc + ww / 2.0)
            ct_src = "DICOM header"
        else:
            ct_window = CT_PRESETS["soft"]
            ct_src = "default soft-tissue"
    else:
        ct_window = tuple(float(v) for v in ct_clip)
        ct_src = "manual"

    mri_window = valid_range(moving, mri_pct[0], mri_pct[1], nonzero=True)

    print(f"\n  CT  clip = [{ct_window[0]:.0f}, {ct_window[1]:.0f}] HU   ({ct_src})")
    print(f"  MRI clip = [{mri_window[0]:.1f}, {mri_window[1]:.1f}] a.u.  "
          f"(percentiles {mri_pct[0]} / {mri_pct[1]})")
    print(f"  applied to = {'display AND metric' if clip_metric else 'display only'}")

    if clip_metric:
        ranges = (ct_window, mri_window)
        report_ranges = ranges
    else:
        ranges = (valid_range(fixed), valid_range(moving))
        report_ranges = None

    # ---- 2. baseline -------------------------------------------------
    head(2, "BASELINE — THE NUMBER TO BEAT")
    base = nmi_score(fixed, moving, ranges=report_ranges, want_joint=False)
    print(f"  valid overlap  = {base.valid} of {fixed.size} px  ({base.coverage*100:.1f}%)")
    print(f"  occupied bins  = {base.used_bins} of {NMI_BINS*NMI_BINS}\n")
    print(f"  H(CT)          = {base.h_fixed:.6f}")
    print(f"  H(MRI)         = {base.h_moving:.6f}")
    print(f"  H(joint)       = {base.h_joint:.6f}\n")
    print(f"  MI             = {base.h_fixed:.4f} + {base.h_moving:.4f} - "
          f"{base.h_joint:.4f} = {base.mi:.6f}")
    print(f"  NMI            = ({base.h_fixed:.4f} + {base.h_moving:.4f}) / "
          f"{base.h_joint:.4f} = {base.nmi:.6f}")

    # ---- 3/4. multi-start --------------------------------------------
    results = {}
    for step_no, kind in ((3, "rigid"), (4, "affine")):
        dof = 3 if kind == "rigid" else 6
        head(step_no, f"{kind.upper()} MULTI-START  ({dof} degrees of freedom)")
        rows = []
        for seed in range(n_starts):
            r = register_once(fixed, moving, kind, seed, ranges=ranges,
                              trace=(show_trace and seed == 0),
                              report_ranges=report_ranges, mm=mm)
            rows.append(r)
            ok, why = scale_verdict(r.scale, ratio)
            extra = (f"theta {math.degrees(r.params[2]):+7.3f}deg"
                     if kind == "rigid" else f"scale {r.scale:.4f}")
            print(f"  seed {seed}:  tx {r.params[0]:+8.3f} mm  ty {r.params[1]:+8.3f} mm  "
                  f"{extra}   NMI {r.nmi:.4f}  {r.evals:>4} evals  "
                  f"{'admissible' if ok else why}")
        valid = [r.nmi for r in rows if not math.isnan(r.nmi)]
        if valid:
            spread = max(valid) - min(valid)
            # The spread across seeds is a direct, free estimate of how much
            # the optimiser is guessing — as valuable as the best score.
            print(f"  -> best {max(valid):.4f}   spread {spread:.4f}"
                  f"{'   (noise floor exceeds MIN_GAIN)' if spread > MIN_GAIN else ''}")
        results[kind] = rows

    # ---- 5. trace ----------------------------------------------------
    if show_trace:
        head(5, "ONE RUN, TRACED — rigid seed 0")
        tr = results["rigid"][0].trace
        print(f"  {'#':>4} {'lvl':>4} {'iter':>5} {'tx mm':>9} {'ty mm':>9} "
              f"{'theta':>8} {'NMI':>9} {'dNMI':>10}  simplex")
        stride = max(1, len(tr) // 24)
        prev = None
        for i, row in enumerate(tr):
            if i % stride and i != len(tr) - 1:
                continue
            d = (row.nmi - prev.nmi) if prev else 0.0
            print(f"  {i:>4} {'/'+str(row.shrink):>4} {row.iteration:>5} "
                  f"{row.params[0]:>9.3f} {row.params[1]:>9.3f} "
                  f"{math.degrees(row.params[2]):>8.3f} {row.nmi:>9.5f} "
                  f"{d:>+10.5f}  {row.op}")
            prev = row
        print(f"\n  {len(tr)} logged iterations, "
              f"{results['rigid'][0].evals} metric evaluations")
        print(f"  NMI {tr[0].nmi:.6f} -> {tr[-1].nmi:.6f}  "
              f"(+{tr[-1].nmi - tr[0].nmi:.6f})")

    # ---- 6. gate -----------------------------------------------------
    head(6, "THE SCALE GATE")
    print(f"  canvas ratio r  = {ratio:.4f}")
    print(f"  suspect scales  = {ratio:.4f}  and  {1/ratio:.4f}")
    print(f"  TEST 1  reject if |scale - 1| > {SCALE_TOL:.2f}")
    print(f"  TEST 2  reject if |scale - suspect| <= {CANVAS_FIT_TOL:.2f}\n")
    for r in results["affine"]:
        ok, why = scale_verdict(r.scale, ratio)
        print(f"  affine seed {r.seed}:  scale {r.scale:.4f}   "
              f"|s-1| {abs(r.scale-1):.4f}   |s-r| {abs(r.scale-ratio):.4f}   "
              f"|s-1/r| {abs(r.scale-1/ratio):.4f}   -> "
              f"{'admissible' if ok else why.upper()}")
    print("  rigid: exempt — it cannot scale")

    # ---- 7. selection ------------------------------------------------
    head(7, "SELECTION")

    def best_of(rows, gated):
        ok = [r for r in rows if not math.isnan(r.nmi)
              and (not gated or scale_verdict(r.scale, ratio)[0])]
        return max(ok, key=lambda r: r.nmi) if ok else None

    best_rigid = best_of(results["rigid"], gated=False)
    best_affine = best_of(results["affine"], gated=True)

    print(f"  best rigid             = "
          f"{best_rigid.nmi:.6f} (seed {best_rigid.seed})" if best_rigid else "  best rigid = none")
    print(f"  best admissible affine = "
          + (f"{best_affine.nmi:.6f} (seed {best_affine.seed}, scale {best_affine.scale:.4f})"
             if best_affine else "none"))

    if best_rigid and best_affine:
        gap = best_affine.nmi - best_rigid.nmi
        cleared = gap > MIN_TRANSFORM_MARGIN
        print(f"\n  affine - rigid         = {gap:+.6f}")
        print(f"  margin required        = {MIN_TRANSFORM_MARGIN:.3f}  "
              f"-> {'CLEARED' if cleared else 'NOT cleared'}")
        chosen = best_affine if cleared else best_rigid
        why = ("affine clears rigid by more than the margin" if cleared
               else "affine did not clear the margin — keep the simpler model")
    else:
        chosen = best_affine or best_rigid
        why = "only one candidate survived"
    print(f"\n  selected               = {chosen.kind if chosen else 'none'}   ({why})")

    # ---- 8. classification -------------------------------------------
    head(8, "CLASSIFICATION")
    best_nmi = chosen.nmi if chosen else float("nan")
    cov = chosen.coverage if chosen else base.coverage
    outcome, failed = classify(best_nmi, base.nmi, cov)
    delta = best_nmi - base.nmi
    print(f"  baseline        = {base.nmi:.6f}")
    print(f"  best registered = {best_nmi:.6f}")
    print(f"  difference      = {delta:+.6f}")
    print(f"  MIN_GAIN        = {MIN_GAIN:.3f}")
    print(f"  MRI coverage    = {cov*100:.1f}%  (fails below {MIN_MRI_COVERAGE*100:.0f}%)")
    print(f"\n  outcome         = {outcome.upper()}   "
          f"{'FAILURE — would trigger the crop fallback' if failed else 'no fallback needed'}")

    # ---- 9. what ships -----------------------------------------------
    head(9, "WHAT SHIPS")
    if chosen and chosen.nmi > base.nmi:
        ship_kind, ship_nmi, ship_img = chosen.kind, chosen.nmi, chosen.warped
    else:
        # Never ship something worse than doing nothing.
        ship_kind, ship_nmi, ship_img = "unregistered", base.nmi, moving
    print(f"  transform = {ship_kind}")
    print(f"  NMI       = {base.nmi:.4f} -> {ship_nmi:.4f}  ({ship_nmi - base.nmi:+.4f})")
    print(f"  outcome   = {outcome}")
    if ship_kind == "unregistered":
        print("  nothing beat doing nothing — a legitimate outcome, not a bug")
    print()

    return {
        "fixed": fixed, "moving": moving, "shipped": ship_img,
        "baseline_nmi": base.nmi, "shipped_nmi": ship_nmi, "shipped_kind": ship_kind,
        "outcome": outcome, "failed": failed, "canvas_ratio": ratio,
        "rigid": results["rigid"], "affine": results["affine"], "chosen": chosen,
    }


# ─────────────────────────── cli ───────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="2D CT/MRI registration with the full derivation printed.")
    ap.add_argument("ct", help="fixed image — DICOM file or raster")
    ap.add_argument("mri", help="moving image — DICOM file or raster")
    ap.add_argument("--mm", type=float, default=1.0,
                    help="isotropic in-plane spacing to resample both to (default 1.0)")
    ap.add_argument("--starts", type=int, default=3, help="multi-start count (default 3)")
    ap.add_argument("--trace", action="store_true",
                    help="print the per-iteration optimiser trace for rigid seed 0")
    ap.add_argument("--save", metavar="PNG",
                    help="write a CT / MRI / before / after figure (needs matplotlib)")
    ap.add_argument("--ct-clip", nargs=2, type=float, metavar=("LO", "HI"),
                    help="CT intensity clip in absolute HU, e.g. --ct-clip -160 240")
    ap.add_argument("--ct-window", choices=sorted(CT_PRESETS),
                    help="named CT window instead of --ct-clip: " +
                         ", ".join(f"{k} {v}" for k, v in sorted(CT_PRESETS.items())))
    ap.add_argument("--mri-pct", nargs=2, type=float, metavar=("LO", "HI"),
                    default=list(NMI_CLIP_PERCENTILES),
                    help="MRI clip percentiles of non-zero voxels (default 0.5 99.5)")
    ap.add_argument("--display-clip-only", action="store_true",
                    help="clip the display but let the metric re-derive its own percentiles")
    a = ap.parse_args(argv)

    ct_clip = a.ct_clip
    if a.ct_window:
        if ct_clip:
            ap.error("use either --ct-clip or --ct-window, not both")
        ct_clip = CT_PRESETS[a.ct_window]

    # Windows consoles default to cp1252; keep output printable regardless.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    for p in (a.ct, a.mri):
        if not os.path.exists(p):
            print(f"no such file: {p}", file=sys.stderr)
            return 2

    out = run_pipeline(a.ct, a.mri, mm=a.mm, n_starts=a.starts, show_trace=a.trace,
                       ct_clip=ct_clip, mri_pct=tuple(a.mri_pct),
                       clip_metric=not a.display_clip_only)

    if a.save:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
        panels = [
            ("CT (fixed)", out["fixed"], None),
            (f"MRI ({out['shipped_kind']})", out["shipped"], None),
            ("overlay before", None, (out["fixed"], out["moving"])),
            ("overlay after", None, (out["fixed"], out["shipped"])),
        ]
        for axis, (title, img, pair) in zip(ax, panels):
            if img is not None:
                axis.imshow(np.nan_to_num(img, nan=np.nanmin(img)), cmap="gray")
            else:
                f, m = pair
                def norm(x):
                    v = x[~np.isnan(x)]
                    lo, hi = np.percentile(v, [0.5, 99.5]) if v.size else (0, 1)
                    return np.clip((np.nan_to_num(x, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1)
                rgb = np.zeros(f.shape + (3,))
                rgb[..., 0] = norm(f)          # CT  -> red/amber channel
                rgb[..., 1] = norm(m) * 0.75   # MRI -> cyan channel
                rgb[..., 2] = norm(m)
                axis.imshow(rgb)
            axis.set_title(title, fontsize=10)
            axis.axis("off")
        fig.suptitle(
            f"{out['shipped_kind']} · NMI {out['baseline_nmi']:.4f} -> "
            f"{out['shipped_nmi']:.4f} · {out['outcome']}", fontsize=11)
        fig.tight_layout()
        fig.savefig(a.save, dpi=130)
        print(f"figure written to {a.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
