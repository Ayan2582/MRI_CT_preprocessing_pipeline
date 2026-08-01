"""
registration_demo_sweep_v3.py
────────────────────────────────
Intersection cropping as a FALLBACK, scored with NMI.

Scoring metric is normalized MI (demo.nmi_score), not raw Mattes MI. Raw MI
is unbounded and grows with the marginal entropies, so its value tracks how
much image is in the frame nearly as much as how well the two images line
up. That is fatal for the comparison made here, which is between two
different canvas sizes. NMI divides that dependence out. The optimizer still
minimizes Mattes MI internally (ITK exposes no NMI metric); MI proposes
candidate alignments, NMI decides which is kept.

Cropping to the FOV intersection is not applied to every slice. Applied
uniformly it is a wash on average and strongly bimodal per series: it
rescues pairs where the MRI covers a sliver of the CT canvas and harms
well-matched pairs by trimming away context the optimizer was using.
Averaging those together answers nothing. So every slice is registered on
the full canvas first, and the crop is attempted only for slices where that
failed, on an explicit and recorded criterion.

────────────────────────────────────────────────────────────────────────────
Bugs found in the first fallback version and fixed here. Each one changed
results, not just tidiness:

  1. FRAME MISMATCH (the big one). The intersection ROI was computed from
     the two volumes' RAW world geometry, but the box was then filled with
     MRI that had the GEOMETRY translation applied - the very correction
     whose job is to make the volumes overlap. The crop therefore kept the
     region where the volumes overlapped BEFORE alignment. On knee/sagittal
     (translation -86.9mm in X) MRI content lands on all 18 CT slices while
     the ROI kept only slices 14-17: the fallback discarded 14 of 18 slices
     that hold real MRI. The ROI is now computed from the MRI's extent
     expressed in the CT frame (corners pushed through the inverse of the
     alignment transform), so box and content share one frame.

  2. "no_gain" WAS NOT A FAILURE TEST. It fired when the result failed to
     beat the baseline by MIN_GAIN - but 12 of the 14 slices it caught had
     IMPROVED, just by less than the margin. Cropping is not a remedy for
     "registration was unnecessary". Failure is now `regressed` (result is
     worse than doing nothing); `marginal` is a recorded outcome that does
     NOT trigger a fallback.

  3. NO UNREGISTERED FALLBACK. The rule is "reject -> fall back to rigid,
     then to unregistered", but only the rigid step existed, so a result
     measurably worse than doing nothing could still ship (knee/sagittal
     last: shipped 1.081 against a 1.127 baseline). Every canvas now ships
     max(best registered result, unregistered baseline).

  4. SCALE GATE APPLIED AFTER SELECTION. Multi-start picked the
     highest-scoring seed with no scale constraint, and the gate then
     rejected that single winner and discarded the whole affine - even when
     a lower-scoring, in-tolerance seed existed. The gate is now an
     `accept_fn` inside the multi-start loop, so it selects the best
     ADMISSIBLE seed. Per-seed scales are recorded so this is auditable.

  5. RIGID-vs-AFFINE DECIDED BELOW THE NOISE FLOOR. Affine was preferred on
     any margin at all; on 18 of 33 slices the affine-rigid gap was smaller
     than the multi-start seed spread, i.e. noise. Affine must now win by
     more than the observed seed spread.

  6. CROP NEVER CHECKED AGAINST THE CROPPED BASELINE. The crop only had to
     beat the full-canvas result, so a crop worse than doing nothing could
     win if the full result was worse still. The choice is now three-way
     between full, crop and unregistered, all scored on the common region.

  7. Bookkeeping: `sparse_overlap` is tested before `regressed` (cause
     before symptom); "why we didn't attempt a crop" and "why we discarded
     one" are separate columns; `final_nmi` carries the region it was scored
     in; PNG filenames include the patient so two patients in one
     region/orientation cannot overwrite each other.

Geometry correctness (carried over from earlier rounds):

  * RegionOfInterest takes a voxel COUNT, so an inclusive [start, stop] span
    is stop - start + 1. Without the +1 every axis loses a voxel and it
    looks exactly like a small genuine crop.
  * The transform is fitted against the FULL CT, never the cropped one.
    Cropping only chooses which part of physical space to keep; it must not
    move the MRI. Fitting against the crop re-centres the MRI on the crop's
    centre (~40mm drift in testing). v2 and v3 therefore apply an identical
    transform and the script asserts that at runtime.
  * A disjoint overlap box is rejected, not clamped into a bogus 1-voxel slab.
  * Multi-start spread is None, not 0.0, when fewer than two seeds survive.

Reuses registration_demo.py's registration/scoring functions unchanged.
"""
import os
import csv
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import io_utils
import image_processing as img_proc
import normalization as norm
import pipeline_config as cfg
import registration_demo as demo
import registration_demo_sweep as sweep

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "registration_demo_output", "sweep_v3")
N_STARTS = 3

# --- thresholds --------------------------------------------------------------
# Starting points from 4 patients, not settled constants.
MIN_GAIN = 0.010          # NMI a result must beat the comparison by to count as better
MIN_MRI_COVERAGE = 0.25   # MRI must fill this fraction of the canvas, else "sparse overlap"

# --- affine scale admissibility ----------------------------------------------
# The old gate was a single tight band, |scale - 1| <= 0.05. That was too crude
# in both directions: it rejected any real scale difference, AND it would have
# admitted a canvas-fit that happened to land near 1.0. It is now two tests.
#
# TEST 1 - outer sanity bound. Generous, because the tight band was the
# complaint. |scale - 1| <= 0.60 permits anything from a 40% shrink to a 60%
# stretch, which is far more than inter-scan anatomy can justify and leaves
# plenty of room for genuine geometric discrepancy.
SCALE_TOL = 0.60
#
# TEST 2 - the canvas-fit veto, which is the test that actually matters here.
#
# IMPORTANT, because it is the thing most likely to be misread: by the time 2D
# registration runs, the CT and MRI are ALREADY ON THE SAME PHYSICAL GRID.
# resample_mri_to_ct_grid_v2 put the MRI on the CT's grid in world millimetres,
# so both slices are 1mm/pixel and a 50mm structure spans 50 pixels in each.
# A field-of-view difference does NOT survive as a magnification difference -
# it survives as the MRI filling less of the canvas. There is no "zoom gap"
# left for affine to bridge, and a large recovered scale is therefore not a
# correction; it is the optimizer stretching the MRI until its border matches
# the frame border, because frame borders are big high-contrast features that
# raise mutual information.
#
# Measured on this dataset:
#   knee/sagittal z=17   MRI box / canvas = 0.779   recovered scales 0.810-0.840
#   spine/coronal z=5    MRI box / canvas = 0.827   recovered scales 1.177-1.206
#                                        (1/0.827 = 1.209)
# The scales sit on the canvas ratio or its reciprocal to within a few percent,
# and NMI RISES when they are allowed - which is exactly why NMI cannot be the
# arbiter here. So: reject scales that coincide with the canvas ratio, and
# allow everything else out to SCALE_TOL.
CANVAS_FIT_TOL = 0.04     # |scale - canvas_ratio| <= this  ->  reject as canvas-fitting
                          # set to 0.0 to disable the veto entirely and trust SCALE_TOL alone

# Floor on the margin affine must clear to be preferred over rigid, so a
# reproducible-but-trivial gap cannot promote the more complex model.
MIN_TRANSFORM_MARGIN = 0.005


# ─────────────────────────── geometry ───────────────────────────

def corners_world(img):
    """The image's 8 index-space corners, in world coordinates."""
    size = img.GetSize()
    idxs = [(i, j, k) for i in (0, size[0] - 1) for j in (0, size[1] - 1) for k in (0, size[2] - 1)]
    return np.array([img.TransformContinuousIndexToPhysicalPoint(c) for c in idxs])


def corners_in_ct_frame(mri_image, transform):
    """
    The MRI's corners expressed in the CT/output frame.

    This is the fix for bug 1. A resampler's transform maps OUTPUT points to
    MOVING points: an output point p samples the MRI at T(p), so p sees real
    MRI exactly when T(p) is inside the MRI's extent - i.e. when p is inside
    T^-1(MRI extent). Intersecting the CT box with the MRI's RAW box instead
    answers a different question: where the two volumes overlapped before
    the alignment transform existed. With translations of -87mm in this
    dataset those are wildly different regions.

    Rotation is identity here by construction, so this reduces to
    (corner - translation), but going through the inverse transform keeps it
    correct if a rotation is ever introduced.
    """
    inv = transform.GetInverse()
    return np.array([inv.TransformPoint(tuple(float(x) for x in c)) for c in corners_world(mri_image)])


def overlap_fracs(ct_image, mri_corners):
    """
    Per-axis world-space overlap between the CT and an MRI given by its
    corners, as a fraction of the smaller field of view. Pass raw corners for
    the scanner-frame number, aligned corners for the one the crop acts on.
    """
    ct_c = corners_world(ct_image)
    ct_lo, ct_hi = ct_c.min(0), ct_c.max(0)
    mri_lo, mri_hi = mri_corners.min(0), mri_corners.max(0)
    fracs = []
    for i in range(3):
        overlap = max(0.0, min(ct_hi[i], mri_hi[i]) - max(ct_lo[i], mri_lo[i]))
        smaller = min(ct_hi[i] - ct_lo[i], mri_hi[i] - mri_lo[i])
        fracs.append(overlap / smaller if smaller > 0 else 0.0)
    return fracs


def compute_intersection_roi(ct_image, mri_corners):
    """
    Returns (start_index, roi_size) in ct_image's own index space: the region
    of ct_image whose world extent overlaps the MRI. `mri_corners` must
    already be in the CT frame (see corners_in_ct_frame) - taking corners
    rather than an image is deliberate, so the caller cannot accidentally
    pass geometry from the wrong frame again.

    Clamped to ct_image's own bounds. None if there is no overlap at all.
    """
    ct_c = corners_world(ct_image)
    world_min = np.maximum(ct_c.min(0), mri_corners.min(0))
    world_max = np.minimum(ct_c.max(0), mri_corners.max(0))
    if np.any(world_max <= world_min):
        return None

    box_corners_world = [
        (world_min[0] if i == 0 else world_max[0],
         world_min[1] if j == 0 else world_max[1],
         world_min[2] if k == 0 else world_max[2])
        for i in (0, 1) for j in (0, 1) for k in (0, 1)
    ]
    idx_corners = np.array([ct_image.TransformPhysicalPointToContinuousIndex(p) for p in box_corners_world])
    idx_min = np.floor(idx_corners.min(axis=0)).astype(int)
    idx_max = np.ceil(idx_corners.max(axis=0)).astype(int)

    ct_size = np.array(ct_image.GetSize())
    # Reject BEFORE clamping. If the overlap box falls entirely outside the CT
    # grid on any axis, clamping would collapse it onto the nearest edge and
    # hand back a thin slab as though it were a real intersection.
    if np.any(idx_max < 0) or np.any(idx_min > ct_size - 1):
        return None

    start = np.clip(idx_min, 0, ct_size - 1)
    stop = np.clip(idx_max, 0, ct_size - 1)
    # RegionOfInterest takes a voxel COUNT, and [start, stop] is an inclusive
    # index range - so the count is stop - start + 1. Without the +1 every
    # axis silently loses one voxel, which looks exactly like a real (but
    # tiny) FOV crop even when the two volumes overlap perfectly.
    roi_size = stop - start + 1
    return tuple(int(x) for x in start), tuple(int(x) for x in roi_size)


def resample_mri_to_ct_grid_v3(mri_image, ct_image, default_pixel_value=0.0):
    """
    v2 (GEOMETRY translation, no rotation search) but cropped to the
    world-space intersection of the two FOVs *as they sit after alignment*,
    so the output canvas isn't mostly territory the MRI never covers.
    """
    mri_f = sitk.Cast(mri_image, sitk.sitkFloat32)
    ct_f = sitk.Cast(ct_image, sitk.sitkFloat32)
    initial_transform = sitk.CenteredTransformInitializer(
        ct_f, mri_f, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    translation = initial_transform.GetTranslation()

    aligned_corners = corners_in_ct_frame(mri_image, initial_transform)
    roi = compute_intersection_roi(ct_image, aligned_corners)
    if roi is None:
        return None, None, translation, None, aligned_corners

    start, size = roi
    ct_cropped = sitk.RegionOfInterest(ct_image, size, start)

    # Fit the transform against the FULL CT, never the cropped one. Cropping
    # only chooses which part of physical space we keep; it must not move the
    # MRI relative to the CT. Because GEOMETRY aligns bounding-box centers,
    # fitting against ct_cropped would re-centre the MRI on the crop's centre
    # instead of the CT's - a different (and wrong) spatial relationship.
    # Only SetReferenceImage below differs from v2.
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_cropped)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetTransform(initial_transform)
    resampled = resampler.Execute(mri_image)

    return resampled, ct_cropped, translation, roi, aligned_corners


# ─────────────────────────── registration ───────────────────────────

def affine_scale(transform):
    """
    Mean singular value of the affine's 2x2 matrix - a rotation/shear-robust
    scale estimate. Used for the gate that catches affine resizing the image
    to fit the canvas instead of aligning anatomy (spine recovered 0.825
    against an FOV ratio of 0.824 - a match to 0.3%).
    """
    if transform is None:
        return None
    try:
        matrix = np.array(transform.GetMatrix()).reshape(2, 2)
    except Exception:
        return None
    return float(np.mean(np.linalg.svd(matrix, compute_uv=False)))


def canvas_fill_ratio(mri_slice):
    """
    How much of the canvas the MRI's content actually spans, as a fraction -
    the mean of its filled bounding box width and height over the canvas
    width and height.

    This is the number an affine will converge on if it decides to fit the
    frame instead of the anatomy, because scaling the moving image by this
    factor (or its reciprocal, depending on which way the fit runs) is what
    makes the MRI's border coincide with the canvas border.

    Distinct from mri_coverage(), which is the fraction of PIXELS filled.
    A ring of anatomy has low pixel coverage but a large bounding box; it is
    the bounding box the optimizer can align a border to.
    """
    filled = sitk.GetArrayFromImage(mri_slice) > 0
    if not filled.any():
        return None
    rows = np.where(filled.any(axis=1))[0]
    cols = np.where(filled.any(axis=0))[0]
    span_y = (rows[-1] - rows[0] + 1) / filled.shape[0]
    span_x = (cols[-1] - cols[0] + 1) / filled.shape[1]
    return float((span_x + span_y) / 2.0)


def veto_applies(canvas_ratio):
    """
    The canvas-fit veto is only meaningful when the MRI's border and the
    canvas border are in DIFFERENT places. If the MRI fills the canvas, the
    ratio is ~1.0, there is no border gap to close, and a scale near 1.0 is
    the correct answer rather than a suspicious one.

    Guarding this is not hypothetical: knee/axial has 100% MRI coverage, so
    its ratio is 1.0, and an unguarded veto would reject every scale within
    CANVAS_FIT_TOL of unity - precisely inverting the gate's purpose.

    Require the suspect band to clear unity by at least CANVAS_FIT_TOL, so
    the vetoed interval can never touch 1.0.
    """
    if not canvas_ratio or CANVAS_FIT_TOL <= 0:
        return False
    return abs(canvas_ratio - 1.0) > 2 * CANVAS_FIT_TOL


def scale_verdict(scale, canvas_ratio):
    """
    Why a scale is or is not admissible. Returns (ok, reason) so the decision
    can be printed and stored rather than silently applied.
    """
    if scale is None:
        return False, "no_scale"
    if abs(scale - 1.0) > SCALE_TOL:
        return False, f"outside_sanity_bound_{SCALE_TOL}"
    if veto_applies(canvas_ratio):
        for suspect, name in ((canvas_ratio, "canvas_ratio"),
                              (1.0 / canvas_ratio, "inverse_canvas_ratio")):
            if abs(scale - suspect) <= CANVAS_FIT_TOL:
                return False, f"canvas_fit_matches_{name}_{suspect:.3f}"
    return True, ""


def make_scale_gate(canvas_ratio):
    """
    Build the admissibility predicate for one slice. It is a factory rather
    than a plain function because the veto depends on that slice's canvas
    ratio, and `accept_fn` receives only the transform.

    Passing canvas_ratio=None disables the veto and leaves only SCALE_TOL,
    which is the behaviour of the old gate with a wider band.
    """
    def gate(transform):
        ok, _ = scale_verdict(affine_scale(transform), canvas_ratio)
        return ok
    return gate


def scale_gate_ok(transform):
    """Canvas-agnostic gate: the outer sanity bound only. Kept for callers
    that have no slice context."""
    return make_scale_gate(None)(transform)


def is_better(a, b, margin=0.0):
    """NaN-safe 'a beats b by margin'. A NaN never wins."""
    if a is None or np.isnan(a):
        return False
    if b is None or np.isnan(b):
        return True
    return a > b + margin


def mri_coverage(mri_slice):
    """
    Fraction of the canvas the resampled MRI actually fills. The resampler's
    default fill is exactly 0.0 and real MRI noise is positive (only 0.8% of
    voxels inside a real MRI FOV are exactly zero, measured on knee/sagittal),
    so nonzero is a good proxy for 'inside the MRI's true FOV'.

    With bug 1 fixed this now captures through-plane mismatch too: a slice
    lying outside the aligned MRI's extent resamples to all-fill and reads 0.
    Before the fix the MRI had been translated to cover every slice, so this
    number was a flat 60.7% across all 18 knee/sagittal slices and could not
    have detected the very case it exists for.
    """
    return float(np.mean(sitk.GetArrayFromImage(mri_slice) > 0))


def register_canvas(ct_slice, mri_slice, label):
    """
    Run rigid and affine multi-start on one canvas and pick a winner.

    Two separate outputs, deliberately:
      best_*    the best REGISTERED result. This is what the failure
                diagnosis is about - did registration achieve anything.
      shipped_* what actually gets used, which is the unregistered baseline
                whenever registration did not beat it. Conflating the two
                would make `regressed` unfireable, since a shipped result is
                never worse than baseline by construction.
    """
    print(f"      -- {label} canvas --")
    nmi_baseline = demo.nmi_score(ct_slice, mri_slice)

    rigid = demo.run_2d_registration_multistart_detailed(
        ct_slice, mri_slice, "rigid", n_starts=N_STARTS)
    # The scale gate goes INSIDE the multi-start (accept_fn), so selection
    # returns the best admissible seed rather than the best seed overall
    # followed by a pass/fail on it.
    ratio = canvas_fill_ratio(mri_slice)
    affine = demo.run_2d_registration_multistart_detailed(
        ct_slice, mri_slice, "affine", n_starts=N_STARTS, accept_fn=make_scale_gate(ratio))

    scales = [affine_scale(t) for t in affine["transforms"]]
    if affine["n_rejected"]:
        ratio_str = f"{ratio:.3f}" if ratio is not None else "n/a"
        print(f"      affine scale gate: {affine['n_rejected']}/{N_STARTS} seeds rejected  "
              f"(sanity |s-1|<={SCALE_TOL}, canvas_ratio={ratio_str}, veto=±{CANVAS_FIT_TOL})")
        for i, s in enumerate(scales):
            if s is None:
                continue
            ok, why = scale_verdict(s, ratio)
            print(f"        seed {i}: scale={s:.3f}  {'OK' if ok else 'REJECTED - ' + why}")

    rigid_ok = not rigid["fell_back_to_unregistered"]
    affine_ok = not affine["fell_back_to_unregistered"]

    # Margin affine must clear to be preferred over rigid.
    #
    # This was max(rigid_spread, affine_spread), which is backwards. The seed
    # spread measures how reproducible a result is, so taking the max means an
    # UNSTABLE RIGID RAISES THE BAR FOR AFFINE - the worse rigid behaves, the
    # harder it becomes to replace it. Observed on knee/sagittal z=17: rigid
    # spread 0.1372 (seeds 1.003-1.140, i.e. barely converging) against affine
    # spread 0.0007 (seeds agreeing to four decimals). Affine beat rigid by
    # 0.042 and was still rejected, because 0.042 < 0.1372.
    #
    # The question is whether THIS affine result is reliably above rigid's
    # best, so the relevant uncertainty is the affine's own reproducibility.
    # Rigid being erratic is rigid's problem and is, if anything, evidence
    # against keeping it.
    #
    # When fewer than two seeds survive the gate, affine["spread"] is None -
    # and falling through to rigid's spread there would reinstate the same bug
    # for exactly the slices the gate is most active on. Use the spread over
    # ALL affine seeds instead: admissibility is a property of the transform,
    # not of the score, so every seed still informs how much the affine
    # optimizer wobbles on this slice. On knee/sagittal z=17 that is 0.0007
    # (seeds 1.181/1.181/1.182) rather than rigid's 0.1441.
    affine_noise = affine["spread"]
    if affine_noise is None:
        affine_noise = demo.score_spread(affine["scores"])
    if affine_noise is None:
        affine_noise = rigid["spread"]
    noise_floor = max(affine_noise or 0.0, MIN_TRANSFORM_MARGIN)

    if rigid_ok and affine_ok:
        if is_better(affine["best_score"], rigid["best_score"], noise_floor):
            best_kind, best = "affine", affine
        else:
            best_kind, best = "rigid", rigid
    elif affine_ok:
        best_kind, best = "affine", affine
    elif rigid_ok:
        best_kind, best = "rigid", rigid
    else:
        best_kind, best = "unregistered", None

    if best is None:
        best_img, best_nmi = mri_slice, nmi_baseline
    else:
        best_img, best_nmi = best["best_img"], best["best_score"]

    # Never ship something worse than doing nothing.
    if best_kind != "unregistered" and not is_better(best_nmi, nmi_baseline):
        shipped_kind, shipped_img, shipped_nmi = "unregistered", mri_slice, nmi_baseline
        print(f"      {best_kind} ({best_nmi:.4f}) does not beat the unregistered baseline "
              f"({nmi_baseline:.4f}) - shipping unregistered")
    else:
        shipped_kind, shipped_img, shipped_nmi = best_kind, best_img, best_nmi

    return {
        "nmi_baseline": nmi_baseline,
        "rigid": rigid, "affine": affine,
        "affine_scales": scales,
        "canvas_ratio": ratio,
        "noise_floor": noise_floor,
        "best_kind": best_kind, "best_img": best_img, "best_nmi": best_nmi,
        "shipped_kind": shipped_kind, "shipped_img": shipped_img, "shipped_nmi": shipped_nmi,
        "all_seeds_failed": rigid["n_failed"] == N_STARTS and affine["n_failed"] == N_STARTS,
        "nothing_admissible": not rigid_ok and not affine_ok,
    }


def classify(attempt, coverage):
    """
    What happened on this canvas, and does it warrant trying the crop?

    Returns (outcome, needs_fallback). Causes are tested before symptoms, so
    the recorded outcome explains the slice rather than describing a
    downstream effect of it - a sparse slice that also fails to gain is
    `sparse_overlap`, not `regressed`.

    The key correction over the previous version: failing to IMPROVE is not
    failing. `marginal` means registration had nothing to add to an already
    good baseline, which is a fine outcome and not something a smaller canvas
    fixes. Only `regressed` - actively worse than doing nothing - is a
    registration failure.
    """
    baseline, best = attempt["nmi_baseline"], attempt["best_nmi"]
    if attempt["all_seeds_failed"]:
        return "all_seeds_failed", True
    if attempt["nothing_admissible"]:
        return "nothing_admissible", True
    if best is None or np.isnan(best):
        return "unevaluable_nmi", True
    if coverage < MIN_MRI_COVERAGE:
        return f"sparse_overlap_{coverage:.2f}", True
    if baseline is None or np.isnan(baseline):
        # Unregistered slice was unscorable and the result is not: that is an
        # improvement, and there is nothing to compare a margin against.
        return "improved", False
    if best < baseline - MIN_GAIN:
        return "regressed", True
    if best <= baseline + MIN_GAIN:
        return "marginal", False
    return "improved", False


# ─────────────────────────── per-slice ───────────────────────────

def blank_crop_columns():
    return {
        "crop_attempted": False, "crop_skipped_reason": "", "crop_rejected_reason": "",
        "nmi_baseline_common": None, "nmi_full_on_common": None,
        "nmi_rigid_crop": None, "nmi_affine_crop": None,
        "crop_best_kind": "", "nmi_crop_best": None, "crop_shipped_kind": "",
        "nmi_crop_shipped": None, "crop_affine_scales": "",
        "crop_affine_rejected": None, "crop_canvas_ratio": None, "crop_noise_floor": None,
        "rigid_crop_spread": None, "affine_crop_spread": None,
        "rigid_crop_failed": None, "affine_crop_failed": None,
        "crop_used": False,
    }


def fmt_scales(scales):
    return "|".join("%.3f" % s if s is not None else "FAIL" for s in scales)


def process_slice(prep, cand, z, position_label):
    """
    One slice: register on the full canvas, and fall back to the
    intersection-cropped canvas only if that actually failed.
    """
    ct_full = prep["ct_res"][:, :, z]
    mri_full = prep["baseline_v2_res"][:, :, z]

    coverage = mri_coverage(mri_full)
    print(f"      [{position_label:6s} z={z:3d}] MRI coverage={coverage * 100:.1f}%")

    full = register_canvas(ct_full, mri_full, "full")
    outcome, needs_fallback = classify(full, coverage)

    row = {
        "region": cand["region"], "patient": cand["patient"], "orientation": cand["orientation"],
        "position": position_label, "slice_index": z, "n_slices": prep["n_slices"],
        "fov_overlap_raw": prep["fov_overlap_raw"],
        "fov_overlap_aligned": prep["fov_overlap_aligned"],
        "mri_coverage_full": coverage,
        "nmi_baseline_full": full["nmi_baseline"],
        "nmi_rigid_full": full["rigid"]["best_score"],
        "nmi_affine_full": full["affine"]["best_score"],
        "full_best_kind": full["best_kind"], "nmi_full_best": full["best_nmi"],
        "full_shipped_kind": full["shipped_kind"], "nmi_full_shipped": full["shipped_nmi"],
        "full_affine_scales": fmt_scales(full["affine_scales"]),
        "full_affine_rejected": full["affine"]["n_rejected"],
        # The canvas ratio is the value a frame-fitting affine converges on.
        # Compare it against full_affine_scales: agreement to a few percent
        # means the optimizer resized to the border, not to the anatomy.
        "full_canvas_ratio": full["canvas_ratio"],
        "full_noise_floor": full["noise_floor"],
        "full_outcome": outcome,
        # spread is None (not 0.0) when fewer than two admissible seeds
        # survived - see demo.score_spread. Read it with the *_failed counts.
        "rigid_full_spread": full["rigid"]["spread"],
        "affine_full_spread": full["affine"]["spread"],
        "rigid_full_failed": full["rigid"]["n_failed"],
        "affine_full_failed": full["affine"]["n_failed"],
        "n_starts": N_STARTS,
    }
    row.update(blank_crop_columns())
    row.update({
        "final_source": "full", "final_kind": full["shipped_kind"],
        "final_nmi": full["shipped_nmi"], "final_scoring_region": "full",
    })

    ct_win_min, ct_win_max = prep["ct_win_min"], prep["ct_win_max"]
    mri_p1, mri_p99 = prep["mri_p1"], prep["mri_p99"]

    def mri_disp(img):
        return norm.normalize_mri_slice(sitk.GetArrayFromImage(img), mri_p1, mri_p99)

    # ---- is the crop fallback available? -----------------------------------
    crop_available, skip_reason, z_crop = True, "", None
    if not needs_fallback:
        crop_available, skip_reason = False, f"full_canvas_ok_{outcome}"
    elif prep["roi"] is None:
        crop_available, skip_reason = False, "no_intersection_box"
    else:
        start, size = prep["roi"]
        z_crop = z - start[2]
        if not (0 <= z_crop < prep["n_slices_cropped"]):
            # This slice lies outside the shared FOV in Z entirely, so there is
            # no cropped counterpart to fall back to. Worth recording rather
            # than hiding: it means the slice holds no dual-modality content.
            crop_available, skip_reason = False, "slice_outside_intersection"

    if not crop_available:
        row["crop_skipped_reason"] = skip_reason
        if needs_fallback:
            print(f"      FAILED ({outcome}) but no crop fallback: {skip_reason}")
        else:
            print(f"      {outcome} on full canvas: shipping {full['shipped_kind']} "
                  f"{full['shipped_nmi']:.4f} (baseline {full['nmi_baseline']:.4f}) - no crop needed")

        ct_disp = norm.normalize_ct_slice(sitk.GetArrayFromImage(ct_full), ct_win_min, ct_win_max)
        fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.9))
        demo.fusion_panel(axes[0], ct_disp, mri_disp(mri_full), f"baseline {full['nmi_baseline']:.3f}")
        demo.fusion_panel(axes[1], ct_disp, mri_disp(full["shipped_img"]),
                          f"{full['shipped_kind']} FULL {full['shipped_nmi']:.3f}")
        title_note = f"{outcome}; full canvas" if not needs_fallback else f"{outcome}; {skip_reason}"
    else:
        print(f"      FAILED ({outcome}) - falling back to the intersection crop")
        start, size = prep["roi"]
        ct_crop = prep["ct_cropped"][:, :, z_crop]
        mri_crop = prep["baseline_v3_res"][:, :, z_crop]

        crop = register_canvas(ct_crop, mri_crop, "crop")

        def to_common(slice_2d):
            """Crop a full-canvas 2D slice to the common intersection box."""
            return sitk.RegionOfInterest(slice_2d, (size[0], size[1]), (start[0], start[1]))

        # Re-score the full-canvas result over the cropped region. NMI still
        # depends on which pixels it sees, so a full-canvas score and a
        # cropped-canvas score are not comparable numbers even for identical
        # anatomy. This common region is what makes the choice well-posed.
        #
        # crop["nmi_baseline"] is the unregistered content over that same
        # region (identical samples to to_common(mri_full)), so all three
        # candidates below are directly comparable.
        nmi_full_common = demo.nmi_score(ct_crop, to_common(full["shipped_img"]))
        nmi_base_common = crop["nmi_baseline"]
        crop_ship = crop["shipped_nmi"]

        # Three-way, not two-way: the crop must beat the full canvas AND beat
        # doing nothing. Comparing it only against the full result let a crop
        # worse than the baseline win whenever the full result was worse still.
        if is_better(crop_ship, nmi_full_common, MIN_GAIN) and is_better(crop_ship, nmi_base_common, MIN_GAIN):
            final_source, final_kind, final_nmi = "crop", crop["shipped_kind"], crop_ship
            rejected_reason, crop_used = "", True
            verdict = "CROP"
        elif is_better(nmi_base_common, nmi_full_common, MIN_GAIN):
            final_source, final_kind, final_nmi = "unregistered", "unregistered", nmi_base_common
            rejected_reason, crop_used = "full_worse_than_baseline_on_common", False
            verdict = "UNREGISTERED (both canvases lost to doing nothing)"
        else:
            final_source, final_kind, final_nmi = "full", full["shipped_kind"], nmi_full_common
            rejected_reason, crop_used = "crop_did_not_help", False
            verdict = "keep FULL (crop_did_not_help)"

        print(f"      common-region NMI: baseline={nmi_base_common:.4f}  "
              f"full({full['shipped_kind']})={nmi_full_common:.4f}  "
              f"crop({crop['shipped_kind']})={crop_ship:.4f}  -> {verdict}")

        row.update({
            "crop_attempted": True,
            "nmi_baseline_common": nmi_base_common,
            "nmi_full_on_common": nmi_full_common,
            "nmi_rigid_crop": crop["rigid"]["best_score"],
            "nmi_affine_crop": crop["affine"]["best_score"],
            "crop_best_kind": crop["best_kind"], "nmi_crop_best": crop["best_nmi"],
            "crop_shipped_kind": crop["shipped_kind"], "nmi_crop_shipped": crop_ship,
            "crop_affine_scales": fmt_scales(crop["affine_scales"]),
            "crop_affine_rejected": crop["affine"]["n_rejected"],
            "crop_canvas_ratio": crop["canvas_ratio"],
            "crop_noise_floor": crop["noise_floor"],
            "rigid_crop_spread": crop["rigid"]["spread"],
            "affine_crop_spread": crop["affine"]["spread"],
            "rigid_crop_failed": crop["rigid"]["n_failed"],
            "affine_crop_failed": crop["affine"]["n_failed"],
            "crop_used": crop_used, "crop_rejected_reason": rejected_reason,
            "final_source": final_source, "final_kind": final_kind,
            "final_nmi": final_nmi, "final_scoring_region": "common",
        })

        ct_disp = norm.normalize_ct_slice(sitk.GetArrayFromImage(ct_crop), ct_win_min, ct_win_max)
        fig, axes = plt.subplots(1, 3, figsize=(8.1, 2.9))
        demo.fusion_panel(axes[0], ct_disp, mri_disp(mri_crop), f"baseline {nmi_base_common:.3f}")
        demo.fusion_panel(axes[1], ct_disp, mri_disp(to_common(full["shipped_img"])),
                          f"{full['shipped_kind']} FULL {nmi_full_common:.3f}")
        demo.fusion_panel(axes[2], ct_disp, mri_disp(crop["shipped_img"]),
                          f"{crop['shipped_kind']} CROP {crop_ship:.3f}")
        axes[{"full": 1, "crop": 2, "unregistered": 0}[final_source]].set_title(
            axes[{"full": 1, "crop": 2, "unregistered": 0}[final_source]].get_title() + "  <- kept",
            fontsize=9.5)
        title_note = f"{outcome} -> kept {final_source} (common region)"

    fig.suptitle(f"{cand['region']} {cand['patient']} {cand['orientation']} {position_label} "
                 f"(z={z}) - NMI - {title_note}", fontsize=8)
    fig.tight_layout()
    # Patient in the filename: two patients in one region/orientation would
    # otherwise overwrite each other's PNG and CSV row silently.
    out_png = os.path.join(
        OUTPUT_DIR, f"{cand['region']}_{cand['patient']}_{cand['orientation']}_{position_label}_v3.png")
    fig.savefig(out_png, dpi=90)
    plt.close(fig)
    row["png"] = out_png
    return row


# ─────────────────────────── per-volume ───────────────────────────

def prepare_volume_v3(cand):
    region, patient, orientation = cand["region"], cand["patient"], cand["orientation"]
    print(f"=== {region.upper()} : {patient} [{orientation}] ===")

    ct_path = os.path.join(cfg.DATA_ROOT, "CT", patient, "ST0", cand["ct_se"])
    mri_path = os.path.join(cfg.DATA_ROOT, "MRI", patient, "ST0", cand["mri_se"])
    ct_image, _ = io_utils.load_dicom_series(ct_path)
    mri_image, _ = io_utils.load_dicom_series(mri_path)
    if ct_image is None or mri_image is None:
        print("    ! Failed to load series, skipping.")
        return None

    mri_corrected = img_proc.apply_n4_bias_correction(mri_image, shrink_factor=4)
    ct_res = img_proc.resample_inplane(ct_image, target_spacing=cfg.TARGET_SPACING_MM, is_ct=True)

    # The full canvas is the PRIMARY path - every slice is registered here
    # first, and the crop below is only ever consulted when that fails.
    baseline_v2_res, v2_t = demo.resample_mri_to_ct_grid_v2(
        mri_corrected, ct_res, default_pixel_value=0.0)

    baseline_v3_res, ct_cropped, v3_t, roi, aligned_corners = resample_mri_to_ct_grid_v3(
        mri_corrected, ct_res, default_pixel_value=0.0)

    # Both overlap numbers are computed against ct_res - the same grid the ROI
    # is indexed in. The raw number is a scanner-frame QC signal; the aligned
    # number is the one the crop actually acts on, and they can differ hugely
    # (knee/sagittal: 12.5% raw against a translation of -86.9mm).
    overlap_raw = float(min(overlap_fracs(ct_res, corners_world(mri_corrected))))
    overlap_aligned = float(min(overlap_fracs(ct_res, aligned_corners)))

    if baseline_v3_res is None:
        # Not fatal: with cropping demoted to a fallback, a pair with no
        # computable intersection can still run the full-canvas path. It just
        # has no fallback available if that path fails.
        print("    ! No FOV overlap box - full canvas only, no fallback available.")
        roi = None
    else:
        drift = np.abs(np.array(v3_t) - np.array(v2_t)).max()
        if drift > 1e-6:
            print(f"    ! WARNING: v2/v3 transforms disagree by {drift:.3f}mm - they should be identical.")
        print(f"    fallback crop available: n_slices {ct_res.GetSize()[2]} -> {ct_cropped.GetSize()[2]}  "
              f"in-plane {ct_res.GetSize()[:2]} -> {ct_cropped.GetSize()[:2]}")

    print(f"    FOV overlap (worst axis): raw={overlap_raw * 100:.1f}%  "
          f"after alignment={overlap_aligned * 100:.1f}%   "
          f"translation=({v2_t[0]:+.1f},{v2_t[1]:+.1f},{v2_t[2]:+.1f})mm")

    prefix = patient.split("_")[0]
    body_region = cfg.PREFIX_TO_REGION.get(prefix, "default")
    profile = cfg.REGION_PROFILES.get(body_region, cfg.REGION_PROFILES["default"])
    ct_win_min, ct_win_max = profile["ct_win_min"], profile["ct_win_max"]

    # Display percentiles come from the full-canvas MRI, which is what the
    # primary path shows; the cropped volume is a subset of it, so the same
    # window applies to both and panels stay visually comparable.
    mri_arr_vol = sitk.GetArrayFromImage(baseline_v2_res).astype(np.float32)
    mri_p1, mri_p99 = norm.compute_mri_percentiles(mri_arr_vol, cfg.MRI_PERCENTILE_LOW, cfg.MRI_PERCENTILE_HIGH)

    return {
        "ct_res": ct_res,
        "ct_cropped": ct_cropped,
        "baseline_v2_res": baseline_v2_res,
        "baseline_v3_res": baseline_v3_res,
        "roi": roi,
        "fov_overlap_raw": overlap_raw,
        "fov_overlap_aligned": overlap_aligned,
        "ct_win_min": ct_win_min, "ct_win_max": ct_win_max,
        "mri_p1": mri_p1, "mri_p99": mri_p99,
        "n_slices": ct_res.GetSize()[2],
        "n_slices_cropped": ct_cropped.GetSize()[2] if roi is not None else 0,
        "v2_translation_mm": v2_t,
    }


def summarize(rows):
    """Print the answers the run exists to give, so they aren't buried in the CSV."""
    n = len(rows)
    attempted = [r for r in rows if r["crop_attempted"]]
    used = [r for r in rows if r["crop_used"]]
    print(f"\n{'=' * 74}\nSUMMARY  ({n} slices, NMI, crop as fallback only)")

    outcomes = {}
    for r in rows:
        key = r["full_outcome"].split("_0.")[0]  # collapse sparse_overlap_0.NN
        outcomes[key] = outcomes.get(key, 0) + 1
    print("  full-canvas outcomes   : " + ", ".join(f"{k}={v}" for k, v in sorted(outcomes.items())))
    print(f"  crop attempted         : {len(attempted)}/{n}")
    print(f"  crop actually kept     : {len(used)}/{n}")

    shipped = {}
    for r in rows:
        shipped[r["final_kind"]] = shipped.get(r["final_kind"], 0) + 1
    print("  transform shipped      : " + ", ".join(f"{k}={v}" for k, v in sorted(shipped.items())))

    rejected = sum(int(r["full_affine_rejected"]) for r in rows)
    print(f"  affine seeds rejected by the scale gate: {rejected}/{n * int(rows[0]['n_starts'])} "
          f"({sum(1 for r in rows if int(r['full_affine_rejected']) == int(r['n_starts']))} slices "
          f"lost every affine seed)")

    if used:
        gains = [r["final_nmi"] - r["nmi_full_on_common"] for r in used
                 if r["nmi_full_on_common"] is not None and not np.isnan(r["nmi_full_on_common"])]
        if gains:
            print(f"  mean NMI gain where crop kept: {np.mean(gains):+.4f}")
        for r in used:
            print(f"    kept crop: {r['region']:8s} {r['orientation']:9s} {r['position']:6s} "
                  f"({r['full_outcome']})  {r['nmi_full_on_common']:.3f} -> {r['final_nmi']:.3f}")
    print("=" * 74)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    for cand in sweep.ORIENTATION_CANDIDATES:
        prep = prepare_volume_v3(cand)
        if prep is None:
            continue
        # Positions are on the FULL canvas - that is the primary path, and
        # first/middle/last there deliberately includes edge slices where the
        # MRI may not reach. Those are exactly the slices the fallback is for.
        n = prep["n_slices"]
        for label, z in [("first", 0), ("middle", n // 2), ("last", n - 1)]:
            rows.append(process_slice(prep, cand, z, label))

    csv_path = os.path.join(OUTPUT_DIR, "sweep_v3_summary.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        summarize(rows)
        print(f"\nWrote {len(rows)} rows to {csv_path}")
    else:
        print("\nNo rows processed.")


if __name__ == "__main__":
    main()
