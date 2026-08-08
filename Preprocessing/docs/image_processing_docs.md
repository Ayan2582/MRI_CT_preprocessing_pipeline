# 📖 Code Docs: `image_processing.py`

This module is responsible for the complex 3D mathematics required to physically manipulate, distort, align, and correct the anatomical scans using SimpleITK.

> **Changed 2026-08-08 —** `apply_n4_bias_correction` was rewritten from a 2D
> slice-by-slice fit to a single 3D fit over the whole volume, with an
> anisotropic B-spline mesh (many control points in-plane, the cubic minimum
> through-plane). `plan_n4_control_points` is new and derives that mesh per
> series from its physical field of view. See
> [`mri_pipeline_docs.md` §5](./mri_pipeline_docs.md) for the reasoning and
> [`pipeline_config_docs.md`](./pipeline_config_docs.md) for the tunables.

---

```python
import math # Used for the ceiling function when back-solving the B-spline mesh.
import logging # Used to print warnings and info to the console without crashing the program.
import numpy as np # Matrix and array operations.
import SimpleITK as sitk # The industry standard library for Medical Image Processing (Simple Insight Toolkit).

import pipeline_config as cfg # Where the N4 control point targets live, so they are tunable in one place.

logger = logging.getLogger(__name__)


def plan_n4_control_points(
    image,
    orientation: str = "default",
    spline_order: int = None,
    fitting_levels: int = None,
):
    """
    [Function 3.1: Called via image_processing.py inside apply_n4_bias_correction]
    Decide how many B-spline control points N4 should use on each axis OF THIS
    VOLUME, from its actual physical field of view.

    Returns a dict with:
      initial   - what to hand to SetNumberOfControlPoints (index order)
      effective - the mesh that will exist at the FINAL fitting level
      spacing   - the physical distance in mm between those control points
      extent    - the field of view in mm on each axis

    `initial` and `effective` differ only when fitting_levels > 1, because ITK
    doubles the mesh on every extra level. We back-solve the initial mesh so the
    final one still lands near the target, instead of silently overshooting it
    by a factor of 8 the way the old 4-level default did.
    """
    # Fall back to the project-wide settings when the caller does not override.
    spline_order   = cfg.N4_SPLINE_ORDER   if spline_order   is None else int(spline_order)
    fitting_levels = cfg.N4_FITTING_LEVELS if fitting_levels is None else int(fitting_levels)
    fitting_levels = max(1, fitting_levels)

    # Physical field of view on each axis, in millimetres. This is the number
    # that actually matters — a 256-pixel knee and a 256-pixel abdomen need
    # completely different meshes, and only the mm extent tells them apart.
    size    = image.GetSize()
    spacing = image.GetSpacing()
    extent  = [size[i] * spacing[i] for i in range(3)]

    # A spline of order p literally cannot be built with fewer than p+1 control
    # points, so this is a hard floor, not a preference.
    floor_cp = spline_order + 1

    # Look up the in-plane target spacing for this acquisition plane.
    target = cfg.N4_CONTROL_POINT_SPACING_MM.get(
        orientation, cfg.N4_CONTROL_POINT_SPACING_MM["default"])

    lo = max(cfg.N4_CONTROL_POINTS_INPLANE_MIN, floor_cp)
    hi = max(cfg.N4_CONTROL_POINTS_INPLANE_MAX, lo)

    effective = []
    # -- Axes 0 and 1 are ALWAYS the in-plane axes, whatever the orientation.
    # (SimpleITK index order is (column, row, slice); the slice axis is index 2
    # by construction for every DICOM series, so "in-plane" needs no lookup.)
    for axis in (0, 1):
        # ncp control points divide an axis into (ncp - order) spans, so the
        # number of spans we want is simply the FOV divided by the target
        # spacing. At least one span, or there is no field to speak of.
        spans = max(1, int(round(extent[axis] / float(target[axis]))))
        # Clamp so a corrupt PixelSpacing tag cannot ask for a 60-point mesh.
        effective.append(int(min(max(spline_order + spans, lo), hi)))

    # -- Axis 2 is the through-plane axis. It is NOT derived from the FOV: we
    # want the stiffest field that exists, which is one single span, and that is
    # true whether the slab is 100 mm or 200 mm deep. See pipeline_config.py.
    effective.append(int(max(floor_cp, cfg.N4_CONTROL_POINTS_THROUGH_PLANE)))

    # -- Back-solve the initial mesh so the final one lands on the target.
    # ITK refines by doubling the number of spans per extra fitting level:
    #     final_spans = initial_spans * 2**(levels - 1)
    factor  = 2 ** (fitting_levels - 1)
    initial = [
        spline_order + max(1, math.ceil((n - spline_order) / factor))
        for n in effective
    ]
    # Rounding up above means we may land slightly above the target. Report what
    # will really happen rather than what was asked for.
    achieved = [spline_order + (n - spline_order) * factor for n in initial]

    return {
        "initial":   initial,
        "effective": achieved,
        "spacing":   [extent[i] / (achieved[i] - spline_order) for i in range(3)],
        "extent":    extent,
        "spline_order":   spline_order,
        "fitting_levels": fitting_levels,
    }


def apply_n4_bias_correction(
    image,
    orientation: str = "default",
    shrink_factor: int = None,
    n_iterations: list = None,
    convergence_threshold: float = None,
    spline_order: int = None,
    fitting_levels: int = None,
):
    """
    [Function 3: Called via pipeline_core.py, originating from main script preprocess_2d.py]
    Apply N4 ITK bias field correction to an MRI volume AS ONE 3D VOLUME.

    This used to run slice by slice, fitting an independent 2D field to each
    slice. That is wrong for a physical reason, not a stylistic one: the bias
    field comes from the receive coil's sensitivity profile, which is a single
    smooth function over the whole bore. It does not reset at slice boundaries.
    Fitting a separate field per slice hands every slice its own free brightness
    scaling, so slices that were consistent with each other come out stepped,
    and any real head-to-foot shading is left in place because no 2D fit can
    see it. A 3D fit removes the shading that is actually there and cannot
    invent a step between neighbours.

    The mesh is deliberately anisotropic: many control points in-plane, the
    bare minimum through-plane. `orientation` selects the in-plane targets,
    because which anatomical direction each in-plane axis corresponds to depends
    on the acquisition plane. See pipeline_config.py for the numbers and why.
    """
    # Resolve the defaults from config so there is one source of truth.
    shrink_factor         = cfg.N4_SHRINK_FACTOR if shrink_factor         is None else int(shrink_factor)
    convergence_threshold = cfg.N4_CONVERGENCE   if convergence_threshold is None else float(convergence_threshold)

    # -- Cast to Float32
    # SimpleITK requires image arrays to be 32-bit floats before it can run the complex N4 math.
    image_f32 = sitk.Cast(image, sitk.sitkFloat32)

    # This function is now genuinely 3D. A 2D input would silently get a
    # 2D-per-slice fit back, which is the exact bug being removed here, so
    # refuse it loudly instead.
    if image_f32.GetDimension() != 3:
        raise ValueError(
            f"apply_n4_bias_correction expects a 3D volume, got "
            f"{image_f32.GetDimension()}D. Reconstruct the series into a volume first."
        )

    size = image_f32.GetSize()

    # -- Tissue mask, computed ONCE on the whole volume --
    # Otsu's method automatically figures out which voxels are "air" (background) and which are "tissue",
    # so N4 does not waste its freedom modelling the lighting of empty air.
    # Computing it on the volume rather than per slice matters: a slice that is
    # nearly all air has no bimodal histogram, and a per-slice Otsu on it picks a
    # threshold somewhere inside the noise and calls the noise tissue.
    mask = sitk.OtsuThreshold(image_f32, 0, 1, 200)

    # If Otsu found nothing (a blank or constant series), N4 has nothing to fit
    # and will throw. Return the volume untouched rather than failing the patient.
    if float(sitk.GetArrayViewFromImage(mask).sum()) == 0.0:
        logger.warning("N4 skipped: Otsu found no tissue in this volume. Returning it uncorrected.")
        return image_f32

    # -- Work out the mesh for this specific volume --
    plan = plan_n4_control_points(image_f32, orientation, spline_order, fitting_levels)

    # -- Shrink for speed, IN-PLANE ONLY --
    # N4 is mathematically heavy, and fitting on a 4x smaller grid is ~16x faster
    # without losing the shape of a field this smooth.
    #
    # But the z axis is NOT shrunk. These MRI series have 15-24 slices; shrinking
    # z by 4 would leave 4-6 of them, which is fewer samples than the through-plane
    # spline has control points. The whole point of moving to 3D is to fit across
    # slices, and there would be almost nothing left to fit across.
    shrink = [1, 1, 1]
    for axis in (0, 1):
        s = max(1, int(shrink_factor))
        # Keep at least two voxels per control point after shrinking. Below that
        # the fit has more freedom than it has data, and N4 starts modelling noise.
        while s > 1 and size[axis] // s < 2 * plan["effective"][axis]:
            s -= 1
        shrink[axis] = s

    if shrink[0] > 1 or shrink[1] > 1:
        image_small = sitk.Shrink(image_f32, shrink)
        mask_small  = sitk.Shrink(mask,      shrink)
    else:
        image_small = image_f32
        mask_small  = mask

    # -- Configure the corrector --
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # Cubic by default. The order is global in ITK — it cannot differ per axis —
    # which is why the anisotropy has to come from the control point counts.
    corrector.SetSplineOrder(plan["spline_order"])
    # The anisotropic mesh: high in-plane, minimal through-plane.
    corrector.SetNumberOfControlPoints(plan["initial"])
    # One entry per fitting level. Default is a single level so the mesh above is
    # exactly the mesh used; see pipeline_config.py for why.
    if n_iterations is None:
        n_iterations = [cfg.N4_ITERATIONS] * plan["fitting_levels"]
    corrector.SetMaximumNumberOfIterations(list(n_iterations))
    # The mathematical threshold where the algorithm decides it has converged.
    corrector.SetConvergenceThreshold(convergence_threshold)

    logger.info(
        f"    N4 (3D, {orientation}): fit on {'x'.join(str(v) for v in image_small.GetSize())} "
        f"(shrink {shrink[0]}x{shrink[1]}x{shrink[2]}), "
        f"control points {tuple(plan['effective'])} "
        f"= {', '.join(f'{s:.0f}' for s in plan['spacing'])} mm apart "
        f"over a {', '.join(f'{e:.0f}' for e in plan['extent'])} mm FOV"
    )

    try:
        # Fit the field on the shrunken volume...
        corrector.Execute(image_small, mask_small)
        # ...then evaluate that same B-spline on the FULL resolution grid. The
        # field is a smooth analytic function, so this is an exact evaluation,
        # not an upsampling of a small image.
        log_bias = corrector.GetLogBiasFieldAsImage(image_f32)
        # Divide the original full-size volume by the exponential of the log
        # field to flatten the lighting. Exp() is strictly positive, so this
        # cannot divide by zero.
        return image_f32 / sitk.Exp(log_bias)
    except Exception as e:
        # If the fit fails, hand back the uncorrected volume rather than killing
        # the patient's run. This is now all-or-nothing per volume, which is the
        # honest behaviour: a partially corrected stack is worse than an
        # uncorrected one because the difference between slices is invisible.
        logger.warning(f"N4 failed on this volume, using it uncorrected. Error: {e}")
        return image_f32


def resample_inplane(image, target_spacing=1.0, is_ct=True):
    """
    [Function 4: Called via pipeline_core.py:94, originating from main script preprocess_2d.py:266]
    Resample a 3-D volume to a uniform in-plane resolution while leaving the
    through-plane (z) spacing unchanged.
    """
    # Extract the original pixel spacing (e.g., 0.48mm x 0.48mm x 3.0mm)
    orig_sp = image.GetSpacing()   # (sx, sy, sz)
    # Extract the total pixel dimensions (e.g., 512 x 512 x 18 slices)
    orig_sz = image.GetSize()      # (nx, ny, nz)

    # We enforce that the X and Y axes must become exactly the target_spacing (e.g., 1.0mm).
    new_sx = float(target_spacing)
    new_sy = float(target_spacing)
    # But we leave the Z axis (thickness of slices) completely alone so we don't hallucinate fake anatomy between slices.
    new_sz = orig_sp[2]            

    # To calculate the new array size (e.g. going from 512 to 256), we multiply the original size by the ratio of the spacings.
    new_nx = max(1, int(round(orig_sz[0] * orig_sp[0] / new_sx)))
    new_ny = max(1, int(round(orig_sz[1] * orig_sp[1] / new_sy)))
    new_nz = orig_sz[2]

    # Initialize the SimpleITK Resampler (the math engine that scales images).
    resampler = sitk.ResampleImageFilter()
    # Tell the resampler what the new physical scale is.
    resampler.SetOutputSpacing((new_sx, new_sy, new_sz))
    # Tell the resampler how many pixels wide the new image will be.
    resampler.SetSize((new_nx, new_ny, new_nz))
    # Preserve the original physical orientation and origin coordinates.
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    # Apply an identity transform (meaning we are only scaling, not rotating).
    resampler.SetTransform(sitk.Transform())
    # Use Linear Interpolation (drawing straight lines between pixels to guess the color of the new pixels).
    resampler.SetInterpolator(sitk.sitkLinear)
    # If the image is scaled up and creates empty borders, fill those borders with -1024 HU (Air) for CTs, or 0 for MRIs.
    resampler.SetDefaultPixelValue(-1024 if is_ct else 0)

    # Execute the heavy math and return the rescaled 3D volume.
    return resampler.Execute(image)


def resample_mri_to_ct_grid(mri_image, ct_image, default_pixel_value=0.0):
    """
    [Function 5: Called via pipeline_core.py:100, originating from main script preprocess_2d.py:266]
    Project the MRI image directly onto the CT image's physical coordinate grid.
    """
    # Create a shallow reference copy of the MRI image to avoid breaking the original memory block.
    mri_aligned = mri_image
    
    # This is a critical mathematical hack! If the MRI patient was tilted by 0.1 degrees compared to the CT patient, 
    # trying to align them across only 18 slices will cause the algorithm to "shear" the MRI and ruin it.
    # By forcing the MRI to mathematically pretend it has the exact same directional tilt as the CT, we prevent shearing!
    mri_aligned.SetDirection(ct_image.GetDirection())
    
    # Initialize the math engine.
    resampler = sitk.ResampleImageFilter()
    # Instead of manually providing sizes and spacings, we tell the engine to perfectly mimic the CT image's grid!
    resampler.SetReferenceImage(ct_image)
    # Use Linear Interpolation.
    resampler.SetInterpolator(sitk.sitkLinear)
    # Fill empty space with black.
    resampler.SetDefaultPixelValue(default_pixel_value)
    # Rely entirely on the physical DICOM origins to map the atoms of the MRI directly onto the atoms of the CT.
    resampler.SetTransform(sitk.Transform())
    
    # Execute the massive mathematical projection.
    return resampler.Execute(mri_aligned)


def estimate_volume_translation(
    ct_slices,
    mri_slices,
    spacing_mm:    float = 1.0,
    search_mm:     float = None,
    coarse_mm:     float = None,
    keep:          int   = None,
    bins:          int   = None,
    n_probes:      int   = None,
    min_probes:    int   = None,
    max_spread_mm: float = None,
    min_gain:      float = None,
):
    """
    [Function 7.5: Called optionally via pipeline_core.py, originating from main script preprocess_2d.py]
    Find ONE in-plane translation that best aligns this MRI stack to this CT
    stack, using registration_idea.py as the underlying method.

    Returns a plain dict, always. It never raises and never prints — everything
    it decided is in the return value for the caller to log and to record in the
    metadata CSV:

        applied      bool  — whether the shift survived every check below
        dy, dx       int   — the shift, in PIXELS of the resampled grid
        dy_mm, dx_mm float — the same shift in millimetres
        mean_gain    float — NMI improvement this shift buys, averaged over the
                             probe slices it was verified on (nan if unmeasured)
        reason       str   — plain English, and the only place a rejection is
                             explained. Worth putting in the log verbatim.
        n_probes / n_usable / spread_mm / hit_edge — the evidence behind it

    WHY ONE SHIFT AND NOT ONE PER SLICE
    ───────────────────────────────────
    Registering each slice independently is the same mistake the N4 rewrite
    removed, in a different variable: it gives every slice its own free
    translation, so the MRI shears through z relative to the CT and continuous
    anatomy comes out as a staircase. See pipeline_config.py for the measured
    per-slice swings on this dataset (85 mm across one shoulder stack) that make
    that concrete.

    HOW THE ANSWER IS DEFENDED
    ──────────────────────────
    Four checks, in order, and a failure of any one of them means no shift at
    all rather than a worse one:

      0. a probe whose best shift landed on the edge of the search square is
         discarded before it can vote — it is a censored observation
      1. enough probes survived that                    (REG_MIN_PROBES)
      2. the survivors agree with each other            (REG_MAX_SPREAD_MM)
      3. the median of them, re-scored on every probe, actually improves NMI
         on average                                     (REG_MIN_GAIN)

    Check 3 is the important one. Checks 1 and 2 ask whether the per-slice
    searches were consistent; check 3 measures the shift that is really about to
    be applied. A median can be a position no probe ever proposed, and this is
    what stops such a position being used on the strength of votes for other
    ones.

    Check 0 has to happen first rather than as a warning afterwards, because
    probes pinned against the same wall all report the same number and would
    otherwise pass check 2 with a spread of zero — unanimous censorship reading
    as unanimous evidence.

    `spacing_mm` is the in-plane pixel size of the slices being passed in, i.e.
    args.target_spacing. The underlying method works in whole pixels; this is
    what keeps every threshold in this function honestly in millimetres, so
    running at 2 mm does not silently double the search range.
    """
    # Resolve the defaults from config so there is one source of truth.
    search_mm     = cfg.REG_SEARCH_MM     if search_mm     is None else float(search_mm)
    coarse_mm     = cfg.REG_COARSE_MM     if coarse_mm     is None else float(coarse_mm)
    keep          = cfg.REG_KEEP          if keep          is None else int(keep)
    bins          = cfg.REG_BINS          if bins          is None else int(bins)
    n_probes      = cfg.REG_N_PROBES      if n_probes      is None else int(n_probes)
    min_probes    = cfg.REG_MIN_PROBES    if min_probes    is None else int(min_probes)
    max_spread_mm = cfg.REG_MAX_SPREAD_MM if max_spread_mm is None else float(max_spread_mm)
    min_gain      = cfg.REG_MIN_GAIN      if min_gain      is None else float(min_gain)

    spacing_mm = float(spacing_mm) if spacing_mm and spacing_mm > 0 else 1.0

    result = {
        "applied": False, "dy": 0, "dx": 0, "dy_mm": 0.0, "dx_mm": 0.0,
        "mean_gain": float("nan"), "n_probes": 0, "n_usable": 0,
        "spread_mm": (0.0, 0.0), "hit_edge": 0, "search_mm": search_mm,
        "probe_slices": [], "reason": "",
    }

    n = min(len(ct_slices), len(mri_slices))
    if n == 0:
        result["reason"] = "no slices to register"
        return result

    # Probe at 10/30/50/70/90% of the way through rather than at the ends: the
    # first and last slices of a stack are the emptiest and the least able to
    # measure anything, so spending probes on them mostly buys back None.
    fractions = np.linspace(0.1, 0.9, max(1, n_probes))
    probes = sorted({int(round(f * (n - 1))) for f in fractions})
    result["n_probes"] = len(probes)
    result["probe_slices"] = probes

    # The search happens in whole pixels; the thresholds are all in mm.
    search_px = max(1, int(round(search_mm / spacing_mm)))
    coarse_px = max(1, int(round(coarse_mm / spacing_mm)))

    # -- Step 1: what does each probe slice think the shift is? --
    votes = []
    for i in probes:
        ct  = np.asarray(ct_slices[i],  dtype=np.float64)
        mri = np.asarray(mri_slices[i], dtype=np.float64)
        r = reg_idea.register(ct, mri, search=search_px, bins=bins,
                              coarse=coarse_px, keep=keep, verbose=False)
        # None means this slice had nothing to measure — a near-empty slice,
        # which is normal at the ends of a stack and not an error.
        if r is None:
            continue
        # A best shift sitting exactly on the boundary of the search square is a
        # wall, not a peak: the true offset is somewhere further out that the
        # search could not see. That is a censored observation, not a
        # measurement, and it is thrown away rather than voted with.
        #
        # Discarding it matters more than it looks. Several probes pinned
        # against the same wall report the SAME number, so they sail through the
        # agreement check below with a spread of zero — the check would read
        # unanimous censorship as unanimous evidence and apply a shift already
        # known to be wrong. The count is kept so the caller can say so and
        # suggest a wider search.
        if abs(r["dy"]) == search_px or abs(r["dx"]) == search_px:
            result["hit_edge"] += 1
            continue
        votes.append((i, r))

    result["n_usable"] = len(votes)
    if len(votes) < min_probes:
        if result["hit_edge"]:
            result["reason"] = (
                f"only {len(votes)} of {len(probes)} probe slices gave a usable answer "
                f"(need {min_probes}); {result['hit_edge']} hit the edge of the "
                f"+/-{search_mm:.0f} mm search and were discarded as unmeasured. The real "
                f"offset is further out than the search reaches - raise the search range. "
                f"Leaving the MRI where it is")
        else:
            result["reason"] = (f"only {len(votes)} of {len(probes)} probe slices could be "
                                f"measured (need {min_probes}); leaving the MRI where it is")
        return result

    # -- Step 2: do the probes agree? --
    dys = [r["dy"] for _, r in votes]
    dxs = [r["dx"] for _, r in votes]
    dy = int(round(float(np.median(dys))))
    dx = int(round(float(np.median(dxs))))
    spread_y = (max(dys) - min(dys)) * spacing_mm
    spread_x = (max(dxs) - min(dxs)) * spacing_mm

    # Report the median even when it is rejected below — a rejected shift that is
    # visible in the CSV is how a bad pair gets noticed later.
    result.update({"dy": dy, "dx": dx,
                   "dy_mm": dy * spacing_mm, "dx_mm": dx * spacing_mm,
                   "spread_mm": (spread_y, spread_x)})

    if max(spread_y, spread_x) > max_spread_mm:
        result["reason"] = (f"probe slices disagree by {spread_y:.0f} mm down / {spread_x:.0f} mm "
                            f"across (limit {max_spread_mm:.0f}) - no single translation fits "
                            f"this pair, leaving the MRI where it is")
        return result

    # -- Step 3: does the shift we actually chose earn its place? --
    # Re-score the median on every probe against that probe's own do-nothing
    # baseline. This is the only check that measures the shift being applied
    # rather than the searches that suggested it.
    gains = []
    for i, r in votes:
        ct  = np.asarray(ct_slices[i],  dtype=np.float64)
        mri = np.asarray(mri_slices[i], dtype=np.float64)
        score = reg_idea.make_scorer(ct, mri, bins)
        if score is None:
            continue
        v, _ = score(dy, dx)
        if not np.isnan(v) and not np.isnan(r["baseline"]):
            gains.append(v - r["baseline"])

    if not gains:
        result["reason"] = "the chosen shift could not be re-scored; leaving the MRI where it is"
        return result

    mean_gain = float(np.mean(gains))
    result["mean_gain"] = mean_gain
    if mean_gain <= min_gain:
        result["reason"] = (f"best shift ({dx:+d}, {dy:+d}) px only improves NMI by "
                            f"{mean_gain:+.4f} (need > {min_gain:.3f}) - not worth moving for")
        return result

    result["applied"] = True
    result["reason"] = (f"shift {result['dx_mm']:+.0f} mm across, {result['dy_mm']:+.0f} mm down, "
                        f"agreed by {len(votes)} probe slices to within "
                        f"{max(spread_y, spread_x):.0f} mm, NMI {mean_gain:+.4f}")
    return result


def apply_translation(mri_slice, ct_shape, dy, dx):
    """
    [Function 7.6: Called optionally via pipeline_core.py, originating from main script preprocess_2d.py]
    Move one MRI slice by the whole-pixel shift estimate_volume_translation
    settled on, onto the CT's frame.

    Whole pixels means no interpolation, so this does not blur the MRI the way
    resampling it a second time would. Anything shifted in from outside the MRI
    becomes 0.0 — the same fill resample_mri_to_ct_grid already uses for MRI
    that does not reach, so the two kinds of "no MRI here" look identical
    downstream instead of one of them being NaN.
    """
    shifted, _ = reg_idea.apply_shift(
        np.asarray(mri_slice, dtype=np.float64), tuple(ct_shape), int(dy), int(dx), fill=0.0)
    # Back to the dtype the rest of the pipeline moves slices around in.
    return shifted.astype(np.asarray(mri_slice).dtype, copy=False)


def volume_to_slices(image):
    """
    [Function 6: Called via pipeline_core.py:106, originating from main script preprocess_2d.py:266]
    Convert a SimpleITK 3-D image to a list of 2-D numpy arrays.
    """
    # SimpleITK stores volumes as (x, y, z) in physical space.
    # GetArrayFromImage converts it to a standard Numpy Matrix, which uses (z, y, x) layout!
    # This is important because Python and AI libraries prefer the slice index (z) to be the first array dimension.
    arr = sitk.GetArrayFromImage(image)   # shape: (z, y, x)
    
    # We loop through the Z axis and pull out every single (y, x) 2D slice, returning them as a neat Python list.
    return [arr[i, :, :] for i in range(arr.shape[0])]
```
