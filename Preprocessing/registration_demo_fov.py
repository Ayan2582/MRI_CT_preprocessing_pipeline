"""
registration_demo_fov.py
──────────────────────────
Diagnostic (not part of the production pipeline, not part of the multi-slice
sweep): computes the physical field-of-view (FOV) of the raw CT and raw MRI
series for every candidate in registration_demo_sweep.ORIENTATION_CANDIDATES,
to check whether an FOV mismatch is what's driving the blank-slice cases and
the brain "v2 makes it worse" result found in the sweep.

For each image:
  - size (voxels) and spacing (mm/voxel) as acquired
  - per-axis FOV in the image's own row/col/slice index directions
  - the true 3D physical center (accounts for Direction, not just Origin)
  - the world-space axis-aligned bounding box (all 8 corners transformed to
    physical space, then min/max per X/Y/Z) - this is orientation-agnostic,
    so it's meaningful even when CT and MRI aren't acquired in the same
    direction convention, unlike comparing index-space FOV directly.

Then reports, per world axis: how much of the smaller FOV's extent actually
overlaps the larger one - the direct, quantitative version of "is there an
FOV mismatch."
"""
import os
import numpy as np
import SimpleITK as sitk

import io_utils
import pipeline_config as cfg
import registration_demo_sweep as sweep


def image_geometry(image):
    size = np.array(image.GetSize(), dtype=float)
    spacing = np.array(image.GetSpacing(), dtype=float)
    fov_local = size * spacing  # extent along the image's own row/col/slice axes

    # All 8 corners in index space -> physical space, to get a world-space AABB
    # that's meaningful regardless of each image's own Direction convention.
    corners_idx = [
        (i, j, k)
        for i in (0, size[0] - 1)
        for j in (0, size[1] - 1)
        for k in (0, size[2] - 1)
    ]
    corners_phys = np.array([image.TransformContinuousIndexToPhysicalPoint(c) for c in corners_idx])
    aabb_min = corners_phys.min(axis=0)
    aabb_max = corners_phys.max(axis=0)

    center = image.TransformContinuousIndexToPhysicalPoint(tuple((size - 1) / 2.0))

    return {
        "size": size, "spacing": spacing, "fov_local": fov_local,
        "aabb_min": aabb_min, "aabb_max": aabb_max, "center": np.array(center),
    }


def overlap_report(ct_geom, mri_geom):
    axis_names = ["X", "Y", "Z"]
    lines = []
    for i, name in enumerate(axis_names):
        ct_lo, ct_hi = ct_geom["aabb_min"][i], ct_geom["aabb_max"][i]
        mri_lo, mri_hi = mri_geom["aabb_min"][i], mri_geom["aabb_max"][i]
        ct_extent = ct_hi - ct_lo
        mri_extent = mri_hi - mri_lo
        overlap = max(0.0, min(ct_hi, mri_hi) - max(ct_lo, mri_lo))
        smaller = min(ct_extent, mri_extent)
        frac = (overlap / smaller * 100.0) if smaller > 0 else 0.0
        lines.append(
            f"    {name}: CT[{ct_lo:7.1f},{ct_hi:7.1f}] ({ct_extent:6.1f}mm)  "
            f"MRI[{mri_lo:7.1f},{mri_hi:7.1f}] ({mri_extent:6.1f}mm)  "
            f"overlap={overlap:6.1f}mm ({frac:5.1f}% of the smaller FOV)"
        )
    return "\n".join(lines)


def main():
    for cand in sweep.ORIENTATION_CANDIDATES:
        region, patient, orientation = cand["region"], cand["patient"], cand["orientation"]
        ct_path = os.path.join(cfg.DATA_ROOT, "CT", patient, "ST0", cand["ct_se"])
        mri_path = os.path.join(cfg.DATA_ROOT, "MRI", patient, "ST0", cand["mri_se"])
        ct_image, _ = io_utils.load_dicom_series(ct_path)
        mri_image, _ = io_utils.load_dicom_series(mri_path)
        if ct_image is None or mri_image is None:
            print(f"=== {region.upper()} {patient} [{orientation}] : failed to load ===")
            continue

        ct_geom = image_geometry(ct_image)
        mri_geom = image_geometry(mri_image)
        center_offset = mri_geom["center"] - ct_geom["center"]

        print(f"=== {region.upper()} : {patient} [{orientation}] ===")
        print(f"    CT  size={tuple(ct_geom['size'].astype(int))}  spacing={tuple(np.round(ct_geom['spacing'],3))}  "
              f"FOV(local axes)={tuple(np.round(ct_geom['fov_local'],1))} mm")
        print(f"    MRI size={tuple(mri_geom['size'].astype(int))}  spacing={tuple(np.round(mri_geom['spacing'],3))}  "
              f"FOV(local axes)={tuple(np.round(mri_geom['fov_local'],1))} mm")
        print(f"    center offset (MRI - CT), world mm: {tuple(np.round(center_offset,1))}  "
              f"|offset|={np.linalg.norm(center_offset):.1f}mm")
        print("    world-space AABB overlap per axis:")
        print(overlap_report(ct_geom, mri_geom))
        print()


if __name__ == "__main__":
    main()
