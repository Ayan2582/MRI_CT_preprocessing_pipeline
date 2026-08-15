"""
manifest.py
───────────
Turn the QC tool's metadata.csv into a portable training manifest.

metadata.csv is the authoritative record of what survived manual QC — 2184
ACCEPTED pairs — but it is not directly usable as a training index for three
reasons, each of which this module fixes:

  1. Its `ct_npy` / `mri_npy` columns hold ABSOLUTE Windows paths. Those are
     meaningless on Kaggle. They are rewritten relative to a configurable root,
     with forward slashes, so the same manifest works on both platforms.

  2. The CT [0,1] values are only invertible to Hounsfield units if you know
     which HU window produced them, and the window is per body region
     (brain 0..80, abdomen -160..240, MSK/spine -200..300). The window is
     resolved once here and written as `hu_min`/`hu_max` columns, so the
     training loop never has to reach back into Preprocessing/ — which matters
     because Kaggle will not have Preprocessing/ on disk.

  3. It carries 35 columns of QC bookkeeping that training does not need.

WHY NOT PAIR BY PATH STRING: 120 of the 2184 MRI files live in a folder whose
name carries an orientation suffix the matching CT folder does not — CT
`PA42_Poonam/ST0/SE1` pairs with MRI `PA42_Poonam/ST0/SE1_axial`. A loader that
derives the MRI path by replacing '/CT/' with '/MRI/' silently drops 5.5% of the
dataset and never reports an error. The two path columns are the only correct
source, which is why this module exists rather than a glob.
"""

import logging
import os
import re

import pandas as pd

logger = logging.getLogger(__name__)

# Columns the loader needs to build a training sample.
KEEP_COLUMNS = [
    "patient_id", "body_region", "orientation", "study",
    "ct_series", "mri_series", "mri_desc", "slice_index",
    "height", "width", "is_background", "erased",
]

# Columns the loader does NOT need, carried as provenance.
#
# Every one of these transformations was ALREADY APPLIED to the saved arrays by
# registration_service.py — the shift, the nudge, the erase and the crop are
# baked in before the .npy is written. So nothing here is an input to the model;
# it is a record of what was done to each slice.
#
# DO NOT use `reg_applied` as a proxy for alignment quality. It records whether
# the AUTOMATIC translation search found a confident shift, not whether the pair
# ended up well aligned. Where it failed, a human corrected the pair by hand:
# 738 slices were nudged individually, and 48 of 119 series carry nudges that
# vary between their own slices. `reg_applied=False` therefore often marks a
# pair that a person then fixed, not a bad one. The same caveat applies to the
# probe-spread fields in the per-series registration JSONs, for the same reason.
#
# They travel anyway because provenance is cheap (12 columns in a 1.2 MB CSV)
# and because reconstructing them later would mean going back to qc.db, which
# does not exist on Kaggle. `roi_*` in particular is directly reusable: it is the
# anatomy bounding box, if crops ever need to be centred on tissue rather than
# taken at random.
DIAGNOSTIC_COLUMNS = [
    "reg_applied", "reg_dx_mm", "reg_dy_mm", "reg_nmi_gain", "reg_note",
    "manual_dx_mm", "manual_dy_mm",
    "roi_x", "roi_y", "roi_w", "roi_h",
]

MANIFEST_COLUMNS = (KEEP_COLUMNS + DIAGNOSTIC_COLUMNS
                    + ["subject_id", "ct_path", "mri_path", "hu_min", "hu_max"])

_SUBJECT_RE = re.compile(r"^(PA\d+)")


def subject_id(patient_id):
    """
    Collapse a patient folder name to the physical subject it belongs to.

    `patient_id` is a folder name, and one physical person can own more than one
    of them: PA32 appears as both 'PA32_Mandbi_ankle' and 'PA32_Mandbi_knee'.
    Treating those as two independent patients lets the same person's tissue,
    scanner settings and acquisition session appear in two different splits —
    which is exactly the leakage the patient-level split exists to prevent.

    The PA-number prefix is the real subject key; it is also what
    pipeline_config.PREFIX_TO_REGION keys on, so this matches how the rest of
    the pipeline already identifies a subject. Names that do not match the
    pattern fall back to the full folder name, which is the conservative
    direction — it can only ever split more finely, never merge two people.
    """
    match = _SUBJECT_RE.match(str(patient_id))
    return match.group(1) if match else str(patient_id)


def _to_relative(abs_path, root_abs):
    """
    Rewrite one absolute path relative to `root_abs`, using forward slashes.

    Raises rather than silently emitting a '..'-prefixed path: a manifest entry
    that escapes the dataset root would break the moment it was packaged for
    Kaggle, and it is far better to learn that here than three hours into an
    upload.
    """
    norm = os.path.normpath(str(abs_path))
    rel = os.path.relpath(norm, root_abs)
    if rel.startswith(".."):
        raise ValueError(
            f"Path escapes the dataset root and cannot be made portable:\n"
            f"  path: {norm}\n  root: {root_abs}\n"
            f"Point --output-root at the folder that contains CT/ and MRI/."
        )
    return rel.replace(os.sep, "/")


def build_manifest(metadata_csv, output_root, hu_windows, drop_background=True,
                   verify_exists=True):
    """
    Build the training manifest DataFrame from metadata.csv.

    Parameters
    ----------
    metadata_csv : path to qc_workspace/output/metadata.csv
    output_root  : the folder the relative paths hang off (contains CT/ and MRI/)
    hu_windows   : {region: (win_min, win_max)}, from bootstrap.region_windows()
    drop_background : drop rows flagged is_background by the QC tool
    verify_exists   : stat every referenced .npy. On by default — a missing file
                      surfaces here in one pass rather than as a crash at a
                      random epoch hours into training.
    """
    if not os.path.isfile(metadata_csv):
        raise FileNotFoundError(f"metadata.csv not found: {metadata_csv}")

    df = pd.read_csv(metadata_csv)
    logger.info("read %d rows from %s", len(df), metadata_csv)

    missing_cols = [c for c in ("ct_npy", "mri_npy") + tuple(KEEP_COLUMNS)
                    if c not in df.columns]
    if missing_cols:
        raise ValueError(f"metadata.csv is missing expected columns: {missing_cols}")

    # Diagnostics are carried when present but never required: an older
    # metadata.csv should still produce a trainable manifest, just one that
    # cannot support the registration-quality analysis.
    absent_diagnostics = [c for c in DIAGNOSTIC_COLUMNS if c not in df.columns]
    if absent_diagnostics:
        logger.warning(
            "metadata.csv has no %s — the manifest will train fine, but metrics "
            "cannot be split by registration quality", absent_diagnostics)

    # The QC tool writes only ACCEPTED rows, but assert it rather than trust it:
    # a future export that included REJECTED pairs would poison training with
    # slices a human already judged unusable.
    if "qc_status" in df.columns:
        bad = df[df["qc_status"] != "ACCEPTED"]
        if len(bad):
            logger.warning("dropping %d rows with qc_status != ACCEPTED", len(bad))
            df = df[df["qc_status"] == "ACCEPTED"].copy()

    if drop_background:
        n_bg = int(df["is_background"].sum())
        if n_bg:
            logger.info("dropping %d slices flagged is_background", n_bg)
            df = df[~df["is_background"].astype(bool)].copy()

    root_abs = os.path.abspath(output_root)
    present_diagnostics = [c for c in DIAGNOSTIC_COLUMNS if c in df.columns]
    out = df[KEEP_COLUMNS + present_diagnostics].copy()
    out["subject_id"] = [subject_id(p) for p in out["patient_id"]]
    n_folders, n_subjects = out["patient_id"].nunique(), out["subject_id"].nunique()
    if n_folders != n_subjects:
        merged = (out[["subject_id", "patient_id"]].drop_duplicates()
                  .groupby("subject_id")["patient_id"].apply(list))
        merged = {k: v for k, v in merged.items() if len(v) > 1}
        logger.info("%d patient folders collapse to %d subjects for splitting: %s",
                    n_folders, n_subjects, merged)
    out["ct_path"] = [_to_relative(p, root_abs) for p in df["ct_npy"]]
    out["mri_path"] = [_to_relative(p, root_abs) for p in df["mri_npy"]]

    # Freeze the HU window into the row. After this point nothing in the model
    # package needs pipeline_config, so the manifest travels to Kaggle alone.
    windows = [hu_windows.get(r, hu_windows["default"]) for r in out["body_region"]]
    out["hu_min"] = [w[0] for w in windows]
    out["hu_max"] = [w[1] for w in windows]

    unknown = sorted(set(out["body_region"]) - set(hu_windows))
    if unknown:
        logger.warning("regions with no HU profile, using 'default': %s", unknown)

    if verify_exists:
        missing = []
        for col in ("ct_path", "mri_path"):
            for rel in out[col]:
                if not os.path.isfile(os.path.join(root_abs, rel)):
                    missing.append(rel)
        if missing:
            preview = "\n  ".join(missing[:10])
            raise FileNotFoundError(
                f"{len(missing)} of {2 * len(out)} referenced .npy files are missing "
                f"under {root_abs}:\n  {preview}"
                + ("\n  ..." if len(missing) > 10 else "")
            )
        logger.info("verified %d .npy files exist", 2 * len(out))

    out = out[[c for c in MANIFEST_COLUMNS if c in out.columns]].reset_index(drop=True)
    logger.info("manifest: %d pairs, %d patients, regions %s",
                len(out), out["patient_id"].nunique(),
                dict(out["body_region"].value_counts()))
    if "reg_applied" in out.columns:
        n_unreg = int((~out["reg_applied"].astype(bool)).sum())
        logger.info("auto-registration applied to %d slices, not applied to %d "
                    "(not a quality signal — those were corrected by hand)",
                    len(out) - n_unreg, n_unreg)
    return out


def write_manifest(df, path):
    """Write the manifest atomically, matching the repo-wide convention."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    logger.info("wrote %s (%d rows)", path, len(df))


def load_manifest(path):
    """Read a manifest written by write_manifest, validating its columns."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Manifest not found: {path}\n"
            f"Run: python model/scripts/make_split.py"
        )
    df = pd.read_csv(path)
    # Only the columns the loader actually reads are required. Diagnostics are
    # optional so a manifest built from an older metadata.csv still trains.
    required = [c for c in MANIFEST_COLUMNS if c not in DIAGNOSTIC_COLUMNS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Manifest {path} is missing columns {missing}. "
            f"It was probably written by an older version — regenerate it with "
            f"python model/scripts/make_split.py --force"
        )
    return df
