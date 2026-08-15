"""
package_for_kaggle.py
─────────────────────
Package the QC output into a self-contained, uploadable dataset.

    python model/scripts/package_for_kaggle.py
    python model/scripts/package_for_kaggle.py --out D:/kaggle_mri_ct --zip

WHY THIS EXISTS. qc_workspace/ is gitignored and 6801 files; the dataset is
treated as regenerable from the raw DICOMs, which is fine locally and useless
remotely. Kaggle needs one directory it can ingest, and three things have to be
true of it that are not true of qc_workspace as it stands:

  1. NO ABSOLUTE PATHS. metadata.csv holds absolute Windows paths
     (C:\\Users\\moham\\...). The packaged manifest carries paths relative to the
     dataset root, with forward slashes, so the same file works on Linux.

  2. NO DEPENDENCY ON Preprocessing/. The HU windows that make the CT arrays
     invertible to Hounsfield units live in pipeline_config.REGION_PROFILES,
     which will not exist on Kaggle. They are baked into the manifest as
     per-row hu_min/hu_max columns.

  3. THE SPLIT TRAVELS WITH THE DATA. If splits.json were regenerated on Kaggle,
     the remote run would validate on different subjects than the local one and
     the two would not be comparable. It is copied, not recomputed.

Only the 2161 QC-accepted pairs referenced by the manifest are copied — not the
previews, the .npz cache, or the rejected slices.
"""

import argparse
import json
import logging
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from model.data.manifest import load_manifest, write_manifest    # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
logger = logging.getLogger("package")

DATASET_SLUG = "mri-ct-paired-slices"
DATASET_TITLE = "MRI-CT paired slices (QC accepted)"


def kaggle_username(explicit=None):
    """
    Resolve the Kaggle username for dataset-metadata.json.

    Order: --username, then $KAGGLE_USERNAME, then ~/.kaggle/kaggle.json. Falls
    back to the literal 'USERNAME' so the file is still written and the error is
    a clear one from the Kaggle CLI rather than a silent upload under the wrong
    account. Only the username is read — never the key.
    """
    if explicit:
        return explicit
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]

    token = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if os.path.isfile(token):
        try:
            with open(token, "r", encoding="utf-8") as fh:
                return json.load(fh).get("username") or "USERNAME"
        except (json.JSONDecodeError, OSError):
            logger.warning("could not read a username from %s", token)
    return "USERNAME"

README = """# MRI → CT paired slices

{n_pairs} QC-accepted 2-D slice pairs from {n_subjects} subjects, produced by the
preprocessing pipeline in the MRI_CT_preprocessing_pipeline repository.

## Layout

    CT/<patient>/<study>/<series>/IM<n>.npy     float32, [0, 1]
    MRI/<patient>/<study>/<series>/IM<n>.npy    float32, [0, 1]
    manifest.csv                                the pair index — USE THIS
    splits.json                                 subject-level train/val/test
    registration/<patient>/<study>/<SE>.json    per-series registration record

## Registration metadata is a record, not an input

The shift, the manual nudge, the erase strokes and the crop were all applied to
the arrays before they were saved. Nothing needs re-applying at load time.

**`reg_applied` is not an alignment-quality signal.** It records whether the
automatic translation search found a confident shift — not whether the pair
ended up well aligned. Where it failed, a human corrected the pair by hand: 738
slices were nudged individually, and 48 of 119 series carry nudges that vary
between their own slices. So `reg_applied=False` frequently marks a pair that a
person then fixed, not a bad pair. Grouping metrics by it measures the
algorithm's behaviour, not the data's quality.

Genuinely reusable here: `roi_x/y/w/h`, the anatomy bounding box in mm, if you
want crops centred on tissue instead of taken at random; and `mri_desc`, the
pulse sequence, whose appearance varies a lot between t1_se, t2_tirm and
t2_BLADE and is worth checking if the model struggles unevenly.

The rest (`reg_dx_mm`, `reg_dy_mm`, `reg_nmi_gain`, `manual_dx_mm`,
`manual_dy_mm`, `reg_note`) is provenance — what was done to each slice, kept
because reconstructing it would mean going back to a qc.db that is not here.

## Pair with the manifest, not with paths

The MRI folder name does not always match the CT folder name: 8 series carry an
orientation suffix on the MRI side only (CT `SE1` pairs with MRI `SE1_axial`).
Deriving one path from the other by string replacement silently loses 120 files.
Always read `ct_path` and `mri_path` from manifest.csv.

## Sizes vary

28 distinct sizes, 180x180 to 430x430, all square, 1 pixel = 1 mm. Nothing is
resized; crop or pad in the dataloader.

## Converting back to Hounsfield units

The CT arrays were windowed per body region before normalisation, so [0,1] is
NOT a fixed HU range:

    hu = value * (hu_max - hu_min) + hu_min

with `hu_min`/`hu_max` taken from the row's own manifest columns. Brain slices
use a 0-80 HU window, which means bone (>150 HU) is saturated to 1.0 and cannot
be recovered on those slices.

## Splits

`splits.json` is keyed by subject (the PA-number), not by patient folder — PA32
owns two folders and they belong to the same person. Use it as given; a fresh
random split would leak anatomy between train and validation.
"""


def human_size(n_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024 or unit == "GB":
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024


def main(argv=None):
    parser = argparse.ArgumentParser(description="Package the dataset for Kaggle")
    parser.add_argument("--manifest",
                        default=os.path.join(REPO_ROOT, "model", "data", "manifest.csv"))
    parser.add_argument("--splits",
                        default=os.path.join(REPO_ROOT, "model", "data", "splits.json"))
    parser.add_argument("--source-root",
                        default=os.path.join(REPO_ROOT, "qc_workspace", "output"),
                        help="Root the manifest's relative paths hang off")
    parser.add_argument("--out",
                        default=os.path.join(REPO_ROOT, "model", "kaggle_dataset"))
    parser.add_argument("--username", default=None,
                        help="Kaggle username for dataset-metadata.json "
                             "(default: $KAGGLE_USERNAME or ~/.kaggle/kaggle.json)")
    parser.add_argument("--zip", action="store_true",
                        help="Also produce a .zip beside the directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be copied and stop")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    manifest = load_manifest(args.manifest)
    source_root = os.path.abspath(args.source_root)
    out_root = os.path.abspath(args.out)

    n_pairs = len(manifest)
    n_subjects = manifest["subject_id"].nunique()

    total_bytes = 0
    missing = []
    for column in ("ct_path", "mri_path"):
        for rel in manifest[column]:
            path = os.path.join(source_root, rel.replace("/", os.sep))
            if os.path.isfile(path):
                total_bytes += os.path.getsize(path)
            else:
                missing.append(rel)

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} referenced files are missing under {source_root}, "
            f"first: {missing[:3]}")

    print()
    print(f"  pairs        {n_pairs}")
    print(f"  subjects     {n_subjects}")
    print(f"  files        {2 * n_pairs}")
    print(f"  total size   {human_size(total_bytes)}")
    print(f"  destination  {out_root}")
    print()

    if args.dry_run:
        print("--dry-run: nothing copied.")
        return 0

    os.makedirs(out_root, exist_ok=True)
    copied = 0
    for _, row in manifest.iterrows():
        for column in ("ct_path", "mri_path"):
            rel = row[column]
            src = os.path.join(source_root, rel.replace("/", os.sep))
            dst = os.path.join(out_root, rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            # copyfile, not copy2: Kaggle does not care about mtimes and
            # preserving them slows a 4322-file copy noticeably.
            shutil.copyfile(src, dst)
            copied += 1
        if copied % 1000 == 0:
            logger.info("copied %d/%d files", copied, 2 * n_pairs)

    # The manifest is already relative — it is written unchanged, which is the
    # point of building it that way in the first place.
    write_manifest(manifest, os.path.join(out_root, "manifest.csv"))
    shutil.copyfile(args.splits, os.path.join(out_root, "splits.json"))

    # Per-series registration records: the authoritative account of what was
    # done to each series, with detail the CSV flattens away — probe spread,
    # whether the search hit its edge, the exact HU window and MRI percentiles.
    # 120 files, ~640 KB, against a 1.7 GB dataset. Cheap enough that leaving it
    # behind only costs you the ability to explain a result later.
    reg_src = os.path.join(source_root, "registration")
    if os.path.isdir(reg_src):
        reg_dst = os.path.join(out_root, "registration")
        if os.path.isdir(reg_dst):
            shutil.rmtree(reg_dst)
        shutil.copytree(reg_src, reg_dst)
        n_reg = sum(len(files) for _, _, files in os.walk(reg_dst))
        logger.info("copied %d per-series registration records", n_reg)
    else:
        logger.warning("no registration/ folder under %s; per-series records "
                       "will not be available on Kaggle", source_root)

    with open(os.path.join(out_root, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(README.format(n_pairs=n_pairs, n_subjects=n_subjects))

    user = kaggle_username(args.username)
    dataset_id = f"{user}/{DATASET_SLUG}"
    with open(os.path.join(out_root, "dataset-metadata.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"title": DATASET_TITLE,
                   "id": dataset_id,
                   "licenses": [{"name": "other"}]}, fh, indent=2)

    if args.zip:
        logger.info("zipping (this takes a while at %s)...", human_size(total_bytes))
        shutil.make_archive(out_root, "zip", out_root)
        logger.info("wrote %s.zip", out_root)

    print()
    print("=" * 74)
    print(f"Packaged as dataset id:  {dataset_id}")
    if user == "USERNAME":
        print("  !! No Kaggle username found. Edit the \"id\" field in")
        print(f"     {os.path.join(out_root, 'dataset-metadata.json')}")
        print("     or re-run with --username <your-kaggle-username>.")
    print()
    print("To upload:")
    print()
    print(f"  python -m kaggle datasets create -p \"{out_root}\" --dir-mode zip")
    print()
    print("  ...or, to update an existing dataset:")
    print(f"  python -m kaggle datasets version -p \"{out_root}\" "
          f"-m \"update\" --dir-mode zip")
    print()
    print("  --dir-mode zip is REQUIRED: without it the CLI silently skips")
    print("  subdirectories, so CT/ and MRI/ would not be uploaded at all.")
    print()
    print("  These are patient images -- keep the dataset PRIVATE.")
    print()
    print("Then in the Kaggle notebook, point the config at the mounted copy:")
    print("  --set data.root=/kaggle/input/mri-ct-paired-slices \\")
    print("        data.manifest=/kaggle/input/mri-ct-paired-slices/manifest.csv \\")
    print("        data.splits=/kaggle/input/mri-ct-paired-slices/splits.json")
    print("=" * 74)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
