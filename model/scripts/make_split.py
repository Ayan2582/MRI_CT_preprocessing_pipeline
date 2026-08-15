"""
make_split.py
─────────────
Build the training manifest and the patient-level train/val/test split.

Run this once, before any training:

    python model/scripts/make_split.py

It produces two files that every experiment then shares:

    model/data/manifest.csv    2161 pairs with portable relative paths + HU windows
    model/data/splits.json     which patients are train / val / test

Both are committed to git on purpose. If each run generated its own split, the
runs in the experiment ladder would be validated on different patients and the
comparison between them — which is the entire point of the ladder — would be
meaningless. Regenerate them only if the underlying QC output changes, and if
you do, rerun every experiment.
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from model import bootstrap                                    # noqa: E402
from model.data import manifest as manifest_mod                # noqa: E402
from model.data import splits as splits_mod                    # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the training manifest and patient-level split.")
    parser.add_argument(
        "--output-root",
        default=os.path.join(REPO_ROOT, "qc_workspace", "output"),
        help="Folder containing CT/, MRI/ and metadata.csv (default: qc_workspace/output)")
    parser.add_argument("--manifest-out",
                        default=os.path.join(REPO_ROOT, "model", "data", "manifest.csv"))
    parser.add_argument("--split-out",
                        default=os.path.join(REPO_ROOT, "model", "data", "splits.json"))
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20240815,
                        help="Split seed. Changing it invalidates every previous run.")
    parser.add_argument("--keep-background", action="store_true",
                        help="Keep the slices the QC tool flagged is_background.")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip stat-ing every referenced .npy (faster, riskier).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing split. Read the warning first.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    log = logging.getLogger("make_split")

    if os.path.exists(args.split_out) and not args.force:
        log.error(
            "%s already exists.\n"
            "Overwriting it changes which patients are validation patients, which "
            "makes every previously recorded metric incomparable to future ones.\n"
            "Pass --force if that is genuinely what you want.", args.split_out)
        return 1

    metadata_csv = os.path.join(args.output_root, "metadata.csv")
    log.info("reading %s", metadata_csv)

    df = manifest_mod.build_manifest(
        metadata_csv=metadata_csv,
        output_root=args.output_root,
        hu_windows=bootstrap.region_windows(),
        drop_background=not args.keep_background,
        verify_exists=not args.no_verify,
    )
    manifest_mod.write_manifest(df, args.manifest_out)

    split = splits_mod.make_patient_split(
        df, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)
    summary = splits_mod.validate_split(split, df)
    splits_mod.write_split(split, summary, args.split_out,
                           seed=args.seed, manifest_rows=len(df))

    print()
    print("=" * 78)
    print(f"{'split':<8}{'subjects':>9}{'folders':>9}{'slices':>8}   regions")
    print("-" * 78)
    for name in splits_mod.SPLIT_NAMES:
        s = summary[name]
        regions = ", ".join(f"{k} {v}" for k, v in sorted(s["regions"].items()))
        print(f"{name:<8}{s['n_subjects']:>9}{s['n_patient_folders']:>9}"
              f"{s['n_slices']:>8}   {regions}")
    print("=" * 78)
    print("\nValidation subjects (held out of training entirely):")
    print("  " + ", ".join(split["val"]))
    print("Test subjects (do not look at these until the very end):")
    print("  " + ", ".join(split["test"]))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
