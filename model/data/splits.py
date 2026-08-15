"""
splits.py
─────────
Patient-level, region-stratified train/val/test split.

TWO RULES, BOTH NON-NEGOTIABLE, AND BOTH EASY TO GET WRONG:

1. SPLIT BY SUBJECT, NEVER BY SLICE.
   Adjacent slices within a series are near-duplicates — same patient, same
   scanner, same anatomy, one millimetre apart. A random slice-level split puts
   slice 7 in train and slice 8 in validation, so the model is validated on data
   it has effectively memorised. The resulting validation MAE looks excellent
   and means nothing. With only 44 subjects this is the single most likely way
   for this project to produce a number that cannot be reproduced on new data.

   Note "subject", not "patient folder": PA32 owns two folders
   (PA32_Mandbi_ankle and PA32_Mandbi_knee) and they are one person. The
   grouping key is manifest['subject_id'], built by manifest.subject_id().

2. STRATIFY BY BODY REGION.
   The dataset spans brain, abdomen, musculoskeletal and spine, and each has a
   different CT HU window and completely different anatomy. An unstratified
   44-subject draw can easily put zero spine subjects in validation — there are
   only four in the entire dataset — and then the validation score is silent
   about a whole quarter of the problem.

The split is computed once, written to splits.json, and reused verbatim by every
experiment. Recomputing it per run — even with a fixed seed, if the patient list
or the seed ever changed — would make runs incomparable, and the comparison
between runs is the entire point of the experiment ladder.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")


def make_patient_split(manifest, val_frac=0.15, test_frac=0.15, seed=20240815):
    """
    Assign whole subjects to train/val/test, stratified by body region.

    Each region is split independently, so every region appears in every split.
    Regions with few subjects (spine has 4) still contribute at least one
    subject to val and one to test — a validation set that is silent about a
    region is worse than a small one.

    Returns {split_name: [subject_id, ...]} with the lists sorted.
    """
    import numpy as np

    if "subject_id" not in manifest.columns:
        raise ValueError(
            "Manifest has no 'subject_id' column. Regenerate it with "
            "python model/scripts/make_split.py --force — splitting on "
            "patient_id would put PA32's ankle and knee folders in different "
            "splits even though they are the same person."
        )

    # One row per subject. Region is normally constant per subject (it is
    # derived from the folder prefix), but take the mode rather than .first():
    # PA32 owns an ankle folder and a knee folder, and a subject whose folders
    # ever disagreed should land in its majority region rather than in whichever
    # one pandas happened to see first.
    per_subject = (manifest.groupby("subject_id")["body_region"]
                   .agg(lambda s: s.value_counts().idxmax()))

    rng = np.random.default_rng(seed)
    assignment = {name: [] for name in SPLIT_NAMES}

    for region in sorted(per_subject.unique()):
        # Sorted first, then shuffled: the sort makes the input order
        # independent of however pandas grouped the rows, so the seed alone
        # determines the outcome and the split is reproducible anywhere.
        subjects = sorted(per_subject[per_subject == region].index)
        rng.shuffle(subjects)
        n = len(subjects)

        if n < 3:
            # Cannot fill three splits. Everything goes to train and the caller
            # is told loudly, because a region absent from val is a real blind
            # spot in every metric that follows.
            logger.warning(
                "region '%s' has only %d subject(s) — all assigned to train, "
                "so validation and test will report nothing for it", region, n)
            assignment["train"].extend(subjects)
            continue

        n_val = max(1, int(round(val_frac * n)))
        n_test = max(1, int(round(test_frac * n)))
        # Guarantee at least one training patient even for a 3-patient region.
        while n - n_val - n_test < 1 and (n_val + n_test) > 2:
            if n_test >= n_val:
                n_test -= 1
            else:
                n_val -= 1

        assignment["val"].extend(subjects[:n_val])
        assignment["test"].extend(subjects[n_val:n_val + n_test])
        assignment["train"].extend(subjects[n_val + n_test:])

        logger.info("region %-16s n=%2d -> train %2d / val %d / test %d",
                    region, n, n - n_val - n_test, n_val, n_test)

    return {name: sorted(subjects) for name, subjects in assignment.items()}


def validate_split(split, manifest):
    """
    Assert the split is sound, and return a per-split summary.

    Checks the failure modes that would invalidate every downstream number:
    a subject in two splits (leakage), a subject in none (silently discarded
    data), an empty split, or a region missing from train or val.
    """
    summary = {}
    seen = {}
    for name in SPLIT_NAMES:
        for subject in split[name]:
            if subject in seen:
                raise ValueError(
                    f"LEAKAGE: subject '{subject}' appears in both "
                    f"'{seen[subject]}' and '{name}'. Every metric computed "
                    f"from this split would be optimistically biased."
                )
            seen[subject] = name

    all_subjects = set(manifest["subject_id"].unique())
    unassigned = all_subjects - set(seen)
    if unassigned:
        raise ValueError(
            f"{len(unassigned)} subject(s) are in the manifest but no split: "
            f"{sorted(unassigned)}. They would be silently excluded from training."
        )
    extra = set(seen) - all_subjects
    if extra:
        raise ValueError(
            f"Split references {len(extra)} subject(s) absent from the manifest: "
            f"{sorted(extra)}. The manifest was probably regenerated with "
            f"different filters — recreate the split."
        )

    for name in SPLIT_NAMES:
        rows = manifest[manifest["subject_id"].isin(split[name])]
        if len(rows) == 0:
            raise ValueError(f"Split '{name}' contains no slices.")
        # int() throughout: pandas value_counts yields numpy int64, which the
        # json encoder refuses, and splits.json has to be readable everywhere.
        summary[name] = {
            "n_subjects": len(split[name]),
            "n_patient_folders": int(rows["patient_id"].nunique()),
            "n_slices": int(len(rows)),
            "regions": {str(k): int(v) for k, v
                        in rows["body_region"].value_counts().sort_index().items()},
            "orientations": {str(k): int(v) for k, v
                             in rows["orientation"].value_counts().sort_index().items()},
        }

    train_regions = set(summary["train"]["regions"])
    for name in ("val", "test"):
        missing = train_regions - set(summary[name]["regions"])
        if missing:
            logger.warning(
                "regions %s are in train but absent from '%s' — that split "
                "reports nothing about them", sorted(missing), name)

    return summary


def write_split(split, summary, path, seed, manifest_rows):
    """Write splits.json atomically, with enough provenance to audit it later."""
    payload = {
        "seed": seed,
        "manifest_rows": int(manifest_rows),
        "splits": {name: split[name] for name in SPLIT_NAMES},
        "summary": summary,
        "_note": (
            "Subject-level (PA-number, so PA32's two folders stay together), "
            "region-stratified. Generated by "
            "model/scripts/make_split.py. Do NOT regenerate between experiments: "
            "every run in the ladder must train and validate on the same "
            "patients or the comparisons are meaningless."
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)
    logger.info("wrote %s", path)


def load_split(path):
    """Read splits.json and return {split_name: [patient_id, ...]}."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Split not found: {path}\nRun: python model/scripts/make_split.py"
        )
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload["splits"]
