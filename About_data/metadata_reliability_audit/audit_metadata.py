"""
audit_metadata.py
──────────────────
Read-only metadata reliability audit for the MRI-CT pipeline.
Runs Stages 0-5 (harvest -> coverage -> orientation ground truth ->
region ground truth -> pairing validation -> verdict) over a chosen
set of patients and writes:
  - tag_inventory.csv        (Stage 0 raw dump, one row per series)
  - tag_reliability_report.md (Stages 1-5 findings)

Does NOT touch the live pipeline. Standalone script.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import pydicom

# ── Config ───────────────────────────────────────────────────────────────
DATA_ROOT = r"C:\Users\moham\MRI_CT_preprocessing_pipeline\Raw_data_mri_ct\Rawdata_dicom"
OUT_DIR   = r"C:\Users\moham\MRI_CT_preprocessing_pipeline\About_data\metadata_reliability_audit"
# Previously: SAMPLE_PATIENTS = ["PA40", "PA7", "PA1", "PA17", "PA15"]  (random.seed(42) sample of 5/45)
# Now set below to every prefix in PREFIX_TO_REGION -> full 45-patient run.

# Same region map as pipeline_config.py, restricted to what we need for lookup.
PREFIX_TO_REGION = {
    "PA0": "brain", "PA1": "brain", "PA4": "brain", "PA5": "brain",
    "PA10": "brain", "PA17": "brain", "PA19": "brain", "PA21": "brain",
    "PA24": "brain", "PA26": "brain", "PA28": "brain", "PA33": "brain",
    "PA34": "brain", "PA38": "brain", "PA44": "brain",
    "PA2": "abdomen", "PA8": "abdomen", "PA9": "abdomen", "PA12": "abdomen",
    "PA15": "abdomen", "PA20": "abdomen", "PA22": "abdomen", "PA25": "abdomen",
    "PA27": "abdomen", "PA29": "abdomen", "PA30": "abdomen", "PA35": "abdomen",
    "PA37": "abdomen", "PA39": "abdomen", "PA41": "abdomen", "PA42": "abdomen",
    "PA3": "musculoskeletal", "PA6": "musculoskeletal", "PA11": "musculoskeletal",
    "PA13": "musculoskeletal", "PA14": "musculoskeletal", "PA16": "musculoskeletal",
    "PA32": "musculoskeletal", "PA36": "musculoskeletal", "PA40": "musculoskeletal",
    "PA43": "musculoskeletal",
    "PA7": "spine", "PA18": "spine", "PA23": "spine", "PA31": "spine",
}

SAMPLE_PATIENTS = sorted(PREFIX_TO_REGION.keys(), key=lambda p: int(p[2:]))  # all 45 patients
SAMPLE_LABEL = f"ALL {len(SAMPLE_PATIENTS)} patients"

TAGS = [
    "Modality", "SeriesDescription", "StudyDescription", "ProtocolName",
    "BodyPartExamined", "ImageOrientationPatient", "PixelSpacing",
    "SliceThickness", "SpacingBetweenSlices", "ScanningSequence",
    "SequenceVariant", "ScanOptions", "MRAcquisitionType",
    "RepetitionTime", "EchoTime", "RescaleSlope", "RescaleIntercept",
    "PatientID", "PatientName", "StudyDate", "SeriesNumber",
    "Manufacturer", "ManufacturerModelName", "FrameOfReferenceUID",
    "PatientPosition", "ImageType", "StudyInstanceUID",
]

# What each raw DICOM tag means, in plain language, for the glossary section.
TAG_DESCRIPTIONS = {
    "Modality":                "Acquisition type of this series. 'CT' or 'MR' in this dataset.",
    "SeriesDescription":       "Free-text label the tech/scanner console assigned to the series (e.g. protocol name, plane). Not standardized — wording is scanner/vendor/operator dependent.",
    "StudyDescription":        "Free-text label for the whole study (all series of one visit), e.g. 'medix^brain'. Set by the ordering/report system, not per-series.",
    "ProtocolName":            "Name of the acquisition protocol used, as configured on the scanner (e.g. '1.1 Routine Head 5mm Axial mode').",
    "BodyPartExamined":        "Coded/free-text body region the operator selected on the scanner console (e.g. 'HEAD', 'PELVIS', 'EXTREMITY'). Optional field per DICOM standard — routinely absent on MR in this dataset.",
    "ImageOrientationPatient": "Direction cosines (6 numbers: row vector + column vector) describing the 3D orientation of the image plane relative to the patient. This is geometry, not text, and is what Stage 2's ground-truth plane is computed from.",
    "PixelSpacing":            "In-plane pixel size in mm (row spacing, column spacing).",
    "SliceThickness":          "Nominal thickness of each slice in mm, as set on the scanner.",
    "SpacingBetweenSlices":    "Center-to-center distance between adjacent slices in mm (can differ from SliceThickness if slices overlap or have gaps). MR-only field in this dataset.",
    "ScanningSequence":        "MR pulse sequence class (e.g. 'SE'=spin echo, 'IR'=inversion recovery). Not applicable to CT.",
    "SequenceVariant":         "MR sequence variant flags (e.g. fat-saturation, spoiling). Not applicable to CT.",
    "ScanOptions":             "Scan option flags used during acquisition (e.g. 'AXIAL MODE' for CT, 'IR'/'FS' for MR).",
    "MRAcquisitionType":       "MR dimensionality of acquisition, '2D' or '3D'. Not applicable to CT.",
    "RepetitionTime":          "MR pulse sequence TR in ms — time between successive excitation pulses. Not applicable to CT.",
    "EchoTime":                "MR pulse sequence TE in ms — time between excitation and echo readout. Along with TR/sequence flags, is what actually distinguishes T1- vs T2-weighting (not the SeriesDescription string).",
    "RescaleSlope":            "Linear scale factor to convert stored CT pixel values to Hounsfield Units: HU = pixel * RescaleSlope + RescaleIntercept. CT-only field in this dataset.",
    "RescaleIntercept":        "Linear offset to convert stored CT pixel values to Hounsfield Units (see RescaleSlope). CT-only field in this dataset.",
    "PatientID":                "Scanner/PACS-assigned patient identifier for this modality's exam. NOT consistent across CT and MRI in this dataset (see Stage 4/5) — cannot be used to link the two modalities for the same patient.",
    "PatientName":              "Patient name as typed at the console, often with age/sex appended (e.g. 'RANJEET 77Y/M'). Formatting differs slightly between CT and MRI consoles (e.g. '77Y/M' vs '77 Y M'), so needs normalization before cross-modality comparison.",
    "StudyDate":                "Date the study was acquired (YYYYMMDD). Same-day CT and MRI for one patient is a useful independent pairing check.",
    "SeriesNumber":             "Scanner-assigned integer index of the series within the study. Not the same thing as the SE folder name; both are separate numbering schemes.",
    "Manufacturer":             "Scanner manufacturer (e.g. 'GE MEDICAL SYSTEMS', 'SIEMENS'). Explains why CT and MRI tag conventions differ so much in this dataset — they come from two different vendors' consoles.",
    "ManufacturerModelName":    "Specific scanner model (e.g. 'Revolution ACTs', 'Symphony').",
    "FrameOfReferenceUID":      "Unique ID of the 3D physical coordinate system this series was acquired in. Differs between CT and MRI here (separate scanners, separate exams) — confirms the two volumes do NOT share a coordinate frame and any alignment must be done by the pipeline's own registration step, not assumed from this UID.",
    "PatientPosition":          "Patient positioning on the table (e.g. 'HFS' = head-first-supine).",
    "ImageType":                "Vendor image-processing flags (e.g. 'DERIVED/SECONDARY/REFORMATTED' for CT in this dataset) — signals the CT was reprocessed/reformatted by console software rather than being raw primary acquisition.",
    "StudyInstanceUID":         "Unique ID of the whole study (visit). Differs between the CT study and the MRI study for the same patient, since they are two separate exams/UIDs even on the same day.",
}

# What each computed/derived column in the CSV and report means (not raw DICOM tags).
COLUMN_DESCRIPTIONS = {
    "modality_folder":                          "Top-level folder this series came from: 'CT' or 'MRI'. Mirrors Modality but is filesystem-derived, independent of the DICOM tag.",
    "patient_folder":                           "Full patient folder name, e.g. 'PA17_Mahi' (ID + name as used on disk).",
    "patient_prefix":                           "Just the 'PAxx' ID portion of patient_folder, used to look up PREFIX_TO_REGION.",
    "series_dir":                               "Series subfolder name, e.g. 'SE0'. This is what the live pipeline's prefix-matching pairing rule keys off.",
    "series_path":                              "Full filesystem path to the series folder that was read.",
    "n_files":                                  "Number of DICOM slice files found in the series folder = slice count (Z-axis) of that series.",
    "SliceThickness_varies_in_series":          "True if SliceThickness differed between the first/middle/last sampled slice of the series (a red flag for inconsistent acquisition).",
    "ImageOrientationPatient_varies_in_series": "True if the orientation direction cosines differed between the first/middle/last sampled slice (would indicate a non-planar or corrupted series).",
    "geom_plane":                               "Ground-truth anatomical plane ('axial'/'coronal'/'sagittal', or 'oblique' if >20° off every axis) computed geometrically from ImageOrientationPatient — independent of any text tag.",
    "geom_angle_off_axis_deg":                  "How many degrees the computed slice normal is off the nearest coordinate axis. 0° = perfectly axis-aligned; larger values mean the gantry/patient was tilted.",
    "live_heuristic_orientation":               "Orientation `discover_series()` computes for THIS series in isolation (folder name keyword check, then SeriesDescription keyword check, else 'unknown'). Note: for CT series this value is computed but never actually used by the live pipeline — `preprocess_2d.py` always uses the PAIRED MRI series' value instead (see Stage 2 'pipeline_orientation').",
    "tag_region_guess":                         "Body region guessed by this audit script from keyword-matching BodyPartExamined + StudyDescription + ProtocolName text.",
    "config_region":                            "Body region this patient is assigned in the pipeline's hardcoded PREFIX_TO_REGION dict (pipeline_config.py) — the value actually used to pick CT windowing/crop size today.",
}


# ── Current live heuristics (mirrored from io_utils.py, unmodified) ────────
def get_orientation_from_desc(description):
    d = (description or "").lower()
    if any(k in d for k in ["_tra", "_ax", "axial", "transv", "tra_"]):
        return "axial"
    if any(k in d for k in ["_cor", "cor_"]):
        return "coronal"
    if any(k in d for k in ["_sag", "sag_"]):
        return "sagittal"
    return "unknown"


def orient_from_folder_name(se_name):
    n = se_name.lower()
    if "axial" in n:
        return "axial"
    if "coronal" in n:
        return "coronal"
    if "sagittal" in n:
        return "sagittal"
    return "unknown"


def live_orientation_heuristic(se_name, series_desc):
    """Reproduces discover_series() orientation logic exactly."""
    orient = orient_from_folder_name(se_name)
    if orient == "unknown":
        orient = get_orientation_from_desc(series_desc) if series_desc else "unknown"
    return orient


# ── Stage 0: Harvest ─────────────────────────────────────────────────────
def geometric_plane(iop, tol_deg=20.0):
    """
    Ground-truth orientation from ImageOrientationPatient direction cosines.
    Computes the slice normal (cross product of row/col vectors) and finds
    the dominant physical axis it points along. Anything more than tol_deg
    off any single axis is reported as 'oblique' rather than forced into a
    bucket.
    """
    if iop is None or len(iop) != 6:
        return "unknown", None
    row = np.array(iop[0:3], dtype=float)
    col = np.array(iop[3:6], dtype=float)
    normal = np.cross(row, col)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        return "unknown", None
    normal = normal / norm
    abs_n = np.abs(normal)
    idx = int(np.argmax(abs_n))
    # angle between normal and the dominant axis unit vector
    angle_off_axis = np.degrees(np.arccos(np.clip(abs_n[idx], -1.0, 1.0)))
    plane = {0: "sagittal", 1: "coronal", 2: "axial"}[idx]
    if angle_off_axis > tol_deg:
        return "oblique", angle_off_axis
    return plane, angle_off_axis


def harvest():
    rows = []
    for modality_dir in ["CT", "MRI"]:
        mod_root = os.path.join(DATA_ROOT, modality_dir)
        for patient_folder in sorted(os.listdir(mod_root)):
            prefix = patient_folder.split("_")[0]
            if prefix not in SAMPLE_PATIENTS:
                continue
            patient_path = os.path.join(mod_root, patient_folder)
            if not os.path.isdir(patient_path):
                continue
            for st_name in sorted(os.listdir(patient_path)):
                st_path = os.path.join(patient_path, st_name)
                if not os.path.isdir(st_path):
                    continue
                for se_name in sorted(os.listdir(st_path)):
                    se_path = os.path.join(st_path, se_name)
                    if not os.path.isdir(se_path):
                        continue
                    files = sorted(glob.glob(os.path.join(se_path, "*")))
                    files = [f for f in files if os.path.isfile(f)]
                    if len(files) < 2:
                        continue

                    # sample first, middle, last slice to check intra-series variance
                    sample_idxs = sorted(set([0, len(files) // 2, len(files) - 1]))
                    slice_records = []
                    for i in sample_idxs:
                        try:
                            ds = pydicom.dcmread(files[i], stop_before_pixels=True)
                        except Exception:
                            continue
                        rec = {t: getattr(ds, t, None) for t in TAGS}
                        slice_records.append(rec)
                    if not slice_records:
                        continue

                    first = slice_records[0]

                    # intra-series variance flags for key geometric tags
                    def varies(tag):
                        vals = {str(r.get(tag)) for r in slice_records}
                        return len(vals) > 1

                    row = {
                        "modality_folder": modality_dir,
                        "patient_folder": patient_folder,
                        "patient_prefix": prefix,
                        "series_dir": se_name,
                        "series_path": se_path,
                        "n_files": len(files),
                    }
                    for t in TAGS:
                        val = first.get(t)
                        if isinstance(val, pydicom.valuerep.PersonName):
                            val = str(val)
                        elif val is not None and hasattr(val, "__iter__") and not isinstance(val, str):
                            # normalize multi-valued DICOM fields (e.g. IOP, ImageType) to plain list for CSV
                            val = list(val)
                        row[t] = val
                    row["SliceThickness_varies_in_series"] = varies("SliceThickness")
                    row["ImageOrientationPatient_varies_in_series"] = varies("ImageOrientationPatient")

                    # ground truth + current heuristic orientation
                    iop = row.get("ImageOrientationPatient")
                    plane, angle_off = geometric_plane(iop)
                    row["geom_plane"] = plane
                    row["geom_angle_off_axis_deg"] = angle_off

                    desc = row.get("SeriesDescription") or ""
                    row["live_heuristic_orientation"] = live_orientation_heuristic(se_name, str(desc))

                    rows.append(row)
    return pd.DataFrame(rows)


# ── Glossary ─────────────────────────────────────────────────────────────
def glossary_section():
    lines = ["## Glossary — Tag & Column Definitions\n"]

    lines.append("### Raw DICOM tags (harvested from the files)\n")
    lines.append("| Tag | Meaning |")
    lines.append("|---|---|")
    for t in TAGS:
        desc = TAG_DESCRIPTIONS.get(t, "")
        lines.append(f"| {t} | {desc} |")

    lines.append("\n### Derived / computed columns (produced by this audit script, not raw DICOM)\n")
    lines.append("| Column | Meaning |")
    lines.append("|---|---|")
    for c, desc in COLUMN_DESCRIPTIONS.items():
        lines.append(f"| {c} | {desc} |")

    return "\n".join(lines)


# ── Stage 1: Coverage ────────────────────────────────────────────────────
def stage1_coverage(df):
    lines = ["## Stage 1 — Tag Coverage (by modality)\n"]
    lines.append("| Tag | CT present | CT n | MRI present | MRI n | CT distinct values | MRI distinct values |")
    lines.append("|---|---|---|---|---|---|---|")
    for t in TAGS:
        ct = df[df.modality_folder == "CT"]
        mr = df[df.modality_folder == "MRI"]
        ct_present = ct[t].notna().sum()
        mr_present = mr[t].notna().sum()
        ct_distinct = ct[t].apply(lambda v: str(v)).nunique()
        mr_distinct = mr[t].apply(lambda v: str(v)).nunique()
        lines.append(
            f"| {t} | {ct_present}/{len(ct)} | {ct_present/max(len(ct),1)*100:.0f}% "
            f"| {mr_present}/{len(mr)} | {mr_present/max(len(mr),1)*100:.0f}% "
            f"| {ct_distinct} | {mr_distinct} |"
        )
    return "\n".join(lines)


# ── Live-pipeline pairing (exact reproduction of preprocess_2d.py's matching loop) ──
def build_live_pairs(df):
    """
    Reproduce preprocess_2d.py's CT<->MRI pairing exactly:
      - Driven by MRI series: the code loops `for m_entry in mri_series`.
      - Both the MRI folder name and each candidate CT folder name are split on '_' and
        only the FIRST token is compared (e.g. 'SE1_saggital' and 'SE1_1' both reduce to 'SE1').
      - The first matching CT (in the same sorted-folder-name order discover_series() produces)
        wins; nothing is removed from consideration, matching the real `next(...)` call.
      - Critically: `orient = m_entry["orientation"]` — the CT series' OWN orientation is never
        read for this decision. Every processed CT slice is labeled with its paired MRI's
        orientation, whatever that is (including 'unknown').
    Returns (pairs, unmatched_ct, unmatched_mri) where pairs is a list of {"ct": row, "mri": row}
    dicts, and the unmatched lists are series the live pipeline would never process at all
    (no partner found by this rule).
    """
    pairs, unmatched_ct, unmatched_mri = [], [], []
    for _, sub in df.groupby("patient_folder", sort=False):
        ct_rows = sub[sub.modality_folder == "CT"].sort_values("series_dir").to_dict("records")
        mr_rows = sub[sub.modality_folder == "MRI"].sort_values("series_dir").to_dict("records")
        matched_ct_ids = set()
        for m in mr_rows:
            m_token = str(m["series_dir"]).split("_")[0]
            match = next((c for c in ct_rows if str(c["series_dir"]).split("_")[0] == m_token), None)
            if match is not None:
                pairs.append({"ct": match, "mri": m})
                matched_ct_ids.add(id(match))
            else:
                unmatched_mri.append(m)
        for c in ct_rows:
            if id(c) not in matched_ct_ids:
                unmatched_ct.append(c)
    return pairs, unmatched_ct, unmatched_mri


# ── Stage 2: Orientation as the live pipeline actually uses it for CT ────
def stage2_orientation(df):
    lines = ["\n## Stage 2 — Orientation: What the Live Pipeline Actually Uses for CT\n"]
    lines.append(
        "`preprocess_2d.py:236` sets `orient = m_entry[\"orientation\"]` — the pipeline NEVER "
        "reads a CT series' own computed orientation, even though `discover_series()` computes "
        "one for every series regardless of modality. Every CT slice that gets processed is "
        "labeled with whatever orientation its PAIRED MRI series resolved to. So scoring CT's own "
        "heuristic against CT's own geometry (as an earlier version of this report did) measures "
        "something the pipeline never actually computes. The question that matches real behavior "
        "is: does the borrowed MRI label correctly describe the CT slice it gets applied to?\n"
    )

    pairs, unmatched_ct, unmatched_mri = build_live_pairs(df)

    lines.append("Per live-pipeline pair (MRI-driven token match, exactly as `preprocess_2d.py` forms it):\n")
    lines.append("| Patient | CT series | MRI series | pipeline_orientation (borrowed from MRI) | "
                  "CT geom_plane | agrees w/ CT? | MRI geom_plane | agrees w/ MRI? | would pipeline process? |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    n_pairs = len(pairs)
    n_would_process = 0
    n_agree_ct = n_agree_mri = 0
    n_agree_ct_proc = n_agree_mri_proc = 0
    for p in pairs:
        c, m = p["ct"], p["mri"]
        orient = m["live_heuristic_orientation"]  # exactly what the pipeline calls `orient`
        would_process = orient in ("axial", "coronal", "sagittal")
        agree_ct = orient == c["geom_plane"]
        agree_mri = orient == m["geom_plane"]
        n_would_process += int(would_process)
        n_agree_ct += int(agree_ct)
        n_agree_mri += int(agree_mri)
        if would_process:
            n_agree_ct_proc += int(agree_ct)
            n_agree_mri_proc += int(agree_mri)
        lines.append(
            f"| {c['patient_folder']} | {c['series_dir']} | {m['series_dir']} | {orient} "
            f"| {c['geom_plane']} | {'agree' if agree_ct else 'DISAGREE'} "
            f"| {m['geom_plane']} | {'agree' if agree_mri else 'DISAGREE'} "
            f"| {'yes' if would_process else 'NO — skipped, orient unknown'} |"
        )

    lines.append(f"\n**Live-pipeline pairs found: {n_pairs}**")
    lines.append(
        f"- Would actually be processed (borrowed orientation resolves to a valid label): "
        f"{n_would_process}/{n_pairs} ({n_would_process/max(n_pairs,1)*100:.0f}%)"
    )
    lines.append(
        f"- Of ALL pairs, borrowed orientation matches CT's OWN geometric plane: "
        f"{n_agree_ct}/{n_pairs} ({n_agree_ct/max(n_pairs,1)*100:.0f}%)"
    )
    lines.append(
        f"- Of ALL pairs, borrowed orientation matches MRI's OWN geometric plane: "
        f"{n_agree_mri}/{n_pairs} ({n_agree_mri/max(n_pairs,1)*100:.0f}%)"
    )
    if n_would_process:
        lines.append(
            f"- **Of pairs that WOULD be processed**, borrowed orientation matches CT's OWN geometric plane: "
            f"{n_agree_ct_proc}/{n_would_process} ({n_agree_ct_proc/n_would_process*100:.0f}%) "
            f"— this is the number that matters: it's the real-world accuracy of the CT orientation "
            f"label actually written into `metadata.csv` and used to route each CT slice into "
            f"axial/coronal/sagittal output folders."
        )
        lines.append(
            f"- Of pairs that WOULD be processed, borrowed orientation matches MRI's OWN geometric plane: "
            f"{n_agree_mri_proc}/{n_would_process} ({n_agree_mri_proc/n_would_process*100:.0f}%)"
        )

    lines.append(f"\n**Unmatched CT series (no MRI partner found by the live rule — never processed, silently dropped): {len(unmatched_ct)}**")
    if unmatched_ct:
        lines.append("| Patient | CT series |")
        lines.append("|---|---|")
        for c in unmatched_ct:
            lines.append(f"| {c['patient_folder']} | {c['series_dir']} |")

    lines.append(f"\n**Unmatched MRI series (no CT partner found by the live rule — never processed, silently dropped): {len(unmatched_mri)}**")
    if unmatched_mri:
        lines.append("| Patient | MRI series |")
        lines.append("|---|---|")
        for m in unmatched_mri:
            lines.append(f"| {m['patient_folder']} | {m['series_dir']} |")

    return "\n".join(lines)


# ── Stage 3: Region ground truth vs PREFIX_TO_REGION ─────────────────────
def guess_region(row):
    """Best-effort region guess from BodyPartExamined / StudyDescription / ProtocolName."""
    text = " ".join(str(row.get(t) or "") for t in ["BodyPartExamined", "StudyDescription", "ProtocolName"]).lower()
    if any(k in text for k in ["head", "brain", "brai"]):
        return "brain"
    if any(k in text for k in ["spine", "cervical", "lumbar", "thoracic", "dorsal"]):
        return "spine"
    if any(k in text for k in ["knee", "ankle", "shoulder", "joint", "msk"]):
        return "musculoskeletal"
    if any(k in text for k in ["abdomen", "pelvis", "chest", "torso", "fistula"]):
        return "abdomen"
    return "unknown"


def stage3_region(df):
    lines = ["\n## Stage 3 — Region: Tag-Derived Guess vs Hardcoded PREFIX_TO_REGION\n"]
    df = df.copy()
    df["tag_region_guess"] = df.apply(guess_region, axis=1)
    df["config_region"] = df["patient_prefix"].map(PREFIX_TO_REGION)

    lines.append("| Patient | Modality | BodyPartExamined | StudyDescription | ProtocolName | tag_guess | config_region | agree? |")
    lines.append("|---|---|---|---|---|---|---|---|")
    n_agree, n_total = 0, 0
    for _, r in df.iterrows():
        n_total += 1
        agree = r["tag_region_guess"] == r["config_region"]
        n_agree += int(agree)
        lines.append(
            f"| {r['patient_folder']} | {r['modality_folder']} | {r['BodyPartExamined']} "
            f"| {r['StudyDescription']} | {r['ProtocolName']} | {r['tag_region_guess']} "
            f"| {r['config_region']} | {'agree' if agree else 'DISAGREE'} |"
        )
    lines.append(f"\n**Per-series agreement: {n_agree}/{n_total} ({n_agree/max(n_total,1)*100:.0f}%)**")

    # per-patient (union across series/modality) agreement -- what actually matters
    lines.append("\nPer-patient (union of all series' tag text) verdict:\n")
    lines.append("| Patient | config_region | any series matched? |")
    lines.append("|---|---|---|")
    n_pat_agree, n_pat = 0, 0
    for patient_folder, sub in df.groupby("patient_folder"):
        n_pat += 1
        cfg = sub["config_region"].iloc[0]
        matched = (sub["tag_region_guess"] == cfg).any()
        n_pat_agree += int(matched)
        lines.append(f"| {patient_folder} | {cfg} | {'yes' if matched else 'NO'} |")
    lines.append(f"\n**Per-patient (best-of-series) agreement: {n_pat_agree}/{n_pat} ({n_pat_agree/max(n_pat,1)*100:.0f}%)**")

    return "\n".join(lines), df


# ── Stage 4: Pairing validation ──────────────────────────────────────────
def normalize_name(name):
    if name is None:
        return ""
    s = str(name).upper()
    return "".join(ch for ch in s if ch.isalnum())


def stage4_pairing(df):
    lines = ["\n## Stage 4 — Pairing Validation (live MRI-driven token-match rule vs independent evidence)\n"]
    lines.append(
        "Live rule (`preprocess_2d.py`), reproduced exactly via `build_live_pairs()` (same "
        "function Stage 2 uses): driven by MRI series; both the MRI folder name and each "
        "candidate CT folder name are split on `_` and only the first token is compared "
        "(so e.g. `SE1_saggital` matches CT `SE1`); first match wins. We check whether matched "
        "pairs agree on StudyDate, normalized PatientName, Stage-2 geometric plane, and slice "
        "count (`n_files`) — none of these are used by the live pairing rule today, so each is a "
        "free independent corroboration (or a red flag) on top of it.\n"
    )

    pairs, _, _ = build_live_pairs(df)

    lines.append("| Patient | CT series | MRI series | CT date | MRI date | date agree | "
                  "CT name | MRI name | name agree | CT plane | MRI plane | plane agree | "
                  "CT n_files | MRI n_files | n_files agree |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    n_pairs, n_date_ok, n_name_ok, n_plane_ok, n_files_ok = 0, 0, 0, 0, 0
    for p in pairs:
        c, m = p["ct"], p["mri"]
        n_pairs += 1
        date_ok = str(c["StudyDate"]) == str(m["StudyDate"])
        name_ok = normalize_name(c["PatientName"]) == normalize_name(m["PatientName"])
        plane_ok = c["geom_plane"] == m["geom_plane"]
        files_ok = c["n_files"] == m["n_files"]
        n_date_ok += int(date_ok)
        n_name_ok += int(name_ok)
        n_plane_ok += int(plane_ok)
        n_files_ok += int(files_ok)
        lines.append(
            f"| {c['patient_folder']} | {c['series_dir']} | {m['series_dir']} "
            f"| {c['StudyDate']} | {m['StudyDate']} | {'agree' if date_ok else 'DISAGREE'} "
            f"| {c['PatientName']} | {m['PatientName']} | {'agree' if name_ok else 'DISAGREE'} "
            f"| {c['geom_plane']} | {m['geom_plane']} | {'agree' if plane_ok else 'DISAGREE'} "
            f"| {c['n_files']} | {m['n_files']} | {'agree' if files_ok else 'DISAGREE'} |"
        )

    if n_pairs:
        lines.append(f"\n**Live-pipeline-matched pairs found: {n_pairs}**")
        lines.append(f"- Date agreement: {n_date_ok}/{n_pairs} ({n_date_ok/n_pairs*100:.0f}%)")
        lines.append(f"- Name agreement: {n_name_ok}/{n_pairs} ({n_name_ok/n_pairs*100:.0f}%)")
        lines.append(f"- Geometric-plane agreement: {n_plane_ok}/{n_pairs} ({n_plane_ok/n_pairs*100:.0f}%)")
        lines.append(f"- Slice-count (n_files) agreement: {n_files_ok}/{n_pairs} ({n_files_ok/n_pairs*100:.0f}%)")
    else:
        lines.append("\n**No live-pipeline-matched CT/MRI pairs found in this sample.**")

    return "\n".join(lines)


# ── Stage 5: Verdict ──────────────────────────────────────────────────────
def stage5_verdict(df):
    lines = ["\n## Stage 5 — Verdict\n"]
    lines.append("| Tag / Signal | Coverage | Agreement w/ ground truth | Verdict |")
    lines.append("|---|---|---|---|")

    total = len(df)
    pairs, _, _ = build_live_pairs(df)
    would_process = [p for p in pairs if p["mri"]["live_heuristic_orientation"] in ("axial", "coronal", "sagittal")]
    n_agree_ct_proc = sum(1 for p in would_process if p["mri"]["live_heuristic_orientation"] == p["ct"]["geom_plane"])
    proc_ratio = n_agree_ct_proc / max(len(would_process), 1)
    lines.append(
        f"| MRI-derived orientation, borrowed for CT (actual live pipeline behavior, see Stage 2) "
        f"| {len(would_process)}/{len(pairs)} pairs processed "
        f"| {n_agree_ct_proc}/{len(would_process)} processed pairs correctly label the CT slice ({proc_ratio*100:.0f}%) "
        f"| {'USABLE' if proc_ratio >= 0.98 else ('FALLBACK-ONLY' if proc_ratio >= 0.90 else 'UNRELIABLE')} |"
    )

    iop_cov = df["ImageOrientationPatient"].notna().sum()
    lines.append(
        f"| ImageOrientationPatient (geometric plane) | {iop_cov}/{total} | ground truth (self) "
        f"| PRIMARY SOURCE (once oblique tolerance is defined) |"
    )

    bpe_cov = df["BodyPartExamined"].notna().sum()
    lines.append(
        f"| BodyPartExamined (region signal) | {bpe_cov}/{total} | see Stage 3 "
        f"| {'FALLBACK-ONLY (CT sparse)' if bpe_cov/max(total,1) < 0.9 else 'USABLE'} |"
    )

    pid_ct = df[df.modality_folder == "CT"]["PatientID"]
    pid_mr = df[df.modality_folder == "MRI"]["PatientID"]
    lines.append(
        f"| PatientID (cross-modality pairing key) | n/a "
        f"| CT distinct={pid_ct.nunique()}, MRI distinct={pid_mr.nunique()} "
        f"| UNUSABLE (see Stage 4 — not comparable across modality) |"
    )

    lines.append(
        "| PatientName (normalized, cross-modality pairing key) | high | see Stage 4 | "
        "USABLE AS SECONDARY CHECK (needs normalization) |"
    )

    lines.append(
        "| StudyDate (cross-modality pairing key) | high | see Stage 4 | "
        "USABLE AS SECONDARY CHECK |"
    )

    lines.append(
        "\n**Bottom line:** no single tag is reliable alone. The safe chain is: "
        "geometric plane from `ImageOrientationPatient` (primary, needs an oblique-tolerance "
        "policy) -> SeriesDescription keyword match (disambiguation/fallback) -> folder name "
        "(last resort, currently tried *first* in the live code -- reversed priority). "
        "For pairing: folder-prefix match (current) should be corroborated with StudyDate + "
        "normalized PatientName, both effectively free checks that are not currently applied."
    )

    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Harvesting DICOM tags for patients: {SAMPLE_PATIENTS} ...")
    df = harvest()
    if df.empty:
        print("No series found for the given sample. Check DATA_ROOT / SAMPLE_PATIENTS.")
        sys.exit(1)

    csv_path = os.path.join(OUT_DIR, "tag_inventory.csv")
    df.to_csv(csv_path, index=False)
    print(f"Stage 0 done: {len(df)} series harvested -> {csv_path}")

    report_parts = [
        f"# Metadata Reliability Audit\n\n"
        f"Sample: {SAMPLE_LABEL}: {SAMPLE_PATIENTS}\n\n"
        f"Series harvested: {len(df)} "
        f"({(df.modality_folder=='CT').sum()} CT, {(df.modality_folder=='MRI').sum()} MRI)\n"
    ]
    report_parts.append(glossary_section())
    report_parts.append(stage1_coverage(df))
    report_parts.append(stage2_orientation(df))
    region_report, df = stage3_region(df)
    report_parts.append(region_report)
    report_parts.append(stage4_pairing(df))
    report_parts.append(stage5_verdict(df))

    report = "\n".join(report_parts)
    report_path = os.path.join(OUT_DIR, "tag_reliability_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written -> {report_path}")


if __name__ == "__main__":
    main()
