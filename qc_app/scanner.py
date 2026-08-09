"""
scanner.py
──────────
Discover the dataset and build a deterministic manifest of CT/MRI pairs.

Two levels, because the pipeline works at one and the reviewer works at the
other:

    SERIES pair   CT/PAx/ST0/SE0  <->  MRI/PAx/ST0/SE0
                  the unit registration is estimated on

    SLICE pair    index i of that series' CT stack <-> index i of its MRI stack
                  the unit a human accepts or rejects

WHAT THE SLICE INDEX ACTUALLY IS
────────────────────────────────
NOT filename order. `IM10` sorts before `IM2` alphabetically, and neither
alphabetical nor numeric filename order is guaranteed to be the order the
slices sit in space. SimpleITK's GetGDCMSeriesFileNames sorts by
ImagePositionPatient — the geometric order — and that is the order
io_utils.load_dicom_series builds its volume in, so it is the order every array
index in this application refers to. The source filename for each index is
recorded alongside it, which is what lets the UI name the DICOM a reviewer is
looking at without ever re-deriving it from a sort.

SERIES MATCHING
───────────────
The production rule, from preprocess_2d.py:253 — compare `name.split('_')[0]`.
Eight MRI series in this dataset are suffix-named (`SE1_axial`) against a plain
`SE1` on the CT side, and only this rule pairs them. Matching is many-to-one by
construction, so a CT series claimed by two MRI series is reported as ambiguous
rather than silently given to whichever was enumerated first.
"""

import os
import re
from dataclasses import dataclass, field

import SimpleITK as sitk

from . import bootstrap

_pp = bootstrap.preprocessing_modules()
io_utils = _pp.io_utils


@dataclass
class SeriesPair:
    patient: str
    study: str
    ct_series: str            # directory name on the CT side, e.g. "SE0"
    mri_series: str           # directory name on the MRI side, e.g. "SE0_axial"
    ct_path: str
    mri_path: str
    orientation: str
    series_desc: str
    n_ct_files: int
    n_mri_files: int
    ct_files: list = field(default_factory=list)    # geometric order, basenames
    mri_files: list = field(default_factory=list)   # geometric order, basenames

    @property
    def key(self) -> str:
        """Stable identifier, safe as a dict key, a DB key and a path fragment."""
        return f"{self.patient}/{self.study}/{self.ct_series}"

    @property
    def n_pairs(self) -> int:
        """
        How many slice pairs this series will produce.

        The MRI is resampled ONTO the CT grid, so the pair count is the CT's
        slice count whatever the MRI's is. A shorter MRI does not shorten the
        output; it produces zero-filled slices where it does not reach.
        """
        return self.n_ct_files


@dataclass
class ScanReport:
    pairs: list = field(default_factory=list)
    problems: list = field(default_factory=list)

    def add_problem(self, level: str, where: str, message: str) -> None:
        self.problems.append({"level": level, "where": where, "message": message})

    @property
    def n_slice_pairs(self) -> int:
        return sum(p.n_pairs for p in self.pairs)


def _ordered_dicom_files(series_dir: str):
    """
    The series' files in geometric order, plus any problem found reading it.

    Returns (basenames, problem_or_None). Never raises: a malformed series has
    to be reportable, not fatal, or one bad folder stops the whole scan.
    """
    try:
        ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(series_dir)
    except Exception as e:
        return [], f"could not be read as DICOM ({e})"

    if not ids:
        return [], "contains no readable DICOM series"

    problem = None
    if len(ids) > 1:
        # More than one series UID in one folder. GetGDCMSeriesFileNames would
        # quietly return only one of them, so say so instead.
        problem = (f"contains {len(ids)} distinct DICOM series UIDs in one folder; "
                   f"only the first is used")

    try:
        names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(series_dir, ids[0])
    except Exception as e:
        return [], f"file ordering failed ({e})"

    return [os.path.basename(n) for n in names], problem


_IMG_NUM = re.compile(r"(\d+)")


def image_number(filename: str):
    """The IMn number in a DICOM filename, or None."""
    m = _IMG_NUM.search(os.path.splitext(filename)[0])
    return int(m.group(1)) if m else None


def pair_slice_names(ct_names, mri_names):
    """
    Pair CT and MRI slices BY IMAGE NUMBER, not by position in the stack.

    Why not by position: GetGDCMSeriesFileNames sorts by ImagePositionPatient,
    and on this dataset that order does not always agree with the file
    numbering - or between the two modalities. Measured over the 120 series
    here, 24 disagree. Some are merely swapped neighbours
    (CT ... 3, 5, 4 ...), some are scrambled (CT IM0, IM3, IM1, IM2 against
    MRI IM0, IM1, IM2, IM3), and four are fully REVERSED on one side, which
    pairs the top of one stack with the bottom of the other.

    Pairing on the number makes IM7 meet IM7 whatever order the headers put
    them in - which is what the dataset's own convention says, and what a
    reviewer sees when they open the same file in a DICOM browser.

    Falls back to positional pairing when the names carry no usable numbers or
    do not intersect, so a differently-named dataset still works.

    Returns [(ct_name, mri_name), ...] in ascending image-number order.
    """
    ct_by, mri_by = {}, {}
    for n in ct_names:
        k = image_number(n)
        if k is not None:
            ct_by.setdefault(k, n)
    for n in mri_names:
        k = image_number(n)
        if k is not None:
            mri_by.setdefault(k, n)

    common = sorted(set(ct_by) & set(mri_by))
    if not common:
        n = min(len(ct_names), len(mri_names))
        return list(zip(ct_names[:n], mri_names[:n]))
    return [(ct_by[k], mri_by[k]) for k in common]


def _list_dirs(path: str):
    if not os.path.isdir(path):
        return []
    return sorted(d for d in os.listdir(path)
                  if os.path.isdir(os.path.join(path, d)))


def scan_dataset(app_cfg, read_order: bool = True) -> ScanReport:
    """
    Walk both modality trees and pair them.

    `read_order=False` skips the DICOM header pass, which makes the scan
    instant but leaves slice filenames unknown. Used only for a fast structural
    validation of a newly added patient.
    """
    report = ScanReport()
    ct_root, mri_root = app_cfg.ct_root, app_cfg.mri_root

    if not os.path.isdir(ct_root):
        report.add_problem("error", ct_root, "CT root directory not found")
        return report
    if not os.path.isdir(mri_root):
        report.add_problem("error", mri_root, "MRI root directory not found")
        return report

    ct_patients = _list_dirs(ct_root)
    mri_patients = _list_dirs(mri_root)

    for p in sorted(set(ct_patients) - set(mri_patients)):
        report.add_problem("warning", p, "patient has CT but no MRI - skipped")
    for p in sorted(set(mri_patients) - set(ct_patients)):
        report.add_problem("warning", p, "patient has MRI but no CT - skipped")

    for patient in sorted(set(ct_patients) & set(mri_patients)):
        ct_studies = _list_dirs(os.path.join(ct_root, patient))
        mri_studies = _list_dirs(os.path.join(mri_root, patient))

        for study in sorted(set(ct_studies) & set(mri_studies)):
            ct_study_dir = os.path.join(ct_root, patient, study)
            mri_study_dir = os.path.join(mri_root, patient, study)

            ct_series = _list_dirs(ct_study_dir)
            mri_series = _list_dirs(mri_study_dir)

            # Index the CT side by its base name, the production matching key.
            ct_by_base = {}
            for name in ct_series:
                ct_by_base.setdefault(name.split("_")[0], []).append(name)

            claimed = {}
            for m_name in mri_series:
                base = m_name.split("_")[0]
                candidates = ct_by_base.get(base, [])

                if not candidates:
                    report.add_problem(
                        "warning", f"{patient}/{study}/{m_name}",
                        f"MRI series has no CT series with base name '{base}'")
                    continue

                if len(candidates) > 1:
                    report.add_problem(
                        "warning", f"{patient}/{study}/{m_name}",
                        f"base name '{base}' matches {len(candidates)} CT series "
                        f"({', '.join(candidates)}); using '{candidates[0]}'")

                c_name = candidates[0]
                if c_name in claimed:
                    report.add_problem(
                        "warning", f"{patient}/{study}/{m_name}",
                        f"CT series '{c_name}' is already paired with MRI series "
                        f"'{claimed[c_name]}'; this MRI series is skipped as ambiguous")
                    continue
                claimed[c_name] = m_name

                ct_dir = os.path.join(ct_study_dir, c_name)
                mri_dir = os.path.join(mri_study_dir, m_name)

                ct_files, mri_files = [], []
                if read_order:
                    ct_files, ct_problem = _ordered_dicom_files(ct_dir)
                    mri_files, mri_problem = _ordered_dicom_files(mri_dir)
                    # Only worth reporting a soft problem when the series was
                    # otherwise usable - an unreadable folder gets one clear
                    # error below instead of two overlapping messages.
                    if ct_problem and ct_files:
                        report.add_problem("warning", f"CT {patient}/{study}/{c_name}", ct_problem)
                    if mri_problem and mri_files:
                        report.add_problem("warning", f"MRI {patient}/{study}/{m_name}", mri_problem)
                    if not ct_files:
                        report.add_problem("error", f"CT {patient}/{study}/{c_name}",
                                           "no readable DICOM files - series skipped")
                        continue
                    if not mri_files:
                        report.add_problem("error", f"MRI {patient}/{study}/{m_name}",
                                           "no readable DICOM files - series skipped")
                        continue
                else:
                    ct_files = sorted(os.listdir(ct_dir)) if os.path.isdir(ct_dir) else []
                    mri_files = sorted(os.listdir(mri_dir)) if os.path.isdir(mri_dir) else []

                if len(ct_files) < 2:
                    # Matches the production rule: a 1-slice folder is a scout,
                    # not a series, and cannot support a 3D N4 fit.
                    report.add_problem("warning", f"{patient}/{study}/{c_name}",
                                       f"CT series has only {len(ct_files)} slice(s) - skipped")
                    continue
                if len(mri_files) < 2:
                    report.add_problem("warning", f"{patient}/{study}/{m_name}",
                                       f"MRI series has only {len(mri_files)} slice(s) - skipped")
                    continue

                if len(ct_files) != len(mri_files):
                    # Not fatal. The MRI is resampled onto the CT grid, so the
                    # output length is the CT's either way - but a reviewer
                    # should know the stacks were not the same depth.
                    report.add_problem(
                        "info", f"{patient}/{study}/{c_name}",
                        f"CT has {len(ct_files)} slices, MRI has {len(mri_files)}; "
                        f"output follows the CT")

                # Pair by image number here, so ct_files[i] and mri_files[i]
                # are the same slice by the dataset's own naming, whatever
                # order the DICOM headers sort them into.
                paired = pair_slice_names(ct_files, mri_files)
                if len(paired) < len(ct_files):
                    report.add_problem(
                        "info", f"{patient}/{study}/{c_name}",
                        f"{len(ct_files)} CT and {len(mri_files)} MRI slices, "
                        f"{len(paired)} share an image number and were paired")
                ct_files = [c for c, _ in paired]
                mri_files = [m for _, m in paired]
                if not ct_files:
                    report.add_problem("error", f"{patient}/{study}/{c_name}",
                                       "no CT/MRI slices could be paired - series skipped")
                    continue

                orientation, desc = _orientation_of(mri_dir, m_name)

                report.pairs.append(SeriesPair(
                    patient=patient, study=study,
                    ct_series=c_name, mri_series=m_name,
                    ct_path=ct_dir, mri_path=mri_dir,
                    orientation=orientation, series_desc=desc,
                    n_ct_files=len(ct_files), n_mri_files=len(mri_files),
                    ct_files=ct_files, mri_files=mri_files,
                ))

    report.pairs.sort(key=lambda p: (p.patient, p.study, p.ct_series))
    return report


def _orientation_of(mri_dir: str, series_name: str):
    """
    Acquisition plane for this series, by the production pipeline's own rules.

    The MRI dictates the orientation (preprocess_2d.py:257) because it is the
    MRI's mesh that N4 sizes from it. Folder name first, then the DICOM series
    description - the same order io_utils.discover_series uses.
    """
    lowered = series_name.lower()
    for key in ("axial", "coronal", "sagittal"):
        if key in lowered:
            return key, ""

    desc = ""
    try:
        reader = sitk.ImageFileReader()
        names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(mri_dir)
        if names:
            reader.SetFileName(names[0])
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()
            if reader.HasMetaDataKey("0008|103e"):
                desc = reader.GetMetaData("0008|103e").strip()
    except Exception:
        desc = ""

    return (io_utils.get_orientation_from_desc(desc) if desc else "unknown"), desc
