# Metadata Reliability Audit

Sample: ALL 45 patients: ['PA0', 'PA1', 'PA2', 'PA3', 'PA4', 'PA5', 'PA6', 'PA7', 'PA8', 'PA9', 'PA10', 'PA11', 'PA12', 'PA13', 'PA14', 'PA15', 'PA16', 'PA17', 'PA18', 'PA19', 'PA20', 'PA21', 'PA22', 'PA23', 'PA24', 'PA25', 'PA26', 'PA27', 'PA28', 'PA29', 'PA30', 'PA31', 'PA32', 'PA33', 'PA34', 'PA35', 'PA36', 'PA37', 'PA38', 'PA39', 'PA40', 'PA41', 'PA42', 'PA43', 'PA44']

Series harvested: 248 (124 CT, 124 MRI)

## Glossary — Tag & Column Definitions

### Raw DICOM tags (harvested from the files)

| Tag | Meaning |
|---|---|
| Modality | Acquisition type of this series. 'CT' or 'MR' in this dataset. |
| SeriesDescription | Free-text label the tech/scanner console assigned to the series (e.g. protocol name, plane). Not standardized — wording is scanner/vendor/operator dependent. |
| StudyDescription | Free-text label for the whole study (all series of one visit), e.g. 'medix^brain'. Set by the ordering/report system, not per-series. |
| ProtocolName | Name of the acquisition protocol used, as configured on the scanner (e.g. '1.1 Routine Head 5mm Axial mode'). |
| BodyPartExamined | Coded/free-text body region the operator selected on the scanner console (e.g. 'HEAD', 'PELVIS', 'EXTREMITY'). Optional field per DICOM standard — routinely absent on MR in this dataset. |
| ImageOrientationPatient | Direction cosines (6 numbers: row vector + column vector) describing the 3D orientation of the image plane relative to the patient. This is geometry, not text, and is what Stage 2's ground-truth plane is computed from. |
| PixelSpacing | In-plane pixel size in mm (row spacing, column spacing). |
| SliceThickness | Nominal thickness of each slice in mm, as set on the scanner. |
| SpacingBetweenSlices | Center-to-center distance between adjacent slices in mm (can differ from SliceThickness if slices overlap or have gaps). MR-only field in this dataset. |
| ScanningSequence | MR pulse sequence class (e.g. 'SE'=spin echo, 'IR'=inversion recovery). Not applicable to CT. |
| SequenceVariant | MR sequence variant flags (e.g. fat-saturation, spoiling). Not applicable to CT. |
| ScanOptions | Scan option flags used during acquisition (e.g. 'AXIAL MODE' for CT, 'IR'/'FS' for MR). |
| MRAcquisitionType | MR dimensionality of acquisition, '2D' or '3D'. Not applicable to CT. |
| RepetitionTime | MR pulse sequence TR in ms — time between successive excitation pulses. Not applicable to CT. |
| EchoTime | MR pulse sequence TE in ms — time between excitation and echo readout. Along with TR/sequence flags, is what actually distinguishes T1- vs T2-weighting (not the SeriesDescription string). |
| RescaleSlope | Linear scale factor to convert stored CT pixel values to Hounsfield Units: HU = pixel * RescaleSlope + RescaleIntercept. CT-only field in this dataset. |
| RescaleIntercept | Linear offset to convert stored CT pixel values to Hounsfield Units (see RescaleSlope). CT-only field in this dataset. |
| PatientID | Scanner/PACS-assigned patient identifier for this modality's exam. NOT consistent across CT and MRI in this dataset (see Stage 4/5) — cannot be used to link the two modalities for the same patient. |
| PatientName | Patient name as typed at the console, often with age/sex appended (e.g. 'RANJEET 77Y/M'). Formatting differs slightly between CT and MRI consoles (e.g. '77Y/M' vs '77 Y M'), so needs normalization before cross-modality comparison. |
| StudyDate | Date the study was acquired (YYYYMMDD). Same-day CT and MRI for one patient is a useful independent pairing check. |
| SeriesNumber | Scanner-assigned integer index of the series within the study. Not the same thing as the SE folder name; both are separate numbering schemes. |
| Manufacturer | Scanner manufacturer (e.g. 'GE MEDICAL SYSTEMS', 'SIEMENS'). Explains why CT and MRI tag conventions differ so much in this dataset — they come from two different vendors' consoles. |
| ManufacturerModelName | Specific scanner model (e.g. 'Revolution ACTs', 'Symphony'). |
| FrameOfReferenceUID | Unique ID of the 3D physical coordinate system this series was acquired in. Differs between CT and MRI here (separate scanners, separate exams) — confirms the two volumes do NOT share a coordinate frame and any alignment must be done by the pipeline's own registration step, not assumed from this UID. |
| PatientPosition | Patient positioning on the table (e.g. 'HFS' = head-first-supine). |
| ImageType | Vendor image-processing flags (e.g. 'DERIVED/SECONDARY/REFORMATTED' for CT in this dataset) — signals the CT was reprocessed/reformatted by console software rather than being raw primary acquisition. |
| StudyInstanceUID | Unique ID of the whole study (visit). Differs between the CT study and the MRI study for the same patient, since they are two separate exams/UIDs even on the same day. |

### Derived / computed columns (produced by this audit script, not raw DICOM)

| Column | Meaning |
|---|---|
| modality_folder | Top-level folder this series came from: 'CT' or 'MRI'. Mirrors Modality but is filesystem-derived, independent of the DICOM tag. |
| patient_folder | Full patient folder name, e.g. 'PA17_Mahi' (ID + name as used on disk). |
| patient_prefix | Just the 'PAxx' ID portion of patient_folder, used to look up PREFIX_TO_REGION. |
| series_dir | Series subfolder name, e.g. 'SE0'. This is what the live pipeline's prefix-matching pairing rule keys off. |
| series_path | Full filesystem path to the series folder that was read. |
| n_files | Number of DICOM slice files found in the series folder = slice count (Z-axis) of that series. |
| SliceThickness_varies_in_series | True if SliceThickness differed between the first/middle/last sampled slice of the series (a red flag for inconsistent acquisition). |
| ImageOrientationPatient_varies_in_series | True if the orientation direction cosines differed between the first/middle/last sampled slice (would indicate a non-planar or corrupted series). |
| geom_plane | Ground-truth anatomical plane ('axial'/'coronal'/'sagittal', or 'oblique' if >20° off every axis) computed geometrically from ImageOrientationPatient — independent of any text tag. |
| geom_angle_off_axis_deg | How many degrees the computed slice normal is off the nearest coordinate axis. 0° = perfectly axis-aligned; larger values mean the gantry/patient was tilted. |
| live_heuristic_orientation | Orientation `discover_series()` computes for THIS series in isolation (folder name keyword check, then SeriesDescription keyword check, else 'unknown'). Note: for CT series this value is computed but never actually used by the live pipeline — `preprocess_2d.py` always uses the PAIRED MRI series' value instead (see Stage 2 'pipeline_orientation'). |
| tag_region_guess | Body region guessed by this audit script from keyword-matching BodyPartExamined + StudyDescription + ProtocolName text. |
| config_region | Body region this patient is assigned in the pipeline's hardcoded PREFIX_TO_REGION dict (pipeline_config.py) — the value actually used to pick CT windowing/crop size today. |
## Stage 1 — Tag Coverage (by modality)

| Tag | CT present | CT n | MRI present | MRI n | CT distinct values | MRI distinct values |
|---|---|---|---|---|---|---|
| Modality | 124/124 | 100% | 124/124 | 100% | 1 | 1 |
| SeriesDescription | 124/124 | 100% | 124/124 | 100% | 1 | 28 |
| StudyDescription | 124/124 | 100% | 124/124 | 100% | 6 | 12 |
| ProtocolName | 124/124 | 100% | 124/124 | 100% | 9 | 1 |
| BodyPartExamined | 124/124 | 100% | 0/124 | 0% | 8 | 1 |
| ImageOrientationPatient | 124/124 | 100% | 124/124 | 100% | 50 | 118 |
| PixelSpacing | 124/124 | 100% | 124/124 | 100% | 38 | 24 |
| SliceThickness | 124/124 | 100% | 124/124 | 100% | 41 | 13 |
| SpacingBetweenSlices | 0/124 | 0% | 115/124 | 93% | 1 | 30 |
| ScanningSequence | 0/124 | 0% | 124/124 | 100% | 1 | 3 |
| SequenceVariant | 0/124 | 0% | 124/124 | 100% | 1 | 6 |
| ScanOptions | 124/124 | 100% | 124/124 | 100% | 2 | 7 |
| MRAcquisitionType | 0/124 | 0% | 124/124 | 100% | 1 | 1 |
| RepetitionTime | 0/124 | 0% | 124/124 | 100% | 1 | 28 |
| EchoTime | 0/124 | 0% | 124/124 | 100% | 1 | 18 |
| RescaleSlope | 124/124 | 100% | 0/124 | 0% | 1 | 1 |
| RescaleIntercept | 124/124 | 100% | 0/124 | 0% | 1 | 1 |
| PatientID | 124/124 | 100% | 124/124 | 100% | 6 | 44 |
| PatientName | 124/124 | 100% | 124/124 | 100% | 43 | 44 |
| StudyDate | 124/124 | 100% | 124/124 | 100% | 27 | 27 |
| SeriesNumber | 124/124 | 100% | 124/124 | 100% | 15 | 28 |
| Manufacturer | 124/124 | 100% | 124/124 | 100% | 1 | 1 |
| ManufacturerModelName | 124/124 | 100% | 124/124 | 100% | 1 | 1 |
| FrameOfReferenceUID | 124/124 | 100% | 124/124 | 100% | 45 | 49 |
| PatientPosition | 124/124 | 100% | 124/124 | 100% | 3 | 2 |
| ImageType | 124/124 | 100% | 124/124 | 100% | 3 | 14 |
| StudyInstanceUID | 124/124 | 100% | 124/124 | 100% | 45 | 46 |

## Stage 2 — Orientation: What the Live Pipeline Actually Uses for CT

`preprocess_2d.py:236` sets `orient = m_entry["orientation"]` — the pipeline NEVER reads a CT series' own computed orientation, even though `discover_series()` computes one for every series regardless of modality. Every CT slice that gets processed is labeled with whatever orientation its PAIRED MRI series resolved to. So scoring CT's own heuristic against CT's own geometry (as an earlier version of this report did) measures something the pipeline never actually computes. The question that matches real behavior is: does the borrowed MRI label correctly describe the CT slice it gets applied to?

Per live-pipeline pair (MRI-driven token match, exactly as `preprocess_2d.py` forms it):

| Patient | CT series | MRI series | pipeline_orientation (borrowed from MRI) | CT geom_plane | agrees w/ CT? | MRI geom_plane | agrees w/ MRI? | would pipeline process? |
|---|---|---|---|---|---|---|---|---|
| PA0_Ranjeet | SE0 | SE0 | axial | axial | agree | axial | agree | yes |
| PA0_Ranjeet | SE1 | SE1 | coronal | coronal | agree | coronal | agree | yes |
| PA0_Ranjeet | SE2 | SE2 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA10_Suman | SE0 | SE0 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA10_Suman | SE1 | SE1 | coronal | coronal | agree | coronal | agree | yes |
| PA10_Suman | SE2 | SE2 | axial | axial | agree | axial | agree | yes |
| PA11_Shivam | SE0 | SE0 | axial | axial | agree | axial | agree | yes |
| PA11_Shivam | SE1 | SE1 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA11_Shivam | SE2 | SE2 | coronal | coronal | agree | coronal | agree | yes |
| PA12_Mamta | SE0 | SE0 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA12_Mamta | SE1 | SE1 | coronal | coronal | agree | coronal | agree | yes |
| PA12_Mamta | SE2 | SE2 | axial | axial | agree | axial | agree | yes |
| PA13_Brajesh | SE0 | SE0 | axial | axial | agree | axial | agree | yes |
| PA13_Brajesh | SE1 | SE1 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA13_Brajesh | SE2 | SE2 | coronal | coronal | agree | coronal | agree | yes |
| PA14_Neeraj | SE0 | SE0 | coronal | coronal | agree | coronal | agree | yes |
| PA14_Neeraj | SE1 | SE1 | axial | axial | agree | axial | agree | yes |
| PA14_Neeraj | SE2 | SE2 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA15_SumanLata1 | SE0 | SE0 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA15_SumanLata1 | SE1 | SE1 | coronal | coronal | agree | coronal | agree | yes |
| PA15_SumanLata1 | SE2 | SE2 | axial | axial | agree | axial | agree | yes |
| PA16_SumanLata2 | SE0 | SE0 | axial | axial | agree | axial | agree | yes |
| PA16_SumanLata2 | SE1 | SE1 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA16_SumanLata2 | SE2 | SE2 | coronal | coronal | agree | coronal | agree | yes |
| PA17_Mahi | SE0 | SE0 | axial | axial | agree | axial | agree | yes |
| PA17_Mahi | SE1 | SE1 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA17_Mahi | SE2 | SE2 | coronal | coronal | agree | coronal | agree | yes |
| PA18_Sangeeta | SE0 | SE0 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA18_Sangeeta | SE1 | SE1 | coronal | coronal | agree | coronal | agree | yes |
| PA19_Ram | SE0 | SE0 | axial | oblique | DISAGREE | axial | agree | yes |
| PA19_Ram | SE1 | SE1 | coronal | coronal | agree | coronal | agree | yes |
| PA19_Ram | SE2 | SE2 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA1_Ravi | SE0 | SE0 | axial | axial | agree | axial | agree | yes |
| PA1_Ravi | SE1 | SE1 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA1_Ravi | SE2 | SE2 | coronal | coronal | agree | coronal | agree | yes |
| PA20_Roshani | SE0 | SE0 | coronal | coronal | agree | coronal | agree | yes |
| PA20_Roshani | SE1 | SE1 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA20_Roshani | SE2 | SE2 | axial | axial | agree | axial | agree | yes |
| PA21_Sanjaykumar | SE0 | SE0 | axial | axial | agree | axial | agree | yes |
| PA21_Sanjaykumar | SE1 | SE1 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA21_Sanjaykumar | SE2 | SE2 | coronal | coronal | agree | coronal | agree | yes |
| PA22_Ishant | SE0 | SE0 | coronal | coronal | agree | coronal | agree | yes |
| PA22_Ishant | SE1 | SE1 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA22_Ishant | SE2 | SE2 | axial | axial | agree | axial | agree | yes |
| PA23_SaleemKhan | SE0 | SE0 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA24_Mahboob | SE0 | SE0 | axial | axial | agree | axial | agree | yes |
| PA25_Dharmanand | SE0 | SE0 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA25_Dharmanand | SE1 | SE1 | coronal | coronal | agree | coronal | agree | yes |
| PA25_Dharmanand | SE2 | SE2 | axial | axial | agree | axial | agree | yes |
| PA26_Manoj | SE0 | SE0 | axial | axial | agree | axial | agree | yes |
| PA27_Ashraf | SE0 | SE0 | axial | axial | agree | axial | agree | yes |
| PA28_Maharani | SE0 | SE0 | axial | axial | agree | axial | agree | yes |
| PA28_Maharani | SE1 | SE1 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA28_Maharani | SE2 | SE2 | coronal | coronal | agree | coronal | agree | yes |
| PA29_Sandeep | SE0 | SE0 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA29_Sandeep | SE1 | SE1 | coronal | coronal | agree | coronal | agree | yes |
| PA29_Sandeep | SE2 | SE2 | axial | axial | agree | axial | agree | yes |
| PA2_Neha | SE0 | SE0 | coronal | coronal | agree | coronal | agree | yes |
| PA2_Neha | SE1 | SE1 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA2_Neha | SE2 | SE2 | axial | axial | agree | axial | agree | yes |
| PA30_Raheesh | SE0 | SE0 | coronal | coronal | agree | coronal | agree | yes |
| PA30_Raheesh | SE1 | SE1 | axial | axial | agree | axial | agree | yes |
| PA30_Raheesh | SE2 | SE2 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA31_Nanhe | SE0 | SE0 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA31_Nanhe | SE1 | SE1 | coronal | coronal | agree | coronal | agree | yes |
| PA32_Mandbi_ankle | SE0 | SE0 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA32_Mandbi_ankle | SE1 | SE1 | coronal | coronal | agree | coronal | agree | yes |
| PA32_Mandbi_ankle | SE2 | SE2 | axial | axial | agree | axial | agree | yes |
| PA32_Mandbi_knee | SE3 | SE3 | axial | axial | agree | axial | agree | yes |
| PA32_Mandbi_knee | SE4 | SE4 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA32_Mandbi_knee | SE5 | SE5 | coronal | coronal | agree | coronal | agree | yes |
| PA33_Reshma | SE0 | SE0 | sagittal | coronal | DISAGREE | sagittal | agree | yes |
| PA33_Reshma | SE1 | SE1 | coronal | sagittal | DISAGREE | coronal | agree | yes |
| PA33_Reshma | SE2 | SE2 | axial | axial | agree | axial | agree | yes |
| PA34_Suresh | SE0 | SE0 | axial | axial | agree | axial | agree | yes |
| PA34_Suresh | SE1 | SE1 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA34_Suresh | SE2 | SE2 | coronal | coronal | agree | coronal | agree | yes |
| PA35_Santosh | SE0 | SE0 | coronal | coronal | agree | coronal | agree | yes |
| PA35_Santosh | SE1 | SE1 | axial | axial | agree | axial | agree | yes |
| PA35_Santosh | SE2 | SE2 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA36_Neeraj | SE0 | SE0 | axial | axial | agree | axial | agree | yes |
| PA36_Neeraj | SE1 | SE1 | coronal | coronal | agree | coronal | agree | yes |
| PA36_Neeraj | SE2 | SE2 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA37_Pratik | SE0 | SE0 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA37_Pratik | SE1 | SE1 | coronal | coronal | agree | coronal | agree | yes |
| PA37_Pratik | SE2 | SE2 | axial | axial | agree | axial | agree | yes |
| PA38_Ravi | SE0 | SE0 | axial | axial | agree | axial | agree | yes |
| PA38_Ravi | SE1 | SE1 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA38_Ravi | SE2 | SE2 | coronal | coronal | agree | coronal | agree | yes |
| PA39_Ajay | SE0 | SE0 | coronal | coronal | agree | coronal | agree | yes |
| PA39_Ajay | SE1 | SE1 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA39_Ajay | SE2 | SE2 | axial | axial | agree | axial | agree | yes |
| PA3_Harshit | SE0 | SE0 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA3_Harshit | SE1 | SE1 | axial | axial | agree | axial | agree | yes |
| PA3_Harshit | SE2 | SE2 | coronal | coronal | agree | coronal | agree | yes |
| PA40_Kabir | SE0 | SE0 | axial | axial | agree | axial | agree | yes |
| PA40_Kabir | SE1 | SE1 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA40_Kabir | SE2 | SE2 | coronal | coronal | agree | coronal | agree | yes |
| PA41_Anshika | SE0 | SE0_coronal | coronal | coronal | agree | coronal | agree | yes |
| PA41_Anshika | SE1 | SE1_axial | axial | axial | agree | axial | agree | yes |
| PA42_Poonam | SE0 | SE0_sagittal | sagittal | sagittal | agree | sagittal | agree | yes |
| PA42_Poonam | SE1 | SE1_axial | axial | axial | agree | axial | agree | yes |
| PA42_Poonam | SE2 | SE2_coronal | coronal | coronal | agree | coronal | agree | yes |
| PA43_Chandan | SE0 | SE0_axial | axial | axial | agree | axial | agree | yes |
| PA43_Chandan | SE1 | SE1_sagittal | sagittal | sagittal | agree | sagittal | agree | yes |
| PA43_Chandan | SE2 | SE2_coronal | coronal | coronal | agree | coronal | agree | yes |
| PA44_Zubair | SE0 | SE0_axial | axial | axial | agree | axial | agree | yes |
| PA4_Kanza | SE0 | SE0 | axial | axial | agree | axial | agree | yes |
| PA4_Kanza | SE1 | SE1 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA4_Kanza | SE2 | SE2 | coronal | coronal | agree | coronal | agree | yes |
| PA5_Hritik | SE0 | SE0 | axial | axial | agree | axial | agree | yes |
| PA5_Hritik | SE1 | SE1 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA5_Hritik | SE2 | SE2 | coronal | coronal | agree | coronal | agree | yes |
| PA6_Vijay | SE0 | SE0 | axial | axial | agree | axial | agree | yes |
| PA6_Vijay | SE1 | SE1 | coronal | oblique | DISAGREE | oblique | DISAGREE | yes |
| PA6_Vijay | SE2 | SE2 | sagittal | oblique | DISAGREE | oblique | DISAGREE | yes |
| PA7_Santosh | SE0 | SE0 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA7_Santosh | SE1 | SE1 | coronal | coronal | agree | coronal | agree | yes |
| PA8_Gunjan | SE0 | SE0 | coronal | coronal | agree | coronal | agree | yes |
| PA8_Gunjan | SE1 | SE1 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA8_Gunjan | SE2 | SE2 | axial | axial | agree | axial | agree | yes |
| PA9_Ghausia | SE0 | SE0 | sagittal | sagittal | agree | sagittal | agree | yes |
| PA9_Ghausia | SE1 | SE1 | coronal | coronal | agree | coronal | agree | yes |
| PA9_Ghausia | SE2 | SE2 | axial | axial | agree | axial | agree | yes |

**Live-pipeline pairs found: 124**
- Would actually be processed (borrowed orientation resolves to a valid label): 124/124 (100%)
- Of ALL pairs, borrowed orientation matches CT's OWN geometric plane: 119/124 (96%)
- Of ALL pairs, borrowed orientation matches MRI's OWN geometric plane: 122/124 (98%)
- **Of pairs that WOULD be processed**, borrowed orientation matches CT's OWN geometric plane: 119/124 (96%) — this is the number that matters: it's the real-world accuracy of the CT orientation label actually written into `metadata.csv` and used to route each CT slice into axial/coronal/sagittal output folders.
- Of pairs that WOULD be processed, borrowed orientation matches MRI's OWN geometric plane: 122/124 (98%)

**Unmatched CT series (no MRI partner found by the live rule — never processed, silently dropped): 0**

**Unmatched MRI series (no CT partner found by the live rule — never processed, silently dropped): 0**

## Stage 3 — Region: Tag-Derived Guess vs Hardcoded PREFIX_TO_REGION

| Patient | Modality | BodyPartExamined | StudyDescription | ProtocolName | tag_guess | config_region | agree? |
|---|---|---|---|---|---|---|---|
| PA0_Ranjeet | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA0_Ranjeet | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA0_Ranjeet | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA10_Suman | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA10_Suman | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA10_Suman | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA11_Shivam | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA11_Shivam | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA11_Shivam | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA12_Mamta | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA12_Mamta | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA12_Mamta | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA13_Brajesh | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA13_Brajesh | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA13_Brajesh | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA14_Neeraj | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA14_Neeraj | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA14_Neeraj | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA15_SumanLata1 | CT | PELVIS | e+1 CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA15_SumanLata1 | CT | PELVIS | e+1 CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA15_SumanLata1 | CT | PELVIS | e+1 CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA16_SumanLata2 | CT | EXTREMITY | e+1 CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA16_SumanLata2 | CT | EXTREMITY | e+1 CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA16_SumanLata2 | CT | EXTREMITY | e+1 CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA17_Mahi | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA17_Mahi | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA17_Mahi | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA18_Sangeeta | CT | LSPINE | CT CUT | 7.1 LS -Spine 3D | spine | spine | agree |
| PA18_Sangeeta | CT | LSPINE | CT CUT | 7.1 LS -Spine 3D | spine | spine | agree |
| PA19_Ram | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA19_Ram | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA19_Ram | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA1_Ravi | CT | HEAD | e+1 CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA1_Ravi | CT | HEAD | e+1 CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA1_Ravi | CT | HEAD | e+1 CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA20_Roshani | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA20_Roshani | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA20_Roshani | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA21_Sanjaykumar | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA21_Sanjaykumar | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA21_Sanjaykumar | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA22_Ishant | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA22_Ishant | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA22_Ishant | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA23_SaleemKhan | CT | LSPINE | CT CUT | 7.1 LS -Spine 3D | spine | spine | agree |
| PA24_Mahboob | CT | HEAD | CT CUT HEAD | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA25_Dharmanand | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA25_Dharmanand | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA25_Dharmanand | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA26_Manoj | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA27_Ashraf | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA28_Maharani | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA28_Maharani | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA28_Maharani | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA29_Sandeep | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA29_Sandeep | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA29_Sandeep | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA2_Neha | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA2_Neha | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA2_Neha | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA30_Raheesh | CT | NECK | CT CUT | 3.1 Soft Tissue Neck | unknown | abdomen | DISAGREE |
| PA30_Raheesh | CT | NECK | CT CUT | 3.1 Soft Tissue Neck | unknown | abdomen | DISAGREE |
| PA30_Raheesh | CT | NECK | CT CUT | 3.1 Soft Tissue Neck | unknown | abdomen | DISAGREE |
| PA31_Nanhe | CT | NECK | CT C-SPINE | 3.3 C-Spine | spine | spine | agree |
| PA31_Nanhe | CT | NECK | CT C-SPINE | 3.3 C-Spine | spine | spine | agree |
| PA32_Mandbi_ankle | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA32_Mandbi_ankle | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA32_Mandbi_ankle | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA32_Mandbi_knee | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA32_Mandbi_knee | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA32_Mandbi_knee | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA33_Reshma | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA33_Reshma | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA33_Reshma | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA34_Suresh | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA34_Suresh | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA34_Suresh | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA35_Santosh | CT | ABDOMEN | CT CUT | 6.1 Routine Abdomen Pelvis | abdomen | abdomen | agree |
| PA35_Santosh | CT | ABDOMEN | CT CUT | 6.1 Routine Abdomen Pelvis | abdomen | abdomen | agree |
| PA35_Santosh | CT | ABDOMEN | CT CUT | 6.1 Routine Abdomen Pelvis | abdomen | abdomen | agree |
| PA36_Neeraj | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA36_Neeraj | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA36_Neeraj | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA37_Pratik | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA37_Pratik | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA37_Pratik | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA38_Ravi | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA38_Ravi | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA38_Ravi | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA39_Ajay | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA39_Ajay | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA39_Ajay | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA3_Harshit | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA3_Harshit | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA3_Harshit | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA40_Kabir | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA40_Kabir | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA40_Kabir | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA41_Anshika | CT | CHEST | CT CUT | 5.1 Routine Chest SmartmA | abdomen | abdomen | agree |
| PA41_Anshika | CT | CHEST | CT CUT | 5.1 Routine Chest SmartmA | abdomen | abdomen | agree |
| PA42_Poonam | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA42_Poonam | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA42_Poonam | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA43_Chandan | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA43_Chandan | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA43_Chandan | CT | EXTREMITY | CT CUT | 9.3 Knee/Ankle 1.25mm | musculoskeletal | musculoskeletal | agree |
| PA44_Zubair | CT | HEAD | NCCT HEAD | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA4_Kanza | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA4_Kanza | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA4_Kanza | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA5_Hritik | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA5_Hritik | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA5_Hritik | CT | HEAD | CT CUT | 1.1 Routine Head 5mm Axial mode | brain | brain | agree |
| PA6_Vijay | CT | SHOULDER | CT CUT | 4.1 Shoulder | musculoskeletal | musculoskeletal | agree |
| PA6_Vijay | CT | SHOULDER | CT CUT | 4.1 Shoulder | musculoskeletal | musculoskeletal | agree |
| PA6_Vijay | CT | SHOULDER | CT CUT | 4.1 Shoulder | musculoskeletal | musculoskeletal | agree |
| PA7_Santosh | CT | NECK | CT CV JUCTION C SPINE | 3.3 C-Spine | spine | spine | agree |
| PA7_Santosh | CT | NECK | CT CV JUCTION C SPINE | 3.3 C-Spine | spine | spine | agree |
| PA8_Gunjan | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA8_Gunjan | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA8_Gunjan | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA9_Ghausia | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA9_Ghausia | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA9_Ghausia | CT | PELVIS | CT CUT | 8.1 Routine Pelvis | abdomen | abdomen | agree |
| PA0_Ranjeet | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA0_Ranjeet | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA0_Ranjeet | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA10_Suman | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA10_Suman | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA10_Suman | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA11_Shivam | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA11_Shivam | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA11_Shivam | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA12_Mamta | MRI | None | medix^female pelvis | Image Filter | abdomen | abdomen | agree |
| PA12_Mamta | MRI | None | medix^female pelvis | Image Filter | abdomen | abdomen | agree |
| PA12_Mamta | MRI | None | medix^female pelvis | Image Filter | abdomen | abdomen | agree |
| PA13_Brajesh | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA13_Brajesh | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA13_Brajesh | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA14_Neeraj | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA14_Neeraj | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA14_Neeraj | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA15_SumanLata1 | MRI | None | medix^hip joints | Image Filter | musculoskeletal | abdomen | DISAGREE |
| PA15_SumanLata1 | MRI | None | medix^hip joints | Image Filter | musculoskeletal | abdomen | DISAGREE |
| PA15_SumanLata1 | MRI | None | medix^hip joints | Image Filter | musculoskeletal | abdomen | DISAGREE |
| PA16_SumanLata2 | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA16_SumanLata2 | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA16_SumanLata2 | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA17_Mahi | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA17_Mahi | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA17_Mahi | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA18_Sangeeta | MRI | None | medix^ls spine | Image Filter | spine | spine | agree |
| PA18_Sangeeta | MRI | None | medix^ls spine | Image Filter | spine | spine | agree |
| PA19_Ram | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA19_Ram | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA19_Ram | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA1_Ravi | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA1_Ravi | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA1_Ravi | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA20_Roshani | MRI | None | medix^female pelvis | Image Filter | abdomen | abdomen | agree |
| PA20_Roshani | MRI | None | medix^female pelvis | Image Filter | abdomen | abdomen | agree |
| PA20_Roshani | MRI | None | medix^female pelvis | Image Filter | abdomen | abdomen | agree |
| PA21_Sanjaykumar | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA21_Sanjaykumar | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA21_Sanjaykumar | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA22_Ishant | MRI | None | medix^fistulogram | Image Filter | unknown | abdomen | DISAGREE |
| PA22_Ishant | MRI | None | medix^fistulogram | Image Filter | unknown | abdomen | DISAGREE |
| PA22_Ishant | MRI | None | medix^fistulogram | Image Filter | unknown | abdomen | DISAGREE |
| PA23_SaleemKhan | MRI | None | medix^ls spine | Image Filter | spine | spine | agree |
| PA24_Mahboob | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA25_Dharmanand | MRI | None | medix^fistulogram | Image Filter | unknown | abdomen | DISAGREE |
| PA25_Dharmanand | MRI | None | medix^fistulogram | Image Filter | unknown | abdomen | DISAGREE |
| PA25_Dharmanand | MRI | None | medix^fistulogram | Image Filter | unknown | abdomen | DISAGREE |
| PA26_Manoj | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA27_Ashraf | MRI | None | medix^fistulogram | Image Filter | unknown | abdomen | DISAGREE |
| PA28_Maharani | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA28_Maharani | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA28_Maharani | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA29_Sandeep | MRI | None | medix^fistulogram | Image Filter | unknown | abdomen | DISAGREE |
| PA29_Sandeep | MRI | None | medix^fistulogram | Image Filter | unknown | abdomen | DISAGREE |
| PA29_Sandeep | MRI | None | medix^fistulogram | Image Filter | unknown | abdomen | DISAGREE |
| PA2_Neha | MRI | None | medix^female pelvis | Image Filter | abdomen | abdomen | agree |
| PA2_Neha | MRI | None | medix^female pelvis | Image Filter | abdomen | abdomen | agree |
| PA2_Neha | MRI | None | medix^female pelvis | Image Filter | abdomen | abdomen | agree |
| PA30_Raheesh | MRI | None | medix^neck | Image Filter | unknown | abdomen | DISAGREE |
| PA30_Raheesh | MRI | None | medix^neck | Image Filter | unknown | abdomen | DISAGREE |
| PA30_Raheesh | MRI | None | medix^neck | Image Filter | unknown | abdomen | DISAGREE |
| PA31_Nanhe | MRI | None | medix^c spine | Image Filter | spine | spine | agree |
| PA31_Nanhe | MRI | None | medix^c spine | Image Filter | spine | spine | agree |
| PA32_Mandbi_ankle | MRI | None | medix^ankle | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA32_Mandbi_ankle | MRI | None | medix^ankle | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA32_Mandbi_ankle | MRI | None | medix^ankle | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA32_Mandbi_knee | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA32_Mandbi_knee | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA32_Mandbi_knee | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA33_Reshma | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA33_Reshma | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA33_Reshma | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA34_Suresh | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA34_Suresh | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA34_Suresh | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA35_Santosh | MRI | None | medix^u abdomen | Image Filter | abdomen | abdomen | agree |
| PA35_Santosh | MRI | None | medix^u abdomen | Image Filter | abdomen | abdomen | agree |
| PA35_Santosh | MRI | None | medix^u abdomen | Image Filter | abdomen | abdomen | agree |
| PA36_Neeraj | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA36_Neeraj | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA36_Neeraj | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA37_Pratik | MRI | None | medix^fistulogram | Image Filter | unknown | abdomen | DISAGREE |
| PA37_Pratik | MRI | None | medix^fistulogram | Image Filter | unknown | abdomen | DISAGREE |
| PA37_Pratik | MRI | None | medix^fistulogram | Image Filter | unknown | abdomen | DISAGREE |
| PA38_Ravi | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA38_Ravi | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA38_Ravi | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA39_Ajay | MRI | None | medix^fistulogram | Image Filter | unknown | abdomen | DISAGREE |
| PA39_Ajay | MRI | None | medix^fistulogram | Image Filter | unknown | abdomen | DISAGREE |
| PA39_Ajay | MRI | None | medix^fistulogram | Image Filter | unknown | abdomen | DISAGREE |
| PA3_Harshit | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA3_Harshit | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA3_Harshit | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA40_Kabir | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA40_Kabir | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA40_Kabir | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA41_Anshika | MRI | None | medix^chest | Image Filter | abdomen | abdomen | agree |
| PA41_Anshika | MRI | None | medix^chest | Image Filter | abdomen | abdomen | agree |
| PA42_Poonam | MRI | None | medix^female pelvis | Image Filter | abdomen | abdomen | agree |
| PA42_Poonam | MRI | None | medix^female pelvis | Image Filter | abdomen | abdomen | agree |
| PA42_Poonam | MRI | None | medix^female pelvis | Image Filter | abdomen | abdomen | agree |
| PA43_Chandan | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA43_Chandan | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA43_Chandan | MRI | None | medix^rt&lt knee | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA44_Zubair | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA4_Kanza | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA4_Kanza | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA4_Kanza | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA5_Hritik | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA5_Hritik | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA5_Hritik | MRI | None | medix^brain | Image Filter | brain | brain | agree |
| PA6_Vijay | MRI | None | medix^shoulder | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA6_Vijay | MRI | None | medix^shoulder | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA6_Vijay | MRI | None | medix^shoulder | Image Filter | musculoskeletal | musculoskeletal | agree |
| PA7_Santosh | MRI | None | medix^c spine | Image Filter | spine | spine | agree |
| PA7_Santosh | MRI | None | medix^c spine | Image Filter | spine | spine | agree |
| PA8_Gunjan | MRI | None | medix^female pelvis | Image Filter | abdomen | abdomen | agree |
| PA8_Gunjan | MRI | None | medix^female pelvis | Image Filter | abdomen | abdomen | agree |
| PA8_Gunjan | MRI | None | medix^female pelvis | Image Filter | abdomen | abdomen | agree |
| PA9_Ghausia | MRI | None | medix^female pelvis | Image Filter | abdomen | abdomen | agree |
| PA9_Ghausia | MRI | None | medix^female pelvis | Image Filter | abdomen | abdomen | agree |
| PA9_Ghausia | MRI | None | medix^female pelvis | Image Filter | abdomen | abdomen | agree |

**Per-series agreement: 223/248 (90%)**

Per-patient (union of all series' tag text) verdict:

| Patient | config_region | any series matched? |
|---|---|---|
| PA0_Ranjeet | brain | yes |
| PA10_Suman | brain | yes |
| PA11_Shivam | musculoskeletal | yes |
| PA12_Mamta | abdomen | yes |
| PA13_Brajesh | musculoskeletal | yes |
| PA14_Neeraj | musculoskeletal | yes |
| PA15_SumanLata1 | abdomen | yes |
| PA16_SumanLata2 | musculoskeletal | yes |
| PA17_Mahi | brain | yes |
| PA18_Sangeeta | spine | yes |
| PA19_Ram | brain | yes |
| PA1_Ravi | brain | yes |
| PA20_Roshani | abdomen | yes |
| PA21_Sanjaykumar | brain | yes |
| PA22_Ishant | abdomen | yes |
| PA23_SaleemKhan | spine | yes |
| PA24_Mahboob | brain | yes |
| PA25_Dharmanand | abdomen | yes |
| PA26_Manoj | brain | yes |
| PA27_Ashraf | abdomen | yes |
| PA28_Maharani | brain | yes |
| PA29_Sandeep | abdomen | yes |
| PA2_Neha | abdomen | yes |
| PA30_Raheesh | abdomen | NO |
| PA31_Nanhe | spine | yes |
| PA32_Mandbi_ankle | musculoskeletal | yes |
| PA32_Mandbi_knee | musculoskeletal | yes |
| PA33_Reshma | brain | yes |
| PA34_Suresh | brain | yes |
| PA35_Santosh | abdomen | yes |
| PA36_Neeraj | musculoskeletal | yes |
| PA37_Pratik | abdomen | yes |
| PA38_Ravi | brain | yes |
| PA39_Ajay | abdomen | yes |
| PA3_Harshit | musculoskeletal | yes |
| PA40_Kabir | musculoskeletal | yes |
| PA41_Anshika | abdomen | yes |
| PA42_Poonam | abdomen | yes |
| PA43_Chandan | musculoskeletal | yes |
| PA44_Zubair | brain | yes |
| PA4_Kanza | brain | yes |
| PA5_Hritik | brain | yes |
| PA6_Vijay | musculoskeletal | yes |
| PA7_Santosh | spine | yes |
| PA8_Gunjan | abdomen | yes |
| PA9_Ghausia | abdomen | yes |

**Per-patient (best-of-series) agreement: 45/46 (98%)**

## Stage 4 — Pairing Validation (live MRI-driven token-match rule vs independent evidence)

Live rule (`preprocess_2d.py`), reproduced exactly via `build_live_pairs()` (same function Stage 2 uses): driven by MRI series; both the MRI folder name and each candidate CT folder name are split on `_` and only the first token is compared (so e.g. `SE1_saggital` matches CT `SE1`); first match wins. We check whether matched pairs agree on StudyDate, normalized PatientName, Stage-2 geometric plane, and slice count (`n_files`) — none of these are used by the live pairing rule today, so each is a free independent corroboration (or a red flag) on top of it.

| Patient | CT series | MRI series | CT date | MRI date | date agree | CT name | MRI name | name agree | CT plane | MRI plane | plane agree | CT n_files | MRI n_files | n_files agree |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PA0_Ranjeet | SE0 | SE0 | 20250402 | 20250402 | agree | RANJEET 77Y/M | RANJEET 77 Y M | agree | axial | axial | agree | 18 | 18 | agree |
| PA0_Ranjeet | SE1 | SE1 | 20250402 | 20250402 | agree | RANJEET 77Y/M | RANJEET 77 Y M | agree | coronal | coronal | agree | 18 | 18 | agree |
| PA0_Ranjeet | SE2 | SE2 | 20250402 | 20250402 | agree | RANJEET 77Y/M | RANJEET 77 Y M | agree | sagittal | sagittal | agree | 18 | 18 | agree |
| PA10_Suman | SE0 | SE0 | 20250320 | 20250320 | agree | SUMAN  32Y/F | SUMAN  32Y/F | agree | sagittal | sagittal | agree | 18 | 18 | agree |
| PA10_Suman | SE1 | SE1 | 20250320 | 20250320 | agree | SUMAN  32Y/F | SUMAN  32Y/F | agree | coronal | coronal | agree | 18 | 18 | agree |
| PA10_Suman | SE2 | SE2 | 20250320 | 20250320 | agree | SUMAN  32Y/F | SUMAN  32Y/F | agree | axial | axial | agree | 18 | 18 | agree |
| PA11_Shivam | SE0 | SE0 | 20250320 | 20250320 | agree | SHIVAM MANJU 27Y/M | SHIVAM MANJU 27Y M | agree | axial | axial | agree | 21 | 21 | agree |
| PA11_Shivam | SE1 | SE1 | 20250320 | 20250320 | agree | SHIVAM MANJU 27Y/M | SHIVAM MANJU 27Y M | agree | sagittal | sagittal | agree | 18 | 18 | agree |
| PA11_Shivam | SE2 | SE2 | 20250320 | 20250320 | agree | SHIVAM MANJU 27Y/M | SHIVAM MANJU 27Y M | agree | coronal | coronal | agree | 18 | 18 | agree |
| PA12_Mamta | SE0 | SE0 | 20250321 | 20250321 | agree | MAMTA 48Y/F | MAMTA 48Y/F | agree | sagittal | sagittal | agree | 24 | 24 | agree |
| PA12_Mamta | SE1 | SE1 | 20250321 | 20250321 | agree | MAMTA 48Y/F | MAMTA 48Y/F | agree | coronal | coronal | agree | 24 | 24 | agree |
| PA12_Mamta | SE2 | SE2 | 20250321 | 20250321 | agree | MAMTA 48Y/F | MAMTA 48Y/F | agree | axial | axial | agree | 24 | 24 | agree |
| PA13_Brajesh | SE0 | SE0 | 20250324 | 20250324 | agree | BRAJESH SHARMA 48Y/M | BRAJESH SHARMA 48Y M | agree | axial | axial | agree | 21 | 21 | agree |
| PA13_Brajesh | SE1 | SE1 | 20250324 | 20250324 | agree | BRAJESH SHARMA 48Y/M | BRAJESH SHARMA 48Y M | agree | sagittal | sagittal | agree | 19 | 18 | DISAGREE |
| PA13_Brajesh | SE2 | SE2 | 20250324 | 20250324 | agree | BRAJESH SHARMA 48Y/M | BRAJESH SHARMA 48Y M | agree | coronal | coronal | agree | 17 | 18 | DISAGREE |
| PA14_Neeraj | SE0 | SE0 | 20250307 | 20250326 | DISAGREE | NEERAJ SINGH 35Y/M | NEERAJ SHARMA 31Y/M | DISAGREE | coronal | coronal | agree | 14 | 14 | agree |
| PA14_Neeraj | SE1 | SE1 | 20250307 | 20250326 | DISAGREE | NEERAJ SINGH 35Y/M | NEERAJ SHARMA 31Y/M | DISAGREE | axial | axial | agree | 21 | 21 | agree |
| PA14_Neeraj | SE2 | SE2 | 20250307 | 20250326 | DISAGREE | NEERAJ SINGH 35Y/M | NEERAJ SHARMA 31Y/M | DISAGREE | sagittal | sagittal | agree | 12 | 12 | agree |
| PA15_SumanLata1 | SE0 | SE0 | 20250326 | 20250326 | agree | SUMAN LATA 50Y/F | SUMAN LATA 50Y F | agree | sagittal | sagittal | agree | 15 | 15 | agree |
| PA15_SumanLata1 | SE1 | SE1 | 20250326 | 20250326 | agree | SUMAN LATA 50Y/F | SUMAN LATA 50Y F | agree | coronal | coronal | agree | 24 | 24 | agree |
| PA15_SumanLata1 | SE2 | SE2 | 20250326 | 20250326 | agree | SUMAN LATA 50Y/F | SUMAN LATA 50Y F | agree | axial | axial | agree | 24 | 24 | agree |
| PA16_SumanLata2 | SE0 | SE0 | 20250326 | 20250326 | agree | SUMAN LATA 50Y/F | SUMAN LATA 50Y F | agree | axial | axial | agree | 21 | 21 | agree |
| PA16_SumanLata2 | SE1 | SE1 | 20250326 | 20250326 | agree | SUMAN LATA 50Y/F | SUMAN LATA 50Y F | agree | sagittal | sagittal | agree | 18 | 18 | agree |
| PA16_SumanLata2 | SE2 | SE2 | 20250326 | 20250326 | agree | SUMAN LATA 50Y/F | SUMAN LATA 50Y F | agree | coronal | coronal | agree | 18 | 18 | agree |
| PA17_Mahi | SE0 | SE0 | 20250328 | 20250328 | agree | MAHI 17Y/F | MAHI  17Y/F | agree | axial | axial | agree | 18 | 18 | agree |
| PA17_Mahi | SE1 | SE1 | 20250328 | 20250328 | agree | MAHI 17Y/F | MAHI  17Y/F | agree | sagittal | sagittal | agree | 18 | 18 | agree |
| PA17_Mahi | SE2 | SE2 | 20250328 | 20250328 | agree | MAHI 17Y/F | MAHI  17Y/F | agree | coronal | coronal | agree | 18 | 18 | agree |
| PA18_Sangeeta | SE0 | SE0 | 20250328 | 20250328 | agree | SANGEETA CHAUHAN 40Y/F | SANGEETA DEVI  40Y/F | DISAGREE | sagittal | sagittal | agree | 9 | 9 | agree |
| PA18_Sangeeta | SE1 | SE1 | 20250328 | 20250328 | agree | SANGEETA CHAUHAN 40Y/F | SANGEETA DEVI  40Y/F | DISAGREE | coronal | coronal | agree | 10 | 10 | agree |
| PA19_Ram | SE0 | SE0 | 20250329 | 20250329 | agree | RAM SAHAY 54Y/M | RAM SAHAY  54Y/M | agree | oblique | axial | DISAGREE | 18 | 18 | agree |
| PA19_Ram | SE1 | SE1 | 20250329 | 20250329 | agree | RAM SAHAY 54Y/M | RAM SAHAY  54Y/M | agree | coronal | coronal | agree | 18 | 18 | agree |
| PA19_Ram | SE2 | SE2 | 20250329 | 20250329 | agree | RAM SAHAY 54Y/M | RAM SAHAY  54Y/M | agree | sagittal | sagittal | agree | 18 | 18 | agree |
| PA1_Ravi | SE0 | SE0 | 20250402 | 20250402 | agree | RAVI KUMAR 30Y/M | RAVI KUMAR 30Y M | agree | axial | axial | agree | 17 | 17 | agree |
| PA1_Ravi | SE1 | SE1 | 20250402 | 20250402 | agree | RAVI KUMAR 30Y/M | RAVI KUMAR 30Y M | agree | sagittal | sagittal | agree | 18 | 18 | agree |
| PA1_Ravi | SE2 | SE2 | 20250402 | 20250402 | agree | RAVI KUMAR 30Y/M | RAVI KUMAR 30Y M | agree | coronal | coronal | agree | 18 | 18 | agree |
| PA20_Roshani | SE0 | SE0 | 20250330 | 20250330 | agree | ROSHANI KHATUN  33Y/F | ROSHANI KHATUN  33Y/F | agree | coronal | coronal | agree | 24 | 24 | agree |
| PA20_Roshani | SE1 | SE1 | 20250330 | 20250330 | agree | ROSHANI KHATUN  33Y/F | ROSHANI KHATUN  33Y/F | agree | sagittal | sagittal | agree | 24 | 24 | agree |
| PA20_Roshani | SE2 | SE2 | 20250330 | 20250330 | agree | ROSHANI KHATUN  33Y/F | ROSHANI KHATUN  33Y/F | agree | axial | axial | agree | 24 | 24 | agree |
| PA21_Sanjaykumar | SE0 | SE0 | 20250331 | 20250331 | agree | SANJAY KUMAR 48Y/M | SANJAY KUMAR 48Y M | agree | axial | axial | agree | 18 | 18 | agree |
| PA21_Sanjaykumar | SE1 | SE1 | 20250331 | 20250331 | agree | SANJAY KUMAR 48Y/M | SANJAY KUMAR 48Y M | agree | sagittal | sagittal | agree | 18 | 18 | agree |
| PA21_Sanjaykumar | SE2 | SE2 | 20250331 | 20250331 | agree | SANJAY KUMAR 48Y/M | SANJAY KUMAR 48Y M | agree | coronal | coronal | agree | 18 | 18 | agree |
| PA22_Ishant | SE0 | SE0 | 20250331 | 20250331 | agree | ISHANT 26Y/M | ISHANT 26Y M | agree | coronal | coronal | agree | 21 | 21 | agree |
| PA22_Ishant | SE1 | SE1 | 20250331 | 20250331 | agree | ISHANT 26Y/M | ISHANT 26Y M | agree | sagittal | sagittal | agree | 23 | 23 | agree |
| PA22_Ishant | SE2 | SE2 | 20250331 | 20250331 | agree | ISHANT 26Y/M | ISHANT 26Y M | agree | axial | axial | agree | 23 | 23 | agree |
| PA23_SaleemKhan | SE0 | SE0 | 20250227 | 20250227 | agree | SALIM KHAN  34Y/M | SALIM KHAN  34Y/M | agree | sagittal | sagittal | agree | 9 | 9 | agree |
| PA24_Mahboob | SE0 | SE0 | 20250227 | 20250227 | agree | MAHBOOB 20Y M | MAHBOOB 20Y M | agree | axial | axial | agree | 16 | 16 | agree |
| PA25_Dharmanand | SE0 | SE0 | 20250228 | 20250228 | agree | DHARMANAND MISHRA 53YM | DHARMANAND MISHRA 53Y/M | agree | sagittal | sagittal | agree | 24 | 24 | agree |
| PA25_Dharmanand | SE1 | SE1 | 20250228 | 20250228 | agree | DHARMANAND MISHRA 53YM | DHARMANAND MISHRA 53Y/M | agree | coronal | coronal | agree | 24 | 24 | agree |
| PA25_Dharmanand | SE2 | SE2 | 20250228 | 20250228 | agree | DHARMANAND MISHRA 53YM | DHARMANAND MISHRA 53Y/M | agree | axial | axial | agree | 24 | 24 | agree |
| PA26_Manoj | SE0 | SE0 | 20250228 | 20250228 | agree | MANOJ KUMAR  55Y/M | MANOJ KUMAR 55Y M | agree | axial | axial | agree | 18 | 18 | agree |
| PA27_Ashraf | SE0 | SE0 | 20250228 | 20250228 | agree | ASHRAF HALEM  38YM | ASHRAF HALIM 38Y M | DISAGREE | axial | axial | agree | 22 | 22 | agree |
| PA28_Maharani | SE0 | SE0 | 20250310 | 20250310 | agree | MAHARANI DEVI  69Y/F | MAHARANI DEVI 69Y F | agree | axial | axial | agree | 18 | 18 | agree |
| PA28_Maharani | SE1 | SE1 | 20250310 | 20250310 | agree | MAHARANI DEVI  69Y/F | MAHARANI DEVI 69Y F | agree | sagittal | sagittal | agree | 18 | 18 | agree |
| PA28_Maharani | SE2 | SE2 | 20250310 | 20250310 | agree | MAHARANI DEVI  69Y/F | MAHARANI DEVI 69Y F | agree | coronal | coronal | agree | 18 | 18 | agree |
| PA29_Sandeep | SE0 | SE0 | 20250301 | 20250301 | agree | SANDEEP SINGH  34Y/M | SANDEEP SINGH 34Y M | agree | sagittal | sagittal | agree | 24 | 24 | agree |
| PA29_Sandeep | SE1 | SE1 | 20250301 | 20250301 | agree | SANDEEP SINGH  34Y/M | SANDEEP SINGH 34Y M | agree | coronal | coronal | agree | 24 | 24 | agree |
| PA29_Sandeep | SE2 | SE2 | 20250301 | 20250301 | agree | SANDEEP SINGH  34Y/M | SANDEEP SINGH 34Y M | agree | axial | axial | agree | 24 | 24 | agree |
| PA2_Neha | SE0 | SE0 | 20250313 | 20250313 | agree | NEHA KUMARI 24Y/F | NEHA KUMARI  24Y/F | agree | coronal | coronal | agree | 24 | 24 | agree |
| PA2_Neha | SE1 | SE1 | 20250313 | 20250313 | agree | NEHA KUMARI 24Y/F | NEHA KUMARI  24Y/F | agree | sagittal | sagittal | agree | 24 | 24 | agree |
| PA2_Neha | SE2 | SE2 | 20250313 | 20250313 | agree | NEHA KUMARI 24Y/F | NEHA KUMARI  24Y/F | agree | axial | axial | agree | 24 | 24 | agree |
| PA30_Raheesh | SE0 | SE0 | 20250310 | 20250310 | agree | MOHD RAHEESH  45Y/M | MOHD. RAHEESH 45Y M | agree | coronal | coronal | agree | 24 | 24 | agree |
| PA30_Raheesh | SE1 | SE1 | 20250310 | 20250310 | agree | MOHD RAHEESH  45Y/M | MOHD. RAHEESH 45Y M | agree | axial | axial | agree | 21 | 21 | agree |
| PA30_Raheesh | SE2 | SE2 | 20250310 | 20250310 | agree | MOHD RAHEESH  45Y/M | MOHD. RAHEESH 45Y M | agree | sagittal | sagittal | agree | 24 | 24 | agree |
| PA31_Nanhe | SE0 | SE0 | 20250302 | 20250302 | agree | NANHE KHAN 78Y M | NANHE KHAN  78Y/M | agree | sagittal | sagittal | agree | 9 | 9 | agree |
| PA31_Nanhe | SE1 | SE1 | 20250302 | 20250302 | agree | NANHE KHAN 78Y M | NANHE KHAN  78Y/M | agree | coronal | coronal | agree | 10 | 10 | agree |
| PA32_Mandbi_ankle | SE0 | SE0 | 20250305 | 20250305 | agree | MANDBI SINGH  48Y/F | MANDBI SINGH 48Y F | agree | sagittal | sagittal | agree | 15 | 15 | agree |
| PA32_Mandbi_ankle | SE1 | SE1 | 20250305 | 20250305 | agree | MANDBI SINGH  48Y/F | MANDBI SINGH 48Y F | agree | coronal | coronal | agree | 24 | 24 | agree |
| PA32_Mandbi_ankle | SE2 | SE2 | 20250305 | 20250305 | agree | MANDBI SINGH  48Y/F | MANDBI SINGH 48Y F | agree | axial | axial | agree | 23 | 23 | agree |
| PA32_Mandbi_knee | SE3 | SE3 | 20250305 | 20250305 | agree | MANDBI SINGH  48Y/F | MANDBI SINGH 48Y F | agree | axial | axial | agree | 21 | 21 | agree |
| PA32_Mandbi_knee | SE4 | SE4 | 20250305 | 20250305 | agree | MANDBI SINGH  48Y/F | MANDBI SINGH 48Y F | agree | sagittal | sagittal | agree | 18 | 18 | agree |
| PA32_Mandbi_knee | SE5 | SE5 | 20250305 | 20250305 | agree | MANDBI SINGH  48Y/F | MANDBI SINGH 48Y F | agree | coronal | coronal | agree | 18 | 18 | agree |
| PA33_Reshma | SE0 | SE0 | 20250306 | 20250306 | agree | RESHMA 50Y/F | RESHMA 50Y F | agree | coronal | sagittal | DISAGREE | 18 | 18 | agree |
| PA33_Reshma | SE1 | SE1 | 20250306 | 20250306 | agree | RESHMA 50Y/F | RESHMA 50Y F | agree | sagittal | coronal | DISAGREE | 18 | 18 | agree |
| PA33_Reshma | SE2 | SE2 | 20250306 | 20250306 | agree | RESHMA 50Y/F | RESHMA 50Y F | agree | axial | axial | agree | 18 | 18 | agree |
| PA34_Suresh | SE0 | SE0 | 20250306 | 20250306 | agree | SURESH KUMAR 44Y/M | SURESH KUMAR 44Y M | agree | axial | axial | agree | 18 | 18 | agree |
| PA34_Suresh | SE1 | SE1 | 20250306 | 20250306 | agree | SURESH KUMAR 44Y/M | SURESH KUMAR 44Y M | agree | sagittal | sagittal | agree | 18 | 18 | agree |
| PA34_Suresh | SE2 | SE2 | 20250306 | 20250306 | agree | SURESH KUMAR 44Y/M | SURESH KUMAR 44Y M | agree | coronal | coronal | agree | 18 | 18 | agree |
| PA35_Santosh | SE0 | SE0 | 20250306 | 20250306 | agree | SANTOSH   50Y/F | SANTOSH  50Y/F | agree | coronal | coronal | agree | 24 | 24 | agree |
| PA35_Santosh | SE1 | SE1 | 20250306 | 20250306 | agree | SANTOSH   50Y/F | SANTOSH  50Y/F | agree | axial | axial | agree | 24 | 24 | agree |
| PA35_Santosh | SE2 | SE2 | 20250306 | 20250306 | agree | SANTOSH   50Y/F | SANTOSH  50Y/F | agree | sagittal | sagittal | agree | 24 | 24 | agree |
| PA36_Neeraj | SE0 | SE0 | 20250307 | 20250307 | agree | NEERAJ SINGH 35Y/M | NEERAJ SINGH  35Y/M | agree | axial | axial | agree | 21 | 21 | agree |
| PA36_Neeraj | SE1 | SE1 | 20250307 | 20250307 | agree | NEERAJ SINGH 35Y/M | NEERAJ SINGH  35Y/M | agree | coronal | coronal | agree | 18 | 18 | agree |
| PA36_Neeraj | SE2 | SE2 | 20250307 | 20250307 | agree | NEERAJ SINGH 35Y/M | NEERAJ SINGH  35Y/M | agree | sagittal | sagittal | agree | 18 | 18 | agree |
| PA37_Pratik | SE0 | SE0 | 20250307 | 20250307 | agree | PRATEEK KUMAR 30Y/M | PRATIK KUMAR  30Y/M | DISAGREE | sagittal | sagittal | agree | 24 | 24 | agree |
| PA37_Pratik | SE1 | SE1 | 20250307 | 20250307 | agree | PRATEEK KUMAR 30Y/M | PRATIK KUMAR  30Y/M | DISAGREE | coronal | coronal | agree | 24 | 24 | agree |
| PA37_Pratik | SE2 | SE2 | 20250307 | 20250307 | agree | PRATEEK KUMAR 30Y/M | PRATIK KUMAR  30Y/M | DISAGREE | axial | axial | agree | 23 | 23 | agree |
| PA38_Ravi | SE0 | SE0 | 20250307 | 20250307 | agree | RAVI SINGH 51Y/M | RAVI SINGH 51Y M | agree | axial | axial | agree | 18 | 18 | agree |
| PA38_Ravi | SE1 | SE1 | 20250307 | 20250307 | agree | RAVI SINGH 51Y/M | RAVI SINGH 51Y M | agree | sagittal | sagittal | agree | 18 | 18 | agree |
| PA38_Ravi | SE2 | SE2 | 20250307 | 20250307 | agree | RAVI SINGH 51Y/M | RAVI SINGH 51Y M | agree | coronal | coronal | agree | 18 | 18 | agree |
| PA39_Ajay | SE0 | SE0 | 20250309 | 20250309 | agree | AJAY SINGH  26Y/M | AJAY SINGH 26Y/M | agree | coronal | coronal | agree | 24 | 24 | agree |
| PA39_Ajay | SE1 | SE1 | 20250309 | 20250309 | agree | AJAY SINGH  26Y/M | AJAY SINGH 26Y/M | agree | sagittal | sagittal | agree | 24 | 24 | agree |
| PA39_Ajay | SE2 | SE2 | 20250309 | 20250309 | agree | AJAY SINGH  26Y/M | AJAY SINGH 26Y/M | agree | axial | axial | agree | 24 | 24 | agree |
| PA3_Harshit | SE0 | SE0 | 20250316 | 20250316 | agree | HARSHIT 22Y/M | HARSHIT  22Y/M | agree | sagittal | sagittal | agree | 18 | 18 | agree |
| PA3_Harshit | SE1 | SE1 | 20250316 | 20250316 | agree | HARSHIT 22Y/M | HARSHIT  22Y/M | agree | axial | axial | agree | 21 | 21 | agree |
| PA3_Harshit | SE2 | SE2 | 20250316 | 20250316 | agree | HARSHIT 22Y/M | HARSHIT  22Y/M | agree | coronal | coronal | agree | 18 | 18 | agree |
| PA40_Kabir | SE0 | SE0 | 20250312 | 20250312 | agree | KABIR 35Y/M | KABIR 35Y/M | agree | axial | axial | agree | 21 | 21 | agree |
| PA40_Kabir | SE1 | SE1 | 20250312 | 20250312 | agree | KABIR 35Y/M | KABIR 35Y/M | agree | sagittal | sagittal | agree | 18 | 18 | agree |
| PA40_Kabir | SE2 | SE2 | 20250312 | 20250312 | agree | KABIR 35Y/M | KABIR 35Y/M | agree | coronal | coronal | agree | 18 | 18 | agree |
| PA41_Anshika | SE0 | SE0_coronal | 20250114 | 20250114 | agree | ANSHIKA VISHWAKRMA 25Y F | ANSHIKA VISHWAKRMA 25Y F | agree | coronal | coronal | agree | 6 | 6 | agree |
| PA41_Anshika | SE1 | SE1_axial | 20250114 | 20250114 | agree | ANSHIKA VISHWAKRMA 25Y F | ANSHIKA VISHWAKRMA 25Y F | agree | axial | axial | agree | 17 | 17 | agree |
| PA42_Poonam | SE0 | SE0_sagittal | 20250211 | 20250211 | agree | POONAM | POONAM  25Y/F | DISAGREE | sagittal | sagittal | agree | 16 | 16 | agree |
| PA42_Poonam | SE1 | SE1_axial | 20250211 | 20250211 | agree | POONAM | POONAM  25Y/F | DISAGREE | axial | axial | agree | 18 | 18 | agree |
| PA42_Poonam | SE2 | SE2_coronal | 20250211 | 20250211 | agree | POONAM | POONAM  25Y/F | DISAGREE | coronal | coronal | agree | 16 | 16 | agree |
| PA43_Chandan | SE0 | SE0_axial | 20250211 | 20250211 | agree | CHANDAN  36Y/M | CHANDAN 36Y M | agree | axial | axial | agree | 13 | 13 | agree |
| PA43_Chandan | SE1 | SE1_sagittal | 20250211 | 20250211 | agree | CHANDAN  36Y/M | CHANDAN 36Y M | agree | sagittal | sagittal | agree | 16 | 16 | agree |
| PA43_Chandan | SE2 | SE2_coronal | 20250211 | 20250211 | agree | CHANDAN  36Y/M | CHANDAN 36Y M | agree | coronal | coronal | agree | 15 | 15 | agree |
| PA44_Zubair | SE0 | SE0_axial | 20250210 | 20250210 | agree | ZUBAIR ANSARI  34Y/M | ZUBAIR ANSARI 34Y/M | agree | axial | axial | agree | 14 | 14 | agree |
| PA4_Kanza | SE0 | SE0 | 20250317 | 20250317 | agree | KANZADANISH 22Y/F | KANAZA DANISH  22Y/F | DISAGREE | axial | axial | agree | 18 | 18 | agree |
| PA4_Kanza | SE1 | SE1 | 20250317 | 20250317 | agree | KANZADANISH 22Y/F | KANAZA DANISH  22Y/F | DISAGREE | sagittal | sagittal | agree | 18 | 18 | agree |
| PA4_Kanza | SE2 | SE2 | 20250317 | 20250317 | agree | KANZADANISH 22Y/F | KANAZA DANISH  22Y/F | DISAGREE | coronal | coronal | agree | 18 | 18 | agree |
| PA5_Hritik | SE0 | SE0 | 20250318 | 20250318 | agree | HRITIK ROSHAN  21Y/M | HRITIK ROSHAN 21Y/M | agree | axial | axial | agree | 18 | 18 | agree |
| PA5_Hritik | SE1 | SE1 | 20250318 | 20250318 | agree | HRITIK ROSHAN  21Y/M | HRITIK ROSHAN 21Y/M | agree | sagittal | sagittal | agree | 18 | 18 | agree |
| PA5_Hritik | SE2 | SE2 | 20250318 | 20250318 | agree | HRITIK ROSHAN  21Y/M | HRITIK ROSHAN 21Y/M | agree | coronal | coronal | agree | 18 | 18 | agree |
| PA6_Vijay | SE0 | SE0 | 20250318 | 20250318 | agree | VIJAY PAL 50Y/M | VIJAY PAL  50Y/M | agree | axial | axial | agree | 18 | 18 | agree |
| PA6_Vijay | SE1 | SE1 | 20250318 | 20250318 | agree | VIJAY PAL 50Y/M | VIJAY PAL  50Y/M | agree | oblique | oblique | agree | 18 | 18 | agree |
| PA6_Vijay | SE2 | SE2 | 20250318 | 20250318 | agree | VIJAY PAL 50Y/M | VIJAY PAL  50Y/M | agree | oblique | oblique | agree | 18 | 18 | agree |
| PA7_Santosh | SE0 | SE0 | 20250318 | 20250318 | agree | SANTOSH DAS 45Y/M | SANTOSH DAS 45Y/M | agree | sagittal | sagittal | agree | 9 | 9 | agree |
| PA7_Santosh | SE1 | SE1 | 20250318 | 20250318 | agree | SANTOSH DAS 45Y/M | SANTOSH DAS 45Y/M | agree | coronal | coronal | agree | 10 | 10 | agree |
| PA8_Gunjan | SE0 | SE0 | 20250319 | 20250319 | agree | GUNJAN TIWARI 20Y/F | GUNJAN TIWARI 20Y F | agree | coronal | coronal | agree | 24 | 24 | agree |
| PA8_Gunjan | SE1 | SE1 | 20250319 | 20250319 | agree | GUNJAN TIWARI 20Y/F | GUNJAN TIWARI 20Y F | agree | sagittal | sagittal | agree | 24 | 24 | agree |
| PA8_Gunjan | SE2 | SE2 | 20250319 | 20250319 | agree | GUNJAN TIWARI 20Y/F | GUNJAN TIWARI 20Y F | agree | axial | axial | agree | 24 | 24 | agree |
| PA9_Ghausia | SE0 | SE0 | 20250319 | 20250319 | agree | GHAUSIA SABA 34Y/F | GHAUSIA SABA 34Y F | agree | sagittal | sagittal | agree | 24 | 24 | agree |
| PA9_Ghausia | SE1 | SE1 | 20250319 | 20250319 | agree | GHAUSIA SABA 34Y/F | GHAUSIA SABA 34Y F | agree | coronal | coronal | agree | 24 | 24 | agree |
| PA9_Ghausia | SE2 | SE2 | 20250319 | 20250319 | agree | GHAUSIA SABA 34Y/F | GHAUSIA SABA 34Y F | agree | axial | axial | agree | 24 | 24 | agree |

**Live-pipeline-matched pairs found: 124**
- Date agreement: 121/124 (98%)
- Name agreement: 109/124 (88%)
- Geometric-plane agreement: 121/124 (98%)
- Slice-count (n_files) agreement: 122/124 (98%)

## Stage 5 — Verdict


**Bottom line:** There's an issue with PA_13 vijay unequal number of slices in coronal and sagittal of ct amd mri data 
The method to determine ct and mri orientation using mri is working perfectly  , one patient PA_6 has oblique data , means that the direction cosine is messed up and doesnt belong to a single category like axial , coronal and sagittal

Perfect Direction Cosines for:
Axial :    [1, 0, 0, 0, 1, 0]
Coronal: [1, 0, 0, 0, 0, -1]
Sagittal:[0, 1, 0, 0, 0, -1]