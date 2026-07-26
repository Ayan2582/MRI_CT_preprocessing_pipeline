"""
pipeline_config.py
──────────────────
Default configuration constants for the 2D MRI-CT preprocessing pipeline.
Edit these values or override via CLI arguments in preprocess_2d.py.
"""

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_ROOT  = r"c:\Users\moham\Downloads\MRI-CT preprocessing pipeline\Raw_data_mri_ct\Rawdata_dicom"
OUTPUT_DIR = r"c:\Users\moham\mri_to_ct_preprocessing_example\processed_2d"

# ── Orientations to process ───────────────────────────────────────────────────
# All three orientations will be attempted; a pair is skipped if either modality
# has no series in that orientation for a given patient.
ORIENTATIONS = ["axial", "coronal", "sagittal"]

# ── In-plane resampling ───────────────────────────────────────────────────────
# Both CT and MRI are resampled to this isotropic in-plane resolution (mm).
# Through-plane (slice) spacing is preserved — slices are NOT interpolated.
# CT native: 0.18–0.84 mm  |  MRI native: 0.36–1.56 mm
TARGET_SPACING_MM = 1.0

# ── Output image size ─────────────────────────────────────────────────────────
# Slices are center-cropped or zero-padded to a square of this side length (px).
# At 1.0 mm/px, 256 px covers a 256 mm FOV — sufficient for joint anatomy.
TARGET_SIZE_PX = 256

# ── CT intensity windowing ────────────────────────────────────────────────────
# Pixels are clipped to [WIN_MIN, WIN_MAX] HU, then normalised to [0.0, 1.0].
# Soft-tissue window (-200 to 300 HU) preserves muscle, fat, cartilage, bone.
CT_WINDOW_MIN_HU = -200
CT_WINDOW_MAX_HU =  300

# ── MRI intensity normalisation ───────────────────────────────────────────────
# Percentiles are computed on the non-zero voxels of the full 3D volume (per
# series), then each 2D slice is clipped and rescaled to [0.0, 1.0].
MRI_PERCENTILE_LOW  =  0.5
MRI_PERCENTILE_HIGH = 99.5

# ── Background slice filtering ────────────────────────────────────────────────
# A slice is discarded if the fraction of normalised pixels below
# BG_INTENSITY_THRESH exceeds BG_PIXEL_FRACTION.
BG_INTENSITY_THRESH = 0.02   # normalised intensity threshold  (0–1 scale)
BG_PIXEL_FRACTION   = 0.90   # fraction above which the slice is background

# ── Output options ────────────────────────────────────────────────────────────
SAVE_PNG      = True   # save side-by-side CT|MRI PNG previews for visual QC
SKIP_EXISTING = True   # skip patients whose output folder already exists

# ── Multi-Region Configurations ───────────────────────────────────────────────
REGION_PROFILES = {
    "brain": {
        "target_size": 256,
        "ct_win_min": 0.0,
        "ct_win_max": 80.0
    },
    "abdomen": {
        "target_size": 384,
        "ct_win_min": -160.0,
        "ct_win_max": 240.0
    },
    "musculoskeletal": {
        "target_size": 256,
        "ct_win_min": -200.0,
        "ct_win_max": 300.0
    },
    "spine": {
        "target_size": 256,
        "ct_win_min": -200.0,
        "ct_win_max": 300.0
    },
    "default": {
        "target_size": 256,
        "ct_win_min": -200.0,
        "ct_win_max": 300.0
    }
}

PREFIX_TO_REGION = {
    # Brain (15 patients)
    "PA0": "brain", "PA1": "brain", "PA4": "brain", "PA5": "brain",
    "PA10": "brain", "PA17": "brain",
    "PA19": "brain", "PA21": "brain", "PA24": "brain", "PA26": "brain", "PA28": "brain",
    "PA33": "brain", "PA34": "brain", "PA38": "brain",
    "PA44": "brain",
    # Abdomen / Pelvis / Torso / Chest / Fistulagram (16 patients)
    "PA2": "abdomen", "PA8": "abdomen", "PA9": "abdomen", "PA12": "abdomen", "PA15": "abdomen",
    "PA20": "abdomen", "PA22": "abdomen", "PA25": "abdomen", "PA27": "abdomen", "PA29": "abdomen",
    "PA30": "abdomen", "PA35": "abdomen", "PA37": "abdomen", "PA39": "abdomen", "PA41": "abdomen",
    "PA42": "abdomen",
    # Knee / Joint / Ankle / Shoulder (10 patients)
    "PA3": "musculoskeletal", "PA6": "musculoskeletal", "PA11": "musculoskeletal", "PA13": "musculoskeletal", 
    "PA14": "musculoskeletal", "PA16": "musculoskeletal", "PA32": "musculoskeletal", 
    "PA36": "musculoskeletal", "PA40": "musculoskeletal", "PA43": "musculoskeletal",
    # Spine (4 patients)
    "PA7": "spine", "PA18": "spine", "PA23": "spine", "PA31": "spine"
}
