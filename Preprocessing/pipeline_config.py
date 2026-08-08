"""
pipeline_config.py
──────────────────
Default configuration constants for the 2D MRI-CT preprocessing pipeline.
Edit these values or override via CLI arguments in preprocess_2d.py.
"""

# ── Data paths ────────────────────────────────────────────────────────────────
# The authoritative copy is the one inside the repository. Resolved relative to
# this file so it does not depend on where the process was started, and so it
# cannot silently drift back to a copy somewhere else on the machine.
#
# There IS another copy at
#     c:\Users\moham\Downloads\MRI-CT preprocessing pipeline\Raw_data_mri_ct\Rawdata_dicom
# and it is NOT the same data: 242 series / 4666 files against 240 / 4626 here.
# Everything registration-related run before 2026-08-08 used that one. Do not
# point DATA_ROOT back at it without re-running the sweeps.
import os as _os

DATA_ROOT  = _os.path.abspath(_os.path.join(
    _os.path.dirname(__file__), "..", "Raw_data_mri_ct", "Rawdata_dicom"))
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
# There is none. This pipeline does not crop, pad, or resize — slices are saved at
# their native post-resample size, and the GAN's dataloader owns the crop decision.
# See Preprocessing/docs/ct_pipeline_docs.md §10 for the per-region sizes that were
# previously used (256 px for brain/MSK/spine, 384 px for abdomen) and why.

# ── N4 bias field correction (MRI only) ───────────────────────────────────────
# N4 is fitted to the WHOLE 3D VOLUME, not slice by slice. A receive-coil bias
# field is a property of the coil in the scanner bore; it does not restart at
# every slice boundary. Fitting one field per slice lets each slice choose its
# own brightness, which manufactures slice-to-slice steps that were not in the
# data and destroys the very consistency the GAN needs across a stack.
#
# This is safe to do here even though docs/data_known_issues_docs.md §3 shows
# the z axis of these volumes is unreliable — that issue is CT-only. All eleven
# MRI series measured there have exactly uniform slice spacing, and N4 only ever
# touches MRI.
#
# ── How many control points ───────────────────────────────────────────────────
# The bias field is a cubic B-spline. Its stiffness is set by how far apart the
# control points are, and the danger is entirely one-sided: too FEW control
# points leaves some shading behind, too MANY lets the "bias field" bend tightly
# enough to follow real anatomy and flatten away genuine tissue contrast.
#
# Control point counts are therefore NOT fixed numbers. The MRI series in this
# dataset span 180 mm (knee sagittal) to 400 mm (abdomen axial) of in-plane
# field of view, so a fixed count would mean a 15 mm mesh on the knee and a
# 33 mm mesh on the abdomen — two completely different amounts of freedom. What
# is held fixed is the CONTROL POINT SPACING IN MILLIMETRES, and the count is
# derived per series from its actual FOV.
#
# For a spline of order p, `ncp` control points divide an axis into `ncp - p`
# spans, so:      ncp = p + round(FOV_mm / target_spacing_mm)
N4_SHRINK_FACTOR   = 4       # in-plane only — see image_processing.py
N4_CONVERGENCE     = 0.001
N4_SPLINE_ORDER    = 3       # cubic
N4_FITTING_LEVELS  = 1       # see note below before changing this
N4_ITERATIONS      = 100     # per fitting level

# Why a SINGLE fitting level: ITK doubles the control point mesh on every extra
# fitting level, in all three axes at once. With 4 levels the (4,4,4) default
# becomes an (11,11,11) mesh — and there is then no way to keep the through-plane
# axis coarse while refining in-plane, because the doubling is not per-axis. One
# level makes the numbers below mean exactly what they say. If you do raise it,
# apply_n4_bias_correction back-solves the initial mesh so the FINAL mesh still
# lands on these targets, and logs what it actually achieved.

# Target control point spacing in mm, per orientation, for the two IN-PLANE axes
# in SimpleITK index order (axis0, axis1). Which anatomical direction each axis
# is depends on the acquisition plane, which is the whole reason these differ:
#
#   orientation   axis0    axis1    axis2 (through-plane)
#   axial         L-R      A-P      S-I
#   coronal       L-R      S-I      A-P
#   sagittal      A-P      S-I      L-R
#
# Chosen so that each anatomical direction gets the same stiffness whichever
# plane it happens to show up in:
#   L-R  35 mm — body/spine coil shading across the patient is broad and roughly
#                symmetric, so it needs the least freedom.
#   A-P  30 mm — anterior array against posterior spine coil is the strongest
#                single gradient in most of these scans.
#   S-I  25 mm — coil arrays are segmented along the bore axis, so sensitivity
#                changes fastest head-to-foot; this axis needs the most freedom.
#
# These are starting values reasoned from coil geometry, not values tuned
# against a measured criterion on this dataset. They are the right knob to turn
# first if N4 is visibly eating anatomy (raise them) or leaving shading behind
# (lower them).
N4_CONTROL_POINT_SPACING_MM = {
    #             axis0   axis1
    "axial":    ( 35.0,  30.0),   # L-R, A-P
    "coronal":  ( 35.0,  25.0),   # L-R, S-I
    "sagittal": ( 30.0,  25.0),   # A-P, S-I
    "default":  ( 35.0,  30.0),
}

# Bounds on the derived in-plane counts, so a freak FOV or a bad PixelSpacing
# tag cannot produce an absurd mesh. Lower bound is also the hard floor: a
# spline of order p needs at least p+1 control points to exist at all.
N4_CONTROL_POINTS_INPLANE_MIN = 6
N4_CONTROL_POINTS_INPLANE_MAX = 20

# Through-plane, we do NOT derive anything — we ask for the fewest control
# points a cubic spline can have, which is 4, i.e. one single span across the
# entire slab. Deliberately the most rigid field expressible.
#
# The reason is that these are 2D multi-slice acquisitions with 5–10 mm slices,
# where through-plane intensity variation is mostly NOT a bias field: it is
# slice profile, cross-talk and per-slice excitation differences. Those are
# genuinely discontinuous between slices, and any mesh with enough freedom to
# follow them will follow anatomy too. Giving the z axis one rigid span means
# N4 can still remove a smooth head-to-foot coil falloff, and cannot express
# anything sharper. Set to 4 = spline order + 1; raising it is the change most
# likely to reintroduce the slice-to-slice stepping this rewrite removed.
N4_CONTROL_POINTS_THROUGH_PLANE = 4

# ── 2D translation registration (optional, --register_2d) ─────────────────────
# The method itself is registration_idea.py: with both images already at 1 mm
# per pixel, slide the MRI over the CT a whole pixel at a time and keep the best
# normalised mutual information. A whole-pixel slide cannot rotate, scale or
# shear, so the three failure modes in docs/registration_gates_docs.md are not
# gated against — they cannot be expressed. There are no random numbers in it,
# so two runs of this pipeline over the same data give the same shift.
#
# ONE SHIFT PER VOLUME, NOT ONE PER SLICE
# ───────────────────────────────────────
# The shift is estimated on a few probe slices and then applied to the WHOLE
# stack. This is the same argument as the N4 rewrite above, in a different
# variable: a per-slice shift hands every slice its own free translation, which
# manufactures slice-to-slice steps that were not in the data. The MRI would
# shear through z relative to the CT and anatomy that was continuous would come
# out as a staircase.
#
# That is not hypothetical on this dataset. In sweep_idea_2_summary.csv, the
# best per-slice shift across one 18-slice shoulder axial stack goes
# (+18,+11) -> (-57,-24) -> (-66,-39) mm, an 85 mm swing in dx; one spine
# sagittal stack swings 54 mm in dy. Those numbers are why the estimate is
# pooled and why REG_MAX_SPREAD_MM exists.
REG_SEARCH_MM     = 40.0   # search +/- this far on each axis. Cost is quadratic
                           # in it. The sweeps used 90 to find out how far the
                           # offsets actually go; 40 covers all but the shoulder
                           # outliers, which REG_MAX_SPREAD_MM rejects anyway.
REG_COARSE_MM     = 4.0    # stride of the first sweep. 1 = exhaustive search.
REG_KEEP          = 5      # coarse positions given a fine search around them.
REG_BINS          = 32     # histogram bins per image for the NMI.

REG_N_PROBES      = 5      # slices sampled through the stack to estimate on.
                           # Taken at 10/30/50/70/90% of the way through rather
                           # than including the very ends, because the first and
                           # last slices are the emptiest and the least able to
                           # measure anything.
REG_MIN_PROBES    = 2      # fewest probes that must return an answer before any
                           # shift is applied. One probe is an unverifiable
                           # estimate being imposed on a whole volume; with two
                           # there is at least something to disagree.
REG_MAX_SPREAD_MM = 20.0   # if the probes' answers disagree by more than this
                           # on either axis, no single translation describes this
                           # pair and the volume is left unshifted. Against the
                           # sweep numbers this passes brain and knee axial and
                           # rejects the shoulder axial and spine sagittal cases
                           # above — which is the intent, since for those the
                           # median would make most slices worse.
REG_MIN_GAIN      = 0.010  # the chosen shift is re-scored on every probe, and
                           # applied only if it raises NMI by more than this on
                           # average. Same threshold the sweeps call MIN_GAIN.
                           # This tests the shift that will actually be used,
                           # rather than trusting the per-probe searches that
                           # produced it.

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
        "ct_win_min": 0.0,
        "ct_win_max": 80.0
    },
    "abdomen": {
        "ct_win_min": -160.0,
        "ct_win_max": 240.0
    },
    "musculoskeletal": {
        "ct_win_min": -200.0,
        "ct_win_max": 300.0
    },
    "spine": {
        "ct_win_min": -200.0,
        "ct_win_max": 300.0
    },
    "default": {
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
