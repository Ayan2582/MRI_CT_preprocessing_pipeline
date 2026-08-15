"""
bootstrap.py
────────────
Make the existing Preprocessing/ modules importable from inside model/.

Mirrors qc_app/bootstrap.py, and for the same reason: the Preprocessing modules
import each other by bare name (`import pipeline_config as cfg`), so they only
resolve when the Preprocessing directory itself is on sys.path. Production files
are not rewritten to accommodate a downstream tool, so the path is fixed here.

The model package needs exactly one thing from Preprocessing: the per-region CT
HU windows in pipeline_config.REGION_PROFILES. Those windows are what produced
the [0,1] floats in qc_workspace/output, so any code that converts a prediction
back to Hounsfield units has to use the same numbers. Re-typing them here would
create a second source of truth that silently rots the first time a window is
retuned, which is precisely the failure this shim exists to prevent.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PREPROCESSING_DIR = os.path.join(REPO_ROOT, "Preprocessing")

if not os.path.isdir(PREPROCESSING_DIR):
    raise RuntimeError(
        f"Cannot find the Preprocessing directory at {PREPROCESSING_DIR}. "
        f"model/ must live next to it inside the repository."
    )

if PREPROCESSING_DIR not in sys.path:
    # Appended rather than prepended, matching qc_app: if a name here ever
    # collides with a stdlib or site-packages module, the established one wins.
    sys.path.append(PREPROCESSING_DIR)


# ── Region → HU window ────────────────────────────────────────────────────────
# Imported lazily-but-once at module load so that a missing/renamed
# REGION_PROFILES fails loudly here rather than as a KeyError deep in a metric.

try:
    import pipeline_config as _cfg
except ImportError as e:                                    # pragma: no cover
    raise RuntimeError(
        f"Failed to import 'pipeline_config' from {PREPROCESSING_DIR}: {e}"
    ) from e

REGION_PROFILES = _cfg.REGION_PROFILES
PREFIX_TO_REGION = _cfg.PREFIX_TO_REGION


def hu_window(body_region):
    """
    Return (win_min, win_max) in Hounsfield units for a body region.

    Unknown regions fall back to the 'default' profile rather than raising,
    matching how registration_service.region_for_patient resolves them, so a
    new patient prefix cannot crash a training run mid-epoch.
    """
    profile = REGION_PROFILES.get(body_region) or REGION_PROFILES["default"]
    return float(profile["ct_win_min"]), float(profile["ct_win_max"])


# Kaggle has no Preprocessing/ directory, so a packaged run cannot rely on the
# import above. package_for_kaggle.py writes these same values into the manifest
# as per-row hu_min/hu_max columns; this dict is the fallback used to build them
# and to sanity-check a manifest that arrives from elsewhere.
def region_windows():
    """Return {region: (win_min, win_max)} for every profile, 'default' included."""
    return {name: (float(p["ct_win_min"]), float(p["ct_win_max"]))
            for name, p in REGION_PROFILES.items()}
