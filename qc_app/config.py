"""
config.py
─────────
Where the application's files live, and which pipeline knobs the UI is allowed
to move.

Registration parameters are NOT redefined here. They are read from
Preprocessing/pipeline_config.py so there is one source of truth: changing
REG_MAX_SPREAD_MM there changes what this tool accepts, with no second copy to
forget about.
"""

import os
from dataclasses import dataclass, field

from . import bootstrap

_pp = bootstrap.preprocessing_modules()
cfg = _pp.cfg

REPO_ROOT = bootstrap.REPO_ROOT

# ── Where things live ─────────────────────────────────────────────────────────
# Input is the repository copy, the same one pipeline_config.DATA_ROOT resolves
# to, and it is treated as strictly read-only everywhere in this application.
DEFAULT_DATA_ROOT = cfg.DATA_ROOT

# Everything this tool generates goes under one workspace directory, so the
# whole thing can be deleted and rebuilt without touching the dataset.
DEFAULT_WORKSPACE = os.path.join(REPO_ROOT, "qc_workspace")


@dataclass
class AppConfig:
    data_root: str = DEFAULT_DATA_ROOT
    workspace: str = DEFAULT_WORKSPACE

    # In-plane resample target, mm. Also the pixel size of every array the UI
    # displays, which is what makes an ROI rectangle measurable in mm.
    target_spacing: float = cfg.TARGET_SPACING_MM

    # Registration search half-width, mm. Cost is quadratic in it.
    reg_search_mm: float = cfg.REG_SEARCH_MM

    # N4 in-plane shrink factor. Never applied through-plane by the pipeline.
    n4_shrink: int = cfg.N4_SHRINK_FACTOR

    # NOTE: there is no world-geometry option here on purpose. Both modalities
    # are brought to 1 mm per pixel in 2D from PixelSpacing alone, the way
    # registration_idea.py does it. See registration_service's module docstring
    # for what projecting through DICOM world coordinates cost on this dataset.

    # How many series the background worker registers ahead of where the
    # reviewer currently is, so paging forward does not stall on a cold series.
    prefetch_depth: int = 2

    # Re-accepting a pair rewrites its own output files. This only ever touches
    # paths under workspace/output that this tool itself created.
    allow_output_overwrite: bool = True

    # Modalities, in the layout the dataset actually uses.
    ct_dirname: str = "CT"
    mri_dirname: str = "MRI"

    @property
    def db_path(self) -> str:
        return os.path.join(self.workspace, "qc.db")

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.workspace, "cache")

    @property
    def output_dir(self) -> str:
        return os.path.join(self.workspace, "output")

    @property
    def ct_root(self) -> str:
        return os.path.join(self.data_root, self.ct_dirname)

    @property
    def mri_root(self) -> str:
        return os.path.join(self.data_root, self.mri_dirname)

    def ensure_dirs(self) -> None:
        for d in (self.workspace, self.cache_dir, self.output_dir):
            os.makedirs(d, exist_ok=True)


# Registration parameters surfaced to the UI read-only, so a reviewer can see
# the thresholds a rejection was measured against without opening the source.
def registration_settings() -> dict:
    return {
        "search_mm":      cfg.REG_SEARCH_MM,
        "coarse_mm":      cfg.REG_COARSE_MM,
        "keep":           cfg.REG_KEEP,
        "bins":           cfg.REG_BINS,
        "n_probes":       cfg.REG_N_PROBES,
        "min_probes":     cfg.REG_MIN_PROBES,
        "max_spread_mm":  cfg.REG_MAX_SPREAD_MM,
        "min_gain":       cfg.REG_MIN_GAIN,
        "target_spacing": cfg.TARGET_SPACING_MM,
    }
