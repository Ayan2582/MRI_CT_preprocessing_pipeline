"""
bootstrap.py
────────────
Make the existing Preprocessing/ modules importable.

Those modules import each other by bare name (`import pipeline_config as cfg`,
`import registration_idea as reg_idea`), so they only resolve when the
Preprocessing directory itself is on sys.path. Rewriting them into a package
would mean touching production files, which this application deliberately does
not do — so the path is fixed up here instead, once, before anything else
imports them.

Import this module before any `import pipeline_core` / `import image_processing`
anywhere in qc_app.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PREPROCESSING_DIR = os.path.join(REPO_ROOT, "Preprocessing")

if not os.path.isdir(PREPROCESSING_DIR):
    raise RuntimeError(
        f"Cannot find the Preprocessing directory at {PREPROCESSING_DIR}. "
        f"qc_app must live next to it inside the repository."
    )

if PREPROCESSING_DIR not in sys.path:
    # Appended rather than prepended: if a name here ever collides with a
    # stdlib or site-packages module, the established one wins, which is the
    # safer direction for a shim.
    sys.path.append(PREPROCESSING_DIR)


def preprocessing_modules():
    """
    Import and return the production modules, as a single namespace object.

    Centralised here so there is exactly one place that reaches into
    Preprocessing/, and so an import failure reports which module was missing
    rather than surfacing as a bare ImportError halfway through a request.
    """
    import types

    ns = types.SimpleNamespace()
    for attr, name in (
        ("cfg",      "pipeline_config"),
        ("io_utils", "io_utils"),
        ("img_proc", "image_processing"),
        ("norm",     "normalization"),
        ("export",   "export_utils"),
        ("reg_idea", "registration_idea"),
    ):
        try:
            setattr(ns, attr, __import__(name))
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import '{name}' from {PREPROCESSING_DIR}: {e}"
            ) from e
    return ns
