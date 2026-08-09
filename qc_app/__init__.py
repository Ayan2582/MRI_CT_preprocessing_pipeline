"""
qc_app — a local review tool wrapped around the existing CT/MRI pipeline.

The registration is not implemented here. Every numerical step is a call into
Preprocessing/: io_utils, image_processing, normalization, export_utils and
registration_idea, in the order pipeline_core.process_orientation_pair runs
them. This package adds dataset discovery, a persistent manifest, a background
worker, and a browser UI for accepting or rejecting each slice pair.

    python -m qc_app                 review the repository dataset
    python -m qc_app --help          all options
"""

__version__ = "1.0.0"
