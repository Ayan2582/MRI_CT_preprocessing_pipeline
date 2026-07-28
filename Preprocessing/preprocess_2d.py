"""
preprocess_2d.py
────────────────
2D MRI-CT Preprocessing Pipeline — Main Entry Point

Processes a paired MRI + CT DICOM dataset across all three orientations
(axial, coronal, sagittal).  For each orientation where both modalities
have a series, it produces:
  • Normalised float32 .npy slice pairs  (CT and MRI, same index)
  • Optional side-by-side PNG previews   (for visual quality control)
  • A metadata.csv with all file paths and stats
"""

import os # Used to manipulate the filesystem, such as creating output directories and checking paths.
import sys # Used to exit the program if an error occurs.
import csv # Used to write the final metadata.csv file for PyTorch.
import logging # Used for safe console printing and writing to a log file.
import argparse # Used to parse command-line arguments when running the script.

import pipeline_config as cfg # Contains all the hardcoded settings like TARGET_SIZE and Windowing bounds.
import io_utils # Our custom module for reading DICOM files.
import pipeline_core # Our custom module containing the massive image processing loop.

def setup_logging(output_dir: str) -> logging.Logger:
    """
    [Function 0.1: Used in the main pipeline at preprocess_2d.py:108]
    Configure root logger to write to both console and pipeline.log.
    """
    # Safely create the output directory so the log file has a place to live.
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the visual format of the log messages (e.g. "14:30:00 [INFO] Pipeline started").
    # This is important for debugging if a patient fails deep into the pipeline.
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    # Get the root logger that controls all print statements.
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Create a FileHandler to write all logs to a permanent 'pipeline.log' text file on the hard drive.
    fh = logging.FileHandler(os.path.join(output_dir, "pipeline.log"), mode="w")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Create a StreamHandler to print all logs directly to the terminal screen in real-time.
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    return logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    [Function 0.2: Used in the main pipeline at preprocess_2d.py:105]
    Parse command-line arguments to override default configuration settings.
    """
    p = argparse.ArgumentParser(
        description="2D MRI-CT Preprocessing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Expose the configuration variables so the user can change them via the terminal (e.g., --target_spacing 2.0).
    # This makes the script highly reusable without having to manually edit the Python code.
    p.add_argument("--data_root",      default=cfg.DATA_ROOT,
                   help="Root DICOM directory with CT/ and MRI/ sub-folders")
    p.add_argument("--output_dir",     default=cfg.OUTPUT_DIR,
                   help="Where to write processed slices")
    p.add_argument("--target_spacing", type=float, default=cfg.TARGET_SPACING_MM,
                   help="In-plane target voxel spacing in mm")
    p.add_argument("--ct_win_min",     type=float, default=cfg.CT_WINDOW_MIN_HU,
                   help="CT window lower bound (HU)")
    p.add_argument("--ct_win_max",     type=float, default=cfg.CT_WINDOW_MAX_HU,
                   help="CT window upper bound (HU)")
    p.add_argument("--register_2d",    action="store_true",
                   help="Perform 2D Rigid Registration on slices to fix patient movement")
    p.add_argument("--mri_p_low",      type=float, default=cfg.MRI_PERCENTILE_LOW,
                   help="MRI lower percentile for clipping")
    p.add_argument("--mri_p_high",     type=float, default=cfg.MRI_PERCENTILE_HIGH,
                   help="MRI upper percentile for clipping")
    p.add_argument("--bg_thresh",      type=float, default=cfg.BG_INTENSITY_THRESH,
                   help="Normalised intensity below which a pixel counts as background")
    p.add_argument("--bg_fraction",    type=float, default=cfg.BG_PIXEL_FRACTION,
                   help="Fraction of bg pixels above which a slice is flagged (kept, not discarded)")
    p.add_argument("--save_png",       action="store_true", default=cfg.SAVE_PNG,
                   help="Save side-by-side CT|MRI PNG previews")
    p.add_argument("--skip_existing",  action="store_true", default=cfg.SKIP_EXISTING,
                   help="Skip patients whose output directory already exists")
    p.add_argument("--n4_shrink",      type=int,   default=4,
                   help="N4 bias correction shrink factor (1=full-res, 4=fast default)")
    p.add_argument("--patient",        default=None,
                   help="Process only this patient ID (for debugging)")
                   
    return p.parse_args()


def main():
    """
    [Function 0.3: Used as the primary entry point when the script runs at preprocess_2d.py:310]
    The main coordination loop that finds the patients and triggers the processing core.
    """
    # Load all user arguments from the terminal.
    args = parse_args()
    
    # Initialize the logging system so we can print to the screen and the log file.
    log  = setup_logging(args.output_dir)

    # Print a nice summary of the settings being used to the console.
    log.info("=" * 65)
    log.info("  2D MRI-CT Preprocessing Pipeline")
    log.info("=" * 65)
    log.info(f"  Data root       : {args.data_root}")
    log.info(f"  Output dir      : {args.output_dir}")
    log.info(f"  Orientations    : {cfg.ORIENTATIONS}")
    log.info(f"  Target spacing  : {args.target_spacing} mm")
    log.info(f"  Cropping        : DISABLED - slices saved at native size, cropped at load time")
    log.info(f"  CT window       : [{args.ct_win_min}, {args.ct_win_max}] HU -> [0, 1]")
    log.info(f"  MRI percentiles : [{args.mri_p_low}, {args.mri_p_high}]  -> [0, 1]")
    log.info(f"  BG filter       : thresh={args.bg_thresh}, fraction={args.bg_fraction}")
    log.info(f"  N4 bias corr.   : MRI only, shrink_factor={args.n4_shrink}")
    log.info(f"  Save PNG        : {args.save_png}")
    log.info("=" * 65)

    # Construct the paths to the massive CT and MRI root folders on the hard drive.
    ct_root  = os.path.join(args.data_root, "CT")
    mri_root = os.path.join(args.data_root, "MRI")

    # If the folders don't exist, kill the script before it crashes unpredictably.
    if not os.path.isdir(ct_root):
        log.error(f"CT root not found: {ct_root}")
        sys.exit(1)
    if not os.path.isdir(mri_root):
        log.error(f"MRI root not found: {mri_root}")
        sys.exit(1)

    # Read the names of all patient folders inside the CT root directory.
    all_patients = sorted(os.listdir(ct_root))
    
    # If the user requested to test a single specific patient (e.g. --patient PA0_Ranjeet)...
    if args.patient:
        if args.patient not in all_patients:
            log.error(f"Patient '{args.patient}' not found in CT root.")
            sys.exit(1)
        patients = [args.patient]
    else:
        # Otherwise, we will loop through all of them.
        patients = all_patients

    log.info(f"\nPatients to process: {len(patients)}\n")

    # Initialize tracking variables so we can generate a CSV file and a summary report at the very end.
    metadata_rows        = []
    total_pairs          = 0
    total_bg_flagged     = 0
    patients_ok          = 0
    patients_skipped_mri = 0
    orient_counts        = {o: 0 for o in cfg.ORIENTATIONS}

    # Start iterating over every patient found on the hard drive!
    for patient_id in patients:
        divider = "-" * 65
        log.info(divider)
        log.info(f"Patient: {patient_id}")
        
        # Check the prefix of the patient's name (e.g. "PA0" -> Brain, "PA3" -> Knee).
        prefix = patient_id.split("_")[0]
        region = cfg.PREFIX_TO_REGION.get(prefix, "default")
        
        # Pull the specific clinical settings for this body region from the config file.
        profile = cfg.REGION_PROFILES.get(region, cfg.REGION_PROFILES["default"])
        
        # Temporarily overwrite the global pipeline arguments with this specific region's settings!
        # This allows the pipeline to dynamically adjust its CT Window per-patient.
        args.ct_win_min = profile["ct_win_min"]
        args.ct_win_max = profile["ct_win_max"]

        log.info(f"  Region          : {region.upper()}")
        log.info(f"  CT window       : [{args.ct_win_min}, {args.ct_win_max}] HU")

        # Construct the path to the patient's actual DICOM scans (ST0 folder).
        ct_study  = os.path.join(ct_root,  patient_id, "ST0")
        mri_study = os.path.join(mri_root, patient_id, "ST0")
        pat_out   = os.path.join(args.output_dir, patient_id)

        # If the patient has a CT scan but no MRI scan, we can't train an AI, so we skip them.
        if not os.path.isdir(mri_study):
            log.warning(f"  MRI study not found - skipping patient.")
            patients_skipped_mri += 1
            continue

        # If --skip_existing is True, check if this patient has already been processed previously to save time.
        if args.skip_existing and os.path.isdir(pat_out):
            existing_dirs = [
                d for d in os.listdir(pat_out)
                if os.path.isdir(os.path.join(pat_out, d))
            ]
            if existing_dirs:
                log.info(f"  Already processed ({existing_dirs}). Skipping.")
                continue

        # Use our custom io_utils module to intelligently scan the folders and determine which axis (Axial, Sagittal) they are.
        log.info("  Discovering CT series ...")
        ct_series  = io_utils.discover_series(ct_study)
        log.info("  Discovering MRI series ...")
        mri_series = io_utils.discover_series(mri_study)

        # Print a quick summary of what was discovered so the user can verify.
        ct_summary  = [(s["orientation"], s["n_slices"]) for s in ct_series]
        mri_summary = [
            (s["orientation"], s["n_slices"], s["series_desc"])
            for s in mri_series
        ]
        log.info(f"  CT  series: {ct_summary}")
        log.info(f"  MRI series: {mri_summary}")

        patient_pairs = 0
        os.makedirs(pat_out, exist_ok=True)

        # We must figure out which CT scan goes with which MRI scan. 
        # Since standard hospitals don't use strict naming, we look for matching folder prefixes (e.g. SE0_1 matches SE0).
        paired_series = []
        for m_entry in mri_series:
            # Extract the raw folder name for the MRI (e.g. "SE0").
            m_name = os.path.basename(m_entry["path"])
            m_base = m_name.split('_')[0]
            
            # Search the CT scans list to see if one has the exact same base name.
            c_entry = next((c for c in ct_series if os.path.basename(c["path"]).split('_')[0] == m_base), None)
            
            # If a match is found...
            if c_entry:
                orient = m_entry["orientation"]
                
                # If we couldn't figure out the orientation (e.g. diagonal scan), skip it.
                if orient not in cfg.ORIENTATIONS:
                    log.warning(f"  Skipping {m_name}: MRI dictates unknown orientation '{orient}'")
                    continue
                    
                # Add the perfectly paired CT and MRI to our processing list!
                paired_series.append((c_entry, m_entry, orient))
                
        # If no scans matched names, this patient is useless for AI training.
        if not paired_series:
            log.info("  No identically named CT and MRI folders found - skipped.")
            continue
            
        # Loop through all the perfectly paired scans...
        for ct_entry, mri_entry, orient in paired_series:
            log.info(
                f"  [{orient}] Pairing (Strict Name Match): "
                f"CT {os.path.basename(ct_entry['path'])} ({ct_entry['n_slices']} sl)  <->  "
                f"MRI {os.path.basename(mri_entry['path'])} ({mri_entry['n_slices']} sl)"
                f"  '{mri_entry['series_desc']}'"
            )

            try:
                # Trigger the massive mathematical pipeline to process this specific pair!
                n_saved, n_bg = pipeline_core.process_orientation_pair(
                    ct_entry, mri_entry, orient,
                    pat_out, args, log, metadata_rows, patient_id
                )

                # Keep track of how many 2D slices were saved, and how many were flagged (but kept) as background.
                patient_pairs    += n_saved
                total_pairs      += n_saved
                total_bg_flagged += n_bg
                if n_saved > 0:
                    orient_counts[orient] += n_saved
            except Exception as exc:
                # If SimpleITK crashes due to a corrupt DICOM file, catch it so it doesn't crash the entire loop!
                log.error(f"  [{orient}] Unhandled error: {exc}", exc_info=True)

        log.info(f"  Patient subtotal: {patient_pairs} paired slices")
        patients_ok += 1

    # Once all patients are done, write the master PyTorch Dataset CSV file.
    meta_path = os.path.join(args.output_dir, "metadata.csv")
    fieldnames = [
        "patient_id", "body_region", "orientation", "slice_index",
        "ct_series", "mri_series", "mri_desc",
        "height", "width",
        "ct_npy", "mri_npy", "is_background",
    ]
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        # DictWriter converts our Python list-of-dictionaries directly into an Excel-style CSV sheet.
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    # Print a final summary so the user knows exactly what the AI has to work with.
    log.info("\n" + "=" * 65)
    log.info("  PIPELINE COMPLETE - SUMMARY")
    log.info("=" * 65)
    log.info(f"  Patients processed       : {patients_ok} / {len(patients)}")
    log.info(f"  Patients skipped (no MRI): {patients_skipped_mri}")
    log.info(f"  Total paired slices saved: {total_pairs}")
    log.info(f"  Slices flagged (bg, kept): {total_bg_flagged}")
    log.info("  Breakdown by orientation :")
    for orient, count in orient_counts.items():
        log.info(f"    {orient:10s}: {count} slices")
    log.info(f"  Metadata CSV             : {meta_path}")
    log.info(f"  Log file                 : {os.path.join(args.output_dir, 'pipeline.log')}")
    log.info("=" * 65)

# Standard Python boilerplate to ensure main() only runs if the file is run directly (not imported).
if __name__ == "__main__":
    main()
