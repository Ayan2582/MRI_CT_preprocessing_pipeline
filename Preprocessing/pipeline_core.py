import os # Used to manipulate the filesystem, such as creating the output directories.
import argparse # Used for passing type hints about the command-line arguments.
import logging # Used for safe console printing and log file writing.
import numpy as np # Matrix and array operations.
import SimpleITK as sitk # Standard toolkit for reading DICOM medical images.

# Import our custom, separated modules.
import pipeline_config as cfg
import image_processing as img_proc
import normalization as norm
import export_utils as export

def process_orientation_pair(
    ct_entry:      dict,
    mri_entry:     dict,
    orient:        str,
    patient_out:   str,
    args:          argparse.Namespace,
    log:           logging.Logger,
    metadata_rows: list,
    patient_id:    str,
) -> tuple[int, int]:
    """
    [Function 2: Used in the main pipeline at preprocess_2d.py:266]
    Run the full preprocessing pipeline for one orientation pair.
    
    EXPECTED VARIABLE CONTENTS:
    
    1. ct_entry / mri_entry: Dictionaries containing the raw data and metadata.
       Example: {
           "path":        "C:/MRI-CT/CT/PA0_Ranjeet/ST0/SE0",
           "image":       <SimpleITK.Image object> (The full loaded 3D Volume),
           "n_slices":    18,
           "orientation": "axial",
           "series_desc": "CT HEAD W/O CONTRAST"
       }
       
    2. orient: A string indicating the anatomical plane.
       Example: "axial" (or "coronal", "sagittal")
       
    3. patient_out: A string path to the specific patient's output folder.
       Example: "C:/Outputs/PA0_Ranjeet"
       
    4. args: The parsed terminal commands (argparse.Namespace).
       Contains dynamic settings like args.ct_win_min and args.ct_win_max.
       
    5. log: The logging.Logger object used to print messages to the console safely.
       
    6. metadata_rows: A Python list passed by reference.
       This function appends dictionaries to this list, which eventually get written to a massive CSV file.
       
    7. patient_id: A string representing the patient's unique folder name.
       Example: "PA0_Ranjeet"
    """
    # Define the exact folder paths where the CTs, MRIs, and PNG previews will be saved.
    orient_dir = os.path.join(patient_out, orient)
    ct_dir     = os.path.join(orient_dir, "ct")
    mri_dir    = os.path.join(orient_dir, "mri")
    prev_dir   = os.path.join(orient_dir, "previews")

    # Safely create those folders on the hard drive. 
    # exist_ok=True prevents crashes if the folders already exist.
    os.makedirs(ct_dir,  exist_ok=True)
    os.makedirs(mri_dir, exist_ok=True)
    if args.save_png:
        os.makedirs(prev_dir, exist_ok=True)

    # -- Step 1: N4 bias field correction (MRI only) --
    log.info(f"  [{orient}] Applying 3D N4 bias field correction to the MRI volume "
             f"({mri_entry['n_slices']} slices, shrink_factor={args.n4_shrink}) ...")
    # Execute the N4 Bias Correction to flatten uneven magnetic lighting in the MRI.
    #
    # This runs on the WHOLE VOLUME AT ONCE, which is why it has to happen here,
    # before any of the steps below. Every later step works one slice at a time,
    # and once the stack has been taken apart the through-plane half of the bias
    # field is no longer visible to anything.
    #
    # `orient` is passed through because the mesh is anisotropic and which
    # anatomical direction each in-plane axis carries depends on the acquisition
    # plane. [Function Origin: image_processing.py]
    mri_corrected = img_proc.apply_n4_bias_correction(
        mri_entry["image"],
        orientation=orient,
        shrink_factor=args.n4_shrink,
    )

    # -- Step 2: Physical Grid Matching & Resampling --
    log.info(f"  [{orient}] Resampling CT  {ct_entry['n_slices']} sl, "
             f"spacing={ct_entry['image'].GetSpacing()[:2]} -> {args.target_spacing} mm")
    
    # Scale the CT image so its X and Y pixels are exactly 1.0mm wide.
    # This standardizes the physical scale of all patients so the neural network doesn't have to learn scale variations.
    # [Function Origin: image_processing.py]
    ct_res  = img_proc.resample_inplane(ct_entry["image"], args.target_spacing, is_ct=True)

    log.info(f"  [{orient}] Projecting MRI {mri_entry['n_slices']} sl onto CT's physical DICOM grid...")
    # Map the MRI's pixels perfectly onto the CT's physical grid coordinate system.
    # This guarantees structural alignment, which is the most critical requirement for Image-to-Image AI networks.
    # [Function Origin: image_processing.py]
    mri_res = img_proc.resample_mri_to_ct_grid(mri_corrected, ct_res, default_pixel_value=0.0)

    # -- Step 3: Extract 2D slices --
    # Convert the 3D SimpleITK volumes into native Python lists of 2D Numpy Arrays.
    # Numpy arrays are natively supported by PyTorch and much easier to crop/manipulate.
    # [Function Origin: image_processing.py]
    ct_slices  = img_proc.volume_to_slices(ct_res)
    mri_slices = img_proc.volume_to_slices(mri_res)

    # Count how many slices there are. Because the MRI was mathematically mapped to the CT,
    # they are guaranteed to have the exact same number of slices!
    n_pair = len(ct_slices)

    # -- Step 4: MRI percentiles --
    # Convert the entire 3D MRI volume into a single Numpy Array to calculate extreme percentiles.
    mri_vol = sitk.GetArrayFromImage(mri_res).astype(np.float32)
    # Calculate the dark and bright tissue thresholds, ignoring the black background air.
    # [Function Origin: normalization.py]
    p1, p99 = norm.compute_mri_percentiles(mri_vol, args.mri_p_low, args.mri_p_high)
    
    log.info(
        f"  [{orient}] MRI intensity percentiles (post-N4): "
        f"p{args.mri_p_low}={p1:.1f}  p{args.mri_p_high}={p99:.1f}"
    )

    # -- Step 4.5: Translation registration (optional, --register_2d) --
    # Step 2 already put the MRI on the CT's physical grid, so the two stacks
    # agree wherever the DICOM origins are trustworthy. Where they are not, this
    # measures the leftover in-plane offset and takes it out.
    #
    # ONE shift is estimated for the whole stack and applied to every slice.
    # Registering slices independently would hand each one its own translation
    # and shear the MRI through z — the same failure the 3D N4 fit above exists
    # to avoid. See image_processing.estimate_volume_translation.
    #
    # This runs AFTER the percentiles are computed, and that ordering is safe:
    # shifting moves pixels around but does not change their values, and the
    # only new values it introduces are the 0.0 fill, which
    # compute_mri_percentiles already excludes along with the existing
    # background. Registering first would give the same p1/p99 at more cost.
    reg = None
    if args.register_2d:
        log.info(f"  [{orient}] Estimating one in-plane shift for the whole stack "
                 f"(search +/-{args.reg_search_mm:.0f} mm, "
                 f"{cfg.REG_N_PROBES} probe slices) ...")
        # [Function Origin: image_processing.py]
        reg = img_proc.estimate_volume_translation(
            ct_slices, mri_slices,
            spacing_mm=args.target_spacing,
            search_mm=args.reg_search_mm,
        )
        verdict = "APPLIED" if reg["applied"] else "NOT applied"
        log.info(f"  [{orient}] Registration {verdict}: {reg['reason']}")
        if reg["hit_edge"]:
            log.warning(
                f"  [{orient}] {reg['hit_edge']} of {reg['n_probes']} probe slices put the best "
                f"shift on the edge of the search square and were discarded. The real offset is "
                f"further out than +/-{args.reg_search_mm:.0f} mm - re-run this pair with a "
                f"larger --reg_search_mm to measure it."
            )

    # -- Steps 5-8: Normalise / flag / save --  (cropping now happens at dataloader time)
    n_saved      = 0
    n_flagged_bg = 0

    # Iterate vertically through the slices one by one.
    for i in range(n_pair):
        # Extract the matching 2D slice from both the CT and MRI.
        ct_slice = ct_slices[i]
        mri_slice = mri_slices[i]
        
        # Apply the single stack-wide shift measured above. Every slice gets the
        # same one, so their alignment relative to each other is untouched.
        if reg is not None and reg["applied"]:
            # [Function Origin: image_processing.py]
            mri_slice = img_proc.apply_translation(
                mri_slice, ct_slice.shape, reg["dy"], reg["dx"])

        # Normalise both arrays so their pixel intensities range strictly between 0.0 and 1.0.
        # [Function Origin: normalization.py]
        ct_norm  = norm.normalize_ct_slice(ct_slice, args.ct_win_min, args.ct_win_max)
        mri_norm = norm.normalize_mri_slice(mri_slice, p1, p99)

        # Background flag: Check if 90% of either slice is just empty black space.
        # We no longer discard these — a slice near the FOV edge can still contain a
        # thin sliver of real anatomy, and silently dropping it risks losing good data.
        # Instead we tag the pair as "is_background" in the metadata CSV so the GAN's
        # dataloader can choose to filter, downweight, or keep them, with the decision
        # visible and reversible instead of baked into this pipeline.
        # [Function Origin: normalization.py]
        is_bg = (
            norm.is_background_slice(ct_norm,  args.bg_thresh, args.bg_fraction) or
            norm.is_background_slice(mri_norm, args.bg_thresh, args.bg_fraction)
        )
        if is_bg:
            n_flagged_bg += 1

        # NOTE: No cropping / padding happens here anymore.
        # Slices are saved at their native post-resample size, which varies per series.
        # Cropping is deferred to the GAN's dataloader so that crop strategy (center,
        # random, body-mask) can be changed and re-tuned without re-running this pipeline.
        ct_final  = ct_norm
        mri_final = mri_norm

        # Save .npy
        # Extract the human-readable folder name (e.g. "SE0").
        ct_sname = os.path.basename(ct_entry["path"])
        mri_sname = os.path.basename(mri_entry["path"])
        # Format the file name so it's easily traceable back to the raw DICOM slice.
        name = f"ct_{ct_sname}_mri_{mri_sname}_{i:03d}"
        
        # Construct the absolute path and save the numpy binaries.
        ct_path  = os.path.join(ct_dir,  f"{name}.npy")
        mri_path = os.path.join(mri_dir, f"{name}.npy")
        
        # [Function Origin: export_utils.py]
        export.save_npy(ct_final,  ct_path)
        export.save_npy(mri_final, mri_path)

        # Save PNG preview
        # If the user requested it, generate a side-by-side picture of the arrays so humans can verify the alignment.
        if args.save_png:
            # [Function Origin: export_utils.py]
            export.save_preview_png(
                ct_final, mri_final,
                os.path.join(prev_dir, f"{name}_pair.png")
            )

        # Accumulate metadata
        # Figure out if this patient is a BRAIN, ABDOMEN, etc., using the prefix string mapping from the config file.
        prefix = patient_id.split("_")[0]
        region = cfg.PREFIX_TO_REGION.get(prefix, "default")

        # Append all the paths and clinical info to a dictionary so the orchestrator can write it into a giant CSV file.
        # Deep Learning Dataloaders (like in PyTorch) use these CSV files to find the images on the hard drive during training!
        # Because slices are no longer cropped to a uniform square, the dataloader cannot
        # assume a shape. We record the actual saved dimensions so the Dataset can size
        # batches and filter undersized slices without opening every .npy file first.
        h, w = ct_final.shape

        metadata_rows.append({
            "patient_id":    patient_id,
            "body_region":   region,
            "orientation":   orient,
            "slice_index":   i,
            "ct_series":     os.path.basename(ct_entry["path"]),
            "mri_series":    os.path.basename(mri_entry["path"]),
            "mri_desc":      mri_entry["series_desc"],
            "height":        h,
            "width":         w,
            "ct_npy":        ct_path,
            "mri_npy":       mri_path,
            "is_background": is_bg,
            # What registration did to this slice, recorded whether or not it
            # was applied. Nobody is going to look at 2313 overlays by hand, so
            # this is how a shift that went wrong stays findable afterwards
            # instead of being baked invisibly into the .npy. reg_dx_mm and
            # reg_dy_mm carry the measured shift even when reg_applied is False,
            # which is what makes the rejections searchable too.
            "reg_applied":   bool(reg["applied"]) if reg else False,
            "reg_dx_mm":     reg["dx_mm"] if reg else 0.0,
            "reg_dy_mm":     reg["dy_mm"] if reg else 0.0,
            "reg_nmi_gain":  reg["mean_gain"] if reg else "",
            "reg_note":      reg["reason"] if reg else "",
        })
        n_saved += 1

    # Print a summary to the console so the user knows how many slices were flagged as background.
    log.info(
        f"  [{orient}] OK {n_saved} pairs saved | "
        f"FLAGGED {n_flagged_bg} as background (kept, not discarded)"
    )
    return n_saved, n_flagged_bg
