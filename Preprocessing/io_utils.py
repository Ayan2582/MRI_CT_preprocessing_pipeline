import os # Used to navigate folders and join paths.
import logging # Used for safe console printing.
import numpy as np # Used for matrix math.
import SimpleITK as sitk # Standard toolkit for reading DICOM medical images.

logger = logging.getLogger(__name__)

def get_orientation_from_direction(direction):
    """
    [Unused Helper Function]
    Derive the slice-plane orientation from a SimpleITK direction cosine matrix.
    """
    # The direction array is a flat list of 9 numbers. We reshape it into a 3x3 matrix.
    # This matrix mathematically describes how the MRI is tilted in physical 3D space.
    d = np.array(direction).reshape(3, 3)
    
    # We only care about the 3rd row (index 2), which represents the "normal" vector.
    # The normal vector is an imaginary arrow pointing straight out of the slice.
    # We take the absolute value because we don't care if it's pointing forward or backward.
    normal = np.abs(d[2, :])
    
    # We find which axis (X=0, Y=1, or Z=2) the arrow is pointing down the hardest.
    idx = int(np.argmax(normal))
    
    # If it points down X, it's a Sagittal slice (ear-to-ear).
    # If it points down Y, it's a Coronal slice (face-to-back).
    # If it points down Z, it's an Axial slice (head-to-toe).
    return {0: "sagittal", 1: "coronal", 2: "axial"}.get(idx, "unknown")


def get_orientation_from_desc(description):
    """
    [Function 1.2: Used by io_utils.py:126 inside discover_series]
    Heuristic orientation guess from a DICOM series description string.
    """
    # Convert the human-written description string (like "T2_AXIAL_1") to all lowercase.
    # This is important because humans are inconsistent with capitalizing file names!
    d = description.lower()
    
    # Check if any common axial keywords are hidden inside the description string.
    if any(k in d for k in ["_tra", "_ax", "axial", "transv", "tra_"]):
        return "axial"
        
    # Check for coronal keywords.
    if any(k in d for k in ["_cor", "cor_"]):
        return "coronal"
        
    # Check for sagittal keywords.
    if any(k in d for k in ["_sag", "sag_"]):
        return "sagittal"
        
    # If the doctor wrote something totally custom, we return unknown.
    return "unknown"


def load_dicom_series(series_path):
    """
    [Function 1.1: Used by io_utils.py:108 inside discover_series]
    Load a DICOM series directory as a SimpleITK image.
    """
    try:
        # Initialize the SimpleITK Series Reader, which knows how to read hundreds of .dcm files at once.
        reader = sitk.ImageSeriesReader()
        
        # Ask SimpleITK to scan the folder and give us the file names of all the DICOM slices in order.
        names = reader.GetGDCMSeriesFileNames(series_path)
        
        # If the folder was empty or contained no DICOMs, safely exit without crashing.
        if not names:
            logger.debug(f"No DICOM files in: {series_path}")
            return None, 0
            
        # Tell the reader exactly which files to read.
        reader.SetFileNames(names)
        # Turn on the metadata dictionaries so we can read the hidden hospital/scanner tags attached to the files!
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        
        # Execute the massive read operation to combine the 2D files into a single 3D volume in memory.
        image = reader.Execute()
        
        try:
            # "0008|103e" is the universal DICOM tag ID for the "Series Description" (what the tech typed into the computer).
            # We strip() it to remove accidental spaces.
            desc = reader.GetMetaData(0, "0008|103e").strip()
        except Exception:
            desc = ""
            
        # We manually attach this description string to our 3D Image object so we can use it later!
        image.SetMetaData("series_desc", desc)
        
        # Return the loaded 3D volume, and the number of slices (Z-axis).
        return image, image.GetSize()[2]
    except Exception as e:
        logger.warning(f"Failed to load DICOM series '{series_path}': {e}")
        return None, 0


def discover_series(study_path):
    """
    [Function 1: Used first in the main pipeline at preprocess_2d.py:207]
    Enumerate all DICOM series (SE* subdirectories) under a study path and
    return metadata for each successfully loaded series.
    """
    series_list = []
    
    # If the folder doesn't exist, don't crash, just return an empty list.
    if not os.path.isdir(study_path):
        logger.warning(f"Study path not found: {study_path}")
        return series_list

    # Loop alphabetically through all the sub-folders (like SE0, SE1, SE2).
    for se_name in sorted(os.listdir(study_path)):
        se_path = os.path.join(study_path, se_name)
        if not os.path.isdir(se_path):
            continue

        # Try to read the entire sub-folder into a 3D volume.
        image, n = load_dicom_series(se_path)
        
        # If the read failed, or if it's just a single 2D scout image (n < 2), skip it!
        if image is None or n < 2:
            continue

        # Extract the Series Description we attached earlier.
        desc = image.GetMetaData("series_desc") if image.HasMetaDataKey("series_desc") else ""

        # First, try to guess the orientation by looking at the folder's name (e.g. "Axial_T2").
        orient = "unknown"
        se_name_lower = se_name.lower()
        if "axial" in se_name_lower:
            orient = "axial"
        elif "coronal" in se_name_lower:
            orient = "coronal"
        elif "sagittal" in se_name_lower:
            orient = "sagittal"

        # If the folder name was just "SE0", we fallback to guessing using the hidden DICOM description!
        if orient == "unknown":
            orient = get_orientation_from_desc(desc) if desc else "unknown"

        logger.debug(f"  {se_name}: orient={orient}, n={n}, desc='{desc}'")
        
        # Append all the information about this scan as a dictionary so the main pipeline can use it.
        series_list.append({
            "path":        se_path,
            "image":       image,
            "n_slices":    n,
            "orientation": orient,
            "series_desc": desc,
        })

    return series_list


def select_best_series(series_list, orientation):
    """
    [Unused Helper Function]
    From a list of series dicts, return the single best candidate for the
    requested orientation.
    """
    # Filter the list so we only have scans matching the orientation we want.
    candidates = [s for s in series_list if s["orientation"] == orientation]
    if not candidates:
        return None
        
    # If the patient had multiple axial scans (e.g. one with contrast, one without),
    # we automatically pick the one with the most slices, as it gives the AI more data!
    return max(candidates, key=lambda s: s["n_slices"])


def get_all_valid_series(series_list, orientation):
    """
    [Unused Helper Function]
    Return all series that match the requested orientation.
    """
    # A simple Python list comprehension to filter the list.
    return [s for s in series_list if s["orientation"] == orientation]
