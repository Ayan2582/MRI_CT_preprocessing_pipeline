# 📖 Code Docs: `export_utils.py`

This module is responsible for saving our processed data structures back to the hard drive in a format that PyTorch can easily read.

---

```python
import os # Used for interacting with the operating system (e.g., creating folders, checking paths).
import numpy as np # Used for high-performance matrix and array mathematics.
from PIL import Image # Python Imaging Library, used here to create and save PNG images.

def save_npy(arr, path):
    """
    [Function 12: Called via pipeline_core.py:178, originating from main script preprocess_2d.py:266]
    Save a float32 2-D array as a .npy file.
    Parent directories are created automatically.
    """
    # Create all necessary parent directories for the given path.
    # 'exist_ok=True' prevents the code from crashing if the folder already exists.
    # This is important to ensure the file save doesn't crash on a missing folder.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the numpy array to the specified path as a binary .npy file.
    # We cast it to 'float32' to ensure it takes up less space and is natively readable by PyTorch.
    # This is important because deep learning models train much faster on .npy binaries than on compressed images.
    np.save(path, arr.astype(np.float32))


def save_preview_png(ct_arr, mri_arr, path):
    """
    [Function 13: Called via pipeline_core.py:185, originating from main script preprocess_2d.py:266]
    Save a side-by-side CT | MRI comparison PNG for visual quality control.
    """
    # Ensure the parent directory (like 'previews/') exists before saving.
    # This prevents the script from crashing if this is the first image being saved.
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Define a small helper function to convert float arrays [0, 1] into 8-bit integers [0, 255].
    # This is important because PNG images can only understand integer values up to 255.
    def to_uint8(a):
        # np.clip ensures no values exceed 1.0 or fall below 0.0, avoiding overflow glitches.
        # Multiplying by 255 scales the decimals to the 0-255 range required for pixels.
        # .astype(np.uint8) strictly converts the data type to 8-bit unsigned integer.
        return (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8)

    # Create a vertical grey divider line (a 2-pixel wide column) to separate the two images.
    # ct_arr.shape[0] gives the height of the image, so the bar is as tall as the image.
    # 180 is the color (light grey), and uint8 is the required pixel data type.
    # This is visually important so humans can easily see where the CT ends and the MRI begins.
    divider = np.full((ct_arr.shape[0], 2), 180, dtype=np.uint8)
    
    # Horizontally stack the CT image, the divider, and the MRI image side-by-side into a single wide matrix.
    # This is important because we want both slices on the same PNG file for easy visual comparison.
    combined = np.hstack([to_uint8(ct_arr), divider, to_uint8(mri_arr)])
    
    # Convert the raw numpy pixel matrix into an actual Image object.
    # mode="L" tells the PIL library that this is a Grayscale (Luminance) image, not RGB.
    # .save(path) writes the PNG file to the hard drive.
    Image.fromarray(combined, mode="L").save(path)
```
