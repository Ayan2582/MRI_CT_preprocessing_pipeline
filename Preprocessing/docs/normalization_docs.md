# 📖 Code Docs: `normalization.py`

This module is responsible for Numpy-level array manipulation, such as clipping, mapping contrast levels, determining if slices are empty, and cropping.

---

```python
import numpy as np # Used for fast numerical matrix operations, which is essential for image processing.

def normalize_ct_slice(slice_2d, window_min=-200.0, window_max=300.0):
    """
    [Function 8: Called via pipeline_core.py:134, originating from main script preprocess_2d.py:262]
    Apply a soft-tissue HU window and normalise a CT slice to [0.0, 1.0].
    """
    # np.clip forces all pixel values below window_min to become window_min, and all values above window_max to become window_max.
    # We cast to float32 first to ensure precision and compatibility with neural networks.
    # This is important because raw CT data (Hounsfield Units) spans a massive range (-1000 for air, +3000 for bone).
    # Clipping acts as an "attention mechanism", forcing the AI to only look at soft tissue differences.
    s = np.clip(slice_2d.astype(np.float32), window_min, window_max)
    
    # Subtracting the minimum value shifts the entire matrix so the lowest value is exactly 0.0.
    # Dividing by the total range (max - min) scales the matrix so the highest value is exactly 1.0.
    # This is important because Deep Learning models train much faster when inputs are normalized between 0 and 1.
    return (s - window_min) / float(window_max - window_min)


def compute_mri_percentiles(volume_arr, p_low=0.5, p_high=99.5):
    """
    [Function 7: Called via pipeline_core.py:110, originating from main script preprocess_2d.py:262]
    Compute robust intensity bounds for MRI normalisation using percentiles
    computed on the non-zero voxels of the entire volume.
    """
    # Extract only pixels that are strictly greater than 0, and flatten them into a 1D list (.ravel()).
    # This is important because the black background (0) dominates MRI images, and including it would skew the percentiles.
    nonzero = volume_arr[volume_arr > 0].ravel()
    
    # If there are fewer than 100 non-zero pixels, the image is mostly empty.
    if len(nonzero) < 100:
        # Fall back to using all pixels (including 0) to prevent the percentile math from crashing.
        nonzero = volume_arr.ravel()
        
    # Calculate the exact pixel value at the bottom 0.5% (ignoring extreme dark noise).
    p1  = float(np.percentile(nonzero, p_low))
    
    # Calculate the exact pixel value at the top 99.5% (ignoring extreme bright noise like metal artifacts).
    # This is important because MRI scanners don't have standard units, so we must calculate min/max dynamically per patient.
    p99 = float(np.percentile(nonzero, p_high))
    
    # Return the robust minimum and maximum values that represent the true tissue contrast.
    return p1, p99


def normalize_mri_slice(slice_2d, p1, p99):
    """
    [Function 9: Called via pipeline_core.py:135, originating from main script preprocess_2d.py:262]
    Clip an MRI slice to [p1, p99] and normalise to [0.0, 1.0].
    """
    # Calculate the total range between the bright and dark tissue percentiles.
    denom = float(p99 - p1)
    
    # If the range is essentially 0, it means the image has no contrast (it's a blank black image).
    if denom < 1e-8:
        # Return an array of pure zeros to prevent a "Divide by Zero" crash in Python.
        return np.zeros_like(slice_2d, dtype=np.float32)
        
    # Clip the MRI slice so any extreme noise spikes are flattened into the p1 (min) and p99 (max) bounds.
    s = np.clip(slice_2d.astype(np.float32), p1, p99)
    
    # Shift the minimum value to 0.0 and divide by the range to scale the entire image into the [0, 1] decimal space.
    return (s - p1) / denom


def center_crop_pad(arr, target=256):
    """
    [Function 11: Called via pipeline_core.py:147, originating from main script preprocess_2d.py:262]
    Resize a 2-D array to (target x target) by symmetrically cropping if
    larger, or zero-padding if smaller.
    """
    # Extract the current height (h) and width (w) of the image array.
    h, w = arr.shape

    # Calculate how much padding is needed. If the target is 256 but height is 200, we need 56 pixels of padding.
    # The max(0, ...) ensures we don't calculate negative padding if the image is already bigger than the target.
    ph = max(0, target - h)
    pw = max(0, target - w)
    
    # If the image is smaller than the target in either dimension...
    if ph > 0 or pw > 0:
        # Pad the array with black pixels (constant_values=0.0).
        # We divide the padding by 2 (e.g. ph // 2) so half the padding goes on top, and the other half goes on bottom.
        # This is important to ensure the anatomical structure remains perfectly centered in the new image.
        arr = np.pad(arr,
                     ((ph // 2, ph - ph // 2),
                      (pw // 2, pw - pw // 2)),
                     mode="constant",
                     constant_values=0.0)

    # Re-calculate the height and width, because if we added padding, the array is now larger.
    h, w = arr.shape
    
    # Calculate how many pixels we need to shave off the top and left edges to get exactly to the target size.
    # By dividing by 2, we find the exact center coordinate of the image.
    sh = (h - target) // 2
    sw = (w - target) // 2
    
    # Crop the array starting from the center offset (sh, sw) and ending exactly at the target width/height.
    # This guarantees the output is always a perfect (target x target) square, which neural networks require.
    return arr[sh : sh + target, sw : sw + target]


def is_background_slice(arr, intensity_thresh=0.02, bg_fraction=0.90):
    """
    [Function 10: Called via pipeline_core.py:140, originating from main script preprocess_2d.py:262]
    Return True if the 2-D slice is predominantly background (uninformative).
    """
    # arr <= intensity_thresh creates a True/False mask where all dark pixels (below 0.02) are True.
    # np.mean calculates the average of this mask, effectively giving the percentage of the image that is dark.
    # If the percentage of dark pixels is greater than the bg_fraction (e.g. 0.90, or 90%)...
    # Then it returns True (this slice is background air and should be discarded).
    # This is important because feeding empty black slices to a neural network wastes computation and hurts learning.
    return float(np.mean(arr <= intensity_thresh)) > bg_fraction

```
