# 📖 Code Docs: `image_processing.py`

This module is responsible for the complex 3D mathematics required to physically manipulate, distort, align, and correct the anatomical scans using SimpleITK.

---

```python
import logging # Used to print warnings and info to the console without crashing the program.
import numpy as np # Matrix and array operations.
import SimpleITK as sitk # The industry standard library for Medical Image Processing (Simple Insight Toolkit).

logger = logging.getLogger(__name__)

def apply_n4_bias_correction(
    image,
    shrink_factor: int = 4,
    n_iterations: list = None,
    convergence_threshold: float = 0.001,
):
    """
    [Function 3: Called via pipeline_core.py:74, originating from main script preprocess_2d.py:262]
    Apply N4 ITK bias field correction to an MRI volume slice-by-slice (2D).
    """
    # If the user doesn't provide a list of iterations, use a default 4-level pyramid of 50 iterations each.
    if n_iterations is None:
        n_iterations = [50, 50, 50, 50]

    # -- Cast to Float32
    # SimpleITK requires image arrays to be 32-bit floats before it can run the complex N4 math.
    image_f32 = sitk.Cast(image, sitk.sitkFloat32)
    
    # Initialize the specific N4 Bias Field Corrector tool from SimpleITK.
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # Set how many times the algorithm will try to fit the B-Spline grid before giving up.
    corrector.SetMaximumNumberOfIterations(n_iterations)
    # Set the mathematical threshold where the algorithm decides it has successfully "converged" on a solution.
    corrector.SetConvergenceThreshold(convergence_threshold)

    # -- Process slice by slice
    # Get the X, Y, Z dimensions of the 3D volume.
    size = image_f32.GetSize()
    # Extract the Z-dimension (number of slices).
    depth = size[2]
    
    # We will temporarily store each mathematically corrected 2D slice in this list.
    corrected_slices = []
    
    # Loop vertically through every single slice in the volume (from Z=0 to the top of the head).
    for z in range(depth):
        # Extract exactly one 2D slice from the 3D volume.
        slice_2d = image_f32[:, :, z]
        
        # Save base origin and direction from the very first slice.
        # This is critically important because when we stitch the 2D slices back into a 3D volume later,
        # SimpleITK will crash if even a single decimal place in their physical coordinates doesn't match perfectly.
        if z == 0:
            base_origin = slice_2d.GetOrigin()
            base_direction = slice_2d.GetDirection()
        
        # Create a Tissue Mask using Otsu Thresholding.
        # Otsu's method automatically figures out which pixels are "air" (background) and which are "tissue".
        # This is important so the N4 algorithm doesn't waste time trying to correct the lighting of empty air.
        mask = sitk.OtsuThreshold(slice_2d, 0, 1, 200)
        
        # Shrink for speed
        if shrink_factor > 1:
            # We shrink the image and the mask by the shrink_factor (default 4x smaller).
            # This is incredibly important because N4 is mathematically heavy. Shrinking it makes the math 16x faster
            # without losing the general shape of the bias field gradient.
            slice_small = sitk.Shrink(slice_2d, [shrink_factor, shrink_factor])
            mask_small  = sitk.Shrink(mask,     [shrink_factor, shrink_factor])
        else:
            slice_small = slice_2d
            mask_small  = mask
            
        # Run N4 on the small 2D slice
        try:
            # Execute the mathematical fitting algorithm on the shrunken tissue mask.
            corrector.Execute(slice_small, mask_small)
            # The algorithm calculated a "bias field" (the lighting gradient). We ask it to expand that small gradient back up to full size.
            log_bias = corrector.GetLogBiasFieldAsImage(slice_2d)
            # We divide the original full-size image by the exponential of the bias field to perfectly flatten the lighting!
            corrected_slice = slice_2d / sitk.Exp(log_bias)
        except Exception as e:
            # If the slice is completely empty air, the math will crash. We catch the crash and just return the raw uncorrected slice.
            logger.warning(f"N4 failed on slice {z}, using uncorrected slice. Error: {e}")
            corrected_slice = slice_2d
            
        # Force identical physical space to satisfy JoinSeries strict checks
        # We manually overwrite the physical coordinates of this slice with the ones we saved at z=0.
        corrected_slice.SetOrigin(base_origin)
        corrected_slice.SetDirection(base_direction)
        
        # Add the perfectly lit slice to our list.
        corrected_slices.append(corrected_slice)

    # -- Stack back into 3D
    # JoinSeries glues our list of 2D slices back together into a single 3D volume block.
    corrected_vol = sitk.JoinSeries(corrected_slices)
    
    # We must explicitly copy the Spacing (mm between pixels), Direction (tilt), and Origin (physical XYZ) 
    # from the original image back to our new image. This ensures the DICOM headers remain medically accurate.
    corrected_vol.SetSpacing(image_f32.GetSpacing())
    corrected_vol.SetDirection(image_f32.GetDirection())
    corrected_vol.SetOrigin(image_f32.GetOrigin())

    return corrected_vol


def resample_inplane(image, target_spacing=1.0, is_ct=True):
    """
    [Function 4: Called via pipeline_core.py:86, originating from main script preprocess_2d.py:262]
    Resample a 3-D volume to a uniform in-plane resolution while leaving the
    through-plane (z) spacing unchanged.
    """
    # Extract the original pixel spacing (e.g., 0.48mm x 0.48mm x 3.0mm)
    orig_sp = image.GetSpacing()   # (sx, sy, sz)
    # Extract the total pixel dimensions (e.g., 512 x 512 x 18 slices)
    orig_sz = image.GetSize()      # (nx, ny, nz)

    # We enforce that the X and Y axes must become exactly the target_spacing (e.g., 1.0mm).
    new_sx = float(target_spacing)
    new_sy = float(target_spacing)
    # But we leave the Z axis (thickness of slices) completely alone so we don't hallucinate fake anatomy between slices.
    new_sz = orig_sp[2]            

    # To calculate the new array size (e.g. going from 512 to 256), we multiply the original size by the ratio of the spacings.
    new_nx = max(1, int(round(orig_sz[0] * orig_sp[0] / new_sx)))
    new_ny = max(1, int(round(orig_sz[1] * orig_sp[1] / new_sy)))
    new_nz = orig_sz[2]

    # Initialize the SimpleITK Resampler (the math engine that scales images).
    resampler = sitk.ResampleImageFilter()
    # Tell the resampler what the new physical scale is.
    resampler.SetOutputSpacing((new_sx, new_sy, new_sz))
    # Tell the resampler how many pixels wide the new image will be.
    resampler.SetSize((new_nx, new_ny, new_nz))
    # Preserve the original physical orientation and origin coordinates.
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    # Apply an identity transform (meaning we are only scaling, not rotating).
    resampler.SetTransform(sitk.Transform())
    # Use Linear Interpolation (drawing straight lines between pixels to guess the color of the new pixels).
    resampler.SetInterpolator(sitk.sitkLinear)
    # If the image is scaled up and creates empty borders, fill those borders with -1024 HU (Air) for CTs, or 0 for MRIs.
    resampler.SetDefaultPixelValue(-1024 if is_ct else 0)

    # Execute the heavy math and return the rescaled 3D volume.
    return resampler.Execute(image)


def resample_mri_to_ct_grid(mri_image, ct_image, default_pixel_value=0.0):
    """
    [Function 5: Called via pipeline_core.py:92, originating from main script preprocess_2d.py:262]
    Project the MRI image directly onto the CT image's physical coordinate grid.
    """
    # Create a shallow reference copy of the MRI image to avoid breaking the original memory block.
    mri_aligned = mri_image
    
    # This is a critical mathematical hack! If the MRI patient was tilted by 0.1 degrees compared to the CT patient, 
    # trying to align them across only 18 slices will cause the algorithm to "shear" the MRI and ruin it.
    # By forcing the MRI to mathematically pretend it has the exact same directional tilt as the CT, we prevent shearing!
    mri_aligned.SetDirection(ct_image.GetDirection())
    
    # Initialize the math engine.
    resampler = sitk.ResampleImageFilter()
    # Instead of manually providing sizes and spacings, we tell the engine to perfectly mimic the CT image's grid!
    resampler.SetReferenceImage(ct_image)
    # Use Linear Interpolation.
    resampler.SetInterpolator(sitk.sitkLinear)
    # Fill empty space with black.
    resampler.SetDefaultPixelValue(default_pixel_value)
    # Rely entirely on the physical DICOM origins to map the atoms of the MRI directly onto the atoms of the CT.
    resampler.SetTransform(sitk.Transform())
    
    # Execute the massive mathematical projection.
    return resampler.Execute(mri_aligned)


def register_2d_rigid(fixed_slice, moving_slice, learning_rate=1.0, num_iters=100):
    """
    [Function 7.5: Called optionally via pipeline_core.py:130, originating from main script preprocess_2d.py:262]
    Perform 2D Rigid Registration (Translation + Rotation) using Mattes Mutual Information.
    Aligns the moving MRI slice to the fixed CT slice.
    """
    # Registration (alignment math) absolutely requires 32-bit floats.
    fixed = sitk.Cast(fixed_slice, sitk.sitkFloat32)
    moving = sitk.Cast(moving_slice, sitk.sitkFloat32)
    
    # We initialize the math by attempting to match the geometric center of the MRI slice with the center of the CT slice.
    # An Euler2DTransform means the MRI is only allowed to Shift (Up/Down/Left/Right) and Rotate (Spin). It cannot Stretch.
    initial_transform = sitk.CenteredTransformInitializer(
        fixed, 
        moving, 
        sitk.Euler2DTransform(), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    
    # Initialize the Deep Learning style Registration Engine.
    registration_method = sitk.ImageRegistrationMethod()
    
    # We use "Mattes Mutual Information" because CT and MRI have completely different colors (bones are white on CT, black on MRI).
    # Mutual Information looks for structural patterns instead of literal color matches!
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    # We only sample 20% of the pixels randomly to make the math run 5x faster.
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.2)
    
    # Use linear drawing between pixels.
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # We use Gradient Descent (just like a Neural Network!) to slowly nudge the MRI until it matches the CT.
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=0.1, 
        numberOfIterations=num_iters, 
        convergenceMinimumValue=1e-6, 
        convergenceWindowSize=10
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # We build an image pyramid (Shrink 4x -> 2x -> 1x).
    # The algorithm aligns a tiny, blurry version first (preventing it from getting stuck on local details),
    # then scales up to fine-tune the alignment on the high-res image.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Load our center-matched starting position.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    try:
        # Execute the Gradient Descent to find the perfect Rotation and Shift.
        final_transform = registration_method.Execute(fixed, moving)
        
        # Now that we have the mathematical transform, we use a Resampler to physically warp the MRI image using that transform.
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        resampler.SetTransform(final_transform)
        return resampler.Execute(moving)
    except Exception as e:
        # If the Gradient Descent crashes (e.g. empty image), warn us and return the raw MRI.
        logger.warning(f"Registration failed: {e}. Returning unregistered slice.")
        return moving


def volume_to_slices(image):
    """
    [Function 6: Called via pipeline_core.py:98, originating from main script preprocess_2d.py:262]
    Convert a SimpleITK 3-D image to a list of 2-D numpy arrays.
    """
    # SimpleITK stores volumes as (x, y, z) in physical space.
    # GetArrayFromImage converts it to a standard Numpy Matrix, which uses (z, y, x) layout!
    # This is important because Python and AI libraries prefer the slice index (z) to be the first array dimension.
    arr = sitk.GetArrayFromImage(image)   # shape: (z, y, x)
    
    # We loop through the Z axis and pull out every single (y, x) 2D slice, returning them as a neat Python list.
    return [arr[i, :, :] for i in range(arr.shape[0])]

```
