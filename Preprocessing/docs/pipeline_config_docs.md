# 📖 Learn the Code: `pipeline_config.py`

This file is the **Configuration Hub** of your entire preprocessing pipeline. By centralizing all the hard-coded variables (like image sizes, Hounsfield windows, and folder paths) into one dictionary, you can easily tweak the pipeline without ever having to touch the complex logic inside the other scripts.

---

## 🎯 Global Settings

These are the default settings that govern the input/output paths and base targets for the pipeline.

- `DATA_ROOT`: Where the raw DICOM files live.
- `OUTPUT_DIR`: Where the final `.npy` and `.png` files will be saved.
- `TARGET_SPACING_MM = 1.0`: The isotropic resolution we want. This means every pixel in our output 2D image will represent exactly 1.0 mm x 1.0 mm of physical space in the real world.

---

## 🖼️ Intensity Normalization Defaults

Neural networks perform best when input data is scaled between `[0, 1]` or `[-1, 1]`. 

### MRI Normalization
MRI scanners do not output standard units (unlike CT). The intensity changes based on the machine.
- `MRI_PERCENTILE_LOW = 0.5`
- `MRI_PERCENTILE_HIGH = 99.5`
> **Concept:** Instead of mapping the absolute minimum to `0` and maximum to `1` (which could include extreme noise spikes), we chop off the bottom `0.5%` and top `0.5%` of pixel values. Everything else is safely squished into the `[0, 1]` range.

### CT Normalization
CT scanners output standard **Hounsfield Units (HU)**. 
- `CT_WINDOW_MIN_HU = -200`
- `CT_WINDOW_MAX_HU = 300`
> **Concept:** By default, anything below `-200` HU becomes pure black (`0`), and anything above `300` HU becomes pure white (`1`). This specific default window emphasizes soft tissues.

---

## 🧠 Region Profiles (`REGION_PROFILES`)

Not all body parts are the same size, nor do they require the same visual contrast. This dictionary allows the pipeline to intelligently adapt its parameters depending on the patient's body part!

### How it works:
When the pipeline runs, it checks which region the patient belongs to, and loads these custom parameters:

1. **`target_size`**: The output pixel dimensions (e.g., `256` means `256x256`).
   - *Example:* Brains are small and fit in `256x256`. Abdomens are large and need `384x384`.
2. **`ct_win_min` & `ct_win_max`**: The Hounsfield Unit bounds.
   - *Example (Brain):* `[0, 80]` - Focuses intensely on gray/white matter, making the dense skull turn pure white and ignored by the AI.
   - *Example (Spine/Knee):* `[-200, 300]` - Focuses on muscles, ligaments, and fluid.
   
> **Pro Tip:** As we discussed, you can easily add the MicroDicom presets (like Lung `[-1200, 400]`) to this dictionary to further enhance the AI's attention mechanism!

---

## 🗺️ Patient Mapping (`PREFIX_TO_REGION`)

This is a simple dictionary mapping tool.
- It maps the patient prefix (like `PA0`) to the region (like `BRAIN`).
- **Why?** When the script reads the folder `PA0_Ranjeet`, it extracts `PA0`, looks it up in this dictionary, realizes it is a `BRAIN`, and immediately loads the `BRAIN` Region Profile mentioned above!
