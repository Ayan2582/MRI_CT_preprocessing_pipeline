# 📖 Learn the Code: `pipeline_config.py`

This file is the **Configuration Hub** of your entire preprocessing pipeline. By centralizing all the hard-coded variables (like Hounsfield windows and folder paths) into one dictionary, you can easily tweak the pipeline without ever having to touch the complex logic inside the other scripts.

> **Note:** there is no output image size setting. The pipeline does not crop, pad, or resize — the GAN's dataloader owns that decision. See [`ct_pipeline_docs.md`](./ct_pipeline_docs.md) §10.

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

## 🧲 N4 Bias Field Correction (MRI only)

*Added 2026-08-08, when N4 was moved from a 2D per-slice fit to a single 3D fit.
The full reasoning is in [`mri_pipeline_docs.md` §5](./mri_pipeline_docs.md).*

The bias field is the smooth brightness gradient an MRI receive coil imposes on the image. N4 models it as a cubic B-spline and divides it out. Everything here is about **how much freedom that spline gets**, because the danger is one-sided: too little freedom leaves shading behind, too much lets the "bias field" bend tightly enough to follow real anatomy and flatten away genuine tissue contrast.

### The mechanical settings

- `N4_SHRINK_FACTOR = 4` — fit on a 4× downsampled grid for ~16× the speed. **In-plane only**; the slice axis is never shrunk, because these stacks are only 15–24 slices deep.
- `N4_SPLINE_ORDER = 3` — cubic. ITK's spline order is a single global number, so it cannot differ per axis; all the anisotropy has to come from the control point counts instead.
- `N4_ITERATIONS = 100`, `N4_CONVERGENCE = 0.001` — per fitting level.
- `N4_FITTING_LEVELS = 1` — **read the note before raising this.** ITK doubles the control point mesh on every extra fitting level, in all three axes at once. With 4 levels the `(4,4,4)` default silently becomes an `(11,11,11)` mesh, and there is no way to refine in-plane while keeping the slice axis coarse. One level makes the numbers below mean exactly what they say. If you do raise it, `plan_n4_control_points()` back-solves the initial mesh so the *final* one still lands near the targets, and logs what it actually achieved.

### `N4_CONTROL_POINT_SPACING_MM` — the in-plane knob

> **Concept:** This is a *spacing in millimetres*, not a count. Counts are derived per series from its actual field of view.

Why it has to work that way: the MRI series here range from a 180 mm knee to a 400 mm abdomen. A fixed count of, say, 12 control points would give the knee a 15 mm mesh and the abdomen a 33 mm mesh — two completely different amounts of freedom, and the 15 mm one is well into the range where N4 starts eating anatomy. Fixing the *spacing* gives every series the same stiffness.

```
ncp = spline_order + round(FOV_mm / target_spacing_mm)     clamped to
                                                           [N4_CONTROL_POINTS_INPLANE_MIN,
                                                            N4_CONTROL_POINTS_INPLANE_MAX]
```

The targets are keyed by orientation, because in SimpleITK index order axes 0 and 1 are always the in-plane ones, but *which anatomical direction* each carries depends on the acquisition plane:

| orientation | axis 0 | axis 1 | axis 2 (through-plane) |
|---|---|---|---|
| `axial` | L-R (35 mm) | A-P (30 mm) | S-I |
| `coronal` | L-R (35 mm) | S-I (25 mm) | A-P |
| `sagittal` | A-P (30 mm) | S-I (25 mm) | L-R |

So each anatomical direction gets the same stiffness whichever plane it appears in:

- **L-R, 35 mm** — body/spine coil shading across the patient is broad and roughly symmetric. Least freedom needed.
- **A-P, 30 mm** — anterior array against posterior spine coil is the strongest single gradient in most of these scans.
- **S-I, 25 mm** — coil arrays are segmented along the bore axis, so sensitivity changes fastest head-to-foot. Most freedom needed.

> ⚠️ These are reasoned from coil geometry, **not** tuned against a measured criterion on this dataset. They are the first thing to change if N4 is visibly flattening anatomy (raise them) or leaving shading behind (lower them).

### `N4_CONTROL_POINTS_THROUGH_PLANE = 4` — the one that is not derived

4 is the fewest control points a cubic spline can have: one single span across the entire slab. Deliberately the most rigid field expressible, and it stays 4 whether the slab is 100 mm or 200 mm deep.

The reason is that these are 2D multi-slice acquisitions with 5–10 mm slices, where through-plane intensity variation is mostly **not** a bias field — it is slice profile, cross-talk, and per-slice excitation differences. Those are genuinely discontinuous between neighbours, and any mesh flexible enough to chase them is flexible enough to chase anatomy. One rigid span lets N4 remove a smooth head-to-foot coil falloff and express nothing sharper.

> **Raising this is the single change most likely to reintroduce the slice-to-slice stepping the 3D rewrite removed.**

---

## 🧠 Region Profiles (`REGION_PROFILES`)

Different body parts require different visual contrast. This dictionary allows the pipeline to intelligently adapt its CT windowing depending on the patient's body part!

### How it works:
When the pipeline runs, it checks which region the patient belongs to, and loads these custom parameters:

1. **`ct_win_min` & `ct_win_max`**: The Hounsfield Unit bounds.
   - *Example (Brain):* `[0, 80]` - Focuses intensely on gray/white matter, making the dense skull turn pure white and ignored by the AI.
   - *Example (Spine/Knee):* `[-200, 300]` - Focuses on muscles, ligaments, and fluid.
   
> **Pro Tip:** As we discussed, you can easily add the MicroDicom presets (like Lung `[-1200, 400]`) to this dictionary to further enhance the AI's attention mechanism!

---

## 🗺️ Patient Mapping (`PREFIX_TO_REGION`)

This is a simple dictionary mapping tool.
- It maps the patient prefix (like `PA0`) to the region (like `BRAIN`).
- **Why?** When the script reads the folder `PA0_Ranjeet`, it extracts `PA0`, looks it up in this dictionary, realizes it is a `BRAIN`, and immediately loads the `BRAIN` Region Profile mentioned above!
