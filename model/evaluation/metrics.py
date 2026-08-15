"""
metrics.py
──────────
Validation metrics, all mask-aware and all region-aware.

READ THIS BEFORE TRUSTING ANY NUMBER THIS FILE PRODUCES.

1. THERE IS NO SINGLE VALID "MAE IN HU" FOR THIS DATASET.
   The CT arrays were normalised to [0,1] with a HU window that depends on body
   region (Preprocessing/pipeline_config.py:208-229):

       brain            0 .. 80 HU     -> 1.0 normalised unit =  80 HU
       abdomen       -160 .. 240 HU    -> 1.0 normalised unit = 400 HU
       MSK / spine   -200 .. 300 HU    -> 1.0 normalised unit = 500 HU

   The same normalised error is therefore worth 80 HU on a brain slice and
   500 HU on a spine slice. Averaging HU errors across regions produces a number
   that is dominated by whichever regions happen to have wide windows, and that
   moves when the composition of the validation set changes rather than when the
   model does.

   So: mae_hu is reported PER REGION and macro-averaged, never pooled. The
   single scalar used to rank epochs and select checkpoints is `mae_norm`, in
   normalised units, which is the only cross-region comparable quantity here.

2. BONE METRICS ARE MEANINGLESS ON BRAIN SLICES.
   The brain window tops out at 80 HU. Cortical bone is 300-2000 HU, so on a
   brain slice every bone voxel — and every other voxel above 80 HU — is clipped
   to exactly 1.0. Bone is not merely hard to measure there, it is not present
   in the data as a distinguishable value. A bone Dice computed over brain
   slices would score agreement about a saturated constant, which is trivially
   near-perfect and completely uninformative.

   So bone Dice and the HU-band MAEs skip the regions named in
   eval.bone_metrics_exclude_regions (brain, by default), and every reported
   value carries the sample count it was computed from so the exclusion is
   visible rather than buried.

3. PADDING IS EXCLUDED EVERYWHERE.
   Validation slices are zero-padded up to a multiple of 256. Those pixels are
   identical in prediction and target, so including them would make a heavily
   padded 180x180 slice appear more accurate than a 430x430 one for reasons that
   have nothing to do with the model. Every metric here is computed over the
   dataset's validity mask.
"""

import logging
import math
from collections import defaultdict

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

METRIC_REGISTRY = {}


def register(name):
    """Register a per-sample metric so adding one is a function, not a refactor."""
    def decorator(fn):
        METRIC_REGISTRY[name] = fn
        return fn
    return decorator


# ── Helpers ──────────────────────────────────────────────────────────────────

def _masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def to_hu(norm, hu_min, hu_max):
    """Normalised [0,1] -> Hounsfield units, using this sample's own window."""
    return norm * (hu_max - hu_min) + hu_min


def hu_to_norm(hu, hu_min, hu_max):
    """Hounsfield units -> normalised [0,1]. May fall outside [0,1] if clipped."""
    return (hu - hu_min) / (hu_max - hu_min)


def _gaussian_window(size=11, sigma=1.5, device=None, dtype=torch.float32):
    coords = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] @ g[None, :]


def ssim_map(pred, target, data_range=1.0, window_size=11, sigma=1.5):
    """
    Per-pixel SSIM map for [B,1,H,W] tensors in [0, data_range].

    Implemented here rather than pulled from scikit-image for three reasons: it
    keeps the dependency list compatible with the repo-wide numpy<2 pin, it runs
    on the GPU, and it stays batched instead of round-tripping every validation
    slice to numpy.
    """
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    window = _gaussian_window(window_size, sigma,
                              device=pred.device, dtype=pred.dtype)
    window = window.expand(pred.size(1), 1, window_size, window_size).contiguous()
    pad = window_size // 2

    def filt(x):
        return F.conv2d(x, window, padding=pad, groups=x.size(1))

    mu_p, mu_t = filt(pred), filt(target)
    mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t

    sigma_p = filt(pred * pred) - mu_p2
    sigma_t = filt(target * target) - mu_t2
    sigma_pt = filt(pred * target) - mu_pt

    numerator = (2 * mu_pt + c1) * (2 * sigma_pt + c2)
    denominator = (mu_p2 + mu_t2 + c1) * (sigma_p + sigma_t + c2)
    return numerator / denominator


# ── Per-sample metrics ───────────────────────────────────────────────────────
# Each takes (pred, target, mask, hu_min, hu_max) with pred/target in [0,1] and
# a single sample's scalars, and returns a float or None ("not applicable").

@register("mae_norm")
def metric_mae_norm(pred, target, mask, hu_min, hu_max):
    """
    Mean absolute error in normalised units.

    THE MODEL-SELECTION SCALAR. Comparable across regions in a way that no
    HU-denominated number is, because it does not carry a region-dependent
    scale factor. Lower is better.
    """
    return float(_masked_mean((pred - target).abs(), mask))


@register("mae_hu")
def metric_mae_hu(pred, target, mask, hu_min, hu_max):
    """
    Mean absolute error in Hounsfield units, for THIS sample's window.

    Only ever aggregated within a region — see the module docstring.
    """
    scale = float(hu_max - hu_min)
    return float(_masked_mean((pred - target).abs(), mask)) * scale


@register("psnr")
def metric_psnr(pred, target, mask, hu_min, hu_max):
    """Peak signal-to-noise ratio over valid pixels, data range 1.0."""
    mse = float(_masked_mean((pred - target) ** 2, mask))
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


@register("ssim")
def metric_ssim(pred, target, mask, hu_min, hu_max):
    """
    Structural similarity over valid pixels.

    The mask is eroded by the window radius before averaging: an SSIM value
    computed at a pixel whose 11x11 neighbourhood straddles the padding boundary
    is contaminated by the zero pad, and those values sit exactly where the
    anatomy meets the background — the region a synthesis model is most likely
    to get wrong, and so the last place to accept a contaminated score.
    """
    smap = ssim_map(pred, target)
    radius = 11 // 2
    eroded = -F.max_pool2d(-mask, kernel_size=2 * radius + 1, stride=1, padding=radius)
    if float(eroded.sum()) < 1.0:
        return float(_masked_mean(smap, mask))
    return float(_masked_mean(smap, eroded))


def hu_band_mae(pred, target, mask, hu_min, hu_max, bands):
    """
    MAE in HU, split by the TARGET's tissue class.

    Bands are defined on the real CT, not the prediction, so each band answers
    "how well does the model reproduce tissue that truly is bone/soft/air"
    rather than "how well does it agree with itself".

    A band that falls entirely outside this sample's HU window returns None —
    it is not zero error, it is an unanswerable question, and reporting 0.0
    there would silently improve the average.
    """
    out = {}
    target_hu = to_hu(target, hu_min, hu_max)
    error_hu = (pred - target).abs() * float(hu_max - hu_min)

    for name, lo, hi in bands:
        # A band is only measurable where the window can represent it. The brain
        # window (0..80 HU) cannot represent bone at all, which is why bone
        # metrics are region-gated upstream as well.
        lo_in = max(float(lo), float(hu_min))
        hi_in = min(float(hi), float(hu_max))
        if hi_in <= lo_in:
            out[name] = None
            continue

        band_mask = ((target_hu >= lo_in) & (target_hu < hi_in)).float() * mask
        n_pixels = float(band_mask.sum())
        if n_pixels < 1.0:
            out[name] = None
            continue
        out[name] = float((error_hu * band_mask).sum() / n_pixels)
    return out


def bone_dice(pred, target, mask, hu_min, hu_max, threshold_hu=150.0):
    """
    Dice overlap of the bone mask, thresholded in HU.

    THE HALLUCINATION DETECTOR. Bone is a small fraction of the pixels in most
    slices, so a model can fabricate or erase it while barely moving global MAE
    or SSIM. Dice on the thresholded bone mask is dominated by exactly the
    pixels those aggregate metrics drown out, which makes it the metric that
    notices when extra sharpness is invention rather than accuracy.

    Returns None when the window cannot represent the threshold (brain), or
    when neither image contains bone (an empty-vs-empty Dice is undefined and
    scoring it 1.0 would inflate the average with slices that tested nothing).
    """
    threshold = hu_to_norm(float(threshold_hu), hu_min, hu_max)
    if not 0.0 < threshold < 1.0:
        return None

    pred_bone = ((pred >= threshold).float() * mask)
    target_bone = ((target >= threshold).float() * mask)

    denominator = float(pred_bone.sum() + target_bone.sum())
    if denominator < 1.0:
        return None
    return float(2.0 * (pred_bone * target_bone).sum() / denominator)


# ── Accumulator ──────────────────────────────────────────────────────────────

class MetricAccumulator:
    """
    Aggregates per-sample metrics across a validation pass.

    Keeps three views of every metric:
      overall           pooled across all samples
      per region        so a brain regression cannot be masked by abdomen gains
      macro             mean of the per-region means, so the metric is not
                        dominated by whichever region has the most slices
                        (abdomen is 41% of the validation set)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.bands = [(str(n), float(lo), float(hi))
                      for n, lo, hi in cfg.get_path("eval.hu_bands", [])]
        self.bone_threshold = float(cfg.get_path("eval.bone_threshold_hu", 150.0))
        self.excluded = set(cfg.get_path("eval.bone_metrics_exclude_regions", []))
        self.reset()

    def reset(self):
        self._values = defaultdict(list)             # metric -> [floats]
        self._by_region = defaultdict(lambda: defaultdict(list))
        self._n_samples = 0
        self._n_bone_eligible = 0

    @torch.no_grad()
    def update(self, pred, target, mask, hu_min, hu_max, regions):
        """
        Add a batch. `pred`/`target` are [B,1,H,W] in [0,1]; `regions` a list
        of body-region strings.
        """
        batch = pred.size(0)
        for i in range(batch):
            p, t, m = pred[i:i + 1], target[i:i + 1], mask[i:i + 1]
            lo, hi = float(hu_min[i]), float(hu_max[i])
            region = regions[i]

            for name, fn in METRIC_REGISTRY.items():
                value = fn(p, t, m, lo, hi)
                if value is None or not math.isfinite(value):
                    # PSNR is legitimately inf for a perfect reconstruction;
                    # letting that into a mean would poison the whole epoch.
                    continue
                self._values[name].append(value)
                self._by_region[name][region].append(value)

            # Bone and band metrics: region-gated, per the module docstring.
            if region in self.excluded:
                continue
            self._n_bone_eligible += 1

            for band_name, value in hu_band_mae(p, t, m, lo, hi, self.bands).items():
                if value is None:
                    continue
                key = f"mae_band_{band_name}"
                self._values[key].append(value)
                self._by_region[key][region].append(value)

            dice = bone_dice(p, t, m, lo, hi, self.bone_threshold)
            if dice is not None:
                self._values["dice_bone"].append(dice)
                self._by_region["dice_bone"][region].append(dice)

        self._n_samples += batch

    def compute(self):
        """Return a flat dict of results plus the counts behind them."""
        out = {}
        for name, values in self._values.items():
            if not values:
                continue
            out[name] = sum(values) / len(values)
            out[f"{name}__n"] = len(values)

            region_means = {}
            for region, region_values in self._by_region[name].items():
                mean = sum(region_values) / len(region_values)
                out[f"{name}/{region}"] = mean
                region_means[region] = mean
            if len(region_means) > 1:
                out[f"{name}__macro"] = sum(region_means.values()) / len(region_means)

        out["n_samples"] = self._n_samples
        out["n_bone_eligible"] = self._n_bone_eligible
        return out

    def format_table(self, results):
        """Human-readable summary for the console and the run log."""
        lines = []
        excluded = ", ".join(sorted(self.excluded)) or "none"

        lines.append(f"  {'metric':<16}{'overall':>10}{'macro':>10}   per region")
        lines.append("  " + "-" * 74)

        for name in ("mae_norm", "psnr", "ssim", "mae_hu",
                     "mae_band_air", "mae_band_soft", "mae_band_bone", "dice_bone"):
            if name not in results:
                continue
            overall = results[name]
            macro = results.get(f"{name}__macro")
            regions = " ".join(
                f"{key.split('/')[1]}={value:.4g}"
                for key, value in sorted(results.items())
                if key.startswith(name + "/"))
            macro_str = f"{macro:>10.4f}" if macro is not None else f"{'-':>10}"
            lines.append(f"  {name:<16}{overall:>10.4f}{macro_str}   {regions}")

        lines.append("  " + "-" * 74)
        lines.append(f"  n_samples={results.get('n_samples', 0)}  "
                     f"bone/band metrics computed on "
                     f"{results.get('n_bone_eligible', 0)} slices "
                     f"(regions excluded: {excluded} — its HU window saturates bone)")
        return "\n".join(lines)
