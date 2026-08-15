"""
diffaug.py
──────────
Differentiable augmentation for the discriminator's inputs.
After Zhao et al., "Differentiable Augmentation for Data-Efficient GAN Training".

THE PROBLEM. This project has 1687 training slices. A PatchGAN has ample capacity
to memorise the texture of 33 specific patient folders, and once it has, it is no
longer answering "does this look like real CT?" — it is answering "have I seen
this exact patch before?". Its gradient stops carrying information about realism,
and the generator starts chasing artifacts that game a critic which is no longer
measuring anything. The tell is a discriminator that is ~100% accurate on
training data while sample quality plateaus or drifts backwards.

THE FIX, AND WHY IT IS NOT ORDINARY AUGMENTATION. Two properties matter:

  1. It is applied to BOTH the real and the fake batch, immediately before D.
     Augmenting only the reals would teach G to reproduce the augmentation.
     Because both sides get the same treatment, the transformation cancels out
     of the objective and G is never asked to produce augmented images — the
     augmentation makes D's job harder without changing what G is aiming for.

  2. It is differentiable. Gradients flow back through the augmentation into G,
     so G still receives a usable learning signal from an augmented critique.
     A non-differentiable augmentation would sever that path and silently turn
     the adversarial term into noise.

This is why the augmentation lives here, coupled to the discriminator step, and
not in the dataset's transform pipeline where ordinary augmentation belongs.

POLICY NOTE. 'color' adjusts brightness/saturation/contrast. On single-channel
medical images saturation is a no-op (it interpolates toward the channel mean,
which for one channel is itself), so it is skipped — brightness and contrast do
the work. Geometric jitter is limited to small translations for the same reason
rotation is absent from the dataset transforms: manual QC corrected translation
on these pairs but could not correct in-plane rotation, so some rotational
residual remains and there is no sense in adding more of it.
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def rand_brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)


def rand_contrast(x):
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    factor = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5
    return (x - mean) * factor + mean


def rand_saturation(x):
    # Single-channel input: the per-pixel channel mean IS the pixel, so this
    # transform is the identity. Returned unchanged rather than wasting the op.
    if x.size(1) == 1:
        return x
    mean = x.mean(dim=1, keepdim=True)
    factor = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2
    return (x - mean) * factor + mean


def rand_translation(x, ratio=0.125):
    """Shift by up to +/- ratio of each dimension, zero-filling what shifts in."""
    shift_x = int(x.size(2) * ratio + 0.5)
    shift_y = int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1],
                                  device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1],
                                  device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing="ij")
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    return (x_pad.permute(0, 2, 3, 1)
            .contiguous()[grid_batch, grid_x, grid_y]
            .permute(0, 3, 1, 2).contiguous())


def rand_cutout(x, ratio=0.5):
    """Zero out one random rectangle per image."""
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2),
                             size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2),
                             size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        indexing="ij")
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2,
                         min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2,
                         min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    return x * mask.unsqueeze(1)


AUGMENT_FNS = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}


def diff_augment(x, policy="color,translation,cutout"):
    """
    Apply the policy to a batch.

    IMPORTANT: the real and fake batches must be augmented by SEPARATE calls
    (each draws its own random parameters) but under the SAME policy. Sharing
    the exact parameters between them is not required and not desirable — what
    must be shared is the distribution, so that neither branch is systematically
    easier for D than the other.
    """
    if not policy:
        return x
    for name in policy.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in AUGMENT_FNS:
            raise ValueError(
                f"Unknown DiffAugment policy '{name}'. "
                f"Valid: {sorted(AUGMENT_FNS)}"
            )
        for fn in AUGMENT_FNS[name]:
            x = fn(x)
    return x.contiguous()


class DiffAugment:
    """Small stateful wrapper so the training loop can treat it as on/off."""

    def __init__(self, enabled=True, policy="color,translation,cutout"):
        self.enabled = bool(enabled)
        self.policy = policy if self.enabled else ""
        if self.enabled:
            logger.info("DiffAugment enabled, policy='%s'", self.policy)

    def __call__(self, x):
        return diff_augment(x, self.policy) if self.enabled else x
