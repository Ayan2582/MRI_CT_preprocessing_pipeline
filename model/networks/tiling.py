"""
tiling.py
─────────
Sliding-window inference for generators that emit a fixed output size.

WHY THIS EXISTS. The U-Net is fully convolutional: hand it a 512x512 slice and it
returns a 512x512 slice. A StyleGAN2 synthesis network is not — it grows a learned
4x4 constant through a fixed number of upsampling blocks, so it emits exactly one
resolution and nothing else. That collides with how this project validates.

Training always sees a 256 crop, but validation keeps whole slices padded up to a
multiple of 256, and the numbers are not marginal:

    45% of validation slices (103 of 230) pad to 512.
    That is EVERY abdomen slice (94) and EVERY spine slice (9).
    Nothing exceeds 512.

Without this module those slices could not be generated at all, `mae_norm` would be
computed on brain and musculoskeletal slices only, and no StyleGAN2 run would be
comparable to the exp0-exp4 ladder that scored all 230.

HOW IT WORKS. Cut the input into overlapping `native`-sized windows, generate each
one, and sum them back into place weighted by a 2-D raised cosine, dividing at the
end by the accumulated weight.

WHY FEATHERING IS NOT OPTIONAL HERE. In ordinary tiled inference the tiles are
nearly consistent with each other and a hard seam is a minor artifact. Here each
tile is encoded to its OWN style vector w, so neighbouring tiles are generated under
genuinely different global styles — brightness and texture really do differ across
a seam. The cosine weights make that transition gradual instead of a visible line
down the middle of the abdomen. It hides the discontinuity; it does not remove it.
"""

import logging
import math

import torch

logger = logging.getLogger(__name__)


def _hann_1d(size, device, dtype):
    """
    A periodic raised cosine, floored away from zero.

    The floor matters. A textbook Hann window is exactly 0.0 at both ends, so the
    outermost row and column of the whole image would be covered only by windows
    whose weight there is zero — the accumulated weight would be 0 and the division
    would produce NaN. Clamping to a small positive value keeps every pixel covered
    while leaving the interior blending unchanged.
    """
    n = torch.arange(size, device=device, dtype=dtype)
    window = 0.5 - 0.5 * torch.cos(2.0 * math.pi * (n + 0.5) / size)
    return window.clamp_min(1e-3)


def _origins(extent, native, stride):
    """
    Top-left offsets covering `extent` with `native`-wide windows.

    The final origin is snapped to `extent - native` so the last window ends flush
    with the edge. Without that, an extent that is not an exact multiple of the
    stride would leave a strip at the far edge that no window covers.
    """
    if extent <= native:
        return [0]
    stops = list(range(0, extent - native + 1, max(1, int(stride))))
    if stops[-1] != extent - native:
        stops.append(extent - native)
    return stops


@torch.no_grad()
def _log_plan(h, w, native, stride, n_tiles):
    logger.debug("tiled inference: %dx%d at native=%d stride=%d -> %d windows",
                 h, w, native, stride, n_tiles)


def tiled_forward(fn, x, native=256, stride=128):
    """
    Apply `fn` over overlapping windows of `x` and blend the results.

    Parameters
    ----------
    fn     : callable taking [B, C, native, native] and returning [B, C_out, native, native].
             Pass the generator's SINGLE-TILE forward, not the public one, or this
             recurses forever.
    x      : [B, C, H, W]
    native : the size `fn` accepts and returns
    stride : window step. Half of `native` gives every interior pixel two windows
             per axis, which is enough for the cosine blend to be smooth.

    Returns
    -------
    [B, C_out, H, W]
    """
    _, _, h, w = x.shape
    native = int(native)

    # The common case by a wide margin: every training crop, and the 55% of
    # validation slices that pad to exactly 256. No blending, no extra cost, and
    # bit-identical to calling fn directly.
    if h <= native and w <= native:
        return fn(x)

    ys = _origins(h, native, stride)
    xs = _origins(w, native, stride)
    _log_plan(h, w, native, stride, len(ys) * len(xs))

    window = None
    out = None
    weight_sum = None

    for top in ys:
        for left in xs:
            tile = x[:, :, top:top + native, left:left + native]
            pred = fn(tile)

            if out is None:
                # Deferred until the first tile so the output channel count comes
                # from fn itself rather than from an assumption about it.
                window = (_hann_1d(native, pred.device, pred.dtype).unsqueeze(1)
                          * _hann_1d(native, pred.device, pred.dtype).unsqueeze(0))
                out = torch.zeros(pred.shape[0], pred.shape[1], h, w,
                                  device=pred.device, dtype=pred.dtype)
                weight_sum = torch.zeros(1, 1, h, w,
                                         device=pred.device, dtype=pred.dtype)

            out[:, :, top:top + native, left:left + native] += pred * window
            weight_sum[:, :, top:top + native, left:left + native] += window

    return out / weight_sum
