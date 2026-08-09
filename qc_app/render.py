"""
render.py
─────────
Turn the cached float32 [0,1] slices into PNG bytes for the browser.

Display only. Nothing here feeds the pipeline or the saved outputs - those stay
float32 arrays throughout. Conversion to 8-bit happens at the last possible
moment, here, for pixels that are about to be looked at by a human.

The four views exist because they fail differently, and a reviewer needs more
than one to judge an alignment:

    gray          one modality on its own - is the image itself sane
    fusion        CT green, MRI magenta - misalignment shows as colour fringing
                  along every edge, which the eye catches faster than an offset
    checker       alternating tiles - a structure crossing a tile boundary
                  shows a step exactly where the two disagree
    difference    |CT - MRI| - bright where they disagree anywhere at all
"""

import io

import numpy as np
from PIL import Image

CHECKER_TILE = 32


def _u8(a: np.ndarray) -> np.ndarray:
    return (np.clip(np.nan_to_num(a, nan=0.0), 0.0, 1.0) * 255).astype(np.uint8)


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=1)
    return buf.getvalue()


def gray(arr: np.ndarray) -> bytes:
    return _png(Image.fromarray(_u8(arr), mode="L"))


def fusion(ct: np.ndarray, mri: np.ndarray) -> bytes:
    """
    CT into the green channel, MRI into red and blue.

    Where the two agree the channels overlap and the result is neutral grey;
    where they do not, one side of every edge goes green and the other magenta.
    That colour fringe is the single most sensitive misalignment cue on this
    kind of pair, because it responds to a one-pixel offset on any edge in the
    frame rather than to overall brightness.
    """
    c, m = _u8(ct), _u8(_match_shape(mri, ct.shape))
    rgb = np.dstack([m, c, m])
    return _png(Image.fromarray(rgb, mode="RGB"))


def checker(ct: np.ndarray, mri: np.ndarray, tile: int = CHECKER_TILE) -> bytes:
    c, m = _u8(ct), _u8(_match_shape(mri, ct.shape))
    h, w = c.shape
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((yy // tile) + (xx // tile)) % 2 == 0
    return _png(Image.fromarray(np.where(mask, c, m), mode="L"))


def difference(ct: np.ndarray, mri: np.ndarray) -> bytes:
    """
    |CT - MRI| on a black-to-hot ramp.

    Both inputs are already normalised to [0,1], but they are different
    modalities, so this is never zero even under perfect alignment - soft
    tissue that is bright on one is not bright on the other. It is read as a
    STRUCTURAL map: sharp bright outlines tracing anatomy mean edges that do
    not coincide; a diffuse glow is just modality contrast.
    """
    d = np.abs(np.clip(ct, 0, 1) - np.clip(_match_shape(mri, ct.shape), 0, 1))
    v = _u8(d)
    r = v
    g = (v.astype(np.uint16) * 180 // 255).astype(np.uint8)
    b = (v.astype(np.uint16) * 60 // 255).astype(np.uint8)
    return _png(Image.fromarray(np.dstack([r, g, b]), mode="RGB"))


def _match_shape(a: np.ndarray, shape) -> np.ndarray:
    """
    Pad or trim `a` to `shape`.

    Cached CT and MRI arrays always share a shape - the MRI was resampled onto
    the CT grid - so this is a guard, not a resampler. It exists so a corrupt
    cache renders a visibly wrong image instead of raising inside a request.
    """
    if a.shape == tuple(shape):
        return a
    out = np.zeros(shape, dtype=a.dtype)
    h = min(a.shape[0], shape[0])
    w = min(a.shape[1], shape[1])
    out[:h, :w] = a[:h, :w]
    return out


def render_view(view: str, ct: np.ndarray, mri_before: np.ndarray,
                mri_after: np.ndarray) -> bytes:
    mri = mri_after
    if view == "ct":
        return gray(ct)
    if view == "mri":
        return gray(mri)
    if view == "mri_before":
        return gray(mri_before)
    if view == "fusion":
        return fusion(ct, mri)
    if view == "fusion_before":
        return fusion(ct, mri_before)
    if view == "checker":
        return checker(ct, mri)
    if view == "difference":
        return difference(ct, mri)
    if view == "difference_before":
        return difference(ct, mri_before)
    raise ValueError(f"unknown view '{view}'")
