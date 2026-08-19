"""
registration.py
───────────────
RegGAN's registration network R and the differentiable warp it drives.

THE PROBLEM THIS SOLVES. Every pair in this dataset went through manual QC, and
that QC could correct translation but not in-plane rotation — configs/base.yaml
says so at line 55, and configs/exp3_nce_heavy.yaml hedges against the leftover
by reweighting its objective. L1 against a target that is a degree or two out of
frame does not merely score the generator unfairly; it actively teaches it to
blur, because a blurred prediction is the L1-optimal hedge when the target's
position is uncertain. RegGAN's answer is to stop pretending the frames agree:
predict the residual deformation, warp the prediction into the target's frame,
and score it there.

WHAT R IS. A small U-Net taking cat[fake_CT, real_CT] and emitting a two-channel
dense displacement field in pixels. It is trained jointly with G by the same
correction loss — neither is given a separate objective, which is what keeps the
field pinned to "the misalignment G could not have known about" rather than
drifting into "whatever makes the loss smallest".

WHY IT IS DELIBERATELY SMALL. nrf=32 and four downsamplings, against the
generator's ngf=64 and eight. R has to represent a smooth, low-frequency,
near-identity field; capacity beyond that is capacity to represent a field that
explains away the generator's mistakes, which is the exact failure the smoothness
penalty in losses/registration.py exists to prevent. Making R weak is the
structural half of the same argument.

NO TAP CONTRACT. Unlike UnetGenerator and StyleGAN2Generator, R implements no
encode_only/tap_layers protocol. PatchNCE samples the generator's encoder, never
this one, so there is nothing here to tap.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .init import get_norm_layer, init_weights, uses_bias

logger = logging.getLogger(__name__)


class SpatialTransformer(nn.Module):
    """
    Differentiable resampling of an image by a dense displacement field.

    The field is in PIXELS, not in grid_sample's normalised [-1, 1] coordinates.
    That choice is load-bearing: a displacement in normalised units means a
    different physical distance on a 180x180 slice than on a 430x430 one, so the
    smoothness penalty and the reported flow magnitude would both silently depend
    on slice size. In pixels — and 1 px = 1 mm throughout this project — they
    mean millimetres on every slice.

    padding_mode="border" rather than "zeros": a field that samples just past the
    edge should pick up the nearest real tissue, not a black pixel that the L1
    term would then read as a large error attributable to the generator.
    """

    def __init__(self, padding_mode="border"):
        super().__init__()
        self.padding_mode = padding_mode
        # Identity grids are pure geometry — same for every batch, every step —
        # so they are cached per (size, device, dtype) rather than rebuilt. Not
        # registered as buffers: they are derived, not learned or restored.
        self._grid_cache = {}

    def _identity_grid(self, height, width, device, dtype):
        key = (height, width, device, dtype)
        grid = self._grid_cache.get(key)
        if grid is None:
            ys, xs = torch.meshgrid(
                torch.arange(height, device=device, dtype=dtype),
                torch.arange(width, device=device, dtype=dtype),
                indexing="ij")
            grid = torch.stack((xs, ys), dim=0)          # [2, H, W], (x, y)
            self._grid_cache[key] = grid
        return grid

    def forward(self, x, flow, mode="bilinear"):
        """
        Warp `x` by `flow`.

        `mode` is a forward argument rather than a constructor setting because
        the same transformer warps images bilinearly and validity masks with
        nearest — interpolating a mask would produce fractional validity, and a
        pixel is either real or padding.
        """
        _, _, height, width = x.shape
        base = self._identity_grid(height, width, flow.device, flow.dtype)
        coords = base.unsqueeze(0) + flow                # [N, 2, H, W] in pixels

        # Pixel centres to normalised coordinates under align_corners=False:
        # index i sits at 2*(i + 0.5)/size - 1.
        norm_x = 2.0 * (coords[:, 0] + 0.5) / width - 1.0
        norm_y = 2.0 * (coords[:, 1] + 0.5) / height - 1.0
        grid = torch.stack((norm_x, norm_y), dim=-1)     # [N, H, W, 2]

        return F.grid_sample(x, grid.to(x.dtype), mode=mode,
                             padding_mode=self.padding_mode, align_corners=False)


def _conv_block(c_in, c_out, norm_layer, bias):
    """Two 3x3 convolutions at one resolution — the standard U-Net rung."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=bias),
        norm_layer(c_out),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=bias),
        norm_layer(c_out),
        nn.LeakyReLU(0.2, inplace=True),
    )


class RegistrationUNet(nn.Module):
    """
    Parameters
    ----------
    in_channels  : 2 for single-channel data (generated CT + real CT)
    out_channels : 2 — the (dx, dy) displacement
    nrf          : filters at full resolution
    num_downs    : downsampling steps; 4 is enough for a field whose content is
                   low-frequency by construction
    norm         : 'instance' | 'batch' | 'none', the same vocabulary the rest of
                   networks/ uses
    """

    def __init__(self, in_channels=2, out_channels=2, nrf=32, num_downs=4,
                 norm="instance"):
        super().__init__()
        norm_layer = get_norm_layer(norm)
        bias = uses_bias(norm)

        widths = [min(nrf * 2 ** i, 256) for i in range(num_downs + 1)]

        self.stem = _conv_block(in_channels, widths[0], norm_layer, bias)
        self.downs = nn.ModuleList([
            nn.Sequential(nn.MaxPool2d(2),
                          _conv_block(widths[i], widths[i + 1], norm_layer, bias))
            for i in range(num_downs)
        ])
        # Decoder rungs run deepest-first and each consumes a skip connection,
        # hence the concatenated input width.
        self.ups = nn.ModuleList([
            _conv_block(widths[i + 1] + widths[i], widths[i], norm_layer, bias)
            for i in reversed(range(num_downs))
        ])
        self.head = nn.Conv2d(widths[0], out_channels, kernel_size=3, padding=1)

        init_weights(self)

        # ── R MUST START AS THE IDENTITY ─────────────────────────────────────
        # This runs AFTER init_weights, which would otherwise overwrite it with
        # N(0, 0.02) like every other conv. The ordering is the whole point.
        #
        # With a normally-initialised head, step 0 warps the target by a random
        # field of tens of pixels. The generator's first gradients then point
        # toward matching a scrambled version of the CT, and it does not recover
        # — while the loss curve falls exactly as it would in a healthy run,
        # because R is simultaneously learning to undo its own noise. There is no
        # symptom to notice. Starting at zero displacement makes epoch 0
        # identical to plain pix2pix, and the field grows only insofar as the
        # data asks it to.
        nn.init.normal_(self.head.weight, 0.0, 1e-5)
        nn.init.constant_(self.head.bias, 0.0)

        logger.info("RegistrationUNet: nrf=%d num_downs=%d params=%.2fM "
                    "(head zero-initialised: starts as the identity transform)",
                    nrf, num_downs,
                    sum(p.numel() for p in self.parameters()) / 1e6)

    def forward(self, x):
        feats = [self.stem(x)]
        for down in self.downs:
            feats.append(down(feats[-1]))

        h = feats[-1]
        for i, up in enumerate(self.ups):
            skip = feats[-2 - i]
            # Interpolating to the skip's size rather than by a fixed factor of
            # two keeps odd spatial sizes working. Training crops are 256 and
            # validation pads to a multiple of 256, so this should never bite —
            # but a silent one-pixel mismatch deep in a decoder is not worth
            # leaving to chance.
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
            h = up(torch.cat([h, skip], dim=1))

        return self.head(h)


def build_registration_net(cfg_registration, in_channels):
    """
    Construct R from cfg.model.registration.

    `in_channels` is passed in rather than read from the config because it is not
    a free choice: R always sees the generated image and the real one stacked, so
    it is twice the generator's out_channels and nothing else.
    """
    cfg_registration = cfg_registration or {}
    return RegistrationUNet(
        in_channels=in_channels,
        out_channels=2,
        nrf=int(cfg_registration.get("nrf", 32)),
        num_downs=int(cfg_registration.get("num_downs", 4)),
        norm=cfg_registration.get("norm", "instance"),
    )
