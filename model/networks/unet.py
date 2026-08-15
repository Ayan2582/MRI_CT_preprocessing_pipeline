"""
unet.py
───────
U-Net generator with skip connections and tappable encoder features.

WHY THIS IS NOT THE STOCK PIX2PIX U-NET

The reference implementation builds its U-Net by recursive nesting: an outermost
UnetSkipConnectionBlock wraps a submodule, which wraps a submodule, and so on
down to the bottleneck. That is elegant and completely opaque from the outside —
by the time you call forward(), the intermediate encoder activations exist only
inside nested closures, and there is no clean way to read them out.

PatchNCE needs exactly those activations. It compares encoder features of the
input MRI against encoder features of the generated CT at matched spatial
locations, so the generator has to be able to hand back what its encoder saw at
several depths. This implementation therefore lays the encoder and decoder out as
two explicit nn.ModuleLists, which makes forward() a readable loop and the taps a
one-line list index.

Everything else — 4x4 stride-2 convolutions, LeakyReLU(0.2) down / ReLU up,
InstanceNorm, dropout in the three innermost decoder blocks, tanh output — is
unchanged from pix2pix.

ON TAP DEPTH: with num_downs=8 and a 256x256 input, the encoder reaches 1x1 at
the bottleneck. A 1x1 feature map offers exactly one spatial location, so it
cannot supply the 256 sampled patches PatchNCE asks for. Useful taps are the
shallow ones; see `layers` in configs/base.yaml.
"""

import logging

import torch
import torch.nn as nn

from .init import get_norm_layer, init_weights, uses_bias

logger = logging.getLogger(__name__)


class UnetGenerator(nn.Module):
    """
    Parameters
    ----------
    in_channels, out_channels : 1 and 1 here (single-channel MRI -> CT)
    ngf        : filters in the first encoder block; doubles up to 8x
    num_downs  : number of downsamplings. 8 takes 256x256 to 1x1.
    norm       : 'instance' | 'batch' | 'none'
    use_dropout: dropout in the three innermost decoder blocks. pix2pix has no
                 noise vector, so this is the generator's only stochasticity.
    """

    def __init__(self, in_channels=1, out_channels=1, ngf=64, num_downs=8,
                 norm="instance", use_dropout=True):
        super().__init__()
        self.num_downs = int(num_downs)
        self.in_channels = in_channels
        norm_layer = get_norm_layer(norm)
        bias = uses_bias(norm)

        # Channel schedule: ngf, 2ngf, 4ngf, then 8ngf all the way down.
        chans = []
        for i in range(self.num_downs):
            chans.append(min(ngf * (2 ** i), ngf * 8))
        self.enc_channels = chans

        # ── Encoder ──────────────────────────────────────────────────────────
        encoder = []
        for i in range(self.num_downs):
            c_in = in_channels if i == 0 else chans[i - 1]
            c_out = chans[i]
            conv = nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=bias)
            if i == 0:
                # Outermost: no activation before it (it consumes the image) and
                # no norm, so the first layer can still see absolute intensity.
                block = nn.Sequential(conv)
            elif i == self.num_downs - 1:
                # Innermost: no norm. At 1x1, instance norm would divide a single
                # value by its own (zero) spatial variance.
                block = nn.Sequential(nn.LeakyReLU(0.2, inplace=True), conv)
            else:
                block = nn.Sequential(nn.LeakyReLU(0.2, inplace=True), conv,
                                      norm_layer(c_out))
            encoder.append(block)
        self.encoder = nn.ModuleList(encoder)

        # ── Decoder ──────────────────────────────────────────────────────────
        # Block j undoes encoder block (num_downs-1-j). Every block except the
        # first receives the matching encoder activation concatenated on, which
        # is what doubles its input channel count.
        decoder = []
        for j in range(self.num_downs):
            enc_idx = self.num_downs - 1 - j
            c_in = chans[enc_idx] if j == 0 else chans[enc_idx] * 2
            if j == self.num_downs - 1:
                up = nn.ConvTranspose2d(c_in, out_channels, kernel_size=4,
                                        stride=2, padding=1, bias=True)
                # tanh, so the network's output range matches the [-1,1] the
                # dataset converts its [0,1] arrays into.
                block = nn.Sequential(nn.ReLU(inplace=True), up, nn.Tanh())
            else:
                c_out = chans[enc_idx - 1]
                up = nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2,
                                        padding=1, bias=bias)
                layers = [nn.ReLU(inplace=True), up, norm_layer(c_out)]
                # pix2pix puts dropout in the three innermost decoder blocks only.
                if use_dropout and 1 <= j <= 3:
                    layers.append(nn.Dropout(0.5))
                block = nn.Sequential(*layers)
            decoder.append(block)
        self.decoder = nn.ModuleList(decoder)

        init_weights(self)
        logger.info("UnetGenerator: num_downs=%d ngf=%d channels=%s params=%.1fM",
                    self.num_downs, ngf, chans,
                    sum(p.numel() for p in self.parameters()) / 1e6)

    # ── Tap bookkeeping ──────────────────────────────────────────────────────

    @property
    def n_taps(self):
        """Number of valid tap indices: 0 (the input) plus one per encoder block."""
        return self.num_downs + 1

    def tap_channels(self, tap):
        """Channel count a given tap yields, for building the MLP heads."""
        self._check_tap(tap)
        return self.in_channels if tap == 0 else self.enc_channels[tap - 1]

    def _check_tap(self, tap):
        if not 0 <= tap < self.n_taps:
            raise ValueError(
                f"Tap {tap} is out of range for num_downs={self.num_downs}. "
                f"Valid taps are 0..{self.num_downs} "
                f"(0 = the input image, k = the output of encoder block k-1)."
            )

    def tap_spatial(self, tap, input_size):
        """Spatial edge length a tap produces for a square input of `input_size`."""
        self._check_tap(tap)
        return input_size // (2 ** tap)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x, tap_layers=None, encode_only=False):
        """
        Parameters
        ----------
        tap_layers  : sorted list of tap indices to collect, or None for none.
                      Tap 0 is the input image itself; tap k is the output of
                      encoder block k-1.
        encode_only : stop after the deepest requested tap and return features
                      only. Used for encoding the real MRI for PatchNCE, where
                      the decoder's output would be computed and thrown away.

        Returns
        -------
        out                       when tap_layers is None
        (out, feats)              when tap_layers is given
        feats                     when encode_only is True
        """
        taps = sorted(set(tap_layers)) if tap_layers else []
        for t in taps:
            self._check_tap(t)

        feats = []
        if 0 in taps:
            feats.append(x)

        deepest = max(taps) if taps else self.num_downs
        h = x
        skips = []
        for i, block in enumerate(self.encoder):
            h = block(h)
            skips.append(h)
            if (i + 1) in taps:
                feats.append(h)
            if encode_only and (i + 1) >= deepest:
                # Nothing below this depth is requested; the remaining encoder
                # and the whole decoder would be wasted compute.
                return feats

        if encode_only:
            return feats

        for j, block in enumerate(self.decoder):
            if j == 0:
                h = block(h)
            else:
                h = block(torch.cat([h, skips[self.num_downs - 1 - j]], dim=1))

        return (h, feats) if taps else h


def build_generator(cfg_generator):
    """Construct the generator described by cfg.model.generator."""
    gen_type = cfg_generator.get("type", "unet")
    if gen_type != "unet":
        raise NotImplementedError(
            f"generator type '{gen_type}' is not implemented; only 'unet' is."
        )
    return UnetGenerator(
        in_channels=cfg_generator.get("in_channels", 1),
        out_channels=cfg_generator.get("out_channels", 1),
        ngf=cfg_generator.get("ngf", 64),
        num_downs=cfg_generator.get("num_downs", 8),
        norm=cfg_generator.get("norm", "instance"),
        use_dropout=cfg_generator.get("dropout", True),
    )
