"""
patchgan.py
───────────
70x70 PatchGAN discriminator, conditional on the input MRI.

WHY A PATCH DISCRIMINATOR. This D does not output one number per image; it
outputs a grid, each element judging one overlapping 70x70 receptive field.
That matters because of how the work is divided in pix2pix: the L1 term already
enforces global anatomy and carries ~99% of the gradient magnitude, so asking D
to police global structure too would be redundant. What L1 cannot do is prevent
blur, because a blurred prediction is the L1-optimal hedge under uncertainty.
Local texture realism is exactly what a patch-level critic can enforce, and
restricting D to a 70x70 window keeps it focused there. It also has far fewer
parameters than a full-image discriminator, which matters on 1687 training
slices, and it applies unchanged to any input size — useful here, where
validation images range from 256 to 512 after padding.

WHY CONDITIONAL. D receives cat[MRI, CT] rather than the CT alone. An
unconditional D only asks "is this a plausible CT?", which a generator can
satisfy by producing a convincing CT of the wrong patient. Feeding it both
modalities makes the question "is this a plausible CT *of this MRI*", so the
adversarial term reinforces correspondence instead of competing with it.

RECEPTIVE FIELD. With n_layers=3 the stack is 4x4 stride-2, 4x4 stride-2,
4x4 stride-2, 4x4 stride-1, 4x4 stride-1, giving a 70x70 receptive field. At
1 px = 1 mm that is a 70 mm window — roughly the scale of an organ, which is a
reasonable unit for "does this texture belong to this tissue".
"""

import logging

import torch.nn as nn
from torch.nn.utils import spectral_norm as apply_spectral_norm

from .init import get_norm_layer, init_weights, uses_bias

logger = logging.getLogger(__name__)


class NLayerDiscriminator(nn.Module):
    """
    Parameters
    ----------
    in_channels   : 2 for a conditional D on single-channel data (MRI + CT)
    ndf           : filters in the first layer
    n_layers      : 3 gives the canonical 70x70 receptive field
    norm          : 'instance' | 'batch' | 'none'
    spectral      : wrap every conv in spectral normalisation.

                    This is the cheapest stabiliser available and the one most
                    worth understanding. It divides each weight matrix by its
                    largest singular value, which bounds how fast D's output can
                    change with its input — its Lipschitz constant. A D with a
                    bounded Lipschitz constant cannot become an arbitrarily
                    confident classifier, and it is exactly that runaway
                    confidence that saturates the adversarial loss and leaves
                    the generator with no usable gradient direction. Costs one
                    extra power iteration per forward pass.
    """

    def __init__(self, in_channels=2, ndf=64, n_layers=3, norm="instance",
                 spectral=True):
        super().__init__()
        norm_layer = get_norm_layer(norm)
        bias = uses_bias(norm)

        def conv(c_in, c_out, stride, use_bias):
            layer = nn.Conv2d(c_in, c_out, kernel_size=4, stride=stride,
                              padding=1, bias=use_bias)
            return apply_spectral_norm(layer) if spectral else layer

        # First layer takes no norm: it should still see absolute intensity,
        # which is meaningful here because both modalities are calibrated to a
        # fixed [0,1] range before training.
        layers = [conv(in_channels, ndf, 2, True), nn.LeakyReLU(0.2, inplace=True)]

        mult = 1
        for i in range(1, n_layers):
            prev, mult = mult, min(2 ** i, 8)
            layers += [
                conv(ndf * prev, ndf * mult, 2, bias),
                norm_layer(ndf * mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # One more stride-1 block widens the receptive field to 70 without
        # shrinking the output grid further.
        prev, mult = mult, min(2 ** n_layers, 8)
        layers += [
            conv(ndf * prev, ndf * mult, 1, bias),
            norm_layer(ndf * mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Output is a one-channel map of per-patch scores. No sigmoid: LSGAN and
        # hinge both consume raw scores, and the vanilla objective uses
        # BCEWithLogits, which applies its own for numerical stability.
        layers += [conv(ndf * mult, 1, 1, True)]

        self.model = nn.Sequential(*layers)
        init_weights(self)

        logger.info("NLayerDiscriminator: n_layers=%d ndf=%d spectral=%s "
                    "params=%.1fM", n_layers, ndf, spectral,
                    sum(p.numel() for p in self.parameters()) / 1e6)

    def forward(self, x):
        return self.model(x)


def build_discriminator(cfg_model, spectral):
    """
    Construct the discriminator described by cfg.model.discriminator.

    `spectral` comes from cfg.stabilizers.spectral_norm_d rather than from the
    discriminator config, because it is a training-stability choice rather than
    an architectural one and belongs beside the other stabilisers.
    """
    disc = cfg_model.discriminator
    if disc.get("type", "patchgan") != "patchgan":
        raise NotImplementedError(
            f"discriminator type '{disc.get('type')}' is not implemented; "
            f"only 'patchgan' is."
        )

    gen = cfg_model.generator
    # Conditional D sees the source and target stacked on the channel axis.
    in_channels = gen.get("out_channels", 1)
    if disc.get("conditional", True):
        in_channels += gen.get("in_channels", 1)

    return NLayerDiscriminator(
        in_channels=in_channels,
        ndf=disc.get("ndf", 64),
        n_layers=disc.get("n_layers", 3),
        norm=gen.get("norm", "instance"),
        spectral=bool(spectral),
    )
