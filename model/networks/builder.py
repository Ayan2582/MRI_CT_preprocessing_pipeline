"""
builder.py
──────────
Architecture dispatch: turn `model.generator.type` / `model.discriminator.type`
into a network.

Until StyleGAN2 arrived there was only one of each, so `unet.build_generator` and
`patchgan.build_discriminator` doubled as both the dispatcher and the constructor.
Adding a second architecture there would have meant `unet.py` importing
`stylegan2.py`, which is backwards. The leaf builders stay exactly where they are
and know only about their own network; this module is the only thing that knows a
choice exists.

Adding a third architecture means one import and one branch in each function here.
"""

import logging

logger = logging.getLogger(__name__)

GENERATOR_TYPES = ("unet", "stylegan2")
DISCRIMINATOR_TYPES = ("patchgan", "stylegan2")


def build_generator(cfg_generator):
    """Construct the generator described by cfg.model.generator."""
    gen_type = cfg_generator.get("type", "unet")

    if gen_type == "unet":
        from .unet import build_generator as build_unet
        return build_unet(cfg_generator)

    if gen_type == "stylegan2":
        from .stylegan2 import build_stylegan2_generator
        return build_stylegan2_generator(cfg_generator)

    raise NotImplementedError(
        f"generator type '{gen_type}' is not implemented. Available: "
        f"{', '.join(GENERATOR_TYPES)}."
    )


def build_discriminator(cfg_model, spectral, batch_size=None):
    """
    Construct the discriminator described by cfg.model.discriminator.

    `spectral` comes from cfg.stabilizers.spectral_norm_d rather than from the
    discriminator config, because it is a training-stability choice rather than an
    architectural one. `batch_size` is only consulted by StyleGAN2, whose
    minibatch-stddev layer needs a group size that divides the batch.
    """
    disc_type = cfg_model.discriminator.get("type", "patchgan")

    if disc_type == "patchgan":
        from .patchgan import build_discriminator as build_patchgan
        return build_patchgan(cfg_model, spectral)

    if disc_type == "stylegan2":
        from .stylegan2 import build_stylegan2_discriminator
        return build_stylegan2_discriminator(cfg_model, spectral, batch_size)

    raise NotImplementedError(
        f"discriminator type '{disc_type}' is not implemented. Available: "
        f"{', '.join(DISCRIMINATOR_TYPES)}."
    )
