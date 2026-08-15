"""
init.py
───────
Weight initialisation, following the pix2pix convention.

Conv and Linear weights are drawn from N(0, 0.02) and normalisation-layer
weights from N(1.0, 0.02). The small standard deviation is not incidental: a
default PyTorch (Kaiming) init makes the discriminator's early outputs large and
confident, which is precisely the state in which the adversarial gradient
saturates and the generator stops learning. Every published pix2pix/CycleGAN
result uses this init, and deviating from it is a common cause of "my GAN
collapsed in the first epoch".
"""

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialise a network in place and return it."""

    def init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, "weight") and ("Conv" in classname or "Linear" in classname):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f"Unknown init_type '{init_type}'")
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif "BatchNorm2d" in classname or "InstanceNorm2d" in classname:
            # InstanceNorm2d is constructed with affine=False here, so it has no
            # weight to initialise; the guard keeps this safe either way.
            if getattr(m, "weight", None) is not None:
                nn.init.normal_(m.weight.data, 1.0, init_gain)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    return net


def get_norm_layer(norm_type="instance"):
    """
    Return a norm-layer factory.

    InstanceNorm is the default and BatchNorm is available but discouraged here:
    batch statistics are noisy at the batch sizes this project trains at, and
    they couple samples within a batch, so a CPU smoke test at batch 2 would not
    behave like the Kaggle run at batch 8. InstanceNorm is per-image and
    batch-size-independent, which keeps the two comparable.
    """
    if norm_type == "instance":
        return lambda c: nn.InstanceNorm2d(c, affine=False, track_running_stats=False)
    if norm_type == "batch":
        return lambda c: nn.BatchNorm2d(c, affine=True, track_running_stats=True)
    if norm_type == "none":
        return lambda c: nn.Identity()
    raise NotImplementedError(f"Unknown norm layer '{norm_type}'")


def uses_bias(norm_type):
    """
    Whether conv layers should carry a bias term.

    InstanceNorm2d(affine=False) applies no learnable shift, so the preceding
    conv must supply one. BatchNorm has its own beta, which would make a conv
    bias redundant (and its gradient degenerate).
    """
    return norm_type in ("instance", "none")
