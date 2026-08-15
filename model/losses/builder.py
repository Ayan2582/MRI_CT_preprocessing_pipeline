"""
builder.py
──────────
Assembles the loss terms a config actually asks for.

    L_total = lambda_gan * L_cGAN + lambda_l1 * L_L1 + lambda_nce * L_PatchNCE

The rule this module enforces is that a zero lambda removes a term's ENTIRE code
path, not merely its contribution. Multiplying a computed loss by 0.0 would give
the same number while still paying for the discriminator forward pass, the
second encoder pass, the MLP heads and their optimizer — so an "ablation" would
cost as much as the full model and, worse, the unused modules would still appear
in checkpoints and still consume optimizer state.

Instead, `LossPlan` reports which terms are live, and the training model consults
it before building anything. That is what makes

    --set loss.lambda_nce=0

genuinely equal to plain pix2pix rather than merely numerically equivalent to it,
and it is what the modularity check in the verification suite asserts.
"""

import logging

logger = logging.getLogger(__name__)

# Below this, a lambda is treated as off. Guards against a config written as
# 1e-12 by accident and against float noise from a scheduler.
EPS = 1e-12


class LossPlan:
    """Which loss terms are active, and at what weight."""

    def __init__(self, cfg):
        self.lambda_gan = float(cfg.get_path("loss.lambda_gan", 1.0))
        self.lambda_l1 = float(cfg.get_path("loss.lambda_l1", 100.0))
        self.lambda_nce = float(cfg.get_path("loss.lambda_nce", 0.0))
        self.gan_mode = cfg.get_path("loss.gan_mode", "lsgan")

        self.nce_layers = list(cfg.get_path("loss.nce.layers", [0, 1, 2, 3, 4]))
        self.num_patches = int(cfg.get_path("loss.nce.num_patches", 256))
        self.temperature = float(cfg.get_path("loss.nce.temperature", 0.07))
        self.use_mlp = bool(cfg.get_path("loss.nce.use_mlp", True))
        self.nce_idt = bool(cfg.get_path("loss.nce.nce_idt", False))
        self.nce_dim = int(cfg.get_path("loss.nce.nce_dim", 256))

        if not (self.use_gan or self.use_l1 or self.use_nce):
            raise ValueError(
                "All three lambdas are zero, so the generator has no training "
                "signal at all. Set at least one of loss.lambda_gan, "
                "loss.lambda_l1, loss.lambda_nce to a non-zero value."
            )

    @property
    def use_gan(self):
        """False means: build no discriminator, run no D pass, save no D state."""
        return self.lambda_gan > EPS

    @property
    def use_l1(self):
        return self.lambda_l1 > EPS

    @property
    def use_nce(self):
        """False means: build no MLP heads, tap no features, create no optimizer_F."""
        return self.lambda_nce > EPS

    def describe(self):
        """One-line human summary, logged at startup and written into the run dir."""
        parts = []
        if self.use_gan:
            parts.append(f"{self.lambda_gan:g}*L_cGAN({self.gan_mode})")
        if self.use_l1:
            parts.append(f"{self.lambda_l1:g}*L_L1")
        if self.use_nce:
            parts.append(f"{self.lambda_nce:g}*L_PatchNCE"
                         f"(layers={self.nce_layers}, patches={self.num_patches})")
        body = "  +  ".join(parts) if parts else "<empty>"

        name = self.nickname()
        return f"L_total = {body}      [{name}]"

    def nickname(self):
        """
        What this configuration is, in the vocabulary of the literature.

        Printed at startup so a run's identity is unambiguous in the log — it is
        surprisingly easy to believe you are running the full objective when an
        override quietly disabled a term.
        """
        if self.use_gan:
            if self.use_l1 and self.use_nce:
                return "pix2pix + PatchNCE"
            if self.use_l1:
                return "pix2pix"
            return "CUT-like (contrastive, no L1)"

        # No adversary: these are regression baselines, not GANs.
        if self.use_l1 and self.use_nce:
            return "L1 + PatchNCE regression (no adversary)"
        if self.use_l1:
            return "L1-only regression (no adversary)"
        return "contrastive-only regression (no adversary)"


def build_losses(cfg):
    """
    Build the loss modules the plan calls for.

    Returns (plan, modules) where `modules` holds only what is active — a key is
    absent rather than present-and-unused, so a downstream `if "gan" in modules`
    is a truthful statement about what this run computes.
    """
    plan = LossPlan(cfg)
    modules = {}

    if plan.use_gan:
        from .gan_loss import build_gan_loss
        modules["gan"] = build_gan_loss(cfg)

    if plan.use_l1:
        import torch.nn as nn
        # reduction='none' so the validity mask from the dataset can exclude
        # padded pixels before the mean is taken. Averaging over padding would
        # make a heavily-padded small slice look artificially accurate.
        modules["l1"] = nn.L1Loss(reduction="none")

    if plan.use_nce:
        from .patch_nce import build_nce_loss
        modules["nce"] = build_nce_loss(cfg)

    logger.info("%s", plan.describe())
    return plan, modules


def masked_l1(pred, target, mask, criterion):
    """
    L1 restricted to valid (non-padded) pixels.

    The dataset zero-pads variable-sized slices up to a common size and records
    which pixels are real. Those padded pixels are identical in the prediction's
    target and in the input, so including them would contribute near-zero error
    over a large area and dilute the reported loss by however much padding a
    given slice happened to need — making a 180x180 slice look better than a
    430x430 one for reasons that have nothing to do with the model.
    """
    per_pixel = criterion(pred, target)
    denom = mask.sum().clamp(min=1.0)
    return (per_pixel * mask).sum() / denom
