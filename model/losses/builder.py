"""
builder.py
──────────
Assembles the loss terms a config actually asks for.

    L_total = lambda_gan   * L_cGAN
            + lambda_l1    * L_L1          (pixel-aligned)
            + lambda_nce   * L_PatchNCE
            + lambda_corr  * L_L1(warp(fake, phi), real)   RegGAN
            + lambda_smooth* ||grad phi||^2                RegGAN
            + lambda_cycle * L_L1(G_B2A(G_A2B(A)), A)      CycleGAN, both ways
            + lambda_cycle * lambda_identity * L_L1(G(x), x)   CycleGAN

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

        # RegGAN. lambda_corr is L1 taken in the frame a learned deformation
        # puts the prediction into, and lambda_smooth is the penalty that keeps
        # that deformation a registration rather than a free reparameterisation.
        # They are inseparable: lambda_corr with lambda_smooth at zero has a
        # degenerate optimum, which losses/registration.py explains at length.
        self.lambda_corr = float(cfg.get_path("loss.lambda_corr", 0.0))
        self.lambda_smooth = float(cfg.get_path("loss.lambda_smooth", 0.0))

        # CycleGAN. lambda_identity is a FRACTION OF lambda_cycle, not an
        # absolute weight — the reference implementation's convention, kept
        # because every published value for it (0.5) is quoted in those terms.
        # Read as an absolute weight, 0.5 against a cycle weight of 10 would be
        # twenty times weaker than intended and the term would do nothing.
        self.lambda_cycle = float(cfg.get_path("loss.lambda_cycle", 0.0))
        self.lambda_identity = float(cfg.get_path("loss.lambda_identity", 0.0))

        self.gan_mode = cfg.get_path("loss.gan_mode", "lsgan")

        self.nce_layers = list(cfg.get_path("loss.nce.layers", [0, 1, 2, 3, 4]))
        self.num_patches = int(cfg.get_path("loss.nce.num_patches", 256))
        self.temperature = float(cfg.get_path("loss.nce.temperature", 0.07))
        self.use_mlp = bool(cfg.get_path("loss.nce.use_mlp", True))
        self.nce_idt = bool(cfg.get_path("loss.nce.nce_idt", False))
        self.nce_dim = int(cfg.get_path("loss.nce.nce_dim", 256))

        if self.use_corr and self.lambda_smooth <= EPS:
            raise ValueError(
                "loss.lambda_corr > 0 with loss.lambda_smooth == 0. The "
                "correction loss warps the prediction by a learned field before "
                "comparing it to the target, and an unpenalised field can warp "
                "almost any prediction onto almost any target — the loss would "
                "fall to zero while the generator learned nothing. Set "
                "loss.lambda_smooth (exp7_reggan.yaml uses 10.0)."
            )

        if not (self.use_gan or self.use_l1 or self.use_nce or self.use_corr
                or self.use_cycle):
            raise ValueError(
                "Every lambda is zero, so the generator has no training signal "
                "at all. Set at least one of loss.lambda_gan, loss.lambda_l1, "
                "loss.lambda_nce, loss.lambda_corr, loss.lambda_cycle to a "
                "non-zero value."
            )

        # GAN warm-up means "train on the reconstruction terms only for N epochs,
        # so G produces something anatomically sane before D starts critiquing".
        # With no reconstruction term there is nothing to warm up ON: the warm-up
        # epochs would leave backward_G with a freshly-created zero scalar that
        # never entered the graph, and .backward() raises "element 0 of tensors
        # does not require grad" — five layers from the config that caused it.
        # A purely adversarial objective (StyleGAN2's own) is a legitimate
        # configuration; combining it with a warm-up is not.
        self.gan_warmup_epochs = int(cfg.get_path("train.gan_warmup_epochs", 0))
        if self.gan_warmup_epochs > 0 and not (self.use_l1 or self.use_nce
                                               or self.use_corr
                                               or self.use_cycle):
            raise ValueError(
                f"train.gan_warmup_epochs is {self.gan_warmup_epochs}, but both "
                f"loss.lambda_l1, loss.lambda_nce, loss.lambda_corr and "
                f"loss.lambda_cycle are zero. Warm-up trains on the "
                f"reconstruction terms while D is held back, "
                f"and there are none here — the first epoch would have no loss at "
                f"all. Set train.gan_warmup_epochs: 0 for a purely adversarial "
                f"run (see configs/exp5_stylegan2_vanilla.yaml)."
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

    @property
    def use_corr(self):
        """
        False means: build no registration network, no optimizer_R, no warp.

        True is what makes this a RegGAN run. Note it is independent of use_l1 —
        exp7 sets lambda_l1 to zero so that the ONLY reconstruction signal is the
        registered one, which is the published objective. Running both is a legal
        ablation, not the default.
        """
        return self.lambda_corr > EPS

    @property
    def use_cycle(self):
        """
        False means: no cycle-consistency term. True is what makes a run CycleGAN.

        It is a reconstruction term, so it satisfies the GAN warm-up requirement
        the same way L1 does — there is something to train on while D is held
        back, even though nothing here compares against a paired target.
        """
        return self.lambda_cycle > EPS

    @property
    def use_identity(self):
        """Identity mapping term. Only meaningful alongside the cycle term."""
        return self.use_cycle and self.lambda_identity > EPS

    def describe(self):
        """One-line human summary, logged at startup and written into the run dir."""
        parts = []
        if self.use_gan:
            parts.append(f"{self.lambda_gan:g}*L_cGAN({self.gan_mode})")
        if self.use_l1:
            parts.append(f"{self.lambda_l1:g}*L_L1")
        if self.use_corr:
            parts.append(f"{self.lambda_corr:g}*L_corr(registered L1)")
            parts.append(f"{self.lambda_smooth:g}*L_smooth(grad phi)")
        if self.use_cycle:
            parts.append(f"{self.lambda_cycle:g}*L_cycle")
            if self.use_identity:
                parts.append(f"{self.lambda_cycle * self.lambda_identity:g}"
                             f"*L_identity")
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
        # CycleGAN and RegGAN are identified by the terms that are unique to
        # them, not by their lambdas' ratios, so both are tested first.
        if self.use_cycle:
            return ("CycleGAN (unpaired)" if self.use_identity
                    else "CycleGAN (unpaired, no identity term)")

        # RegGAN is identified by its correction term, not by its lambdas'
        # ratios, so it is tested before everything else.
        if self.use_corr:
            name = "RegGAN (registration-corrected L1)"
            if not self.use_gan:
                return name + ", no adversary"
            if self.use_l1:
                return name + " + unwarped L1"
            if self.use_nce:
                return name + " + PatchNCE"
            return name

        if self.use_gan:
            if self.use_l1 and self.use_nce:
                return "pix2pix + PatchNCE"
            if self.use_l1:
                return "pix2pix"
            if self.use_nce:
                return "CUT-like (contrastive, no L1)"
            # Purely adversarial — StyleGAN2's own objective. This branch used to
            # fall through to the CUT label, which named a contrastive term that a
            # lambda_nce=0 run does not have. That is exactly the confusion this
            # method exists to prevent.
            return "adversarial-only (no reconstruction term)"

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

    if plan.use_l1 or plan.use_corr or plan.use_cycle:
        # `or plan.use_corr` because RegGAN's correction term is itself a masked
        # L1 — just taken after the warp — and `or plan.use_cycle` because
        # CycleGAN's cycle and identity terms are too. Both exp7 and exp8 run
        # with lambda_l1 == 0, so gating this on use_l1 alone would leave
        # criteria["l1"] missing and both models would KeyError on step one.
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
