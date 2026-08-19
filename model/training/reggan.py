"""
reggan.py
─────────
RegGAN: pix2pix with its reconstruction loss moved into a registered frame.

    L_total = lambda_gan    * L_cGAN
            + lambda_corr   * L1(warp(G(A), phi), B)
            + lambda_smooth * ||grad phi||^2          , phi = R(G(A), B)

WHY THIS EXPERIMENT EXISTS. configs/base.yaml:55-59 records that manual QC
corrected translation on these pairs but could not correct in-plane rotation, and
that the residual is unmeasured. configs/exp3_nce_heavy.yaml then reweights its
objective as a hedge against that residual, while saying plainly that it does not
know how large it is. RegGAN closes the loop: R predicts the residual, L1 is
taken after it has been applied, and the field's mean magnitude is logged every
epoch as R_flow_px. That number is the answer exp3 was written without.

WHY IT SHOULD HELP IF THE RESIDUAL IS REAL. L1 against a misaligned target does
not merely mis-score the generator, it teaches it to blur: under uncertainty
about where an edge belongs, the L1-optimal prediction is a smeared edge. Every
millimetre of uncontrolled residual buys blur. Scoring after registration removes
the incentive.

WHAT IT INHERITS. Everything. Pix2PixNCEModel already owns G, D, the
discriminator step, DiffAugment, EMA, R1, the AMP scaler dance and the
checkpoint contract. This class adds one network, one optimizer and one loss
term, through the two extension points declared in that file — extra_G_terms and
g_step_optimizers — rather than by reimplementing backward_G.

THE DEGENERATE SOLUTION, AND WHY IT DOES NOT HAPPEN HERE. An unconstrained phi
can warp nearly any prediction onto nearly any target, driving L_corr to zero
while G learns nothing. Three things prevent it, and all three are load-bearing:
the smoothness penalty (LossPlan refuses to run without it), R's deliberately
small capacity (networks/registration.py), and R's zero-initialised head, which
means training begins at the identity and the field grows only as far as the data
pushes it.
"""

import logging

import torch

from ..losses.builder import masked_l1
from ..losses.registration import flow_magnitude, flow_smoothness
from ..networks.registration import SpatialTransformer, build_registration_net
from .pix2pix_nce import Pix2PixNCEModel

logger = logging.getLogger(__name__)


class RegGANModel(Pix2PixNCEModel):

    def __init__(self, cfg, device):
        super().__init__(cfg, device)

        if not self.plan.use_corr:
            raise ValueError(
                "model.name is 'reggan' but loss.lambda_corr is zero, so there "
                "would be no registration term and this would be an ordinary "
                "pix2pix run carrying an untrained extra network. Set "
                "loss.lambda_corr (exp7_reggan.yaml uses 100.0), or set "
                "model.name back to 'pix2pix_nce'."
            )

        # R sees the generated image and the real one stacked. Not a free
        # choice, so it is derived rather than configured.
        in_channels = 2 * int(cfg.get_path("model.generator.out_channels", 1))
        self.netR = build_registration_net(
            cfg.get_path("model.registration", {}), in_channels).to(device)
        self.stn = SpatialTransformer()

        # Same learning rate as G. R and G are two halves of one objective —
        # giving R its own rate would let the field adapt faster than the image
        # it is correcting, which is the direction that ends in the degenerate
        # solution described in the module docstring.
        self.optimizer_R = torch.optim.Adam(
            self.netR.parameters(),
            lr=float(cfg.train.lr),
            betas=(float(cfg.train.beta1), float(cfg.train.beta2)))

        logger.info("RegGAN: correction loss active "
                    "(lambda_corr=%g, lambda_smooth=%g); L1 in the registered "
                    "frame%s", self.plan.lambda_corr, self.plan.lambda_smooth,
                    "" if not self.plan.use_l1 else
                    f", ALONGSIDE unwarped L1 at {self.plan.lambda_l1:g}")

    # ── Extension points ─────────────────────────────────────────────────────

    def train_mode(self):
        super().train_mode()
        self.netR.train()

    def g_step_optimizers(self):
        """
        R is stepped by the generator's backward, not by one of its own.

        R and G minimise the SAME scalar. Training R on a separate objective — or
        on its own alternating step — would make the field an independent actor
        with its own incentive to explain the residual away. Here it can only
        reduce L_corr by finding a deformation that genuinely improves the match,
        because the smoothness term is in the same sum.
        """
        return super().g_step_optimizers() + [self.optimizer_R]

    def extra_G_terms(self, stats):
        """
        The correction loss, and the diagnostics that make exp7 worth running.

        RUNS IN FP32, DELIBERATELY. The field is a displacement in pixels, and at
        half precision a value near 400 — the size of the largest validation
        slices — quantises to steps of about 0.25 px. The residual this whole
        experiment exists to measure is itself only a few pixels, so an fp16
        field would put the measurement noise at the same order as the
        measurement. R is small (nrf=32), so the fp32 pass is cheap. This is a
        disabled-autocast REGION inside the ordinary scaled backward, not a
        separate unscaled step — unlike the path-length penalty, there is no
        double backward here, so nothing needs splitting off.
        """
        with torch.autocast(device_type=self.device.type, enabled=False):
            fake_B = self.fake_B.float()
            real_B = self.real_B.float()
            mask = self.mask.float()

            flow = self.netR(torch.cat([fake_B, real_B], dim=1))

            # DIRECTION: the GENERATED image is warped toward the target, never
            # the reverse. Warping the target instead would make the supervision
            # signal itself mobile, and G could then be rewarded for producing an
            # image that is easy to warp rather than one that is correct.
            warped = self.stn(fake_B, flow)

            # The valid region moves with the warp, so score only pixels valid in
            # BOTH frames. Using the unwarped mask alone lets the deformation
            # drag zero-padding into the scored area, where it agrees with the
            # target's padding perfectly and reads as accuracy.
            warped_mask = self.stn(mask, flow, mode="nearest") * mask

            g_corr = masked_l1(warped, real_B, warped_mask, self.criteria["l1"])
            g_smooth = flow_smoothness(flow, mask)

            mean_flow, max_flow = flow_magnitude(flow, mask)

        stats["G_corr"] = g_corr.detach()
        stats["G_smooth"] = g_smooth.detach()
        # Millimetres: 1 px = 1 mm throughout this project. THIS is the number
        # exp3 was missing.
        stats["R_flow_px"] = mean_flow
        stats["R_flow_max"] = max_flow

        return (self.plan.lambda_corr * g_corr
                + self.plan.lambda_smooth * g_smooth)

    # ── Checkpointing ────────────────────────────────────────────────────────

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state["netR"] = self.netR.state_dict()
        state["optimizer_R"] = self.optimizer_R.state_dict()
        state["loss_plan"]["lambda_corr"] = self.plan.lambda_corr
        state["loss_plan"]["lambda_smooth"] = self.plan.lambda_smooth
        return state

    def load_state_dict(self, state, strict=True):
        super().load_state_dict(state, strict=strict)

        if "netR" not in state:
            raise ValueError(
                "This is a RegGAN run but the checkpoint holds no registration "
                "network — it was produced by a plain pix2pix run. Resuming "
                "would start R from the identity against an already-trained G, "
                "so the first epochs would score in an unregistered frame while "
                "reporting themselves as RegGAN. That is not a continuation of "
                "either run."
            )
        self.netR.load_state_dict(state["netR"])
        self.optimizer_R.load_state_dict(state["optimizer_R"])
        return self
