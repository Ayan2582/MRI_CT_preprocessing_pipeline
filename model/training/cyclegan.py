"""
cyclegan.py
───────────
The unpaired baseline: two generators, two discriminators, no correspondence.

    L_G = lambda_gan   * ( L_GAN(D_B, G_A2B(A)) + L_GAN(D_A, G_B2A(B)) )
        + lambda_cycle * ( ||G_B2A(G_A2B(A)) - A||_1 + ||G_A2B(G_B2A(B)) - B||_1 )
        + lambda_cycle * lambda_identity
                       * ( ||G_A2B(B) - B||_1 + ||G_B2A(A) - A||_1 )

WHAT QUESTION THIS ANSWERS. Every other experiment here is handed 2161 QC-accepted
MRI/CT correspondences. exp8 is handed two piles of images from disjoint sets of
patients and told to find the mapping itself. The gap between exp8 and exp1 is
what the pairing is worth — a number this project has never measured, and the
one figure a reader coming from the unpaired literature will look for first.
exp8 is EXPECTED to score worse on mae_norm. That is the result, not a failure.

WHY IT IS A SIBLING OF Pix2PixNCEModel RATHER THAN A SUBCLASS. RegGAN adds a term
to the pix2pix step, so it subclasses and overrides two hooks. CycleGAN replaces
the step: four networks, no conditioning, no target to compare against, and a
reconstruction loss that closes a loop instead of pointing at a label. Inheriting
would mean overriding almost everything it inherited. What IS shared — GANLoss,
DiffAugment, EMA, masked_l1, the network builders, the LR schedulers — is shared
by import, which is the honest kind of reuse.

WHY THE DISCRIMINATORS MUST BE UNCONDITIONAL. pix2pix's D sees cat[MRI, CT] so it
judges correspondence rather than plausibility. Here the MRI and the CT in a batch
come from different patients and no correspondence is asserted, so stacking them
would train D on a relationship that does not exist. The constructor refuses
model.discriminator.conditional: true rather than letting that happen quietly.

WHAT THE IDENTITY TERM IS FOR. G_A2B applied to a real CT should return it
unchanged. Without that constraint nothing stops both generators from applying a
consistent global shift — a systematic HU offset, say — since the cycle would
still close perfectly. It is weighted by lambda_cycle * lambda_identity, which is
the reference implementation's convention: lambda_identity is a FRACTION OF the
cycle weight, not an absolute one, so the documented 0.5 means 5.0 here.
"""

import itertools
import logging

import torch
import torch.nn as nn

from ..losses.builder import build_losses, masked_l1
from ..networks.builder import build_discriminator, build_generator
from .diffaug import DiffAugment
from .ema import ModelEMA
from .image_pool import ImagePool

logger = logging.getLogger(__name__)


class CycleGANModel(nn.Module):

    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.plan, self.criteria = build_losses(cfg)

        if not self.plan.use_gan:
            raise ValueError(
                "model.name is 'cyclegan' but loss.lambda_gan is 0. CycleGAN's "
                "only link between the two domains is adversarial — with no "
                "discriminators the cycle loss is minimised perfectly by making "
                "both generators the identity, and the run would learn nothing."
            )
        if not self.plan.use_cycle:
            raise ValueError(
                "model.name is 'cyclegan' but loss.lambda_cycle is 0. Without "
                "the cycle term the two directions are unconstrained and G_A2B "
                "may emit any plausible CT of any patient — see exp8's config."
            )
        if cfg.get_path("model.discriminator.conditional", True):
            raise ValueError(
                "model.discriminator.conditional must be false for CycleGAN. A "
                "conditional D judges whether a CT corresponds to a given MRI, "
                "but here the two come from different patients and no "
                "correspondence is claimed — it would be trained on a "
                "relationship that does not exist in the data."
            )

        # ── Networks ─────────────────────────────────────────────────────────
        # Both directions use the same architecture as exp0-exp4, so exp8 differs
        # from exp1 in its objective and its data pairing, not in its capacity.
        self.netG_A2B = build_generator(cfg.model.generator).to(device)   # MRI -> CT
        self.netG_B2A = build_generator(cfg.model.generator).to(device)   # CT -> MRI

        spectral = cfg.get_path("stabilizers.spectral_norm_d", True)
        batch_size = int(cfg.train.batch_size)
        self.netD_B = build_discriminator(cfg.model, spectral, batch_size).to(device)
        self.netD_A = build_discriminator(cfg.model, spectral, batch_size).to(device)

        # ── Optimizers ───────────────────────────────────────────────────────
        # One optimizer per player, not per network. The two generators minimise
        # a single scalar and so do the two discriminators, so splitting them
        # would only add state. It also means .optimizer_G / .optimizer_D still
        # mean what the trainer's two LR schedulers expect.
        lr = float(cfg.train.lr)
        betas = (float(cfg.train.beta1), float(cfg.train.beta2))
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
            lr=lr, betas=betas)

        lr_d = lr
        if cfg.get_path("stabilizers.ttur.enabled", False):
            lr_d = float(cfg.get_path("stabilizers.ttur.lr_d", lr))
            logger.info("TTUR enabled: lr_G=%.2e, lr_D=%.2e", lr, lr_d)
        self.optimizer_D = torch.optim.Adam(
            itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
            lr=lr_d, betas=betas)

        # ── Stabilisers ──────────────────────────────────────────────────────
        pool_size = int(cfg.get_path("train.image_pool_size", 50))
        self.pool_A = ImagePool(pool_size)
        self.pool_B = ImagePool(pool_size)

        self.diffaug = DiffAugment(
            enabled=cfg.get_path("stabilizers.diffaug.enabled", False),
            policy=cfg.get_path("stabilizers.diffaug.policy", ""))

        # Only the MRI->CT generator is averaged: it is the only one validation,
        # metrics and sample rendering ever call. Averaging G_B2A too would
        # double the shadow's memory to shadow a network nothing evaluates.
        self.ema = None
        if cfg.get_path("stabilizers.ema.enabled", True):
            self.ema = ModelEMA(
                self.netG_A2B,
                decay=cfg.get_path("stabilizers.ema.decay", 0.999),
                start_epoch=cfg.get_path("stabilizers.ema.start_epoch", 1))

        self.warmup_epochs = int(cfg.get_path("train.gan_warmup_epochs", 0))
        self.grad_clip = float(cfg.get_path("train.grad_clip", 0.0))
        self.global_step = 0

        logger.info("CycleGAN: 2 generators + 2 unconditional discriminators, "
                    "lambda_cycle=%g, identity=%g (effective %g), pool=%d",
                    self.plan.lambda_cycle, self.plan.lambda_identity,
                    self.plan.lambda_cycle * self.plan.lambda_identity, pool_size)

    # ── Trainer-facing surface ───────────────────────────────────────────────

    def train_mode(self):
        for net in (self.netG_A2B, self.netG_B2A, self.netD_A, self.netD_B):
            net.train()

    def gan_active(self, epoch):
        return epoch >= self.warmup_epochs

    def generator_for_eval(self):
        """
        The MRI->CT direction, EMA-averaged when enabled.

        Returning this — rather than the model — is what lets Trainer.validate,
        MetricAccumulator and render_samples treat exp8 exactly like exp1. They
        call it with a real MRI and get a synthetic CT back; that the model also
        contains a CT->MRI generator is none of their business.
        """
        if self.cfg.get_path("eval.use_ema", True) and self.ema is not None:
            return self.ema.module
        return self.netG_A2B

    def set_eval_mode(self, net):
        """Deterministic at eval, for the reasons Pix2PixNCEModel documents."""
        net.eval()
        if self.cfg.get_path("model.generator.dropout_at_eval", False):
            for module in net.modules():
                if isinstance(module, nn.Dropout):
                    module.train()
        return net

    # ── Data plumbing ────────────────────────────────────────────────────────

    def set_input(self, batch):
        self.real_A = batch["A"].to(self.device, non_blocking=True)     # MRI
        self.real_B = batch["B"].to(self.device, non_blocking=True)     # CT
        # UnpairedSliceDataset supplies two masks because the two images are
        # different patients. Falling back to the shared key keeps the model
        # runnable on the paired dataset, which is what an "is the pairing worth
        # anything?" ablation would want.
        self.mask_A = batch.get("mask_A", batch["mask"]).to(self.device, non_blocking=True)
        self.mask_B = batch.get("mask_B", batch["mask"]).to(self.device, non_blocking=True)

    def forward(self):
        self.fake_B = self.netG_A2B(self.real_A)        # synthetic CT
        self.rec_A = self.netG_B2A(self.fake_B)         # back to MRI
        self.fake_A = self.netG_B2A(self.real_B)        # synthetic MRI
        self.rec_B = self.netG_A2B(self.fake_A)         # back to CT
        return self.fake_B

    # ── Optimisation ─────────────────────────────────────────────────────────

    def backward_D(self, scaler, amp_ctx):
        """
        Both discriminators, one backward.

        Each sees real images of its own domain against a sample from the pool of
        past fakes — see image_pool.py for why the pool and not just the current
        batch.
        """
        stats = {}
        with amp_ctx:
            loss_D = torch.zeros((), device=self.device)
            for tag, net, real, fake, pool in (
                ("A", self.netD_A, self.real_A, self.fake_A, self.pool_A),
                ("B", self.netD_B, self.real_B, self.fake_B, self.pool_B),
            ):
                real_in = self.diffaug(real)
                fake_in = self.diffaug(pool.query(fake))

                loss, sub_stats = self.criteria["gan"].d_loss(net(real_in),
                                                              net(fake_in))
                # Halved, as in the reference implementation: D takes one step
                # per G step and two full-strength domain losses would make the
                # discriminators advance at twice the generators' rate.
                loss_D = loss_D + 0.5 * loss
                for key, value in sub_stats.items():
                    stats[f"{key}_{tag}"] = value

        self.optimizer_D.zero_grad(set_to_none=True)
        scaler.scale(loss_D).backward()
        if self.grad_clip > 0:
            scaler.unscale_(self.optimizer_D)
            nn.utils.clip_grad_norm_(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                self.grad_clip)
        scaler.step(self.optimizer_D)

        stats["D_total"] = loss_D.detach()
        return stats

    def backward_G(self, epoch, scaler, amp_ctx):
        """Both generators: adversarial, cycle and identity, one backward."""
        stats = {}
        with amp_ctx:
            loss_G = torch.zeros((), device=self.device)

            if self.gan_active(epoch):
                g_a2b = self.criteria["gan"].g_loss(self.netD_B(self.diffaug(self.fake_B)))
                g_b2a = self.criteria["gan"].g_loss(self.netD_A(self.diffaug(self.fake_A)))
                loss_G = loss_G + self.plan.lambda_gan * (g_a2b + g_b2a)
                stats["G_GAN_A2B"] = g_a2b.detach()
                stats["G_GAN_B2A"] = g_b2a.detach()

            # Masked like every other reconstruction term here: the slices are
            # zero-padded to a common size and padding must not be scored.
            cyc_A = masked_l1(self.rec_A, self.real_A, self.mask_A, self.criteria["l1"])
            cyc_B = masked_l1(self.rec_B, self.real_B, self.mask_B, self.criteria["l1"])
            loss_G = loss_G + self.plan.lambda_cycle * (cyc_A + cyc_B)
            stats["G_cycle_A"] = cyc_A.detach()
            stats["G_cycle_B"] = cyc_B.detach()

            if self.plan.use_identity:
                # A generator handed an image already in its target domain must
                # leave it alone. Both directions are single-channel here, so
                # G_A2B(real_B) is well defined.
                idt_B = masked_l1(self.netG_A2B(self.real_B), self.real_B,
                                  self.mask_B, self.criteria["l1"])
                idt_A = masked_l1(self.netG_B2A(self.real_A), self.real_A,
                                  self.mask_A, self.criteria["l1"])
                weight = self.plan.lambda_cycle * self.plan.lambda_identity
                loss_G = loss_G + weight * (idt_A + idt_B)
                stats["G_idt_A"] = idt_A.detach()
                stats["G_idt_B"] = idt_B.detach()

        self.optimizer_G.zero_grad(set_to_none=True)
        scaler.scale(loss_G).backward()
        if self.grad_clip > 0:
            scaler.unscale_(self.optimizer_G)
            nn.utils.clip_grad_norm_(
                itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
                self.grad_clip)
        scaler.step(self.optimizer_G)

        stats["G_total"] = loss_G.detach()
        return stats

    def optimize_parameters(self, batch, epoch, scaler, amp_ctx):
        self.set_input(batch)
        self.forward()

        stats = {}
        if self.gan_active(epoch):
            stats.update(self.backward_D(scaler, amp_ctx))
        stats.update(self.backward_G(epoch, scaler, amp_ctx))
        scaler.update()

        if self.ema is not None:
            self.ema.update(self.netG_A2B, epoch)

        self.global_step += 1
        return {k: float(v) for k, v in stats.items()}

    # ── Checkpointing ────────────────────────────────────────────────────────

    def state_dict(self, *args, **kwargs):
        state = {
            "netG_A2B": self.netG_A2B.state_dict(),
            "netG_B2A": self.netG_B2A.state_dict(),
            "netD_A": self.netD_A.state_dict(),
            "netD_B": self.netD_B.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            "global_step": self.global_step,
            "loss_plan": {
                "lambda_gan": self.plan.lambda_gan,
                "lambda_l1": self.plan.lambda_l1,
                "lambda_nce": self.plan.lambda_nce,
                "lambda_cycle": self.plan.lambda_cycle,
                "lambda_identity": self.plan.lambda_identity,
                "nickname": self.plan.nickname(),
            },
        }
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()
        # The image pools are NOT saved; see image_pool.py for why. A resumed run
        # refills them within an epoch.
        return state

    def load_state_dict(self, state, strict=True):
        if "netG_A2B" not in state:
            raise ValueError(
                "This is a CycleGAN run but the checkpoint holds a single "
                "generator — it was produced by a pix2pix-family run. The two "
                "have no compatible weights to continue from."
            )
        self.netG_A2B.load_state_dict(state["netG_A2B"])
        self.netG_B2A.load_state_dict(state["netG_B2A"])
        self.netD_A.load_state_dict(state["netD_A"])
        self.netD_B.load_state_dict(state["netD_B"])
        self.optimizer_G.load_state_dict(state["optimizer_G"])
        self.optimizer_D.load_state_dict(state["optimizer_D"])
        self.global_step = state.get("global_step", 0)

        if self.ema is not None and "ema" in state:
            self.ema.load_state_dict(state["ema"])
        return self
