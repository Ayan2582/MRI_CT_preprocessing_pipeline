"""
pix2pix_nce.py
──────────────
The composite model: generator, discriminator, projection heads, three
optimizers, and one training step that computes whichever loss terms the config
actually asked for.

    L_total = lambda_gan * L_cGAN + lambda_l1 * L_L1 + lambda_nce * L_PatchNCE

THE DESIGN RULE. A zero lambda removes a term's whole code path. With
lambda_gan=0 there is no discriminator object, no D forward pass and no D state
in the checkpoint; with lambda_nce=0 there are no MLP heads and no third
optimizer. This is what makes `--set loss.lambda_nce=0` an honest pix2pix run
rather than the full model multiplied by zero, and it is asserted directly in
scripts/smoke_test.py by inspecting checkpoint keys.

THE ORDER OF A STEP.
    1. forward           fake_B = G(real_A)
    2. D step            real and fake both through DiffAugment, then D.
                         Skipped during GAN warm-up and when lambda_gan == 0.
    3. G step            adversarial + L1 + PatchNCE, one backward
    4. path-length step  StyleGAN2 only, and only every pl_every steps: a SECOND,
                         separate optimisation of G. It needs a double backward,
                         which is fragile and overflows in fp16 under autocast, so
                         it runs unscaled in fp32 — and a scaled and an unscaled
                         backward must not meet inside one GradScaler step.
    5. EMA update        shadow generator folds in the new weights

ARCHITECTURE IS A CONFIG CHOICE. netG and netD come from networks/builder.py,
which dispatches on model.generator.type / model.discriminator.type. Nothing in
this file knows whether it is driving a U-Net or a StyleGAN2 synthesis network —
the tap protocol PatchNCE relies on is implemented by both.

THE LAZY-OPTIMIZER GOTCHA. PatchSampleF's MLP heads cannot be built until a real
tensor has passed through, because their input widths are the encoder's channel
counts at the tapped depths. So optimizer_F does not exist at construction time;
it is created immediately after the first NCE forward and before the first
backward. Checkpoint save and load both tolerate its absence, which matters on
step 0 and on any resume of a run whose first step has not completed.
"""

import logging
import math

import torch
import torch.nn as nn

from ..losses.builder import build_losses, masked_l1
from ..losses.gan_loss import r1_penalty
from ..networks.builder import build_discriminator, build_generator
from ..networks.patch_sampler import PatchSampleF
from .diffaug import DiffAugment
from .ema import ModelEMA

logger = logging.getLogger(__name__)


class Pix2PixNCEModel(nn.Module):

    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.plan, self.criteria = build_losses(cfg)

        # ── Networks ─────────────────────────────────────────────────────────
        self.netG = build_generator(cfg.model.generator).to(device)

        self.netD = None
        if self.plan.use_gan:
            spectral = cfg.get_path("stabilizers.spectral_norm_d", True)
            # batch_size reaches the builder because StyleGAN2's minibatch-stddev
            # layer groups the batch and needs a group size that divides it.
            self.netD = build_discriminator(
                cfg.model, spectral, int(cfg.train.batch_size)).to(device)
        else:
            logger.info("lambda_gan == 0: no discriminator built at all "
                        "(this run is a plain regression)")

        self.netF = None
        if self.plan.use_nce:
            self.netF = PatchSampleF(use_mlp=self.plan.use_mlp,
                                     nce_dim=self.plan.nce_dim).to(device)
            self._validate_nce_taps()
        else:
            logger.info("lambda_nce == 0: no projection heads, no PatchNCE pass")

        # ── Optimizers ───────────────────────────────────────────────────────
        lr = float(cfg.train.lr)
        betas = (float(cfg.train.beta1), float(cfg.train.beta2))

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=betas)

        self.optimizer_D = None
        if self.plan.use_gan:
            # TTUR: a separate, usually lower, learning rate for D. The handicap
            # for the player that tends to win. Off by default.
            lr_d = lr
            if cfg.get_path("stabilizers.ttur.enabled", False):
                lr_d = float(cfg.get_path("stabilizers.ttur.lr_d", lr))
                logger.info("TTUR enabled: lr_G=%.2e, lr_D=%.2e", lr, lr_d)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=lr_d, betas=betas)

        # Built after the first NCE forward — see the module docstring.
        self.optimizer_F = None
        self._f_lr, self._f_betas = lr, betas

        # ── Stabilisers ──────────────────────────────────────────────────────
        self.diffaug = DiffAugment(
            enabled=cfg.get_path("stabilizers.diffaug.enabled", False) and self.plan.use_gan,
            policy=cfg.get_path("stabilizers.diffaug.policy", ""))

        self.ema = None
        if cfg.get_path("stabilizers.ema.enabled", True):
            self.ema = ModelEMA(
                self.netG,
                decay=cfg.get_path("stabilizers.ema.decay", 0.999),
                start_epoch=cfg.get_path("stabilizers.ema.start_epoch", 1))

        self.r1_enabled = (cfg.get_path("stabilizers.r1.enabled", False)
                           and self.plan.use_gan)
        self.r1_gamma = float(cfg.get_path("stabilizers.r1.gamma", 10.0))
        self.r1_every = int(cfg.get_path("stabilizers.r1.every", 16))
        if self.r1_enabled:
            logger.info("R1 penalty enabled: gamma=%.1f, every %d D steps "
                        "(this D step runs in fp32)", self.r1_gamma, self.r1_every)

        # ── Path-length regularization (StyleGAN2) ───────────────────────────
        self.pl_enabled = bool(cfg.get_path("stabilizers.path_length.enabled", False))
        self.pl_weight = float(cfg.get_path("stabilizers.path_length.weight", 2.0))
        self.pl_every = int(cfg.get_path("stabilizers.path_length.every", 4))
        self.pl_decay = float(cfg.get_path("stabilizers.path_length.decay", 0.01))
        # Running mean of the path length, the moving target the penalty pulls
        # toward. Training state, so it is checkpointed.
        self.pl_mean = torch.zeros((), device=device)
        if self.pl_enabled:
            if not hasattr(self.netG, "synthesise_with_styles"):
                raise ValueError(
                    "stabilizers.path_length.enabled is true, but generator type "
                    f"'{cfg.get_path('model.generator.type', 'unet')}' has no style "
                    "vector to regularise. Path-length regularization measures how "
                    "far the image moves per unit step in W, which only exists for "
                    "a style-based generator. Set it false, or use "
                    "model.generator.type: stylegan2."
                )
            logger.info("path-length regularization enabled: weight=%.1f, every %d "
                        "G steps (those steps run in fp32, unscaled)",
                        self.pl_weight, self.pl_every)

        self.warmup_epochs = int(cfg.get_path("train.gan_warmup_epochs", 0))
        self.grad_clip = float(cfg.get_path("train.grad_clip", 0.0))

        self.global_step = 0
        self._buffers = {}

    # ── Setup helpers ────────────────────────────────────────────────────────

    def _validate_nce_taps(self):
        """
        Fail early if a tap cannot supply the requested number of patches.

        A tap deeper than the image can support degenerates into sampling the
        same handful of locations over and over — the loss still decreases, so
        nothing announces the problem. Checking it here converts a silent
        quality bug into a startup error.
        """
        crop = int(self.cfg.data.crop_size)
        for tap in self.plan.nce_layers:
            side = self.netG.tap_spatial(tap, crop)
            locations = side * side
            if locations < self.plan.num_patches:
                logger.warning(
                    "NCE tap %d yields %dx%d = %d locations at crop %d, fewer "
                    "than num_patches=%d. Sampling will be clamped to %d, which "
                    "weakens this tap's contrastive signal. Consider a shallower "
                    "tap in loss.nce.layers.",
                    tap, side, side, locations, crop,
                    self.plan.num_patches, locations)

    def _ensure_optimizer_F(self):
        """Create optimizer_F once the MLP heads exist. Idempotent."""
        if self.optimizer_F is not None or self.netF is None:
            return
        params = list(self.netF.parameters())
        if not params:
            # use_mlp=False: PatchSampleF is a pure sampler with nothing to train.
            return
        self.optimizer_F = torch.optim.Adam(params, lr=self._f_lr, betas=self._f_betas)
        logger.info("built optimizer_F over %d MLP-head tensors "
                    "(deferred until feature widths were known)", len(params))

    # ── Data plumbing ────────────────────────────────────────────────────────

    def set_input(self, batch):
        self.real_A = batch["A"].to(self.device, non_blocking=True)   # MRI
        self.real_B = batch["B"].to(self.device, non_blocking=True)   # CT
        self.mask = batch["mask"].to(self.device, non_blocking=True)

    def forward(self):
        self.fake_B = self.netG(self.real_A)
        return self.fake_B

    def _d_input(self, source, target):
        """Assemble D's input: conditional D sees both modalities stacked."""
        if self.cfg.get_path("model.discriminator.conditional", True):
            return torch.cat([source, target], dim=1)
        return target

    def gan_active(self, epoch):
        """Whether the adversarial term contributes this epoch."""
        return self.plan.use_gan and epoch >= self.warmup_epochs

    # ── Loss terms ───────────────────────────────────────────────────────────

    def compute_nce(self, source, target):
        """
        PatchNCE between the encoder's view of `source` and of `target`.

        Both go through the SAME encoder (the generator's), and both are sampled
        at the SAME spatial locations — the ids drawn for the source are handed
        back for the target. Sampling independently would make every pair a
        negative, and the loss would still decrease, which is what makes that
        particular bug so easy to ship.
        """
        layers = self.plan.nce_layers
        feat_k = self.netG(source, tap_layers=layers, encode_only=True)
        feat_q = self.netG(target, tap_layers=layers, encode_only=True)

        k_pool, sample_ids = self.netF(feat_k, self.plan.num_patches, None)
        q_pool, _ = self.netF(feat_q, self.plan.num_patches, sample_ids)

        # The heads exist now; the optimizer that trains them may not.
        self._ensure_optimizer_F()

        return self.criteria["nce"](q_pool, k_pool, batch_size=source.size(0))

    def path_length_penalty(self):
        """
        StyleGAN2's path-length regularization.

        WHAT IT ASKS FOR. That a fixed-size step in W moves the image by a fixed
        amount, everywhere in W — i.e. that the generator's mapping is well
        conditioned rather than wildly sensitive in some directions and flat in
        others. It is measured by pushing a random unit image-space direction back
        through the Jacobian and comparing the resulting length against a running
        mean of past lengths.

        HOW IT DIFFERS FROM THE PAPER HERE. StyleGAN2 draws w from the mapping
        network's Gaussian prior, so it has unlimited free samples. There is no
        prior in a translation model — every w is the encoding of a real patient —
        so the estimate comes from the batch that happens to be in flight, which
        makes it noisier than the published version at these batch sizes.

        Returns (penalty, mean_path_length) with the running mean already updated.
        """
        fake, ws = self.netG.synthesise_with_styles(self.real_A)

        # Unit-norm random direction in image space, scaled so the expected
        # magnitude is independent of resolution.
        noise = torch.randn_like(fake) / math.sqrt(fake.shape[2] * fake.shape[3])
        grad = torch.autograd.grad(outputs=(fake * noise).sum(), inputs=ws,
                                   create_graph=True, only_inputs=True)[0]

        lengths = grad.square().sum(dim=2).mean(dim=1).sqrt()
        mean = self.pl_mean.lerp(lengths.detach().mean(), self.pl_decay)
        with torch.no_grad():
            self.pl_mean.copy_(mean)

        return (lengths - mean).square().mean(), lengths.detach().mean()

    # ── Optimisation ─────────────────────────────────────────────────────────

    def backward_D(self, scaler, amp_ctx):
        """Discriminator step. Returns a stats dict."""
        do_r1 = self.r1_enabled and (self.global_step % self.r1_every == 0)

        # R1 differentiates D's output with respect to its input, and a
        # double-backward under autocast is both fragile and prone to inf in
        # fp16. This one step therefore runs in fp32.
        ctx = torch.autocast(device_type=self.device.type, enabled=False) if do_r1 else amp_ctx

        with ctx:
            real_in = self._d_input(self.real_A, self.real_B)
            fake_in = self._d_input(self.real_A, self.fake_B.detach())

            real_in = self.diffaug(real_in)
            fake_in = self.diffaug(fake_in)

            if do_r1:
                real_in.requires_grad_(True)

            pred_real = self.netD(real_in)
            pred_fake = self.netD(fake_in)
            loss_D, stats = self.criteria["gan"].d_loss(pred_real, pred_fake)

            if do_r1:
                penalty = r1_penalty(pred_real, real_in)
                # Lazy regularisation: applied every r1_every steps and scaled
                # by that factor, which recovers most of the benefit at a
                # fraction of the cost.
                loss_D = loss_D + (self.r1_gamma / 2.0) * penalty * self.r1_every
                stats["D_r1"] = penalty.detach()

        self.optimizer_D.zero_grad(set_to_none=True)
        if do_r1:
            loss_D.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.netD.parameters(), self.grad_clip)
            self.optimizer_D.step()
        else:
            scaler.scale(loss_D).backward()
            if self.grad_clip > 0:
                scaler.unscale_(self.optimizer_D)
                nn.utils.clip_grad_norm_(self.netD.parameters(), self.grad_clip)
            scaler.step(self.optimizer_D)

        stats["D_total"] = loss_D.detach()
        return stats

    def backward_G(self, epoch, scaler, amp_ctx):
        """Generator step: every active term, one backward. Returns stats."""
        stats = {}
        with amp_ctx:
            loss_G = torch.zeros((), device=self.device)

            if self.gan_active(epoch):
                fake_in = self.diffaug(self._d_input(self.real_A, self.fake_B))
                pred_fake = self.netD(fake_in)
                g_gan = self.criteria["gan"].g_loss(pred_fake)
                loss_G = loss_G + self.plan.lambda_gan * g_gan
                stats["G_GAN"] = g_gan.detach()

            if self.plan.use_l1:
                # Masked so zero-padding on variable-sized slices cannot dilute
                # the term toward zero.
                g_l1 = masked_l1(self.fake_B, self.real_B, self.mask,
                                 self.criteria["l1"])
                loss_G = loss_G + self.plan.lambda_l1 * g_l1
                stats["G_L1"] = g_l1.detach()

            if self.plan.use_nce:
                g_nce, per_layer = self.compute_nce(self.real_A, self.fake_B)
                loss_G = loss_G + self.plan.lambda_nce * g_nce
                stats["G_NCE"] = g_nce.detach()
                for i, value in enumerate(per_layer):
                    stats[f"G_NCE_L{self.plan.nce_layers[i]}"] = value

                if self.plan.nce_idt:
                    # Identity NCE: pass the real CT through G and require it to
                    # be left alone. Off by default here because L1 already
                    # pins the output to the target and this costs a further
                    # full generator pass.
                    idt_B = self.netG(self.real_B)
                    g_idt, _ = self.compute_nce(self.real_B, idt_B)
                    loss_G = loss_G + self.plan.lambda_nce * g_idt
                    stats["G_NCE_idt"] = g_idt.detach()

        self.optimizer_G.zero_grad(set_to_none=True)
        if self.optimizer_F is not None:
            self.optimizer_F.zero_grad(set_to_none=True)

        scaler.scale(loss_G).backward()

        if self.grad_clip > 0:
            scaler.unscale_(self.optimizer_G)
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.grad_clip)

        scaler.step(self.optimizer_G)
        if self.optimizer_F is not None:
            scaler.step(self.optimizer_F)

        stats["G_total"] = loss_G.detach()

        # ── Path-length regularization, as its own optimisation step ─────────
        # Lazy regularisation: applied every pl_every steps and scaled by that
        # factor, which recovers most of the benefit at a fraction of the cost —
        # the same trick R1 uses in backward_D.
        #
        # It is a SEPARATE step, not another term added to loss_G, for two
        # reasons. It needs a double backward through G, which is fragile and
        # overflows in fp16 under autocast, so it has to run unscaled in fp32 —
        # and mixing a scaled and an unscaled backward into one GradScaler step is
        # exactly where silent gradient corruption lives. Splitting it also
        # matches the reference implementation, where regularisation is its own
        # pass.
        if self.pl_enabled and (self.global_step % self.pl_every == 0):
            with torch.autocast(device_type=self.device.type, enabled=False):
                penalty, length = self.path_length_penalty()
                loss_pl = self.pl_weight * penalty * self.pl_every

            self.optimizer_G.zero_grad(set_to_none=True)
            loss_pl.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.netG.parameters(), self.grad_clip)
            self.optimizer_G.step()

            stats["G_PL"] = penalty.detach()
            stats["G_PL_len"] = length

        return stats

    def optimize_parameters(self, batch, epoch, scaler, amp_ctx):
        """One full training step. Returns a flat dict of scalars for logging."""
        self.set_input(batch)
        self.forward()

        stats = {}
        if self.gan_active(epoch):
            stats.update(self.backward_D(scaler, amp_ctx))
        stats.update(self.backward_G(epoch, scaler, amp_ctx))
        scaler.update()

        if self.ema is not None:
            self.ema.update(self.netG, epoch)

        self.global_step += 1
        return {k: float(v) for k, v in stats.items()}

    # ── Evaluation mode ──────────────────────────────────────────────────────

    def generator_for_eval(self):
        """
        The generator that validation and sample rendering should use.

        Returns the EMA shadow when enabled, because ranking epochs by the live
        weights means ranking a point on an oscillation rather than the model.
        """
        use_ema = self.cfg.get_path("eval.use_ema", True)
        if use_ema and self.ema is not None:
            return self.ema.module
        return self.netG

    def set_eval_mode(self, net):
        """
        Put a generator into evaluation mode.

        pix2pix conventionally leaves dropout ON at test time, using it as its
        only source of output variation. That is the wrong default for medical
        image synthesis: two runs of the same checkpoint on the same slice would
        disagree, so a reported metric could not be reproduced and a clinician
        could not be shown a stable image. config's model.generator.dropout_at_eval
        controls it and defaults to off.
        """
        net.eval()
        if self.cfg.get_path("model.generator.dropout_at_eval", False):
            for module in net.modules():
                if isinstance(module, nn.Dropout):
                    module.train()
        return net

    # ── Checkpointing ────────────────────────────────────────────────────────

    def state_dict(self, *args, **kwargs):
        """
        Everything needed to resume exactly, and nothing that is not built.

        Keys for inactive components are absent rather than None: the modularity
        check asserts on their absence, and a None would make "was this a
        pix2pix run?" a question about values rather than about structure.
        """
        state = {
            "netG": self.netG.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
            "global_step": self.global_step,
            "loss_plan": {
                "lambda_gan": self.plan.lambda_gan,
                "lambda_l1": self.plan.lambda_l1,
                "lambda_nce": self.plan.lambda_nce,
                "nickname": self.plan.nickname(),
            },
        }
        if self.netD is not None:
            state["netD"] = self.netD.state_dict()
            state["optimizer_D"] = self.optimizer_D.state_dict()
        if self.netF is not None:
            state["netF"] = self.netF.state_dict()
        if self.optimizer_F is not None:
            state["optimizer_F"] = self.optimizer_F.state_dict()
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()
        if self.pl_enabled:
            # The running path length is training state, not a derived value: a
            # resume that restarted it from zero would spend the first hundred
            # steps regularising toward a target that is still warming up, and the
            # bit-exact resume check in smoke_test.py would fail.
            state["pl_mean"] = self.pl_mean.detach().cpu()
        return state

    def load_state_dict(self, state, strict=True):
        self.netG.load_state_dict(state["netG"])
        self.optimizer_G.load_state_dict(state["optimizer_G"])
        self.global_step = state.get("global_step", 0)

        if self.netD is not None:
            if "netD" not in state:
                raise ValueError(
                    "This run has lambda_gan > 0 but the checkpoint holds no "
                    "discriminator — it was produced by an L1-only run. Resuming "
                    "would start D from scratch against an already-trained G, "
                    "which is not a continuation of either run."
                )
            self.netD.load_state_dict(state["netD"])
            self.optimizer_D.load_state_dict(state["optimizer_D"])

        if self.netF is not None and "netF" in state:
            self.netF.load_state_dict(state["netF"])
            if "optimizer_F" in state:
                # The heads were rebuilt by netF.load_state_dict above, so the
                # optimizer can now be constructed and its state restored.
                self._ensure_optimizer_F()
                if self.optimizer_F is not None:
                    self.optimizer_F.load_state_dict(state["optimizer_F"])

        if self.ema is not None and "ema" in state:
            self.ema.load_state_dict(state["ema"])

        # Absent when resuming a run that had path-length off, which is a normal
        # thing to do — zero is the same value a fresh run starts from.
        if self.pl_enabled and "pl_mean" in state:
            self.pl_mean.copy_(state["pl_mean"].to(self.pl_mean.device))
        return self


def build_lr_scheduler(optimizer, cfg):
    """
    Constant learning rate for the first half of training, then linear decay to
    zero over the second half.

    This is the pix2pix schedule and the decay half is not optional garnish —
    it is where fine detail settles. Stopping a run at the start of the decay
    phase reliably gives worse samples than letting it finish, so a run cut
    short by a Kaggle session limit should be resumed, not called done.
    """
    policy = cfg.get_path("train.lr_policy", "linear_decay")
    n_epochs = int(cfg.train.n_epochs)

    if policy == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    if policy != "linear_decay":
        raise NotImplementedError(f"Unknown train.lr_policy '{policy}'")

    start = int(n_epochs * float(cfg.get_path("train.lr_decay_start_frac", 0.5)))

    def lr_lambda(epoch):
        if epoch < start:
            return 1.0
        return max(0.0, 1.0 - (epoch - start) / max(1, n_epochs - start))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
