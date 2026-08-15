"""
gan_loss.py
───────────
The adversarial objective, in three interchangeable forms.

All three consume RAW discriminator scores (no sigmoid in D) and all three work
on the PatchGAN's [B,1,h,w] score grid, where the reduction averages over every
patch position.

  lsgan   (default)  D: (D(real)-1)^2 + D(fake)^2      G: (D(fake)-1)^2
      Least-squares. Penalises how FAR a sample sits from the decision boundary
      rather than which side of it it is on, so the gradient stays informative
      even once D is winning comfortably. This is the pix2pix default and the
      reason to prefer it here is entirely about gradient supply, not about
      producing a better-calibrated critic.

  vanilla            binary cross-entropy with logits
      The original Goodfellow objective. Included for completeness and for
      teaching. Its failure mode is instructive: once D is confident, the
      sigmoid saturates, its gradient goes to zero, and G stops learning while
      the loss numbers still look busy. Not recommended.

  hinge              D: relu(1-D(real)) + relu(1+D(fake))    G: -D(fake)
      The modern large-scale default (SAGAN, BigGAN). D is only penalised until
      each sample clears a margin of 1, after which that sample contributes
      nothing — a built-in brake on over-confidence that plays well with
      spectral normalisation. Roughly as stable as lsgan, sometimes crisper.

ON LABEL SMOOTHING. Training D toward 0.9 instead of 1.0 for reals removes its
incentive to drive confidence toward infinity, which is the mechanism behind the
saturation described above. It applies to lsgan and vanilla, which regress
toward an explicit target. It does NOT apply to hinge, which has no target to
smooth — its margin already serves the same purpose — and it is silently ignored
there rather than pretending to do something.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

GAN_MODES = ("lsgan", "vanilla", "hinge")


class GANLoss(nn.Module):
    """
    Parameters
    ----------
    gan_mode     : one of GAN_MODES
    real_target  : target for real samples. 0.9 with label smoothing on, 1.0 off.
    fake_target  : target for fake samples, normally 0.0. Not smoothed — one-sided
                   smoothing is the established form; smoothing the fake target
                   too has been shown to encourage the generator to match a
                   blurred version of the data distribution.
    """

    def __init__(self, gan_mode="lsgan", real_target=1.0, fake_target=0.0):
        super().__init__()
        if gan_mode not in GAN_MODES:
            raise ValueError(f"gan_mode must be one of {GAN_MODES}, got {gan_mode!r}")

        self.gan_mode = gan_mode
        self.real_target = float(real_target)
        self.fake_target = float(fake_target)

        if gan_mode == "hinge" and real_target != 1.0:
            logger.warning(
                "label smoothing (real_target=%.2f) has no effect with gan_mode="
                "'hinge': the hinge objective has no regression target to smooth. "
                "Its margin already limits D's confidence.", real_target)

    def _target_like(self, pred, is_real):
        value = self.real_target if is_real else self.fake_target
        return torch.full_like(pred, value)

    def d_loss(self, pred_real, pred_fake):
        """Discriminator loss. Returns (loss, {stats})."""
        if self.gan_mode == "lsgan":
            loss_real = F.mse_loss(pred_real, self._target_like(pred_real, True))
            loss_fake = F.mse_loss(pred_fake, self._target_like(pred_fake, False))
        elif self.gan_mode == "vanilla":
            loss_real = F.binary_cross_entropy_with_logits(
                pred_real, self._target_like(pred_real, True))
            loss_fake = F.binary_cross_entropy_with_logits(
                pred_fake, self._target_like(pred_fake, False))
        else:  # hinge
            loss_real = F.relu(1.0 - pred_real).mean()
            loss_fake = F.relu(1.0 + pred_fake).mean()

        # The 0.5 is the pix2pix convention: it halves D's effective learning
        # rate relative to G, a mild handicap on the player that usually wins.
        loss = (loss_real + loss_fake) * 0.5

        stats = {
            "D_real": loss_real.detach(),
            "D_fake": loss_fake.detach(),
            # Accuracy against the 0-crossing (lsgan/hinge) or 0.5 probability
            # (vanilla, whose logit 0 is the same crossing). These two numbers
            # are the primary health signal: see docs/gan_evaluation_guide.md.
            "D_acc_real": (pred_real.detach() > 0).float().mean(),
            "D_acc_fake": (pred_fake.detach() <= 0).float().mean(),
            "D_score_real": pred_real.detach().mean(),
            "D_score_fake": pred_fake.detach().mean(),
        }
        return loss, stats

    def g_loss(self, pred_fake):
        """Generator's adversarial loss — G wants D to call its output real."""
        if self.gan_mode == "lsgan":
            return F.mse_loss(pred_fake, self._target_like(pred_fake, True))
        if self.gan_mode == "vanilla":
            # Non-saturating form: maximise log D(fake) rather than minimise
            # log(1 - D(fake)). The latter has vanishing gradient exactly when G
            # is doing badly, which is when it most needs one.
            return F.binary_cross_entropy_with_logits(
                pred_fake, torch.ones_like(pred_fake))
        return -pred_fake.mean()   # hinge


def r1_penalty(pred_real, real_input):
    """
    R1 gradient penalty: the squared gradient norm of D at real samples.

    The intuition is geometric. A discriminator that has memorised its training
    set carries very sharp decision boundaries around each individual real
    image — tall spikes in a mostly flat landscape — and a spike has a large
    gradient. Penalising the gradient at real points flattens those spikes,
    which forces D to separate real from fake with general rules rather than
    with per-image lookups. On a 1687-slice training set that is a real risk,
    which is why this is offered at all.

    Costs a second backward pass through D, so it is normally applied lazily
    (every N steps, scaled by N) rather than every step.
    """
    grad = torch.autograd.grad(
        outputs=pred_real.sum(), inputs=real_input,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    return grad.pow(2).flatten(1).sum(1).mean()


def build_gan_loss(cfg):
    """Construct the GANLoss described by cfg.loss and cfg.stabilizers."""
    smoothing = cfg.get_path("stabilizers.label_smoothing.enabled", False)
    real_target = (cfg.get_path("stabilizers.label_smoothing.real_target", 0.9)
                   if smoothing else 1.0)
    return GANLoss(gan_mode=cfg.loss.gan_mode, real_target=real_target)
