"""
ema.py
──────
Exponential moving average of the generator's weights.

THE PROBLEM IT SOLVES. GAN training does not converge to a point; it orbits one.
G improves, D adapts, G's advantage evaporates, repeat. The practical
consequence is that epoch 47 can genuinely be better than epoch 48 and worse
than epoch 49, with no trend — so "take the last checkpoint" is arbitrary and
"take the best validation epoch" is largely sampling noise.

An EMA copy of the weights,

    ema = decay * ema + (1 - decay) * live

averages over roughly 1/(1-decay) recent steps — 1000 at the default 0.999 — so
it sits near the CENTRE of the orbit rather than at whatever random point on it
the current step landed. Two things follow, and both are worth having:

  * the EMA generator's samples are usually visibly better than the live one's,
    for free, with no change to the training dynamics whatsoever; and
  * the validation curve computed from it is smooth enough to read, which is
    what makes model selection a decision rather than a coin flip.

It is never trained. It is a read-only shadow updated after each generator step.

COST: one extra copy of G's parameters (~54M floats, ~210 MB in fp32) and one
lerp per step. Both are negligible next to the training itself.
"""

import copy
import logging

import torch

logger = logging.getLogger(__name__)


class ModelEMA:
    """
    Parameters
    ----------
    model        : the live generator
    decay        : 0.999 averages over ~1000 steps. At ~210 iterations/epoch that
                   is about five epochs of memory, which is the right order for
                   smoothing GAN oscillation without lagging real progress.
    start_epoch  : epochs before which the EMA simply tracks the live weights.
                   Averaging in the random initialisation would leave the shadow
                   contaminated by noise for its whole warm-up window.
    """

    def __init__(self, model, decay=0.999, start_epoch=1):
        self.decay = float(decay)
        self.start_epoch = int(start_epoch)
        self.ema = copy.deepcopy(model).eval()
        for param in self.ema.parameters():
            param.requires_grad_(False)
        self.n_updates = 0
        logger.info("EMA: decay=%.4f (~%.0f-step window), start_epoch=%d",
                    self.decay, 1.0 / max(1e-8, 1.0 - self.decay), self.start_epoch)

    @torch.no_grad()
    def update(self, model, epoch):
        """Fold the live weights into the shadow. Call once per generator step."""
        if epoch < self.start_epoch:
            # Track exactly, rather than average: before start_epoch the live
            # weights are still leaving their random initialisation and there is
            # nothing worth remembering about where they have been.
            self.copy_from(model)
            return

        decay = self.decay
        ema_params = dict(self.ema.named_parameters())
        for name, param in model.named_parameters():
            ema_params[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)

        # Buffers (none with InstanceNorm(track_running_stats=False), but
        # BatchNorm's running stats would land here) are copied, not averaged:
        # they are already running estimates and averaging them twice biases
        # them toward staleness.
        ema_buffers = dict(self.ema.named_buffers())
        for name, buf in model.named_buffers():
            ema_buffers[name].copy_(buf)

        self.n_updates += 1

    @torch.no_grad()
    def copy_from(self, model):
        """Hard-set the shadow to the live weights."""
        self.ema.load_state_dict(model.state_dict())

    def state_dict(self):
        return {"ema": self.ema.state_dict(),
                "decay": self.decay,
                "n_updates": self.n_updates}

    def load_state_dict(self, state):
        self.ema.load_state_dict(state["ema"])
        self.decay = state.get("decay", self.decay)
        self.n_updates = state.get("n_updates", 0)

    @property
    def module(self):
        """The shadow generator, for evaluation and sample rendering."""
        return self.ema
