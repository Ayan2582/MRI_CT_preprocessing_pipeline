"""
builder.py
──────────
Model dispatch: turn `model.name` into a training model.

This is the same extension point networks/builder.py provides one level down,
and it exists for the same reason. Until RegGAN arrived there was exactly one
model class, so trainer.py imported Pix2PixNCEModel directly and constructed it
by name. Adding a second would have meant the trainer importing every model
class and choosing between them, which puts architecture knowledge in the epoch
loop — the one place in this package that is supposed to be indifferent to what
it is driving.

Adding a model means one import and one branch here.

WHAT A MODEL MUST PROVIDE. The trainer talks to models through a small, fixed
surface, and anything added here has to implement all of it:

    optimize_parameters(batch, epoch, scaler, amp_ctx) -> {str: float}
    train_mode()
    gan_active(epoch) -> bool
    generator_for_eval() -> nn.Module        # takes real_A, returns fake_B
    set_eval_mode(net) -> nn.Module
    state_dict() / load_state_dict(state)
    .optimizer_G, .optimizer_D (may be None), .plan, .warmup_epochs

Pix2PixNCEModel defines that surface, RegGANModel inherits it, and
CycleGANModel implements it independently — which is the point of writing the
surface down: the epoch loop drives four networks and one network through the
same calls.
"""

import logging

logger = logging.getLogger(__name__)

MODEL_TYPES = ("pix2pix_nce", "reggan", "cyclegan")


def build_model(cfg, device):
    """
    Construct the model described by cfg.model.name.

    The default is 'pix2pix_nce', which is what every config written before this
    dispatch existed resolves to — so exp0 through exp6 and their checkpoints are
    unaffected by the indirection.
    """
    name = cfg.get_path("model.name", "pix2pix_nce")

    if name == "pix2pix_nce":
        from .pix2pix_nce import Pix2PixNCEModel
        return Pix2PixNCEModel(cfg, device)

    if name == "reggan":
        from .reggan import RegGANModel
        return RegGANModel(cfg, device)

    if name == "cyclegan":
        from .cyclegan import CycleGANModel
        return CycleGANModel(cfg, device)

    raise NotImplementedError(
        f"model '{name}' is not implemented. Available: {', '.join(MODEL_TYPES)}."
    )
