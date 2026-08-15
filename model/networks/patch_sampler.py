"""
patch_sampler.py
────────────────
PatchSampleF — samples spatial locations from encoder features and projects
them through small MLP heads, for the PatchNCE loss.

WHAT IT DOES. Given a list of feature maps (one per tap depth), it picks the
same set of random spatial locations in each, gathers the feature vector at each
location, runs it through a two-layer MLP, and L2-normalises the result. The
output is a list of [B*num_patches, nce_dim] tensors ready for InfoNCE.

THE CRITICAL DETAIL: SHARED LOCATIONS. PatchNCE's positive pair is "the same
place in the image, before and after translation". That only holds if the MRI
encoding and the generated-CT encoding are sampled at *identical* locations, so
the first call returns the location ids it drew and the second call is given
them back. Sampling independently would make every pair a negative and the loss
would train the model toward nothing in particular — while still going down,
which is what makes this specific bug so hard to notice.

WHY AN MLP AT ALL. Raw encoder features live in a space shaped by the
reconstruction task, where cosine similarity is not especially meaningful. The
projection head learns a space where it is, which is the same argument SimCLR
makes for its projection head. The heads are trained, so they need their own
optimizer — the third one in this project.

LAZY CONSTRUCTION. The heads cannot be built at __init__ because their input
widths are the encoder's channel counts at the tapped depths, which are only
known once a real tensor has flowed through. So they are created on the first
forward pass. Everything downstream has to tolerate that: the optimizer is
created after the first forward, and checkpoint save/load must cope with the
heads being absent at step 0.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .init import init_weights

logger = logging.getLogger(__name__)


class PatchSampleF(nn.Module):
    """
    Parameters
    ----------
    use_mlp  : project sampled features through a learned head. With this off
               the module is a pure sampler and needs no optimizer.
    nce_dim  : width of the projection head's output.
    """

    def __init__(self, use_mlp=True, nce_dim=256):
        super().__init__()
        self.use_mlp = bool(use_mlp)
        self.nce_dim = int(nce_dim)
        self.mlp_init = False
        self._device = None

    def create_mlp(self, feats):
        """Build one 2-layer head per tapped feature map. Idempotent."""
        if self.mlp_init:
            return
        for i, feat in enumerate(feats):
            channels = feat.shape[1]
            mlp = nn.Sequential(
                nn.Linear(channels, self.nce_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.nce_dim, self.nce_dim),
            )
            if self._device is not None:
                mlp = mlp.to(self._device)
            init_weights(mlp)
            setattr(self, f"mlp_{i}", mlp)
        self.mlp_init = True
        logger.info("PatchSampleF: built %d MLP heads, channels %s -> %d",
                    len(feats), [f.shape[1] for f in feats], self.nce_dim)

    def to(self, *args, **kwargs):
        # Remember the device so heads built later land in the right place.
        out = super().to(*args, **kwargs)
        for arg in args:
            if isinstance(arg, (str, torch.device)):
                out._device = torch.device(arg)
        if "device" in kwargs and kwargs["device"] is not None:
            out._device = torch.device(kwargs["device"])
        return out

    def forward(self, feats, num_patches=256, patch_ids=None):
        """
        Parameters
        ----------
        feats       : list of [B, C, H, W] feature maps, one per tap
        num_patches : locations to sample per map. Clamped to H*W — the shallow
                      taps have plenty, but a deep tap can have fewer.
        patch_ids   : location ids returned by a previous call. Pass these on
                      the second call so both encodings sample the same places.

        Returns
        -------
        (sampled, patch_ids) — sampled[i] is [B*n_i, nce_dim], L2-normalised.
        """
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)

        sampled, out_ids = [], []
        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            # [B, C, H, W] -> [B, H*W, C]; each row is one location's vector.
            flat = feat.permute(0, 2, 3, 1).flatten(1, 2)

            if patch_ids is not None:
                ids = patch_ids[i].to(flat.device)
            else:
                n = min(int(num_patches), h * w)
                if n < num_patches:
                    logger.debug("tap %d has only %d locations; sampling %d "
                                 "instead of %d", i, h * w, n, num_patches)
                ids = torch.randperm(h * w, device=feat.device)[:n]

            # One shared permutation across the batch, as in the reference CUT
            # implementation: the negatives for an anchor are then other
            # locations from the same image, which is what makes the loss a
            # statement about spatial correspondence rather than about identity.
            patch = flat[:, ids, :].flatten(0, 1)          # [B*n, C]

            if self.use_mlp:
                patch = getattr(self, f"mlp_{i}")(patch)

            # L2-normalise so the dot product in the InfoNCE numerator is a
            # cosine similarity and the temperature has a consistent meaning.
            sampled.append(F.normalize(patch, p=2, dim=1))
            out_ids.append(ids)

        return sampled, out_ids

    # ── Checkpointing ────────────────────────────────────────────────────────
    # The heads may not exist yet when a checkpoint is written (nothing has been
    # forwarded) or when one is loaded (the run has not started). Both cases are
    # normal and neither should raise.

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state["_mlp_init"] = torch.tensor(bool(self.mlp_init))
        return state

    def load_state_dict(self, state_dict, strict=True):
        state = dict(state_dict)
        was_init = bool(state.pop("_mlp_init", torch.tensor(False)))

        if was_init and not self.mlp_init:
            # Rebuild the heads from the shapes recorded in the checkpoint, so a
            # resumed run can load them before any forward pass has happened.
            widths = {}
            for key, value in state.items():
                if key.endswith(".0.weight") and key.startswith("mlp_"):
                    widths[int(key.split(".")[0].split("_")[1])] = value.shape[1]
            if widths:
                dummy = [torch.zeros(1, widths[i], 1, 1)
                         for i in sorted(widths)]
                self.create_mlp(dummy)

        return super().load_state_dict(state, strict=strict)
