"""
patch_nce.py
────────────
PatchNCE — the contrastive term of the loss, and the one this dataset most needs.

WHAT IT MEASURES. Take a location in the input MRI and the same location in the
generated CT. Encode both through the generator's encoder. Those two feature
vectors should be more similar to each other than either is to a feature vector
from a *different* location in the same image. That is an InfoNCE classification
problem with one positive and N-1 negatives, and minimising it maximises a lower
bound on the mutual information between input and output at matched positions.

In plain terms: it forces the model to keep the input's structure where the
input put it, without ever asking the output to have particular pixel values.

WHY IT MIGHT MATTER HERE — AN OPEN QUESTION, NOT A MEASURED ONE.

A tempting argument runs: the raw DICOM frames carry a median 5.2 degree CT/MRI
mismatch (qc_app/registration_service.py:26-37), L1 charges full price for a
displaced target even when the model is right, and its optimal response to that
is to blur — so a misalignment-tolerant term should help.

That argument does NOT hold as stated on this dataset, and it is worth being
precise about why. The 5.2 degrees describes the RAW frames, before quality
control. Every accepted pair was then reviewed by hand: 129 pairs were rejected
outright, 738 slices were nudged INDIVIDUALLY (48 of 119 series carry per-slice
nudges spanning up to 84 mm), and artifacts were erased on 560. Translation
error, including the slice-to-slice variation that out-of-plane tilt produces,
was therefore corrected by a human rather than left in the data.

What survives is in-plane rotation, which no translation can fix at any
granularity — and its residual magnitude is unmeasured, because the QC process
recorded what was done to each slice, not an alignment score afterwards.

So the honest position: PatchNCE is worth testing here, but as an empirical
question. A contrastive term supplies structural signal that L1 and a patch
discriminator do not, and 1687 training slices is few enough that extra signal
may earn its place regardless of alignment. exp3_nce_heavy.yaml tests whether it
does. Do not describe it as a targeted fix for a measured defect.

THE NEGATIVES ARE INTERNAL. Both the positive and all negatives for an anchor
come from the same image. That makes the loss a statement about spatial
correspondence within a slice rather than about telling patients apart — which
is what you want, since every slice in this dataset is the same handful of
tissue types and cross-image negatives would mostly be trivially easy.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PatchNCELoss(nn.Module):
    """
    InfoNCE over patch features.

    Parameters
    ----------
    temperature : scales the logits before the softmax. 0.07 is the CUT/SimCLR
                  value. Lower sharpens the distribution and weights the hardest
                  negatives more heavily; higher flattens it. It is not a
                  free parameter to fiddle with casually — it interacts with the
                  effective loss scale, so changing it changes what lambda_nce
                  means.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = float(temperature)
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")
        # Set by the caller before each forward. It cannot be inferred from the
        # tensors alone: a [B*n, C] tensor is ambiguous between (B=2, n=256) and
        # (B=1, n=512), and guessing wrong silently mixes negatives across
        # images — which would still produce a loss that goes down.
        self.batch_size = 1

    def forward(self, feat_q, feat_k):
        """
        Parameters
        ----------
        feat_q : [B*n, C] query features, from the GENERATED CT
        feat_k : [B*n, C] key features, from the REAL MRI, at the same locations

        Both are already L2-normalised by PatchSampleF, so every dot product
        below is a cosine similarity.
        """
        n_total, dim = feat_q.shape
        # Keys are detached: the contrastive signal should shape the generated
        # output, not drag the encoding of the (fixed) input around to make the
        # problem easier. This is standard contrastive practice and skipping it
        # lets the model reduce the loss by degrading its own representation.
        feat_k = feat_k.detach()

        # Positive: each query with its own key. [B*n, 1]
        l_pos = torch.bmm(feat_q.view(n_total, 1, -1),
                          feat_k.view(n_total, -1, 1)).view(n_total, 1)

        # Negatives: every query against every key from the SAME image.
        # feat_* arrive as [B*n, C] laid out image-major, so reshaping to
        # [B, n, C] recovers the per-image grouping.
        batch = self.batch_size
        n_patches = n_total // batch
        q = feat_q.view(batch, n_patches, dim)
        k = feat_k.view(batch, n_patches, dim)
        l_neg = torch.bmm(q, k.transpose(2, 1))            # [B, n, n]

        # Remove the diagonal — that entry is the positive, and leaving it in
        # the negatives would ask the model to be dissimilar from itself.
        diagonal = torch.eye(n_patches, device=feat_q.device, dtype=torch.bool)
        l_neg.masked_fill_(diagonal[None, :, :], -10.0)
        l_neg = l_neg.view(-1, n_patches)                  # [B*n, n]

        # Positive sits at column 0, so the target class is always 0.
        logits = torch.cat((l_pos, l_neg), dim=1) / self.temperature
        loss = self.cross_entropy(
            logits, torch.zeros(n_total, dtype=torch.long, device=feat_q.device))
        return loss.mean()


class PatchNCECollection(nn.Module):
    """
    PatchNCE summed over all tapped layers.

    Each tap gets its own loss instance so that per-layer values can be logged
    separately — a layer whose loss refuses to move is a useful signal that the
    tap is too deep to have enough spatial locations to sample from.
    """

    def __init__(self, n_layers, temperature=0.07):
        super().__init__()
        self.losses = nn.ModuleList(
            [PatchNCELoss(temperature) for _ in range(n_layers)])

    def forward(self, feats_q, feats_k, batch_size):
        if len(feats_q) != len(feats_k):
            raise ValueError(
                f"PatchNCE got {len(feats_q)} query feature maps and "
                f"{len(feats_k)} key maps; they must be tapped at the same layers."
            )

        total = 0.0
        per_layer = []
        for crit, q, k in zip(self.losses, feats_q, feats_k):
            crit.batch_size = batch_size
            value = crit(q, k)
            per_layer.append(value.detach())
            total = total + value

        # Mean over layers, not sum: this keeps the magnitude of the term — and
        # therefore the meaning of lambda_nce — independent of how many taps are
        # configured, so changing loss.nce.layers does not silently rescale the
        # loss you spent five experiments tuning.
        return total / max(1, len(self.losses)), per_layer


def build_nce_loss(cfg):
    """Construct the PatchNCECollection described by cfg.loss.nce."""
    layers = cfg.get_path("loss.nce.layers", [0, 1, 2, 3, 4])
    temperature = cfg.get_path("loss.nce.temperature", 0.07)
    return PatchNCECollection(n_layers=len(layers), temperature=temperature)
