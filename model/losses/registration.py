"""
registration.py
───────────────
The smoothness penalty on RegGAN's deformation field.

WHY THIS TERM IS NOT OPTIONAL. RegGAN computes its reconstruction loss in a
registered frame: a network R predicts a dense field phi, the generated CT is
warped by it, and L1 is taken against the target there. Left unconstrained, the
optimal phi is not "the residual misalignment" — it is whatever field maps the
generator's output onto the target pixel for pixel, which for a sufficiently
flexible field is achievable from almost any input. The correction loss would go
to zero, the generator would receive no pressure to be correct, and nothing in
the logs would look wrong: L_corr falls, which is what a loss curve is supposed
to do.

Penalising the field's spatial gradient is what makes it a *registration* rather
than a free reparameterisation. A smooth field can translate, rotate and mildly
deform; it cannot shuffle individual pixels. That is exactly the class of error
manual QC left behind on these pairs — see the note at configs/base.yaml:55-59.
"""

import torch


def flow_smoothness(flow, mask=None):
    """
    Mean squared first-order spatial gradient of a deformation field.

    Parameters
    ----------
    flow : [N, 2, H, W] displacement in PIXELS (dx, dy), the convention
           networks/registration.py emits and SpatialTransformer consumes.
    mask : optional [N, 1, H, W] validity mask. Restricting the penalty to valid
           pixels matters because the slices are zero-padded to a common size:
           the field over padding is unconstrained by any data term, so its
           gradient is noise, and averaging that noise into the penalty makes a
           heavily-padded small slice look rougher than a large one for reasons
           that have nothing to do with the registration.

    Returns a scalar. Both axes contribute equally; 1 px = 1 mm here, so the two
    directions are already in the same units and need no reweighting.
    """
    d_x = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    d_y = flow[:, :, 1:, :] - flow[:, :, :-1, :]

    if mask is None:
        return d_x.pow(2).mean() + d_y.pow(2).mean()

    # A difference is valid only where BOTH of the pixels it was taken between
    # are valid, so the mask is ANDed with itself shifted by one.
    m_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    m_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]

    def _masked_mean(diff, m):
        # m is single-channel and diff has the field's 2; broadcasting applies
        # the same validity to dx and dy, and the denominator counts both.
        return (diff.pow(2) * m).sum() / (m.sum() * diff.shape[1]).clamp(min=1.0)

    return _masked_mean(d_x, m_x) + _masked_mean(d_y, m_y)


def flow_magnitude(flow, mask=None):
    """
    Mean and max displacement in pixels — a diagnostic, not a loss.

    This is the number exp7 exists to produce. configs/base.yaml:55-59 records
    that manual QC corrected translation on these pairs but could not correct
    in-plane rotation, and that the residual is *unmeasured*; exp3_nce_heavy.yaml
    then reweights its objective as a hedge against a quantity nobody has looked
    at. The mean returned here is that quantity, in millimetres (1 px = 1 mm),
    logged once per epoch.
    """
    with torch.no_grad():
        norm = flow.pow(2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
        if mask is None:
            return norm.mean(), norm.max()
        denom = mask.sum().clamp(min=1.0)
        return (norm * mask).sum() / denom, (norm * mask).max()
