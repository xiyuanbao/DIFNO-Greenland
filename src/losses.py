"""Loss functions. The forward surrogates and the default inverse use a plain
``torch.nn.L1Loss``; the coastal inverse uses ``AmplitudeWeightedLoss``, which
up-weights points where the target ice change is large.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AmplitudeWeightedLoss(nn.Module):
    def __init__(
        self,
        base_loss="L2",      # "L2" or "L1"
        min_weight=1,
        normalize_weight=True,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.min_weight = min_weight
        self.normalize_per_sample = normalize_weight

    def forward(self, pred, target, epoch=0):
        """
        pred, target: (B, ...)
        """
        B = target.shape[0]

        # -------- pointwise weights --------

        weight = target.abs()
        weight = weight/weight.max()*2
        weight = torch.clamp(weight, min=self.min_weight)
        weight[weight>=self.min_weight] = weight[weight>=self.min_weight]*100

        # -------- pointwise loss --------
        if self.base_loss == "L2":
            loss_point = (pred - target).abs() ** 2
        elif self.base_loss == "L1":
            loss_point = (pred - target).abs()
        else:
            raise ValueError("base_loss must be 'L1' or 'L2'")

        # -------- per-sample normalization --------
        weight = weight.view(B, -1)
        loss_point = loss_point.view(B, -1)

        weighted_loss = weight * loss_point

        if self.normalize_per_sample:
            # keeps loss scale invariant to resolution
            loss_per_sample = weighted_loss.sum(dim=1) / (weight.sum(dim=1) + 1e-12)
        else:
            loss_per_sample = weighted_loss.mean(dim=1)

        return loss_per_sample.mean()
