#!/usr/bin/env python3
"""Loss functions for supervised fine-tuning (Part A).

Combined objective = classification loss (cross-entropy or focal, with
class-balanced effective-number weighting for minority classes) + a supervised
contrastive term on the projection head. See Cui et al. 2019 (effective number
of samples) and Khosla et al. 2020 (SupCon).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def effective_number_weights(class_counts: Sequence[int], beta: float = 0.999) -> torch.Tensor:
    """Class-balanced weights w_c = (1 - beta) / (1 - beta**n_c), mean-normalized to 1."""
    counts = np.clip(np.asarray(class_counts, dtype=np.float64), 1.0, None)
    effective = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective
    weights = weights / weights.sum() * len(counts)  # mean weight == 1
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class alpha weights."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        loss = -((1.0 - pt) ** self.gamma) * logpt
        if self.weight is not None:
            loss = loss * self.weight.to(logits.device)[targets]
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class SupConLoss(nn.Module):
    """Supervised contrastive loss over L2-normalized projection features.

    Single-view formulation: positives for an anchor are the other in-batch
    samples sharing its label. Anchors with no in-batch positive are skipped;
    if the whole batch has none, returns a graph-connected zero.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        features = F.normalize(features, dim=1)
        batch = features.size(0)
        sim = (features @ features.T) / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()  # numerical stability

        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float().to(device)
        self_mask = torch.eye(batch, device=device)
        pos_mask = pos_mask * (1.0 - self_mask)          # drop self-pairs
        logits_mask = 1.0 - self_mask

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        pos_counts = pos_mask.sum(dim=1)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_counts.clamp(min=1.0)
        valid = pos_counts > 0
        if valid.any():
            return -mean_log_prob_pos[valid].mean()
        return features.sum() * 0.0


class CombinedLoss(nn.Module):
    """ce_weight * classification + supcon_weight * supervised-contrastive."""

    def __init__(
        self,
        class_counts: Sequence[int],
        ce_weight: float = 1.0,
        supcon_weight: float = 0.5,
        class_balancing: str = "effective_number",
        cb_beta: float = 0.999,
        focal_gamma: float = 2.0,
        supcon_temperature: float = 0.07,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.supcon_weight = supcon_weight
        self.class_balancing = class_balancing

        en_weights = effective_number_weights(class_counts, beta=cb_beta)
        if class_balancing == "effective_number":
            self.classification = nn.CrossEntropyLoss(weight=en_weights)
        elif class_balancing == "focal":
            self.classification = FocalLoss(gamma=focal_gamma, weight=en_weights)
        else:  # "none"
            self.classification = nn.CrossEntropyLoss()

        self.supcon = SupConLoss(temperature=supcon_temperature)

    def forward(
        self,
        logits: torch.Tensor,
        projections: Optional[torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cls_loss = self.classification(logits, targets)
        if self.supcon_weight > 0 and projections is not None:
            sc_loss = self.supcon(projections, targets)
        else:
            sc_loss = logits.sum() * 0.0
        total = self.ce_weight * cls_loss + self.supcon_weight * sc_loss
        return total, {
            "classification": float(cls_loss.detach()),
            "supcon": float(sc_loss.detach()),
            "total": float(total.detach()),
        }
