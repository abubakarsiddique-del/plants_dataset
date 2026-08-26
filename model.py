#!/usr/bin/env python3
"""Model definitions (Part A): backbone fallback chain + heads + SimSiam.

``build_backbone`` prefers **EfficientNetV2-S** and degrades gracefully so the
pipeline runs offline with ImageNet-pretrained weights:

    timm ``tf_efficientnetv2_s``  →  torchvision ``efficientnet_v2_s``
    →  torchvision ``efficientnet_b0``  (ImageNet weights cached in this env)

Pretrained weights are only requested when their checkpoint is already in the
local torch-hub cache, so no (blocked) download is attempted. ``PlantModel``
adds a linear classifier head and a SupCon projection head; ``SimSiam`` wraps
the same backbone for optional self-supervised pretraining.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm

    _HAS_TIMM = True
except Exception:  # pragma: no cover
    _HAS_TIMM = False

_HUB_CACHE = os.path.expanduser("~/.cache/torch/hub/checkpoints")


def _weights_if_cached(weights_enum):
    """Return the weights enum only if its checkpoint is already downloaded."""
    try:
        filename = weights_enum.url.split("/")[-1]
    except Exception:
        return None
    return weights_enum if os.path.exists(os.path.join(_HUB_CACHE, filename)) else None


def _tv_efficientnet(kind: str, pretrained: bool):
    """Build a torchvision EfficientNet, using cached weights if available."""
    import torchvision.models as tvm

    table = {
        "efficientnet_v2_s": (tvm.efficientnet_v2_s, tvm.EfficientNet_V2_S_Weights.DEFAULT),
        "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.DEFAULT),
    }
    ctor, default_weights = table[kind]
    weights = _weights_if_cached(default_weights) if pretrained else None
    model = ctor(weights=weights)
    features = model.features
    feat_dim = model.classifier[-1].in_features
    return features, feat_dim, weights is not None


def build_backbone(cfg_model: Dict[str, Any]) -> Tuple[nn.Module, int, str, bool]:
    """Return (features_module, feature_dim, backbone_name, pretrained_used)."""
    requested = str(cfg_model.get("backbone", "efficientnetv2_s"))
    pretrained = bool(cfg_model.get("pretrained", True))
    allow_fallback = bool(cfg_model.get("allow_backbone_fallback", True))

    # 1) timm EfficientNetV2-S (best; absent in this env)
    if _HAS_TIMM and "efficientnetv2" in requested.replace("-", "").lower():
        try:
            m = timm.create_model("tf_efficientnetv2_s", pretrained=pretrained, num_classes=0, global_pool="")
            feat_dim = m.num_features
            return m, feat_dim, "timm:tf_efficientnetv2_s", pretrained
        except Exception:
            if not allow_fallback:
                raise

    # 2) torchvision efficientnet_v2_s (weights not cached here → skipped if pretrained wanted)
    if "efficientnetv2" in requested.replace("-", "").lower() or requested == "efficientnet_v2_s":
        try:
            features, feat_dim, used = _tv_efficientnet("efficientnet_v2_s", pretrained)
            if used or not pretrained or not allow_fallback:
                return features, feat_dim, "torchvision:efficientnet_v2_s", used
        except Exception:
            if not allow_fallback:
                raise

    # 3) torchvision efficientnet_b0 (ImageNet weights cached → offline pretrained)
    features, feat_dim, used = _tv_efficientnet("efficientnet_b0", pretrained)
    return features, feat_dim, "torchvision:efficientnet_b0", used


# ---------------------------------------------------------------------------
# Supervised model: backbone + classifier + SupCon projection head
# ---------------------------------------------------------------------------
class PlantModel(nn.Module):
    def __init__(self, num_classes: int, cfg_model: Dict[str, Any]):
        super().__init__()
        self.features, self.feat_dim, self.backbone_name, self.pretrained_used = build_backbone(cfg_model)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(float(cfg_model.get("dropout", 0.3)))
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        proj_dim = int(cfg_model.get("proj_dim", 128))
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, proj_dim),
        )
        self.num_classes = num_classes

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x: torch.Tensor, return_projection: bool = True):
        feats = self.forward_features(x)
        logits = self.classifier(self.dropout(feats))
        if return_projection:
            proj = F.normalize(self.projector(feats), dim=1)
            return logits, proj
        return logits, None

    def last_conv_module(self) -> nn.Module:
        """Deepest conv-bearing module, for Grad-CAM hooks."""
        last = None
        for module in self.features.modules():
            if isinstance(module, nn.Conv2d):
                last = module
        return last if last is not None else self.features[-1]


# ---------------------------------------------------------------------------
# SimSiam (optional self-supervised pretraining)
# ---------------------------------------------------------------------------
class SimSiam(nn.Module):
    """SimSiam over the shared backbone (Chen & He, 2021)."""

    def __init__(self, features: nn.Module, feat_dim: int, proj_dim: int = 2048, pred_dim: int = 512):
        super().__init__()
        self.features = features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, proj_dim, bias=False), nn.BatchNorm1d(proj_dim), nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim, bias=False), nn.BatchNorm1d(proj_dim), nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim, bias=False), nn.BatchNorm1d(proj_dim, affine=False),
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim, bias=False), nn.BatchNorm1d(pred_dim), nn.ReLU(inplace=True),
            nn.Linear(pred_dim, proj_dim),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.pool(self.features(x)).flatten(1))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        z1, z2 = self._encode(x1), self._encode(x2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()


def simsiam_loss(p1, p2, z1, z2) -> torch.Tensor:
    def d(p, z):
        return -F.cosine_similarity(p, z, dim=1).mean()

    return 0.5 * (d(p1, z2) + d(p2, z1))


def build_model(num_classes: int, cfg_model: Dict[str, Any]) -> PlantModel:
    return PlantModel(num_classes, cfg_model)
