#!/usr/bin/env python3
"""Stage 3 — preprocessing & augmentation (deterministic config, stochastic ops).

Builds the **12-technique Albumentations training pool** (train split only), a
deterministic eval transform, and a **domain-randomization** corruption
transform used to synthesize the external-domain shift. Also provides
``PlantDataset`` (consumed by train/evaluate/explain) and an
``augmentation_summary`` for the state.

Techniques (12): H-flip, V-flip, rotate, shift-scale-rotate (Affine),
brightness/contrast, hue/sat/value, RGB-shift, Gaussian noise, blur,
CLAHE|Sharpen, coarse-dropout, elastic|grid. Albumentations 2.x APIs
(``GaussNoise(std_range=...)``, ``CoarseDropout(num_holes_range=..., fill=...)``,
``ImageCompression(quality_range=...)``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from augmentation_pipeline import letterbox_standardize, load_rgb_image
from common import get_logger, profile_value, save_json

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_REFLECT = cv2.BORDER_REFLECT_101


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------
def _train_ops(techniques: List[str]) -> List[A.BasicTransform]:
    """Map technique names → stochastic ops. 'flip' expands to H+V (→ 12 total)."""
    catalog: Dict[str, List[A.BasicTransform]] = {
        "flip": [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)],
        "rotate": [A.Rotate(limit=30, border_mode=_REFLECT, p=0.5)],
        "shift_scale_rotate": [
            A.Affine(translate_percent=(-0.0625, 0.0625), scale=(0.9, 1.1),
                     rotate=(-15, 15), border_mode=_REFLECT, p=0.5)
        ],
        "brightness_contrast": [A.RandomBrightnessContrast(0.2, 0.2, p=0.5)],
        "hue_saturation": [A.HueSaturationValue(15, 25, 10, p=0.5)],
        "rgb_shift": [A.RGBShift(15, 15, 15, p=0.4)],
        "gaussian_noise": [A.GaussNoise(std_range=(0.04, 0.12), p=0.3)],
        "blur": [A.GaussianBlur(blur_limit=(3, 7), p=0.3)],
        "clahe_sharpen": [A.OneOf([A.CLAHE(clip_limit=2.0, p=1.0), A.Sharpen(p=1.0)], p=0.3)],
        "coarse_dropout": [
            A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.15),
                            hole_width_range=(0.05, 0.15), fill=0, p=0.3)
        ],
        "elastic_grid": [A.OneOf([A.ElasticTransform(p=1.0), A.GridDistortion(p=1.0)], p=0.3)],
    }
    ops: List[A.BasicTransform] = []
    for name in techniques:
        ops.extend(catalog.get(name, []))
    return ops


def _dr_ops(names: List[str], prob: float) -> List[A.BasicTransform]:
    """Domain-randomization group folded into training (each applies w.p. `prob`)."""
    catalog = {
        "sensor_noise": A.OneOf([A.GaussNoise(std_range=(0.05, 0.15), p=1.0), A.ISONoise(p=1.0)], p=prob),
        "white_balance": A.OneOf(
            [A.RGBShift(20, 20, 20, p=1.0), A.RandomGamma(gamma_limit=(80, 120), p=1.0)], p=prob
        ),
        "jpeg": A.ImageCompression(quality_range=(35, 70), p=prob),
    }
    return [catalog[n] for n in names if n in catalog]


def count_train_techniques(techniques: List[str]) -> int:
    return len(_train_ops(techniques))


def build_train_transform(
    image_size: int,
    techniques: List[str],
    domain_randomization: Optional[List[str]] = None,
    dr_prob: float = 0.5,
) -> A.Compose:
    ops = [A.Resize(image_size, image_size)]
    ops += _train_ops(techniques)
    if domain_randomization:
        ops += _dr_ops(domain_randomization, dr_prob)
    ops += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(ops)


def build_eval_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [A.Resize(image_size, image_size), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    )


def build_domain_randomization_transform(corruptions: List[str]) -> A.Compose:
    """uint8→uint8 device-shift applied to synthetic external images before the eval transform."""
    catalog = {
        "white_balance": A.RGBShift(25, 25, 25, p=1.0),
        "jpeg_recompress": A.ImageCompression(quality_range=(30, 50), p=1.0),
        "jpeg": A.ImageCompression(quality_range=(30, 50), p=1.0),
        "sensor_noise": A.GaussNoise(std_range=(0.08, 0.15), p=1.0),
    }
    ops = [catalog[c] for c in corruptions if c in catalog]
    return A.Compose(ops if ops else [A.NoOp()])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PlantDataset(Dataset):
    """Reads split_manifest entries → (CHW float tensor, label_index).

    Letterboxes to a square ``working_size`` first (aspect-preserving), then
    applies ``transform``. For entries flagged ``synthetic_corruption`` and when
    a ``corruption`` transform is supplied, applies the device-shift first.
    """

    def __init__(
        self,
        entries: List[Dict[str, Any]],
        transform: A.Compose,
        working_size: int = 640,
        corruption: Optional[A.Compose] = None,
    ):
        self.entries = entries
        self.transform = transform
        self.working_size = working_size
        self.corruption = corruption

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        rec = self.entries[index]
        img = load_rgb_image(Path(rec["path"]))
        img = letterbox_standardize(img, (self.working_size, self.working_size))
        if self.corruption is not None and rec.get("synthetic_corruption"):
            img = self.corruption(image=img)["image"]
        tensor = self.transform(image=img)["image"]
        return tensor, int(rec["label_index"])

    def labels(self) -> List[int]:
        return [int(e["label_index"]) for e in self.entries]


# ---------------------------------------------------------------------------
# Stage entry
# ---------------------------------------------------------------------------
def run_augment_stage(
    config: Dict[str, Any],
    run_paths,
    profile: str,
    split_manifest: Dict[str, Any],
    logger=None,
) -> Dict[str, Any]:
    logger = logger or get_logger("stage3.augment", run_paths.stage("stage3_augment") / "augment.log")
    aug_cfg = config["augment"]
    image_size = int(split_manifest.get("image_size", config["data"]["image_size"]))
    techniques = list(aug_cfg["techniques"])
    dr_group = list(aug_cfg.get("domain_randomization", []))
    dr_prob = float(aug_cfg.get("domain_randomization_prob", 0.5))
    corruptions = list(split_manifest.get("corruptions", []))

    n_techniques = count_train_techniques(techniques)
    summary = {
        "image_size": image_size,
        "working_size": int(split_manifest.get("working_size", config["data"]["working_size"])),
        "train_techniques": techniques,
        "num_train_ops": n_techniques,
        "reconciliation_note": (
            f"'12-technique pool' = {n_techniques} distinct train ops "
            f"({len(techniques)} named entries; 'flip' expands to horizontal+vertical)."
        ),
        "domain_randomization": {"group": dr_group, "prob": dr_prob},
        "external_corruptions": corruptions,
        "external_mode": split_manifest.get("external_mode"),
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "applies_to": "train split only (eval/test/external use deterministic resize+normalize)",
        "split_counts": split_manifest.get("counts", {}),
    }
    save_json(summary, run_paths.stage("stage3_augment") / "augmentation_summary.json")
    logger.info("Stage 3 done. %d train ops; DR group=%s (p=%.2f); external corruptions=%s",
                n_techniques, dr_group, dr_prob, corruptions or "n/a")
    return {"augmentation_summary": summary}
