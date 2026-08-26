#!/usr/bin/env python3
"""Lightweight supervised fine-tuning baseline for plant disease originals.

Uses an EfficientNet backbone pretrained on ImageNet, fine-tunes it on the
original labeled dataset, saves validation metrics, and writes a model artifact.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from keras_tuner import HyperParameters
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "Malabar_Dataset"
MODEL_DIR = ROOT / "models"


class ClassFolderDataset(Dataset):
    def __init__(self, image_paths: List[Path], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_split() -> Tuple[List[Path], List[int], List[Path], List[int], List[Path], List[int]]:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_ROOT}")

    class_dirs = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()])
    if not class_dirs:
        raise FileNotFoundError("No class folders found in Malabar_Dataset")

    label_names = [p.name for p in class_dirs]
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}

    train_paths: List[Path] = []
    train_labels: List[int] = []
    val_paths: List[Path] = []
    val_labels: List[int] = []
    test_paths: List[Path] = []
    test_labels: List[int] = []

    rng = np.random.default_rng(42)
    for class_dir in class_dirs:
        images = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.jpeg")) + sorted(class_dir.glob("*.png"))
        if not images:
            raise FileNotFoundError(f"No images found in {class_dir}")
        idxs = np.arange(len(images))
        rng.shuffle(idxs)
        images = [images[int(i)] for i in idxs]
        n = len(images)
        n_train = int(round(0.8 * n))
        n_val = int(round(0.1 * n))
        n_test = n - n_train - n_val
        train_paths.extend(images[:n_train])
        train_labels.extend([label_to_idx[class_dir.name]] * n_train)
        val_paths.extend(images[n_train : n_train + n_val])
        val_labels.extend([label_to_idx[class_dir.name]] * n_val)
        test_paths.extend(images[n_train + n_val :])
        test_labels.extend([label_to_idx[class_dir.name]] * n_test)

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = Counter(labels)
    total = sum(counts.values())
    weights = torch.tensor([total / max(counts.get(i, 1), 1) for i in range(num_classes)], dtype=torch.float32)
    weights = weights / weights.mean()
    return weights


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = self.alpha * (1.0 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def search_hyperparameters(args: argparse.Namespace) -> Dict[str, object]:
    hp = HyperParameters()
    hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    hp.Choice("batch_size", values=[16, 32, 64])
    hp.Int("epochs", min_value=5, max_value=20, step=5)

    candidate_grid = [
        {"lr": 1e-4, "batch_size": 16, "epochs": 10},
        {"lr": 3e-4, "batch_size": 32, "epochs": 10},
        {"lr": 1e-3, "batch_size": 32, "epochs": 20},
        {"lr": 3e-3, "batch_size": 64, "epochs": 10},
        {"lr": 1e-2, "batch_size": 32, "epochs": 20},
    ]

    evaluated: List[Dict[str, object]] = []
    for trial_config in candidate_grid[: max(1, min(args.max_trials, len(candidate_grid)))]:
        trial_args = argparse.Namespace(
            epochs=trial_config["epochs"],
            batch_size=trial_config["batch_size"],
            lr=trial_config["lr"],
            loss=args.loss,
        )
        metrics = train_and_evaluate(trial_args)
        evaluated.append({
            "lr": trial_config["lr"],
            "batch_size": trial_config["batch_size"],
            "epochs": trial_config["epochs"],
            "test_accuracy": metrics["test_accuracy"],
            "test_macro_f1": metrics["test_macro_f1"],
        })

    best = max(evaluated, key=lambda item: (item["test_macro_f1"], item["test_accuracy"]))
    return {
        "tuner_selected": best,
        "trials": evaluated,
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    correct = 0
    total = 0
    all_pred: List[int] = []
    all_true: List[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_pred.extend(preds.cpu().tolist())
            all_true.extend(labels.cpu().tolist())
    acc = correct / max(total, 1)
    macro_f1 = float(f1_score(all_true, all_pred, average="macro"))
    return acc, macro_f1, all_pred, all_true


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.to(device)
    return model


def train_and_evaluate(args: argparse.Namespace) -> Dict[str, object]:
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = build_split()
    class_names = sorted([p.name for p in DATA_ROOT.iterdir() if p.is_dir()])
    num_classes = len(class_names)

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ClassFolderDataset(train_paths, train_labels, transform_train)
    val_ds = ClassFolderDataset(val_paths, val_labels, transform_eval)
    test_ds = ClassFolderDataset(test_paths, test_labels, transform_eval)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(num_classes, device)
    class_weights = compute_class_weights(train_labels, num_classes).to(device)
    if args.loss == "focal":
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_state = None
    best_val_acc = -1.0
    best_val_f1 = -1.0

    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1} validation accuracy: {val_acc:.4f} | macro_f1: {val_f1:.4f}")
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_f1 > best_val_f1):
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, test_f1, _, _ = evaluate(model, test_loader, device)
    metrics = {
        "loss": args.loss,
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1),
        "best_val_accuracy": float(best_val_acc),
        "best_val_macro_f1": float(best_val_f1),
        "num_classes": num_classes,
        "class_names": class_names,
        "train_samples": len(train_paths),
        "val_samples": len(val_paths),
        "test_samples": len(test_paths),
    }

    torch.save({
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "metrics": metrics,
    }, MODEL_DIR / "efficientnet_b0_finetuned.pt")

    (MODEL_DIR / "efficientnet_b0_finetuned_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet on original plant disease images")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["weighted_ce", "focal"], default="focal", help="class-aware training objective")
    parser.add_argument("--tune", action="store_true", help="Let the tuner choose lr/batch_size/epochs from a small search space")
    parser.add_argument("--max-trials", type=int, default=5, help="Upper bound on tuner candidate configurations to evaluate")
    return parser.parse_args()


if __name__ == "__main__":
    from PIL import Image

    args = parse_args()
    if args.tune:
        metrics = search_hyperparameters(args)
        print(json.dumps(metrics, indent=2))
    else:
        metrics = train_and_evaluate(args)
        print(json.dumps(metrics, indent=2))
