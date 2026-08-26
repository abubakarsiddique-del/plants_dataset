#!/usr/bin/env python3
"""Stage 4a — training (deterministic given seed).

Optional SimSiam SSL pretraining of the backbone, then supervised fine-tuning
with the combined CE/focal (class-balanced) + supervised-contrastive loss, a
class-balanced sampler, cosine LR with warmup, and best-checkpoint selection by
validation macro-F1. Profile-scoped epochs keep the ``fast`` run to CPU-minutes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from augment import PlantDataset, build_eval_transform, build_train_transform
from common import get_logger, profile_value, save_json
from losses import CombinedLoss
from model import SimSiam, build_model, simsiam_loss


# ---------------------------------------------------------------------------
# Loaders & helpers
# ---------------------------------------------------------------------------
def class_counts_from(entries: List[Dict[str, Any]], num_classes: int) -> List[int]:
    counts = [0] * num_classes
    for e in entries:
        counts[int(e["label_index"])] += 1
    return counts


def make_weighted_sampler(entries: List[Dict[str, Any]], num_classes: int) -> WeightedRandomSampler:
    counts = class_counts_from(entries, num_classes)
    per_class_w = [0.0 if c == 0 else 1.0 / c for c in counts]
    sample_w = [per_class_w[int(e["label_index"])] for e in entries]
    return WeightedRandomSampler(sample_w, num_samples=len(entries), replacement=True)


def make_loader(
    entries: List[Dict[str, Any]],
    transform,
    batch_size: int,
    working_size: int,
    num_workers: int = 0,
    sampler: Optional[WeightedRandomSampler] = None,
    shuffle: bool = False,
    corruption=None,
) -> DataLoader:
    ds = PlantDataset(entries, transform, working_size=working_size, corruption=corruption)
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        drop_last=False,
    )


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device, num_classes: int) -> Dict[str, Any]:
    model.eval()
    all_probs: List[np.ndarray] = []
    all_targets: List[int] = []
    for x, y in loader:
        x = x.to(device)
        logits, _ = model(x, return_projection=False)
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_targets.extend(int(t) for t in y)
    probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, num_classes))
    preds = probs.argmax(axis=1) if len(probs) else np.zeros((0,), dtype=int)
    targets = np.asarray(all_targets, dtype=int)
    acc = float(accuracy_score(targets, preds)) if len(targets) else 0.0
    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0)) if len(targets) else 0.0
    return {"accuracy": acc, "macro_f1": macro_f1, "preds": preds, "targets": targets, "probs": probs}


def _cosine_warmup(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# SimSiam pretraining (optional)
# ---------------------------------------------------------------------------
class _TwoViewDataset(Dataset):
    def __init__(self, entries, transform, working_size):
        self.inner = PlantDataset(entries, transform, working_size=working_size)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        x1, _ = self.inner[i]
        x2, _ = self.inner[i]
        return x1, x2


def simsiam_pretrain(model, entries, cfg, profile, device, working_size, image_size, logger) -> Dict[str, Any]:
    ss_cfg = cfg["model"]["simsiam"]
    epochs = int(profile_value(ss_cfg.get("epochs", {"fast": 1, "full": 10}), profile))
    transform = build_train_transform(image_size, cfg["augment"]["techniques"],
                                       cfg["augment"].get("domain_randomization"), cfg["augment"].get("domain_randomization_prob", 0.5))
    loader = DataLoader(_TwoViewDataset(entries, transform, working_size),
                        batch_size=int(cfg["train"]["batch_size"]), shuffle=True, num_workers=int(cfg["train"].get("num_workers", 0)))
    ss = SimSiam(model.features, model.feat_dim, int(ss_cfg.get("proj_dim", 2048)), int(ss_cfg.get("pred_dim", 512))).to(device)
    opt = torch.optim.SGD(ss.parameters(), lr=float(ss_cfg.get("lr", 0.05)), momentum=0.9, weight_decay=1e-4)
    ss.train()
    history = []
    for epoch in range(epochs):
        total = 0.0
        for x1, x2 in loader:
            x1, x2 = x1.to(device), x2.to(device)
            p1, p2, z1, z2 = ss(x1, x2)
            loss = simsiam_loss(p1, p2, z1, z2)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach())
        avg = total / max(1, len(loader))
        history.append(avg)
        logger.info("[SimSiam] epoch %d/%d  loss=%.4f", epoch + 1, epochs, avg)
    return {"epochs": epochs, "loss_history": history}


# ---------------------------------------------------------------------------
# Supervised training
# ---------------------------------------------------------------------------
def train_model(
    config: Dict[str, Any],
    run_paths,
    profile: str,
    split_manifest: Dict[str, Any],
    device,
    logger=None,
    tracker=None,
    epochs_override: Optional[int] = None,
) -> Dict[str, Any]:
    logger = logger or get_logger("stage4.train", run_paths.stage("stage4_model") / "train.log")
    classes = split_manifest["classes"]
    num_classes = len(classes)
    working_size = int(split_manifest.get("working_size", config["data"]["working_size"]))
    image_size = int(split_manifest.get("image_size", config["data"]["image_size"]))
    train_entries = split_manifest["splits"]["train"]
    val_entries = split_manifest["splits"]["val"]

    train_cfg = config["train"]
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 0))
    epochs = epochs_override or int(profile_value(train_cfg["epochs"], profile))
    patience = int(profile_value(train_cfg.get("early_stopping_patience", {"fast": 3, "full": 8}), profile))

    model = build_model(num_classes, config["model"]).to(device)
    logger.info("Model: %s (pretrained=%s), %d classes, %d train / %d val",
                model.backbone_name, model.pretrained_used, num_classes, len(train_entries), len(val_entries))

    simsiam_info = None
    if bool(config["model"].get("simsiam", {}).get("enabled", False)):
        logger.info("SimSiam SSL pretraining enabled")
        simsiam_info = simsiam_pretrain(model, train_entries, config, profile, device, working_size, image_size, logger)

    train_tf = build_train_transform(image_size, config["augment"]["techniques"],
                                      config["augment"].get("domain_randomization"),
                                      float(config["augment"].get("domain_randomization_prob", 0.5)))
    eval_tf = build_eval_transform(image_size)

    sampler = make_weighted_sampler(train_entries, num_classes) if train_cfg.get("balanced_sampler", True) else None
    train_loader = make_loader(train_entries, train_tf, batch_size, working_size, num_workers, sampler=sampler, shuffle=sampler is None)
    val_loader = make_loader(val_entries, eval_tf, batch_size, working_size, num_workers)

    counts = class_counts_from(train_entries, num_classes)
    loss_cfg = config["loss"]
    loss_fn = CombinedLoss(
        class_counts=counts,
        ce_weight=float(loss_cfg.get("ce_weight", 1.0)),
        supcon_weight=float(loss_cfg.get("supcon_weight", 0.5)),
        class_balancing=str(loss_cfg.get("class_balancing", "effective_number")),
        cb_beta=float(loss_cfg.get("cb_beta", 0.999)),
        focal_gamma=float(loss_cfg.get("focal_gamma", 2.0)),
        supcon_temperature=float(loss_cfg.get("supcon_temperature", 0.07)),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    scheduler = _cosine_warmup(optimizer, int(train_cfg.get("warmup_epochs", 1)), epochs)

    ckpt_path = run_paths.checkpoints / "best.pt"
    best_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(epochs):
        model.train()
        running = {"total": 0.0, "classification": 0.0, "supcon": 0.0}
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, proj = model(x, return_projection=True)
            loss, parts = loss_fn(logits, proj, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            for k in running:
                running[k] += parts[k]
            n_batches += 1
        scheduler.step()
        val = evaluate_model(model, val_loader, device, num_classes)
        avg_loss = running["total"] / max(1, n_batches)
        lr_now = optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch + 1, "train_loss": avg_loss, "val_accuracy": val["accuracy"],
               "val_macro_f1": val["macro_f1"], "lr": lr_now}
        history.append(row)
        logger.info("epoch %d/%d  loss=%.4f  val_acc=%.4f  val_macroF1=%.4f  lr=%.2e",
                    epoch + 1, epochs, avg_loss, val["accuracy"], val["macro_f1"], lr_now)
        if tracker is not None:
            tracker.log_metrics(row, step=epoch + 1)

        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]; best_epoch = epoch + 1; epochs_no_improve = 0
            torch.save(
                {"model_state": model.state_dict(), "classes": classes, "cfg_model": config["model"],
                 "backbone_name": model.backbone_name, "num_classes": num_classes,
                 "image_size": image_size, "working_size": working_size},
                ckpt_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping at epoch %d (no val macro-F1 improvement for %d epochs)", epoch + 1, patience)
                break

    if best_epoch < 0:  # degenerate (0 epochs) — still save a checkpoint
        torch.save({"model_state": model.state_dict(), "classes": classes, "cfg_model": config["model"],
                    "backbone_name": model.backbone_name, "num_classes": num_classes,
                    "image_size": image_size, "working_size": working_size}, ckpt_path)
        best_epoch = 0

    summary = {
        "backbone": model.backbone_name,
        "pretrained_used": model.pretrained_used,
        "epochs_run": len(history),
        "epochs_planned": epochs,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "train_class_counts": {classes[i]: counts[i] for i in range(num_classes)},
        "loss_config": {k: loss_cfg.get(k) for k in ("ce_weight", "supcon_weight", "class_balancing", "cb_beta", "focal_gamma")},
        "history": history,
        "checkpoint": str(ckpt_path),
        "simsiam": simsiam_info,
    }
    save_json(summary, run_paths.stage("stage4_model") / "train_summary.json")
    return summary
