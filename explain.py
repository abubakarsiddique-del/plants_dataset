#!/usr/bin/env python3
"""Stage 5 — explainability (Grad-CAM++ + faithfulness).

Self-contained **Grad-CAM++** (hooks on the backbone's last conv module; prefers
``pytorch-grad-cam`` if installed) producing K per-class overlays, plus
**deletion / insertion AUC** faithfulness (RISE-style) so the explanations are
quantitatively checked, not just drawn. Emits ``explainability_report`` with
per-class **coverage** (Agent 2, gap 4) and a ``faithfulness_score`` = insertion
AUC − deletion AUC compared against ``thresholds.faithfulness_min``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from augment import IMAGENET_MEAN, IMAGENET_STD, build_eval_transform
from augmentation_pipeline import letterbox_standardize, load_rgb_image
from common import cfg_get, get_logger, profile_value, save_json
from model import build_model


# ---------------------------------------------------------------------------
# Grad-CAM++
# ---------------------------------------------------------------------------
class GradCAMpp:
    """Grad-CAM++ (Chattopadhay et al., 2018) on a single conv layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd = target_layer.register_forward_hook(self._save_activation)
        self._bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()

    def __call__(self, x: torch.Tensor, target: Optional[int] = None) -> Tuple[np.ndarray, int]:
        self.model.zero_grad()
        logits, _ = self.model(x, return_projection=False)
        if target is None:
            target = int(logits.argmax(dim=1).item())
        score = logits[0, target]
        score.backward(retain_graph=False)

        grads = self.gradients            # (1,C,h,w)
        acts = self.activations           # (1,C,h,w)
        grads2 = grads.pow(2)
        grads3 = grads2 * grads
        global_sum = acts.sum(dim=(2, 3), keepdim=True)
        eps = 1e-8
        alpha = grads2 / (2.0 * grads2 + global_sum * grads3 + eps)
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)   # (1,C,1,1)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))           # (1,1,h,w)
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, target


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
_STD = np.array(IMAGENET_STD, dtype=np.float32)


def _load_input(rec: Dict[str, Any], image_size: int, working_size: int, eval_tf) -> Tuple[torch.Tensor, np.ndarray]:
    """Return (normalized CHW tensor with batch dim, uint8 RGB display image)."""
    img = load_rgb_image(Path(rec["path"]))
    img = letterbox_standardize(img, (working_size, working_size))
    tensor = eval_tf(image=img)["image"].unsqueeze(0)
    display = np.asarray(letterbox_standardize(img, (image_size, image_size)), dtype=np.uint8)
    return tensor, display


def _save_overlay(display: np.ndarray, cam: np.ndarray, title: str, path: Path) -> None:
    heat = cm.jet(cam)[:, :, :3]
    base = display.astype(np.float32) / 255.0
    overlay = np.clip(0.55 * base + 0.45 * heat, 0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
    axes[0].imshow(base); axes[0].set_title("input", fontsize=9); axes[0].axis("off")
    axes[1].imshow(overlay); axes[1].set_title("Grad-CAM++", fontsize=9); axes[1].axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Faithfulness: deletion / insertion AUC
# ---------------------------------------------------------------------------
@torch.no_grad()
def _del_ins_auc(model, x: torch.Tensor, cam: np.ndarray, target: int, steps: int, device) -> Tuple[float, float]:
    _, _, H, W = x.shape
    n = H * W
    order = np.argsort(-cam.reshape(-1))          # most-important pixels first
    step = max(1, n // steps)
    x_flat = x.squeeze(0).reshape(3, n)           # (3, n) view onto data
    baseline = torch.zeros_like(x_flat)           # normalized zero == ImageNet mean image

    def _score(t: torch.Tensor) -> float:
        logits, _ = model(t.reshape(1, 3, H, W).to(device), return_projection=False)
        return float(torch.softmax(logits, dim=1)[0, target])

    # deletion: start from full image, remove important pixels
    cur = x_flat.clone()
    del_scores = [_score(cur)]
    for i in range(0, n, step):
        idx = torch.as_tensor(order[i:i + step], dtype=torch.long)
        cur[:, idx] = baseline[:, idx]
        del_scores.append(_score(cur))

    # insertion: start from baseline, insert important pixels
    cur = baseline.clone()
    ins_scores = [_score(cur)]
    for i in range(0, n, step):
        idx = torch.as_tensor(order[i:i + step], dtype=torch.long)
        cur[:, idx] = x_flat[:, idx]
        ins_scores.append(_score(cur))

    frac = np.linspace(0.0, 1.0, len(del_scores))
    deletion_auc = float(np.trapz(del_scores, frac))
    insertion_auc = float(np.trapz(ins_scores, np.linspace(0.0, 1.0, len(ins_scores))))
    return deletion_auc, insertion_auc


# ---------------------------------------------------------------------------
# Stage entry
# ---------------------------------------------------------------------------
def run_explain_stage(
    config: Dict[str, Any],
    run_paths,
    profile: str,
    split_manifest: Dict[str, Any],
    device,
    checkpoint_path: str,
    logger=None,
    tracker=None,
) -> Dict[str, Any]:
    logger = logger or get_logger("stage5.explain", run_paths.explain / "explain.log")
    classes = split_manifest["classes"]
    num_classes = len(classes)
    image_size = int(split_manifest.get("image_size", config["data"]["image_size"]))
    working_size = int(split_manifest.get("working_size", config["data"]["working_size"]))
    k = int(profile_value(cfg_get(config, "explain.samples_per_class", {"fast": 2, "full": 5}), profile))
    steps = int(cfg_get(config, "explain.faithfulness_steps", 20))
    method = str(cfg_get(config, "explain.method", "gradcampp"))

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(num_classes, ckpt.get("cfg_model", config["model"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    eval_tf = build_eval_transform(image_size)

    # group test images by true class
    by_class: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(num_classes)}
    for rec in split_manifest["splits"]["test"]:
        by_class[int(rec["label_index"])].append(rec)

    cam_engine = GradCAMpp(model, model.last_conv_module())
    overlays: List[Dict[str, Any]] = []
    del_aucs: List[float] = []
    ins_aucs: List[float] = []
    covered: List[str] = []

    for cls_idx in range(num_classes):
        recs = by_class[cls_idx][:k]
        if recs:
            covered.append(classes[cls_idx])
        for j, rec in enumerate(recs):
            tensor, display = _load_input(rec, image_size, working_size, eval_tf)
            tensor = tensor.to(device).requires_grad_(True)
            cam, pred = cam_engine(tensor, target=cls_idx)
            out_path = run_paths.explain / f"cam_{classes[cls_idx]}_{j}.png"
            _save_overlay(display, cam,
                          f"true={classes[cls_idx]}  pred={classes[pred]}", out_path)
            d_auc, i_auc = _del_ins_auc(model, tensor.detach(), cam, cls_idx, steps, device)
            del_aucs.append(d_auc); ins_aucs.append(i_auc)
            overlays.append({"class": classes[cls_idx], "true": classes[cls_idx],
                             "pred": classes[pred], "path": str(out_path),
                             "deletion_auc": d_auc, "insertion_auc": i_auc,
                             "source": rec["path"]})
            logger.info("CAM %s [%d]: pred=%s del_auc=%.3f ins_auc=%.3f",
                        classes[cls_idx], j, classes[pred], d_auc, i_auc)
    cam_engine.remove()

    deletion_mean = float(np.mean(del_aucs)) if del_aucs else 0.0
    insertion_mean = float(np.mean(ins_aucs)) if ins_aucs else 0.0
    faithfulness_score = insertion_mean - deletion_mean
    coverage_complete = set(covered) == set(classes)

    report = {
        "method": method,
        "samples_per_class": k,
        "faithfulness_steps": steps,
        "num_active_classes": num_classes,
        "classes_covered": covered,
        "coverage_complete": coverage_complete,
        "coverage_missing": sorted(set(classes) - set(covered)),
        "num_overlays": len(overlays),
        "overlays": overlays,
        "faithfulness": {
            "deletion_auc_mean": deletion_mean,
            "insertion_auc_mean": insertion_mean,
            "faithfulness_score": faithfulness_score,
            "n_images": len(del_aucs),
            "note": "faithfulness_score = insertion_auc_mean - deletion_auc_mean (higher is better)",
        },
    }
    save_json(report, run_paths.explain / "explainability_report.json")
    if tracker is not None:
        tracker.log_metrics({"deletion_auc": deletion_mean, "insertion_auc": insertion_mean,
                             "faithfulness_score": faithfulness_score})
        for ov in overlays:
            tracker.log_artifact(ov["path"], Path(ov["path"]).name)
    logger.info("Explainability done. coverage_complete=%s faithfulness_score=%.3f (del=%.3f ins=%.3f)",
                coverage_complete, faithfulness_score, deletion_mean, insertion_mean)
    return report
