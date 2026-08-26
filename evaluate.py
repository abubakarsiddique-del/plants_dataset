#!/usr/bin/env python3
"""Stage 4b — evaluation (deterministic given seed).

* **5-fold stratified cross-validation** over train+val (``fast`` uses 2 folds,
  1 epoch each) — a variance estimate independent of the single fitted model.
* Final metrics for the best checkpoint on the **in-domain test** split and the
  **external_domain_test** split: accuracy, per-class precision/recall/F1,
  confusion matrices, and **bootstrap 95% CIs** (1000 resamples; fewer in
  ``fast``) for accuracy and macro-F1.
* A ``comparison`` block with the in-domain→external accuracy/macro-F1 drop and
  the minimum per-class F1 — the exact evidence Agent 2's gap checklist reads.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold

from augment import build_eval_transform, build_train_transform
from common import cfg_get, get_logger, profile_value, save_json
from losses import CombinedLoss
from model import build_model
from train import class_counts_from, evaluate_model, make_loader, make_weighted_sampler


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def per_class_prf(targets: np.ndarray, preds: np.ndarray, num_classes: int,
                  class_names: List[str]) -> Dict[str, Dict[str, float]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=list(range(num_classes)), zero_division=0)
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(class_names):
        out[name] = {"precision": float(precision[i]), "recall": float(recall[i]),
                     "f1": float(f1[i]), "support": int(support[i])}
    return out


def bootstrap_ci(targets: np.ndarray, preds: np.ndarray, n_resamples: int,
                 ci: float, seed: int) -> Dict[str, Dict[str, float]]:
    """Percentile bootstrap CIs for accuracy and macro-F1."""
    rng = np.random.default_rng(seed)
    n = len(targets)
    if n == 0:
        empty = {"point": 0.0, "lo": 0.0, "hi": 0.0}
        return {"accuracy": dict(empty), "macro_f1": dict(empty), "n_resamples": n_resamples, "ci": ci}
    accs, f1s = [], []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        t, p = targets[idx], preds[idx]
        accs.append(accuracy_score(t, p))
        f1s.append(f1_score(t, p, average="macro", zero_division=0))
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {
        "accuracy": {"point": float(accuracy_score(targets, preds)),
                     "lo": float(np.percentile(accs, lo_q)), "hi": float(np.percentile(accs, hi_q))},
        "macro_f1": {"point": float(f1_score(targets, preds, average="macro", zero_division=0)),
                     "lo": float(np.percentile(f1s, lo_q)), "hi": float(np.percentile(f1s, hi_q))},
        "n_resamples": n_resamples, "ci": ci,
    }


def _metric_block(targets: np.ndarray, preds: np.ndarray, num_classes: int,
                  class_names: List[str], n_boot: int, ci: float, seed: int) -> Dict[str, Any]:
    per_class = per_class_prf(targets, preds, num_classes, class_names)
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes))).tolist() if len(targets) else []
    return {
        "n": int(len(targets)),
        "accuracy": float(accuracy_score(targets, preds)) if len(targets) else 0.0,
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)) if len(targets) else 0.0,
        "weighted_f1": float(f1_score(targets, preds, average="weighted", zero_division=0)) if len(targets) else 0.0,
        "per_class": per_class,
        "confusion_matrix": cm,
        "bootstrap": bootstrap_ci(targets, preds, n_boot, ci, seed),
    }


def _min_class_f1(per_class: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Minimum F1 over classes with at least one true example (support>0)."""
    populated = {k: v for k, v in per_class.items() if v["support"] > 0}
    if not populated:
        return {"class": None, "f1": None}
    worst = min(populated.items(), key=lambda kv: kv[1]["f1"])
    return {"class": worst[0], "f1": float(worst[1]["f1"])}


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def _train_eval_fold(train_entries, val_entries, config, device, num_classes, class_names,
                     image_size, working_size, cv_epochs, seed) -> Dict[str, float]:
    torch.manual_seed(seed)
    train_cfg = config["train"]
    train_tf = build_train_transform(image_size, config["augment"]["techniques"],
                                     config["augment"].get("domain_randomization"),
                                     float(config["augment"].get("domain_randomization_prob", 0.5)))
    eval_tf = build_eval_transform(image_size)
    sampler = make_weighted_sampler(train_entries, num_classes) if train_cfg.get("balanced_sampler", True) else None
    batch_size = int(train_cfg["batch_size"])
    train_loader = make_loader(train_entries, train_tf, batch_size, working_size,
                               int(train_cfg.get("num_workers", 0)), sampler=sampler, shuffle=sampler is None)
    val_loader = make_loader(val_entries, eval_tf, batch_size, working_size, int(train_cfg.get("num_workers", 0)))

    model = build_model(num_classes, config["model"]).to(device)
    counts = class_counts_from(train_entries, num_classes)
    loss_cfg = config["loss"]
    loss_fn = CombinedLoss(
        class_counts=counts, ce_weight=float(loss_cfg.get("ce_weight", 1.0)),
        supcon_weight=float(loss_cfg.get("supcon_weight", 0.5)),
        class_balancing=str(loss_cfg.get("class_balancing", "effective_number")),
        cb_beta=float(loss_cfg.get("cb_beta", 0.999)), focal_gamma=float(loss_cfg.get("focal_gamma", 2.0)),
        supcon_temperature=float(loss_cfg.get("supcon_temperature", 0.07)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]),
                                  weight_decay=float(train_cfg["weight_decay"]))
    for _ in range(cv_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, proj = model(x, return_projection=True)
            loss, _ = loss_fn(logits, proj, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    res = evaluate_model(model, val_loader, device, num_classes)
    return {"accuracy": res["accuracy"], "macro_f1": res["macro_f1"], "n_val": len(val_entries)}


def cross_validate(config, split_manifest, device, profile, logger, seed) -> Dict[str, Any]:
    classes = split_manifest["classes"]
    num_classes = len(classes)
    image_size = int(split_manifest.get("image_size", config["data"]["image_size"]))
    working_size = int(split_manifest.get("working_size", config["data"]["working_size"]))
    pool = list(split_manifest["splits"]["train"]) + list(split_manifest["splits"]["val"])
    labels = np.array([int(e["label_index"]) for e in pool])

    n_folds = int(profile_value(cfg_get(config, "eval.cv_folds", {"fast": 2, "full": 5}), profile))
    cv_epochs = int(profile_value(cfg_get(config, "eval.cv_epochs", {"fast": 1, "full": 15}), profile))
    # StratifiedKFold needs at least n_folds samples in the rarest class
    min_class = int(np.bincount(labels, minlength=num_classes)[np.bincount(labels, minlength=num_classes) > 0].min())
    n_folds = max(2, min(n_folds, min_class))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: List[Dict[str, float]] = []
    for k, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(pool)), labels), start=1):
        tr = [pool[i] for i in tr_idx]
        va = [pool[i] for i in va_idx]
        fold_res = _train_eval_fold(tr, va, config, device, num_classes, classes,
                                    image_size, working_size, cv_epochs, seed + k)
        fold_res["fold"] = k
        folds.append(fold_res)
        logger.info("CV fold %d/%d: acc=%.4f macroF1=%.4f (n_val=%d)",
                    k, n_folds, fold_res["accuracy"], fold_res["macro_f1"], fold_res["n_val"])

    accs = np.array([f["accuracy"] for f in folds])
    f1s = np.array([f["macro_f1"] for f in folds])
    return {
        "n_folds": n_folds, "cv_epochs": cv_epochs, "folds": folds,
        "mean_accuracy": float(accs.mean()), "std_accuracy": float(accs.std()),
        "mean_macro_f1": float(f1s.mean()), "std_macro_f1": float(f1s.std()),
    }


# ---------------------------------------------------------------------------
# Final evaluation on test + external
# ---------------------------------------------------------------------------
def _predict_split(model, entries, config, device, num_classes, image_size, working_size,
                   corruption=None) -> Tuple[np.ndarray, np.ndarray]:
    eval_tf = build_eval_transform(image_size)
    loader = make_loader(entries, eval_tf, int(config["train"]["batch_size"]), working_size,
                         int(config["train"].get("num_workers", 0)), corruption=corruption)
    res = evaluate_model(model, loader, device, num_classes)
    return res["targets"], res["preds"]


def run_evaluate_stage(
    config: Dict[str, Any],
    run_paths,
    profile: str,
    split_manifest: Dict[str, Any],
    device,
    checkpoint_path: str,
    logger=None,
    tracker=None,
) -> Dict[str, Any]:
    logger = logger or get_logger("stage4.evaluate", run_paths.stage("stage4_model") / "evaluate.log")
    from augment import build_domain_randomization_transform

    classes = split_manifest["classes"]
    num_classes = len(classes)
    image_size = int(split_manifest.get("image_size", config["data"]["image_size"]))
    working_size = int(split_manifest.get("working_size", config["data"]["working_size"]))
    seed = int(config.get("seed", 42))
    ci = float(cfg_get(config, "eval.bootstrap.ci", 0.95))
    n_boot = int(profile_value({"fast": int(cfg_get(config, "eval.bootstrap_fast_n", 200)),
                                "full": int(cfg_get(config, "eval.bootstrap.n", 1000))}, profile))

    # ---- cross-validation ----
    logger.info("Cross-validation (%s profile)...", profile)
    cv = cross_validate(config, split_manifest, device, profile, logger, seed)

    # ---- load best checkpoint ----
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(num_classes, ckpt.get("cfg_model", config["model"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- in-domain test ----
    test_entries = split_manifest["splits"]["test"]
    t_targets, t_preds = _predict_split(model, test_entries, config, device, num_classes, image_size, working_size)
    test_metrics = _metric_block(t_targets, t_preds, num_classes, classes, n_boot, ci, seed)
    logger.info("In-domain test: acc=%.4f macroF1=%.4f (n=%d)",
                test_metrics["accuracy"], test_metrics["macro_f1"], test_metrics["n"])

    # ---- external-domain test ----
    ext_entries = split_manifest["splits"]["external_domain_test"]
    corruption = None
    if split_manifest.get("external_mode") == "synthetic_corruption":
        corruption = build_domain_randomization_transform(list(split_manifest.get("corruptions", [])))
    e_targets, e_preds = _predict_split(model, ext_entries, config, device, num_classes,
                                        image_size, working_size, corruption=corruption)
    external_metrics = _metric_block(e_targets, e_preds, num_classes, classes, n_boot, ci, seed + 1)
    external_metrics["mode"] = split_manifest.get("external_mode")
    external_metrics["domain"] = split_manifest.get("external_domain")
    logger.info("External-domain test: acc=%.4f macroF1=%.4f (n=%d, mode=%s)",
                external_metrics["accuracy"], external_metrics["macro_f1"], external_metrics["n"],
                split_manifest.get("external_mode"))

    # ---- comparison block (feeds Agent 2 gaps 1 & 2) ----
    comparison = {
        "accuracy_drop_pts": float((test_metrics["accuracy"] - external_metrics["accuracy"]) * 100.0),
        "macro_f1_drop_pts": float((test_metrics["macro_f1"] - external_metrics["macro_f1"]) * 100.0),
        "min_class_f1_in_domain": _min_class_f1(test_metrics["per_class"]),
        "min_class_f1_external": _min_class_f1(external_metrics["per_class"]),
        "in_domain_accuracy": test_metrics["accuracy"],
        "external_accuracy": external_metrics["accuracy"],
    }

    confusion_matrices = {
        "labels": classes,
        "in_domain_test": test_metrics["confusion_matrix"],
        "external_domain_test": external_metrics["confusion_matrix"],
    }

    out = {
        "cv_metrics": cv,
        "test_metrics": test_metrics,
        "external_domain_metrics": external_metrics,
        "confusion_matrices": confusion_matrices,
        "comparison": comparison,
    }
    save_json(out, run_paths.stage("stage4_model") / "evaluation.json")
    if tracker is not None:
        tracker.log_metrics({"test_accuracy": test_metrics["accuracy"], "test_macro_f1": test_metrics["macro_f1"],
                             "external_accuracy": external_metrics["accuracy"],
                             "external_macro_f1": external_metrics["macro_f1"],
                             "cv_mean_macro_f1": cv["mean_macro_f1"]})
        tracker.log_artifact(run_paths.stage("stage4_model") / "evaluation.json", "evaluation.json")
    logger.info("Evaluation done. accuracy drop in→ext = %.1f pts; min class F1 (in-domain) = %s",
                comparison["accuracy_drop_pts"], comparison["min_class_f1_in_domain"])
    return out
