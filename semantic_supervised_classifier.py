#!/usr/bin/env python3
"""Train a supervised classifier on the exported semantic feature matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from augmentation_pipeline import WORKING_SIZE, letterbox_standardize, load_rgb_image
from feature_extraction import DeepEmbeddingModel, resolve_device

ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURE_MATRIX = ROOT / "features" / "feature_matrix.npy"
DEFAULT_FEATURE_INDEX = ROOT / "features" / "feature_index.csv"
DEFAULT_MODEL_OUTPUT = ROOT / "models" / "semantic_svm.joblib"
DEFAULT_METRICS_OUTPUT = ROOT / "models" / "semantic_svm_metrics.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_rows(feature_index_path: Path) -> List[Dict[str, str]]:
    with feature_index_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_dataset(feature_matrix_path: Path, feature_index_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feature_matrix = np.load(feature_matrix_path)
    rows = load_rows(feature_index_path)
    if feature_matrix.shape[0] != len(rows):
        raise ValueError(
            f"Feature matrix size {feature_matrix.shape[0]} does not match feature index size {len(rows)}"
        )

    class_names = sorted({row["class_label"] for row in rows})
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    labels = np.array([label_to_idx[row["class_label"]] for row in rows], dtype=np.int64)

    features = np.nan_to_num(feature_matrix.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return features, labels, class_names


def train_and_evaluate(feature_matrix_path: Path, feature_index_path: Path) -> Dict[str, object]:
    features, labels, class_names = build_dataset(feature_matrix_path, feature_index_path)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10.0, gamma="scale", probability=False, random_state=42),
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    test_accuracy = float(accuracy_score(y_test, preds))
    test_macro_f1 = float(f1_score(y_test, preds, average="macro"))

    metrics = {
        "model": "svc_rbf",
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "class_names": class_names,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_dim": int(features.shape[1]),
    }

    return metrics, model


def load_trained_model(model_path: Path, metrics_path: Path) -> Tuple[object, List[str]]:
    payload = joblib.load(model_path)
    if isinstance(payload, dict):
        model = payload.get("model")
        class_names = payload.get("class_names") or []
    else:
        model = payload
        class_names = []

    if not class_names and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        class_names = metrics.get("class_names", [])

    if model is None:
        raise ValueError(f"Could not load a valid model from {model_path}")
    if not class_names:
        raise ValueError(f"No class labels were found for {model_path}")

    return model, class_names


def resolve_input_paths(paths: Sequence[Path]) -> List[Path]:
    resolved: List[Path] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                    resolved.append(child)
        else:
            resolved.append(path)
    return resolved


def load_image_for_embedding(image_path: Path) -> np.ndarray:
    rgb = load_rgb_image(image_path)
    h, w = rgb.shape[:2]
    if (w, h) != WORKING_SIZE:
        rgb = letterbox_standardize(rgb, WORKING_SIZE)
    return rgb


def embed_image_paths(image_paths: Sequence[Path], backbone: str = "resnet50") -> np.ndarray:
    device = resolve_device("auto")
    model = DeepEmbeddingModel(backbone_name=backbone).to(device)
    rgb_images = [load_image_for_embedding(path) for path in image_paths]
    with torch.inference_mode():
        embeddings = model.forward_batch(rgb_images, device)
    return embeddings.astype(np.float32)


def predict_images(image_paths: Sequence[Path], model_path: Path, metrics_path: Path, backbone: str = "resnet50") -> Dict[str, object]:
    model, class_names = load_trained_model(model_path, metrics_path)
    resolved_paths = resolve_input_paths([Path(path) for path in image_paths])
    if not resolved_paths:
        raise ValueError("No image files were found to classify")

    features = embed_image_paths(resolved_paths, backbone=backbone)
    preds = model.predict(features)

    results = []
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)
        for image_path, pred_idx, prob_vector in zip(resolved_paths, preds, probs):
            class_idx = int(pred_idx)
            class_name = class_names[class_idx]
            confidences = {
                class_names[idx]: float(prob_vector[idx]) for idx in range(len(class_names))
            }
            results.append(
                {
                    "image": str(image_path),
                    "predicted_class": class_name,
                    "confidence": round(float(prob_vector[class_idx]), 4),
                    "confidences": {k: round(v, 4) for k, v in confidences.items()},
                }
            )
    else:
        for image_path, pred_idx in zip(resolved_paths, preds):
            class_idx = int(pred_idx)
            results.append(
                {
                    "image": str(image_path),
                    "predicted_class": class_names[class_idx],
                    "confidence": None,
                }
            )

    return {"model": str(model_path), "class_names": class_names, "results": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a supervised classifier from the semantic feature matrix")
    parser.add_argument("--feature-matrix", type=Path, default=DEFAULT_FEATURE_MATRIX)
    parser.add_argument("--feature-index", type=Path, default=DEFAULT_FEATURE_INDEX)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_OUTPUT)
    parser.add_argument("--predict", nargs="+", type=Path, default=None, help="Predict classes for one or more image files or folders")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_OUTPUT, help="Path to the trained model artifact")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_OUTPUT, help="Path to the metrics JSON file")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet18", "resnet50"], help="Backbone used to compute the semantic embedding")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.predict is not None:
        predictions = predict_images(args.predict, args.model, args.metrics, backbone=args.backbone)
        print(json.dumps(predictions, indent=2))
    else:
        metrics, model = train_and_evaluate(args.feature_matrix, args.feature_index)

        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "class_names": metrics["class_names"]}, args.model_output)
        args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        print(json.dumps(metrics, indent=2))
