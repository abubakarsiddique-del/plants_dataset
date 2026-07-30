#!/usr/bin/env python3
"""Fuzzy C-Means clustering pipeline for originals-only features.

Produces membership, centroids, hard labels, entropy scores, and validation
metrics against ground truth class labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
import yaml
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

try:
    import skfuzzy as fuzz
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-fuzzy"])
    import skfuzzy as fuzz


@dataclass
class PipelineConfig:
    feature_path: Path
    feature_index_path: Path
    output_dir: Path
    cs: Sequence[int]
    m: float
    error: float
    maxiter: int
    seed: int


def load_feature_index(index_path: Path) -> List[Dict[str, str]]:
    with index_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_CLUSTERS = [5, 6, 7]


def load_yaml_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def merge_config(args: argparse.Namespace, config_data: Dict[str, object]) -> PipelineConfig:
    fcm_conf = config_data.get("fcm", {}) if config_data else {}

    clusters = args.clusters
    if clusters == DEFAULT_CLUSTERS and isinstance(fcm_conf.get("clusters"), list):
        clusters = fcm_conf["clusters"]

    return PipelineConfig(
        feature_path=Path(fcm_conf.get("feature_path", args.feature_path)),
        feature_index_path=Path(fcm_conf.get("feature_index_path", args.feature_index)),
        output_dir=Path(fcm_conf.get("output_dir", args.output_dir)),
        cs=clusters,
        m=float(fcm_conf.get("m", args.m)),
        error=float(fcm_conf.get("error", args.error)),
        maxiter=int(fcm_conf.get("maxiter", args.maxiter)),
        seed=int(fcm_conf.get("seed", args.seed)),
    )


def load_features(feature_path: Path) -> np.ndarray:
    return np.load(feature_path)


def compute_membership_entropy(U: np.ndarray) -> np.ndarray:
    # U shape: (c, n_samples)
    eps = np.finfo(float).eps
    log_u = np.log(U + eps)
    entropy = -np.sum(U * log_u, axis=0)
    # normalize by log(c) so values range [0, 1]
    norm = np.log(U.shape[0])
    if norm <= 0:
        return np.zeros_like(entropy)
    return entropy / norm


def contingency_matrix(labels_true: Sequence[str], labels_pred: Sequence[int]) -> Dict[str, Dict[int, int]]:
    matrix: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for true, pred in zip(labels_true, labels_pred):
        matrix[true][int(pred)] += 1
    return matrix


def write_confusion_matrix(matrix: Dict[str, Dict[int, int]], path: Path) -> None:
    classes = sorted(matrix.keys())
    clusters = sorted({cluster for row in matrix.values() for cluster in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_label"] + [f"cluster_{c}" for c in clusters])
        for cls in classes:
            counts = [matrix[cls].get(c, 0) for c in clusters]
            writer.writerow([cls] + counts)


def run_fcm(config: PipelineConfig) -> Dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_features(config.feature_path)
    index_rows = load_feature_index(config.feature_index_path)
    if data.ndim != 2:
        raise ValueError(f"Expected feature matrix shape (n_samples, n_features), got {data.shape}")

    n_samples = data.shape[0]
    labels_true = [row["class_label"] for row in index_rows if row["is_original"].strip().lower() == "true"]
    if len(labels_true) != n_samples:
        raise ValueError(f"Label count {len(labels_true)} does not match sample count {n_samples}")

    labels_map = sorted(set(labels_true))
    label_ids = {label: idx for idx, label in enumerate(labels_map)}
    truth_ids = np.array([label_ids[label] for label in labels_true], dtype=int)

    best_summary: Optional[Dict[str, object]] = None

    # transpose for skfuzzy cmeans: features x samples
    data_T = data.T

    for c in config.cs:
        result_dir = config.output_dir / f"fcm_c{c}"
        result_dir.mkdir(parents=True, exist_ok=True)

        # run fuzzy c-means
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data_T,
            c=c,
            m=config.m,
            error=config.error,
            maxiter=config.maxiter,
            init=None,
            seed=config.seed,
        )

        hard_labels = np.argmax(u, axis=0)
        entropy = compute_membership_entropy(u)

        nmi = float(normalized_mutual_info_score(truth_ids, hard_labels, average_method="arithmetic"))
        ari = float(adjusted_rand_score(truth_ids, hard_labels))

        matrix = contingency_matrix(labels_true, hard_labels.tolist())
        write_confusion_matrix(matrix, result_dir / "confusion_matrix.csv")

        np.save(result_dir / "membership_matrix.npy", u)
        np.save(result_dir / "cluster_centers.npy", cntr)
        np.save(result_dir / "hard_labels.npy", hard_labels)
        np.save(result_dir / "membership_entropy.npy", entropy)

        label_csv = result_dir / "fcm_hard_labels.csv"
        with label_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            header = ["feature_row", "source_id", "class_label", "hard_label", "entropy"] + [f"membership_{k}" for k in range(c)]
            writer.writerow(header)
            for row, hard, ent, memberships in zip(index_rows, hard_labels, entropy, u.T):
                if row["is_original"].strip().lower() != "true":
                    continue
                writer.writerow([
                    row["feature_row"],
                    row["source_id"],
                    row["class_label"],
                    int(hard),
                    float(ent),
                    *[float(x) for x in memberships],
                ])

        summary = {
            "c": c,
            "m": config.m,
            "seed": config.seed,
            "error": config.error,
            "maxiter": config.maxiter,
            "n_samples": n_samples,
            "n_features": int(data.shape[1]),
            "fpc": float(fpc),
            "nmi": nmi,
            "ari": ari,
            "cluster_counts": {int(k): int(v) for k, v in Counter(hard_labels).items()},
            "class_labels": labels_map,
            "label_counts": {label: int(count) for label, count in Counter(labels_true).items()},
            "membership_entropy_mean": float(entropy.mean()),
            "membership_entropy_std": float(entropy.std()),
            "feature_path": str(config.feature_path),
            "feature_index_path": str(config.feature_index_path),
        }

        with (result_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        if best_summary is None or fpc > best_summary["fpc"]:
            best_summary = {**summary, "result_dir": str(result_dir)}

    if best_summary is not None:
        with (config.output_dir / "fcm_best_run.json").open("w", encoding="utf-8") as handle:
            json.dump(best_summary, handle, indent=2)

    return best_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Fuzzy C-Means clustering and validate with NMI/ARI.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to YAML config file")
    parser.add_argument("--feature-path", type=Path, default=Path("features/originals_pca64_features.npy"))
    parser.add_argument("--feature-index", type=Path, default=Path("features/feature_index.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("features/fcm_results"))
    parser.add_argument("--clusters", type=int, nargs="+", default=[5, 6, 7])
    parser.add_argument("--m", type=float, default=2.0)
    parser.add_argument("--error", type=float, default=1e-5)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_data = load_yaml_config(args.config)
    config = merge_config(args, config_data)
    best_summary = run_fcm(config)
    if best_summary is None:
        raise RuntimeError("No FCM runs completed successfully")
    print("Best run:", json.dumps(best_summary, indent=2))


if __name__ == "__main__":
    main()
