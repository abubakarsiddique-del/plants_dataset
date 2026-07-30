#!/usr/bin/env python3
"""Generate evaluation and QA artifacts for FCM clustering.

Outputs:
- t-SNE and UMAP plots colored by true label and cluster assignment
- per-cluster montages for top membership images
- augmentation stability report
- uncertainty report for high-entropy images
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "umap-learn"])
    import umap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent


def load_index(index_path: Path) -> List[Dict[str, str]]:
    with index_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compute_fcm_membership(data: np.ndarray, centers: np.ndarray, m: float) -> np.ndarray:
    # data shape: (n_samples, n_features), centers shape: (c, n_features)
    eps = np.finfo(float).eps
    dist = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
    dist = np.maximum(dist, eps)
    exponent = 2.0 / (m - 1.0)
    inv = dist[:, :, None] / dist[:, None, :]
    inv = np.power(inv, exponent)
    denominator = np.sum(inv, axis=2)
    U = 1.0 / denominator
    return U


def compute_entropy(U: np.ndarray) -> np.ndarray:
    eps = np.finfo(float).eps
    U_safe = np.clip(U, eps, 1.0)
    entropy = -np.sum(U_safe * np.log(U_safe), axis=1)
    norm = math.log(U.shape[1]) if U.shape[1] > 1 else 1.0
    return entropy / norm


def create_scatter(points: np.ndarray, labels: Sequence, title: str, path: Path, cmap: str = "tab10") -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(set(labels), key=lambda x: str(x))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    color_idx = [label_to_idx[label] for label in labels]
    scatter = ax.scatter(points[:, 0], points[:, 1], c=color_idx, cmap=cmap, s=8, alpha=0.75)
    handles = []
    for i, lab in enumerate(unique_labels):
        handles.append(plt.Line2D([0], [0], marker="o", color="w", label=str(lab), markerfacecolor=plt.cm.get_cmap(cmap)(i % 10), markersize=8))
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def montage_image(paths: Sequence[Path], out_path: Path, thumb_size: Tuple[int, int] = (224, 224), grid_size: Tuple[int, int] = (5, 2)) -> None:
    cols, rows = grid_size
    width, height = thumb_size
    montage = Image.new("RGB", (cols * width, rows * height), color=(255, 255, 255))
    for idx, img_path in enumerate(paths[: cols * rows]):
        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((width, height), Image.LANCZOS)
            x = (idx % cols) * width
            y = (idx // cols) * height
            montage.paste(img, (x, y))
        except Exception as exc:
            print(f"WARNING: failed to open {img_path}: {exc}")
    montage.save(out_path)


def save_csv(rows: List[Dict[str, object]], path: Path, fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    feature_raw_path = ROOT / "features" / "feature_matrix_raw.npy"
    feature_index_path = ROOT / "features" / "feature_index.csv"
    centers_path = ROOT / "features" / "fcm_results" / "fcm_c5" / "cluster_centers.npy"
    membership_path = ROOT / "features" / "fcm_results" / "fcm_c5" / "membership_matrix.npy"
    output_dir = ROOT / "features" / "fcm_results" / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    index_rows = load_index(feature_index_path)
    raw_features = np.load(feature_raw_path)
    centers = np.load(centers_path)
    membership = np.load(membership_path)

    originals_mask = np.array([row["is_original"].strip().lower() == "true" for row in index_rows])
    if membership.shape[1] != originals_mask.sum():
        raise ValueError("membership count does not match originals count")

    original_indices = np.nonzero(originals_mask)[0]
    original_rows = [index_rows[i] for i in original_indices]
    original_raw = raw_features[original_indices]

    # Fit an originals-only projection model to embed all images in same PCA space.
    scaler = StandardScaler().fit(original_raw)
    original_scaled = scaler.transform(original_raw)
    from sklearn.decomposition import PCA

    pca = PCA(n_components=centers.shape[1], random_state=42).fit(original_scaled)
    all_scaled = scaler.transform(raw_features)
    all_proj = pca.transform(all_scaled)
    print("Projected all images into originals PCA space", all_proj.shape)

    # Compute membership for all images against original FCM centers.
    all_membership = compute_fcm_membership(all_proj, centers, m=2.0)
    all_hard = np.argmax(all_membership, axis=1)
    all_entropy = compute_entropy(all_membership)

    # t-SNE and UMAP on originals only
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
    originals_emb = tsne.fit_transform(original_proj := pca.transform(original_scaled))
    save_path = output_dir / "tsne_true_class.png"
    create_scatter(originals_emb, [row["class_label"] for row in original_rows], "t-SNE: true class (originals)", save_path)
    save_path = output_dir / "tsne_cluster_assignment.png"
    create_scatter(originals_emb, [int(np.argmax(membership[:, i])) for i in range(membership.shape[1])], "t-SNE: cluster assignment (originals)", save_path)

    umap_emb = umap.UMAP(n_components=2, random_state=42).fit_transform(original_proj)
    save_path = output_dir / "umap_true_class.png"
    create_scatter(umap_emb, [row["class_label"] for row in original_rows], "UMAP: true class (originals)", save_path)
    save_path = output_dir / "umap_cluster_assignment.png"
    create_scatter(umap_emb, [int(np.argmax(membership[:, i])) for i in range(membership.shape[1])], "UMAP: cluster assignment (originals)", save_path)

    # Per-cluster montages from originals
    clusters = sorted(set(int(np.argmax(membership[:, i])) for i in range(membership.shape[1])))
    for cluster_id in clusters:
        cluster_members = [i for i in range(membership.shape[1])]
        memberships = membership[cluster_id]
        sorted_idx = np.argsort(-memberships)
        top_idx = sorted_idx[:10]
        top_paths = []
        for idx in top_idx:
            original_row = original_rows[idx]
            top_paths.append(ROOT / original_row["output_path"])
        montage_image(top_paths, output_dir / f"cluster_{cluster_id}_top10.png")

    # Augmentation stability
    group_by_parent: Dict[str, List[int]] = defaultdict(list)
    for i, row in enumerate(index_rows):
        parent = row["parent_filename"].strip()
        if parent:
            group_by_parent[parent].append(i)

    stability_rows = []
    stable_count = 0
    for parent, idxs in sorted(group_by_parent.items()):
        hard_labels = [int(all_hard[i]) for i in idxs]
        unique_clusters = sorted(set(hard_labels))
        stable = len(unique_clusters) == 1
        if stable:
            stable_count += 1
        stability_rows.append({
            "parent_filename": parent,
            "n_children": len(idxs),
            "clusters": ";".join(str(c) for c in unique_clusters),
            "stable": stable,
        })
    save_csv(stability_rows, output_dir / "augmentation_stability.csv", ["parent_filename", "n_children", "clusters", "stable"])
    with (output_dir / "augmentation_stability_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "total_parents": len(stability_rows),
            "stable_parents": stable_count,
            "unstable_parents": len(stability_rows) - stable_count,
            "stable_pct": stable_count / len(stability_rows) if stability_rows else 0,
        }, handle, indent=2)

    # High entropy images
    entropy_rows: List[Dict[str, object]] = []
    for idx, row in enumerate(index_rows):
        entropy_rows.append({
            "feature_row": idx,
            "source_id": row["source_id"],
            "class_label": row["class_label"],
            "output_path": row["output_path"],
            "hard_label": int(all_hard[idx]),
            "entropy": float(all_entropy[idx]),
        })
    entropy_rows.sort(key=lambda x: -x["entropy"])
    save_csv(entropy_rows[:200], output_dir / "entropy_highest_200.csv", ["feature_row", "source_id", "class_label", "output_path", "hard_label", "entropy"])

    high_entropy_paths = [ROOT / row["output_path"] for row in entropy_rows[:20]]
    montage_image(high_entropy_paths, output_dir / "high_entropy_top20.png")

    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "tsne_path": str(output_dir / "tsne_true_class.png"),
            "tsne_cluster_path": str(output_dir / "tsne_cluster_assignment.png"),
            "umap_path": str(output_dir / "umap_true_class.png"),
            "umap_cluster_path": str(output_dir / "umap_cluster_assignment.png"),
            "cluster_montages": [str(output_dir / f"cluster_{cluster_id}_top10.png") for cluster_id in clusters],
            "augmentation_stability_csv": str(output_dir / "augmentation_stability.csv"),
            "entropy_csv": str(output_dir / "entropy_highest_200.csv"),
            "high_entropy_montage": str(output_dir / "high_entropy_top20.png"),
            "n_parents": len(stability_rows),
            "n_stable_parents": stable_count,
            "stable_percent": stable_count / len(stability_rows) if stability_rows else 0,
        }, handle, indent=2)

    print("Evaluation artifacts saved to", output_dir)


if __name__ == "__main__":
    main()
