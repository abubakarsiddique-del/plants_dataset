#!/usr/bin/env python3
"""
Hybrid feature extraction pipeline (Option C) for fuzzy C-means clustering.

Combines frozen deep embeddings (ResNet18) with hand-crafted color/texture/shape
features extracted from letterboxed 640x640 images, then applies StandardScaler
+ PCA to produce fixed-length vectors for FCM.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from torchvision import models, transforms
from tqdm import tqdm

from augmentation_pipeline import (
    PROJECT_ROOT,
    RANDOM_SEED,
    WORKING_SIZE,
    letterbox_standardize,
    load_rgb_image,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MANIFEST_PATH = PROJECT_ROOT / "logs" / "master_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "features"
LOG_DIR = PROJECT_ROOT / "logs"

DEEP_BACKBONE = "resnet18"
DEEP_EMBED_DIM = 512
PCA_COMPONENTS = 128
BATCH_SIZE = 32
CHECKPOINT_INTERVAL = 256
CLASSICAL_TEXTURE_SIZE = (256, 256)
RANDOM_STATE = RANDOM_SEED

HSV_BINS = (8, 8, 8)
LAB_BINS = (8, 8, 8)
GLCM_DISTANCES = (1,)
GLCM_ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GLCM_LEVELS = 32
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_BINS = 26  # uniform LBP for P=8

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASSICAL_FEATURE_GROUPS = {
    "hsv_histogram": int(np.prod(HSV_BINS)),
    "lab_histogram": int(np.prod(LAB_BINS)),
    "glcm": 4,  # contrast, dissimilarity, homogeneity, energy (mean over angles)
    "lbp_histogram": LBP_BINS,
    "hu_moments": 7,
    "lesion_stats": 6,
}


@dataclass
class ExtractionStats:
    rows_processed: int = 0
    rows_failed: int = 0
    start_time: float = field(default_factory=time.time)

    def elapsed(self) -> float:
        return time.time() - self.start_time


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("feature_extraction")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_dir / "feature_extraction.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def classical_feature_dim() -> int:
    return sum(CLASSICAL_FEATURE_GROUPS.values())


def load_manifest_rows(manifest_path: Path, limit: Optional[int] = None) -> List[Dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        rows = rows[:limit]
    return rows


def load_letterboxed_rgb(row: Dict[str, str], project_root: Path) -> np.ndarray:
    """Return a 640x640 RGB uint8 array (letterboxed for originals)."""
    path = project_root / row["output_path"]
    rgb = load_rgb_image(path)

    if row["is_original"].lower() == "true":
        rgb = letterbox_standardize(rgb, WORKING_SIZE)
    else:
        target_w, target_h = WORKING_SIZE
        h, w = rgb.shape[:2]
        if (w, h) != (target_w, target_h):
            rgb = letterbox_standardize(rgb, WORKING_SIZE)

    if rgb.shape[0] != WORKING_SIZE[1] or rgb.shape[1] != WORKING_SIZE[0]:
        raise ValueError(
            f"Expected letterboxed size {WORKING_SIZE}, got {(rgb.shape[1], rgb.shape[0])} for {path}"
        )
    return rgb


def resize_for_classical(rgb: np.ndarray) -> np.ndarray:
    """Downsample letterboxed image for faster texture/shape descriptors."""
    tw, th = CLASSICAL_TEXTURE_SIZE
    if rgb.shape[1] == tw and rgb.shape[0] == th:
        return rgb
    return cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)


def normalized_histogram(channel: np.ndarray, bins: int) -> np.ndarray:
    hist, _ = np.histogram(channel.ravel(), bins=bins, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def extract_color_histograms(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    hsv_values = np.stack(
        [hsv[:, :, 0].ravel(), hsv[:, :, 1].ravel(), hsv[:, :, 2].ravel()],
        axis=1,
    )
    lab_values = np.stack(
        [lab[:, :, 0].ravel(), lab[:, :, 1].ravel(), lab[:, :, 2].ravel()],
        axis=1,
    )

    hsv_hist, _ = np.histogramdd(
        hsv_values,
        bins=HSV_BINS,
        range=((0, 180), (0, 256), (0, 256)),
    )
    lab_hist, _ = np.histogramdd(
        lab_values,
        bins=LAB_BINS,
        range=((0, 256), (0, 256), (0, 256)),
    )

    hsv_hist = hsv_hist.astype(np.float64).ravel()
    lab_hist = lab_hist.astype(np.float64).ravel()
    total = hsv_hist.sum()
    if total > 0:
        hsv_hist /= total
    total = lab_hist.sum()
    if total > 0:
        lab_hist /= total

    return np.concatenate([hsv_hist, lab_hist])


def extract_glcm_features(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    quantized = (gray.astype(np.float32) / 256.0 * GLCM_LEVELS).astype(np.uint8)
    quantized = np.clip(quantized, 0, GLCM_LEVELS - 1)

    matrix = graycomatrix(
        quantized,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    contrast = graycoprops(matrix, "contrast").mean()
    dissimilarity = graycoprops(matrix, "dissimilarity").mean()
    homogeneity = graycoprops(matrix, "homogeneity").mean()
    energy = graycoprops(matrix, "energy").mean()
    return np.array([contrast, dissimilarity, homogeneity, energy], dtype=np.float64)


def extract_lbp_histogram(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=LBP_POINTS, R=LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=LBP_BINS,
        range=(0, LBP_BINS),
        density=True,
    )
    return hist.astype(np.float64)


def extract_hu_moments(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    # Log-transform for scale invariance; sign preserved
    with np.errstate(divide="ignore", invalid="ignore"):
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu.astype(np.float64)


def extract_lesion_stats(rgb: np.ndarray) -> np.ndarray:
    """
    Heuristic leaf/lesion descriptors using HSV green segmentation.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Broad green vegetation mask in OpenCV HSV (H: 0-179)
    green_mask = cv2.inRange(hsv, (25, 30, 30), (95, 255, 255))
    leaf_ratio = green_mask.mean() / 255.0
    lesion_ratio = 1.0 - leaf_ratio

    leaf_pixels = green_mask > 0
    if leaf_pixels.any():
        mean_sat = float(s[leaf_pixels].mean())
        std_hue = float(h[leaf_pixels].std())
        std_sat = float(s[leaf_pixels].std())
        mean_val = float(v[leaf_pixels].mean())
    else:
        mean_sat = float(s.mean())
        std_hue = float(h.std())
        std_sat = float(s.std())
        mean_val = float(v.mean())

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(edges.mean() / 255.0)

    return np.array(
        [leaf_ratio, lesion_ratio, mean_sat, std_hue, std_sat, edge_density],
        dtype=np.float64,
    )


def extract_classical_features(rgb: np.ndarray) -> np.ndarray:
    classical_rgb = resize_for_classical(rgb)
    parts = [
        extract_color_histograms(classical_rgb),
        extract_glcm_features(classical_rgb),
        extract_lbp_histogram(classical_rgb),
        extract_hu_moments(classical_rgb),
        extract_lesion_stats(classical_rgb),
    ]
    vector = np.concatenate(parts).astype(np.float32)
    expected = classical_feature_dim()
    if vector.shape[0] != expected:
        raise ValueError(f"Classical feature dim mismatch: got {vector.shape[0]}, expected {expected}")
    return vector


class DeepEmbeddingModel(nn.Module):
    def __init__(self, backbone_name: str = DEEP_BACKBONE) -> None:
        super().__init__()
        if backbone_name != "resnet18":
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        weights = models.ResNet18_Weights.IMAGENET1K_V1
        backbone = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.out_dim = 512
        self.eval()

        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    @torch.inference_mode()
    def forward_batch(self, rgb_batch: Sequence[np.ndarray], device: torch.device) -> np.ndarray:
        tensors = [self.preprocess(Image.fromarray(rgb)) for rgb in rgb_batch]
        batch = torch.stack(tensors).to(device)
        embeddings = self.features(batch).flatten(1)
        return embeddings.cpu().numpy().astype(np.float32)


def iter_batches(items: Sequence, batch_size: int) -> Iterator[Tuple[int, Sequence]]:
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def build_feature_index_row(row: Dict[str, str], feature_row: int) -> Dict[str, object]:
    return {
        "feature_row": feature_row,
        "manifest_row": row.get("manifest_row", ""),
        "source_id": row["source_id"],
        "class_label": row["class_label"],
        "is_original": row["is_original"],
        "augmentation_type": row["augmentation_type"],
        "augmentation_slug": row["augmentation_slug"],
        "parent_filename": row["parent_filename"],
        "filename": row["filename"],
        "output_path": row["output_path"],
        "sha256": row["sha256"],
    }


def checkpoint_dir_for(output_dir: Path) -> Path:
    return output_dir / "checkpoint"


def save_checkpoint(
    checkpoint_dir: Path,
    *,
    next_manifest_idx: int,
    raw_features: np.ndarray,
    feature_index_rows: List[Dict[str, object]],
    failed_rows: List[Dict[str, str]],
    stats: ExtractionStats,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    np.save(checkpoint_dir / "raw_features.npy", raw_features)
    index_path = checkpoint_dir / "feature_index.csv"
    if feature_index_rows:
        with index_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(feature_index_rows[0].keys()))
            writer.writeheader()
            writer.writerows(feature_index_rows)
    if failed_rows:
        with (checkpoint_dir / "failed_rows.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["manifest_row", "output_path", "error"])
            writer.writeheader()
            writer.writerows(failed_rows)
    state = {
        "next_manifest_idx": next_manifest_idx,
        "rows_processed": stats.rows_processed,
        "rows_failed": stats.rows_failed,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (checkpoint_dir / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def load_checkpoint(checkpoint_dir: Path) -> Tuple[int, np.ndarray, List[Dict[str, object]], List[Dict[str, str]], ExtractionStats]:
    state_path = checkpoint_dir / "state.json"
    if not state_path.exists():
        return 0, np.empty((0, 0), dtype=np.float32), [], [], ExtractionStats()

    state = json.loads(state_path.read_text(encoding="utf-8"))
    raw_features = np.load(checkpoint_dir / "raw_features.npy")
    feature_index_rows: List[Dict[str, object]] = []
    index_path = checkpoint_dir / "feature_index.csv"
    if index_path.exists():
        with index_path.open(newline="", encoding="utf-8") as handle:
            feature_index_rows = list(csv.DictReader(handle))

    failed_rows: List[Dict[str, str]] = []
    failed_path = checkpoint_dir / "failed_rows.csv"
    if failed_path.exists():
        with failed_path.open(newline="", encoding="utf-8") as handle:
            failed_rows = list(csv.DictReader(handle))

    stats = ExtractionStats(
        rows_processed=int(state.get("rows_processed", len(feature_index_rows))),
        rows_failed=int(state.get("rows_failed", len(failed_rows))),
    )
    return int(state["next_manifest_idx"]), raw_features, feature_index_rows, failed_rows, stats


def clear_checkpoint(checkpoint_dir: Path) -> None:
    if checkpoint_dir.exists():
        for path in checkpoint_dir.iterdir():
            path.unlink()
        checkpoint_dir.rmdir()


def run_extraction(
    *,
    manifest_path: Path = MANIFEST_PATH,
    output_dir: Path = OUTPUT_DIR,
    project_root: Path = PROJECT_ROOT,
    pca_components: int = PCA_COMPONENTS,
    batch_size: int = BATCH_SIZE,
    device_name: str = "auto",
    limit: Optional[int] = None,
    resume: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, object]:
    logger = logger or setup_logging(LOG_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = checkpoint_dir_for(output_dir)

    device = resolve_device(device_name)
    logger.info("Using device: %s", device)
    logger.info("PyTorch %s | scikit-learn PCA target dim: %d", torch.__version__, pca_components)

    manifest_rows = load_manifest_rows(manifest_path, limit=limit)
    for idx, row in enumerate(manifest_rows):
        row["manifest_row"] = str(idx)
    logger.info("Loaded %d manifest rows from %s", len(manifest_rows), manifest_path)

    deep_model = DeepEmbeddingModel(DEEP_BACKBONE).to(device)

    start_idx = 0
    stats = ExtractionStats()
    raw_feature_list: List[np.ndarray] = []
    feature_index_rows: List[Dict[str, object]] = []
    failed_rows: List[Dict[str, str]] = []

    if resume:
        start_idx, raw_ckpt, feature_index_rows, failed_rows, stats = load_checkpoint(ckpt_dir)
        if start_idx > 0:
            raw_feature_list = [raw_ckpt[i] for i in range(raw_ckpt.shape[0])]
            logger.info(
                "Resuming from manifest row %d (%d features already extracted)",
                start_idx,
                len(raw_feature_list),
            )

    classical_dim = classical_feature_dim()
    raw_dim = DEEP_EMBED_DIM + classical_dim
    n_rows = len(manifest_rows)

    pending_indices: List[int] = []
    pending_rgb: List[np.ndarray] = []
    pending_manifest_rows: List[Dict[str, str]] = []
    images_since_checkpoint = 0

    def flush_batch() -> None:
        nonlocal pending_indices, pending_rgb, pending_manifest_rows, images_since_checkpoint
        if not pending_rgb:
            return

        classical_batch = np.stack(
            [extract_classical_features(rgb) for rgb in pending_rgb],
            axis=0,
        )
        deep_batch = deep_model.forward_batch(pending_rgb, device)
        hybrid_batch = np.concatenate([deep_batch, classical_batch], axis=1)

        for local_idx, _manifest_idx in enumerate(pending_indices):
            raw_feature_list.append(hybrid_batch[local_idx])
            feature_index_rows.append(
                build_feature_index_row(pending_manifest_rows[local_idx], len(raw_feature_list) - 1)
            )
            stats.rows_processed += 1

        images_since_checkpoint += len(pending_rgb)
        pending_indices = []
        pending_rgb = []
        pending_manifest_rows = []

    def maybe_checkpoint(next_manifest_idx: int) -> None:
        nonlocal images_since_checkpoint
        if images_since_checkpoint < CHECKPOINT_INTERVAL:
            return
        raw_arr = np.stack(raw_feature_list, axis=0).astype(np.float32)
        save_checkpoint(
            ckpt_dir,
            next_manifest_idx=next_manifest_idx,
            raw_features=raw_arr,
            feature_index_rows=feature_index_rows,
            failed_rows=failed_rows,
            stats=stats,
        )
        logger.info(
            "Checkpoint saved at manifest row %d (%d features)",
            next_manifest_idx,
            len(raw_feature_list),
        )
        images_since_checkpoint = 0

    progress = tqdm(
        total=n_rows - start_idx,
        initial=0,
        desc="Extracting hybrid features",
        unit="img",
    )
    next_manifest_idx = start_idx
    try:
        for row_idx in range(start_idx, n_rows):
            row = manifest_rows[row_idx]
            try:
                rgb = load_letterboxed_rgb(row, project_root)
                pending_indices.append(row_idx)
                pending_rgb.append(rgb)
                pending_manifest_rows.append(row)

                if len(pending_rgb) >= batch_size:
                    flush_batch()
                    maybe_checkpoint(row_idx + 1)
            except Exception as exc:
                stats.rows_failed += 1
                failed_rows.append(
                    {
                        "manifest_row": str(row_idx),
                        "output_path": row.get("output_path", ""),
                        "error": str(exc),
                    }
                )
                logger.error("Failed row %d (%s): %s", row_idx, row.get("output_path"), exc)
            next_manifest_idx = row_idx + 1
            progress.update(1)

        flush_batch()
        maybe_checkpoint(next_manifest_idx)
    finally:
        progress.close()

    if stats.rows_processed == 0:
        raise RuntimeError("No features extracted successfully")

    raw_features = np.stack(raw_feature_list, axis=0).astype(np.float32)

    logger.info("Fitting StandardScaler on %d samples x %d raw features", stats.rows_processed, raw_dim)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw_features)

    n_components = min(pca_components, scaled.shape[0], scaled.shape[1])
    logger.info("Fitting PCA with n_components=%d", n_components)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(scaled).astype(np.float32)

    # Persist artifacts
    np.save(output_dir / "feature_matrix.npy", reduced)
    np.save(output_dir / "feature_matrix_raw.npy", raw_features)
    np.save(output_dir / "feature_matrix_scaled.npy", scaled.astype(np.float32))

    joblib.dump(scaler, output_dir / "scaler.joblib")
    joblib.dump(pca, output_dir / "pca.joblib")

    index_path = output_dir / "feature_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(feature_index_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_index_rows)

    if failed_rows:
        failed_path = output_dir / "failed_rows.csv"
        with failed_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["manifest_row", "output_path", "error"])
            writer.writeheader()
            writer.writerows(failed_rows)

    clear_checkpoint(ckpt_dir)

    explained = pca.explained_variance_ratio_
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "strategy": "hybrid_option_c",
        "device": str(device),
        "config": {
            "deep_backbone": DEEP_BACKBONE,
            "deep_embed_dim": DEEP_EMBED_DIM,
            "classical_feature_dim": classical_dim,
            "raw_hybrid_dim": raw_dim,
            "pca_components": n_components,
            "batch_size": batch_size,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "classical_texture_size": list(CLASSICAL_TEXTURE_SIZE),
            "working_size": list(WORKING_SIZE),
            "random_state": RANDOM_STATE,
            "classical_feature_groups": CLASSICAL_FEATURE_GROUPS,
        },
        "totals": {
            "manifest_rows_requested": n_rows,
            "rows_processed": stats.rows_processed,
            "rows_failed": stats.rows_failed,
            "final_feature_shape": list(reduced.shape),
        },
        "pca": {
            "explained_variance_ratio_sum": float(explained.sum()),
            "explained_variance_ratio_top10": [float(v) for v in explained[:10]],
            "n_components": n_components,
        },
        "artifacts": {
            "feature_matrix": str(output_dir / "feature_matrix.npy"),
            "feature_matrix_raw": str(output_dir / "feature_matrix_raw.npy"),
            "feature_matrix_scaled": str(output_dir / "feature_matrix_scaled.npy"),
            "feature_index_csv": str(index_path),
            "scaler": str(output_dir / "scaler.joblib"),
            "pca": str(output_dir / "pca.joblib"),
        },
        "runtime_sec": round(stats.elapsed(), 2),
    }

    summary_path = output_dir / "extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("=" * 60)
    logger.info("FEATURE EXTRACTION COMPLETE")
    logger.info("Processed: %d | Failed: %d", stats.rows_processed, stats.rows_failed)
    logger.info("Raw dim: %d -> PCA dim: %d", raw_dim, n_components)
    logger.info("PCA explained variance (sum): %.4f", explained.sum())
    logger.info("Feature matrix: %s", output_dir / "feature_matrix.npy")
    logger.info("Summary: %s", summary_path)
    logger.info("Runtime: %.2f sec", stats.elapsed())

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid feature extraction for FCM")
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--pca-components", type=int, default=PCA_COMPONENTS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--limit", type=int, default=None, help="Process first N manifest rows")
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoint and start fresh")
    args = parser.parse_args()

    logger = setup_logging(LOG_DIR)
    summary = run_extraction(
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        pca_components=args.pca_components,
        batch_size=args.batch_size,
        device_name=args.device,
        limit=args.limit,
        resume=not args.no_resume,
        logger=logger,
    )
    if summary["totals"]["rows_failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
