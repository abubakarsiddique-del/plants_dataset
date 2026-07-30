#!/usr/bin/env python3
"""
Production augmentation pipeline for the Malabar plant leaf disease dataset.

Preprocesses all images with letterbox standardization, applies 12 isolated
Albumentations transforms per image, saves individual augmented files for
downstream fuzzy clustering, and builds labeled review grids for QA.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Step 1 — Config block (all tunables)
# ---------------------------------------------------------------------------
INPUT_DIR = "Malabar_Dataset"
OUTPUT_AUGMENTED_DIR = "augmented_dataset"
OUTPUT_GRIDS_DIR = "augmentation_review_grids"
LOG_DIR = "logs"
WORKING_SIZE = (640, 640)
THUMB_SIZE = (300, 300)
GRID_COLS = 4
JPEG_QUALITY = 90
RANDOM_SEED = 42

EXPECTED_CLASS_COUNTS: Dict[str, int] = {
    "Anthracnose(102)": 102,
    "Bacterial-Spot(752)": 752,
    "Downy-Mildew(240)": 240,
    "Healthy-Leaf(1399)": 1399,
    "Pest-Damage(513)": 513,
}
EXPECTED_TOTAL_IMAGES = 3006
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AugmentationSpec:
    index: int
    name: str
    slug: str
    transform: A.BasicTransform


@dataclass
class RunStats:
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    corrupt: int = 0
    per_class_processed: Dict[str, int] = field(default_factory=dict)
    per_class_skipped: Dict[str, int] = field(default_factory=dict)
    per_class_failed: Dict[str, int] = field(default_factory=dict)
    total_augmented_files: int = 0
    total_review_grids: int = 0
    start_time: float = field(default_factory=time.time)

    def elapsed(self) -> float:
        return time.time() - self.start_time


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("augmentation_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_dir / "pipeline.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def print_dependency_versions(logger: logging.Logger) -> None:
    import albumentations
    import cv2 as cv2_mod
    import numpy as np_mod
    from PIL import __version__ as pil_version

    versions = {
        "albumentations": albumentations.__version__,
        "opencv-python-headless": cv2_mod.__version__,
        "pillow": pil_version,
        "numpy": np_mod.__version__,
        "tqdm": __import__("tqdm").__version__,
    }
    logger.info("Dependency versions:")
    for name, version in versions.items():
        logger.info("  %s: %s", name, version)


def resolve_input_dir(project_root: Path, input_dir: str) -> Path:
    """Auto-detect dataset root if nested one level deeper."""
    candidates = [
        project_root / input_dir,
        project_root / input_dir / input_dir,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate

    for path in project_root.rglob(input_dir):
        if path.is_dir():
            return path

    raise FileNotFoundError(
        f"Could not locate dataset directory '{input_dir}' under {project_root}"
    )


def discover_class_folders(input_path: Path) -> List[Path]:
    folders = sorted(
        p for p in input_path.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    return folders


def list_image_files(class_folder: Path) -> List[Path]:
    files = [
        p
        for p in class_folder.iterdir()
        if p.is_file() and p.suffix in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def validate_dataset(
    input_path: Path,
    logger: logging.Logger,
    corrupt_log: Path,
) -> Tuple[List[Tuple[str, Path]], List[dict]]:
    """
    Walk dataset, validate counts, probe readability.
    Returns list of (class_name, image_path) and corrupt file records.
    """
    corrupt_records: List[dict] = []
    dataset_entries: List[Tuple[str, Path]] = []

    class_folders = discover_class_folders(input_path)
    logger.info("Discovered %d class folders under %s", len(class_folders), input_path)

    if len(class_folders) != len(EXPECTED_CLASS_COUNTS):
        logger.warning(
            "Expected %d class folders, found %d",
            len(EXPECTED_CLASS_COUNTS),
            len(class_folders),
        )

    for folder in class_folders:
        class_name = folder.name
        images = list_image_files(folder)
        expected = EXPECTED_CLASS_COUNTS.get(class_name)
        if expected is not None and len(images) != expected:
            logger.warning(
                "Class %s: expected %d images, found %d",
                class_name,
                expected,
                len(images),
            )
        else:
            logger.info("Class %s: %d images (OK)", class_name, len(images))

        for image_path in images:
            try:
                with Image.open(image_path) as img:
                    img.verify()
                with Image.open(image_path) as img:
                    img.convert("RGB").load()
            except Exception as exc:
                corrupt_records.append(
                    {
                        "class": class_name,
                        "filename": image_path.name,
                        "path": str(image_path),
                        "error": str(exc),
                    }
                )
                logger.warning("Corrupt/unreadable: %s (%s)", image_path, exc)
                continue

            dataset_entries.append((class_name, image_path))

    if corrupt_records:
        corrupt_log.parent.mkdir(parents=True, exist_ok=True)
        with corrupt_log.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["class", "filename", "path", "error"]
            )
            writer.writeheader()
            writer.writerows(corrupt_records)

    logger.info(
        "Validation complete: %d readable images, %d corrupt",
        len(dataset_entries),
        len(corrupt_records),
    )
    if len(dataset_entries) != EXPECTED_TOTAL_IMAGES - len(corrupt_records):
        logger.warning(
            "Readable image count %d differs from expected total %d",
            len(dataset_entries),
            EXPECTED_TOTAL_IMAGES,
        )

    return dataset_entries, corrupt_records


def per_image_seed(filename: str, base_seed: int = RANDOM_SEED) -> int:
    digest = hashlib.md5(filename.encode("utf-8")).hexdigest()
    return base_seed + (int(digest, 16) % 100_000)


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))


def letterbox_standardize(
    image_rgb: np.ndarray,
    target_size: Tuple[int, int] = WORKING_SIZE,
    fill: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    target_w, target_h = target_size
    h, w = image_rgb.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), fill, dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
    return np.array(rgb, dtype=np.uint8)


def build_augmentation_specs() -> List[AugmentationSpec]:
    specs = [
        AugmentationSpec(1, "Horizontal Flip", "horizontal_flip", A.HorizontalFlip(p=1.0)),
        AugmentationSpec(2, "Vertical Flip", "vertical_flip", A.VerticalFlip(p=1.0)),
        AugmentationSpec(
            3,
            "Rotation",
            "rotation",
            A.Rotate(limit=40, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        ),
        AugmentationSpec(
            4,
            "Zoom / Scale",
            "zoom_scale",
            A.Affine(scale=(0.75, 1.25), p=1.0),
        ),
        AugmentationSpec(
            5,
            "Brightness",
            "brightness",
            A.RandomBrightnessContrast(
                brightness_limit=0.4, contrast_limit=0.0, p=1.0
            ),
        ),
        AugmentationSpec(
            6,
            "Contrast",
            "contrast",
            A.RandomBrightnessContrast(
                brightness_limit=0.0, contrast_limit=0.4, p=1.0
            ),
        ),
        AugmentationSpec(
            7,
            "Hue/Saturation Jitter",
            "hue_saturation",
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=0, p=1.0
            ),
        ),
        AugmentationSpec(
            8,
            "Gaussian Blur",
            "gaussian_blur",
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ),
        AugmentationSpec(
            9,
            "Gaussian Noise",
            "gaussian_noise",
            # Albumentations 2.x: std_range is normalized to [0, 1] (~sigma 10–25 on 0–255)
            A.GaussNoise(
                std_range=(10.0 / 255.0, 25.0 / 255.0),
                mean_range=(0.0, 0.0),
                p=1.0,
            ),
        ),
        AugmentationSpec(
            10,
            "Shear",
            "shear",
            A.Affine(shear=(-15, 15), p=1.0),
        ),
        AugmentationSpec(
            11,
            "CLAHE",
            "clahe",
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        ),
        AugmentationSpec(
            12,
            "Translation/Shift",
            "translation",
            A.Affine(translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, p=1.0),
        ),
    ]
    return specs


def apply_transform(
    image: np.ndarray, transform: A.BasicTransform, seed: int
) -> np.ndarray:
    set_deterministic_seed(seed)
    pipeline = A.Compose([transform])
    result = pipeline(image=image)["image"]
    return np.clip(result, 0, 255).astype(np.uint8)


def save_jpeg(image_rgb: np.ndarray, path: Path, quality: int = JPEG_QUALITY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def load_font(size: int = 16) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for font_path in candidates:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def resize_thumb(image_rgb: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image_rgb, size, interpolation=cv2.INTER_AREA)


def build_review_grid(
    original: np.ndarray,
    augmented: Sequence[Tuple[str, np.ndarray]],
    class_name: str,
    filename: str,
    thumb_size: Tuple[int, int] = THUMB_SIZE,
    grid_cols: int = GRID_COLS,
) -> Image.Image:
    label_height = 28
    header_height = 36
    border = 3
    pad = 8
    thumb_w, thumb_h = thumb_size

    cells: List[Tuple[str, np.ndarray, bool]] = [("ORIGINAL", original, True)]
    cells.extend((name, img, False) for name, img in augmented)

    total_cells = grid_cols * int(np.ceil(len(cells) / grid_cols))
    while len(cells) < total_cells:
        cells.append(("", np.full((thumb_h, thumb_w, 3), 240, dtype=np.uint8), False))

    rows = int(np.ceil(len(cells) / grid_cols))
    grid_w = grid_cols * thumb_w + (grid_cols + 1) * pad
    grid_h = header_height + rows * (thumb_h + label_height) + (rows + 1) * pad

    canvas = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(18)
    label_font = load_font(14)

    header_text = f"{class_name} / {filename}"
    draw.rectangle([0, 0, grid_w, header_height], fill=(30, 30, 30))
    draw.text((pad, 8), header_text, fill=(255, 255, 255), font=title_font)

    for idx, (label, image_rgb, is_original) in enumerate(cells):
        row = idx // grid_cols
        col = idx % grid_cols
        x = pad + col * (thumb_w + pad)
        y = header_height + pad + row * (thumb_h + label_height + pad)

        thumb = resize_thumb(image_rgb, thumb_size)
        pil_thumb = Image.fromarray(thumb)
        canvas.paste(pil_thumb, (x, y))

        outline = (220, 50, 50) if is_original else (180, 180, 180)
        width = border + 1 if is_original else border
        draw.rectangle(
            [x - 1, y - 1, x + thumb_w, y + thumb_h],
            outline=outline,
            width=width,
        )

        if label:
            text_y = y + thumb_h + 4
            draw.text((x, text_y), label, fill=(20, 20, 20), font=label_font)

    return canvas


def augmented_output_path(
    output_root: Path,
    class_name: str,
    stem: str,
    spec: AugmentationSpec,
) -> Path:
    return (
        output_root
        / class_name
        / f"{stem}_aug{spec.index:02d}_{spec.slug}.jpg"
    )


def review_grid_path(output_root: Path, class_name: str, stem: str) -> Path:
    return output_root / class_name / f"{stem}_review.jpg"


def process_single_image(
    class_name: str,
    image_path: Path,
    specs: Sequence[AugmentationSpec],
    augmented_dir: Path,
    grids_dir: Path,
) -> Tuple[str, float, Optional[str], int]:
    """
    Process one image. Returns (status, elapsed_seconds, error_message, num_saved_aug).
    status: 'processed' | 'skipped'
    """
    stem = image_path.stem
    review_path = review_grid_path(grids_dir, class_name, stem)
    if review_path.exists():
        return "skipped", 0.0, None, 0

    start = time.time()
    seed_base = per_image_seed(image_path.name)

    rgb = load_rgb_image(image_path)
    standardized = letterbox_standardize(rgb, WORKING_SIZE)

    augmented_outputs: List[Tuple[str, np.ndarray]] = []
    for spec in specs:
        aug_image = apply_transform(standardized, spec.transform, seed_base + spec.index)
        out_path = augmented_output_path(augmented_dir, class_name, stem, spec)
        save_jpeg(aug_image, out_path)
        augmented_outputs.append((spec.name, aug_image))

    grid = build_review_grid(
        standardized,
        augmented_outputs,
        class_name,
        image_path.name,
        THUMB_SIZE,
        GRID_COLS,
    )
    review_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(review_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)

    elapsed = time.time() - start
    return "processed", elapsed, None, len(specs)


def increment_counter(counter: Dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


def format_bytes(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"


def run_pipeline(
    limit: Optional[int] = None,
    smoke_test: bool = False,
) -> RunStats:
    logger = setup_logging(PROJECT_ROOT / LOG_DIR)
    print_dependency_versions(logger)

    input_path = resolve_input_dir(PROJECT_ROOT, INPUT_DIR)
    augmented_dir = PROJECT_ROOT / OUTPUT_AUGMENTED_DIR
    grids_dir = PROJECT_ROOT / OUTPUT_GRIDS_DIR
    log_dir = PROJECT_ROOT / LOG_DIR
    corrupt_log = log_dir / "corrupt_files.csv"
    augmentation_log = log_dir / "augmentation_log.csv"
    summary_path = log_dir / "augmentation_summary.json"

    specs = build_augmentation_specs()
    logger.info("Configured %d isolated augmentation techniques", len(specs))

    dataset_entries, corrupt_records = validate_dataset(input_path, logger, corrupt_log)
    stats = RunStats(corrupt=len(corrupt_records))

    if smoke_test:
        dataset_entries = dataset_entries[:3]
        logger.info("Smoke test mode: processing first %d images only", len(dataset_entries))
    elif limit is not None:
        dataset_entries = dataset_entries[:limit]
        logger.info("Limited run: processing first %d images", len(dataset_entries))

    log_fields = [
        "timestamp_utc",
        "class",
        "filename",
        "status",
        "elapsed_sec",
        "augmented_files_written",
        "error",
    ]
    write_header = not augmentation_log.exists()
    log_file = augmentation_log.open("a", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    if write_header:
        log_writer.writeheader()

    try:
        for class_name, image_path in tqdm(
            dataset_entries,
            desc="Augmenting images",
            unit="img",
        ):
            try:
                status, elapsed, error, num_aug = process_single_image(
                    class_name,
                    image_path,
                    specs,
                    augmented_dir,
                    grids_dir,
                )
                if status == "skipped":
                    stats.skipped += 1
                    increment_counter(stats.per_class_skipped, class_name)
                else:
                    stats.processed += 1
                    stats.total_augmented_files += num_aug
                    stats.total_review_grids += 1
                    increment_counter(stats.per_class_processed, class_name)

                log_writer.writerow(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "class": class_name,
                        "filename": image_path.name,
                        "status": status,
                        "elapsed_sec": f"{elapsed:.4f}",
                        "augmented_files_written": num_aug if status == "processed" else 0,
                        "error": error or "",
                    }
                )
            except Exception as exc:
                stats.failed += 1
                increment_counter(stats.per_class_failed, class_name)
                logger.exception("Failed on %s: %s", image_path, exc)
                log_writer.writerow(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "class": class_name,
                        "filename": image_path.name,
                        "status": "failed",
                        "elapsed_sec": "0.0000",
                        "augmented_files_written": 0,
                        "error": str(exc),
                    }
                )
            log_file.flush()
    finally:
        log_file.close()

    elapsed_total = stats.elapsed()
    aug_disk = directory_size_bytes(augmented_dir)
    grid_disk = directory_size_bytes(grids_dir)
    total_disk = aug_disk + grid_disk
    avg_time = (
        elapsed_total / stats.processed if stats.processed > 0 else 0.0
    )

    summary = {
        "run_completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_path),
        "output_augmented_dir": str(augmented_dir),
        "output_grids_dir": str(grids_dir),
        "config": {
            "WORKING_SIZE": WORKING_SIZE,
            "THUMB_SIZE": THUMB_SIZE,
            "GRID_COLS": GRID_COLS,
            "JPEG_QUALITY": JPEG_QUALITY,
            "RANDOM_SEED": RANDOM_SEED,
            "num_augmentations_per_image": len(specs),
        },
        "totals": {
            "expected_images": EXPECTED_TOTAL_IMAGES,
            "readable_images": len(dataset_entries) + stats.corrupt,
            "processed": stats.processed,
            "skipped": stats.skipped,
            "failed": stats.failed,
            "corrupt_at_validation": stats.corrupt,
            "augmented_files_generated_this_run": stats.total_augmented_files,
            "review_grids_generated_this_run": stats.total_review_grids,
            "expected_augmented_files_full_dataset": EXPECTED_TOTAL_IMAGES * len(specs),
            "expected_review_grids_full_dataset": EXPECTED_TOTAL_IMAGES,
        },
        "per_class_processed": stats.per_class_processed,
        "per_class_skipped": stats.per_class_skipped,
        "per_class_failed": stats.per_class_failed,
        "runtime_sec": round(elapsed_total, 2),
        "average_time_per_processed_image_sec": round(avg_time, 4),
        "disk_usage_bytes": {
            "augmented_dataset": aug_disk,
            "augmentation_review_grids": grid_disk,
            "total": total_disk,
        },
        "disk_usage_human": {
            "augmented_dataset": format_bytes(aug_disk),
            "augmentation_review_grids": format_bytes(grid_disk),
            "total": format_bytes(total_disk),
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("=" * 60)
    logger.info("AUGMENTATION RUN COMPLETE")
    logger.info("Processed: %d | Skipped: %d | Failed: %d", stats.processed, stats.skipped, stats.failed)
    logger.info("Augmented files this run: %d", stats.total_augmented_files)
    logger.info("Review grids this run: %d", stats.total_review_grids)
    logger.info("Total runtime: %.2f sec (avg %.4f sec/image processed)", elapsed_total, avg_time)
    logger.info("Disk usage — augmented: %s | grids: %s | total: %s",
                format_bytes(aug_disk), format_bytes(grid_disk), format_bytes(total_disk))
    logger.info("Summary saved to %s", summary_path)

    if not smoke_test and limit is None:
        post_run_sanity_check(augmented_dir, grids_dir, logger)

    return stats


def post_run_sanity_check(
    augmented_dir: Path,
    grids_dir: Path,
    logger: logging.Logger,
    sample_size: int = 10,
) -> None:
    aug_files = sorted(augmented_dir.rglob("*.jpg"))
    grid_files = sorted(grids_dir.rglob("*_review.jpg"))

    if not aug_files or not grid_files:
        logger.warning("Sanity check skipped: no output files found")
        return

    rng = random.Random(RANDOM_SEED)
    aug_sample = rng.sample(aug_files, min(sample_size, len(aug_files)))
    grid_sample = rng.sample(grid_files, min(sample_size, len(grid_files)))

    logger.info("Post-run sanity check (%d augmented + %d grids)...", len(aug_sample), len(grid_sample))

    ok_aug = 0
    for path in aug_sample:
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                img.load()
            ok_aug += 1
        except Exception as exc:
            logger.error("Invalid augmented file: %s (%s)", path, exc)

    ok_grid = 0
    for path in grid_sample:
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                img.load()
            ok_grid += 1
        except Exception as exc:
            logger.error("Invalid review grid: %s (%s)", path, exc)

    logger.info(
        "Sanity check passed: %d/%d augmented, %d/%d review grids valid",
        ok_aug,
        len(aug_sample),
        ok_grid,
        len(grid_sample),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Malabar leaf image augmentation pipeline")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Process only 3 images to verify pipeline logic",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N images (for debugging)",
    )
    args = parser.parse_args()
    run_pipeline(limit=args.limit, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
