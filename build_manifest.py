#!/usr/bin/env python3
"""
Build a master manifest CSV linking every original and augmented image to
traceability metadata for downstream fuzzy clustering.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

from augmentation_pipeline import (
    EXPECTED_CLASS_COUNTS,
    EXPECTED_TOTAL_IMAGES,
    IMAGE_EXTENSIONS,
    INPUT_DIR,
    OUTPUT_AUGMENTED_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    WORKING_SIZE,
    build_augmentation_specs,
    discover_class_folders,
    list_image_files,
    per_image_seed,
)

MANIFEST_VERSION = "1.0.0"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "logs" / "master_manifest.csv"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "logs" / "master_manifest_summary.json"

AUGMENTED_NAME_RE = re.compile(
    r"^(?P<stem>.+)_aug(?P<index>\d{2})_(?P<slug>[a-z0-9_]+)\.jpg$",
    re.IGNORECASE,
)

MANIFEST_FIELDNAMES = [
    "source_id",
    "class_label",
    "is_original",
    "augmentation_type",
    "augmentation_index",
    "augmentation_slug",
    "parent_filename",
    "filename",
    "output_path",
    "sha256",
    "original_width",
    "original_height",
    "letterbox_target_width",
    "letterbox_target_height",
    "letterbox_scale",
    "letterbox_resized_width",
    "letterbox_resized_height",
    "letterbox_x_offset",
    "letterbox_y_offset",
    "letterbox_fill_r",
    "letterbox_fill_g",
    "letterbox_fill_b",
    "per_image_seed",
    "manifest_version",
    "pipeline_random_seed",
    "run_id",
]


@dataclass(frozen=True)
class LetterboxParams:
    original_width: int
    original_height: int
    target_width: int
    target_height: int
    scale: float
    resized_width: int
    resized_height: int
    x_offset: int
    y_offset: int
    fill_r: int
    fill_g: int
    fill_b: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "original_width": self.original_width,
            "original_height": self.original_height,
            "letterbox_target_width": self.target_width,
            "letterbox_target_height": self.target_height,
            "letterbox_scale": round(self.scale, 8),
            "letterbox_resized_width": self.resized_width,
            "letterbox_resized_height": self.resized_height,
            "letterbox_x_offset": self.x_offset,
            "letterbox_y_offset": self.y_offset,
            "letterbox_fill_r": self.fill_r,
            "letterbox_fill_g": self.fill_g,
            "letterbox_fill_b": self.fill_b,
        }


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    class_label: str
    parent_filename: str
    parent_stem: str
    original_path: Path
    letterbox: LetterboxParams
    per_image_seed: int


def compute_source_id(class_label: str, parent_filename: str) -> str:
    payload = f"{class_label}|{parent_filename}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_letterbox_params(
    original_width: int,
    original_height: int,
    target_size: Tuple[int, int] = WORKING_SIZE,
    fill: Tuple[int, int, int] = (255, 255, 255),
) -> LetterboxParams:
    target_w, target_h = target_size
    scale = min(target_w / original_width, target_h / original_height)
    resized_w = max(1, int(round(original_width * scale)))
    resized_h = max(1, int(round(original_height * scale)))
    x_offset = (target_w - resized_w) // 2
    y_offset = (target_h - resized_h) // 2
    return LetterboxParams(
        original_width=original_width,
        original_height=original_height,
        target_width=target_w,
        target_height=target_h,
        scale=scale,
        resized_width=resized_w,
        resized_height=resized_h,
        x_offset=x_offset,
        y_offset=y_offset,
        fill_r=fill[0],
        fill_g=fill[1],
        fill_b=fill[2],
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def relative_output_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def build_slug_lookup() -> Dict[str, Tuple[int, str]]:
    return {spec.slug: (spec.index, spec.name) for spec in build_augmentation_specs()}


def index_original_sources(input_dir: Path) -> Tuple[Dict[Tuple[str, str], SourceRecord], List[str]]:
    """
    Index originals by (class_label, stem).
    Returns source index and warning messages.
    """
    warnings: List[str] = []
    sources: Dict[Tuple[str, str], SourceRecord] = {}

    for class_folder in discover_class_folders(input_dir):
        class_label = class_folder.name
        for image_path in list_image_files(class_folder):
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception as exc:
                warnings.append(f"Unreadable original skipped: {image_path} ({exc})")
                continue

            parent_filename = image_path.name
            parent_stem = image_path.stem
            source_id = compute_source_id(class_label, parent_filename)
            letterbox = compute_letterbox_params(width, height)
            key = (class_label, parent_stem)
            if key in sources:
                warnings.append(f"Duplicate stem in class {class_label}: {parent_filename}")
            sources[key] = SourceRecord(
                source_id=source_id,
                class_label=class_label,
                parent_filename=parent_filename,
                parent_stem=parent_stem,
                original_path=image_path,
                letterbox=letterbox,
                per_image_seed=per_image_seed(parent_filename),
            )

    return sources, warnings


def make_row(
    *,
    source: SourceRecord,
    is_original: bool,
    augmentation_type: str,
    augmentation_index: str,
    augmentation_slug: str,
    output_path: Path,
    sha256: str,
    project_root: Path,
) -> Dict[str, object]:
    row = {
        "source_id": source.source_id,
        "class_label": source.class_label,
        "is_original": str(is_original).lower(),
        "augmentation_type": augmentation_type,
        "augmentation_index": augmentation_index,
        "augmentation_slug": augmentation_slug,
        "parent_filename": source.parent_filename,
        "filename": output_path.name,
        "output_path": relative_output_path(output_path, project_root),
        "sha256": sha256,
        "per_image_seed": source.per_image_seed,
        "manifest_version": MANIFEST_VERSION,
        "pipeline_random_seed": RANDOM_SEED,
        "run_id": RUN_ID,
    }
    row.update(source.letterbox.as_dict())
    return row


def collect_manifest_rows(
    project_root: Path,
    input_dir: Path,
    augmented_dir: Path,
    compute_hashes: bool = True,
) -> Tuple[List[Dict[str, object]], List[str]]:
    warnings: List[str] = []
    slug_lookup = build_slug_lookup()
    sources, source_warnings = index_original_sources(input_dir)
    warnings.extend(source_warnings)

    rows: List[Dict[str, object]] = []
    hash_jobs: List[Tuple[Path, Dict[str, object]]] = []

    for source in sorted(
        sources.values(),
        key=lambda item: (item.class_label.lower(), item.parent_filename.lower()),
    ):
        row = make_row(
            source=source,
            is_original=True,
            augmentation_type="ORIGINAL",
            augmentation_index="",
            augmentation_slug="",
            output_path=source.original_path,
            sha256="",
            project_root=project_root,
        )
        rows.append(row)
        if compute_hashes:
            hash_jobs.append((source.original_path, row))

    augmented_files = sorted(augmented_dir.rglob("*.jpg"))
    for aug_path in augmented_files:
        class_label = aug_path.parent.name
        match = AUGMENTED_NAME_RE.match(aug_path.name)
        if not match:
            warnings.append(f"Unrecognized augmented filename: {aug_path}")
            continue

        stem = match.group("stem")
        aug_index = match.group("index")
        aug_slug = match.group("slug").lower()
        source = sources.get((class_label, stem))
        if source is None:
            warnings.append(f"Augmented file without source parent: {aug_path}")
            continue

        if aug_slug not in slug_lookup:
            warnings.append(f"Unknown augmentation slug '{aug_slug}' in {aug_path.name}")
            aug_name = aug_slug.replace("_", " ").title()
        else:
            expected_index, aug_name = slug_lookup[aug_slug]
            if f"{expected_index:02d}" != aug_index:
                warnings.append(
                    f"Augmentation index mismatch for {aug_path.name}: "
                    f"expected {expected_index:02d}, found {aug_index}"
                )

        row = make_row(
            source=source,
            is_original=False,
            augmentation_type=aug_name,
            augmentation_index=aug_index,
            augmentation_slug=aug_slug,
            output_path=aug_path,
            sha256="",
            project_root=project_root,
        )
        rows.append(row)
        if compute_hashes:
            hash_jobs.append((aug_path, row))

    if compute_hashes:
        for path, row in tqdm(hash_jobs, desc="Computing SHA256", unit="file"):
            row["sha256"] = sha256_file(path)

    return rows, warnings


def write_manifest_csv(rows: Iterable[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    rows: List[Dict[str, object]],
    warnings: List[str],
    *,
    input_dir: Path,
    augmented_dir: Path,
    manifest_path: Path,
) -> Dict[str, object]:
    originals = [row for row in rows if row["is_original"] == "true"]
    augmented = [row for row in rows if row["is_original"] == "false"]

    per_class_original = {}
    per_class_augmented = {}
    for row in originals:
        per_class_original[row["class_label"]] = per_class_original.get(row["class_label"], 0) + 1
    for row in augmented:
        per_class_augmented[row["class_label"]] = per_class_augmented.get(row["class_label"], 0) + 1

    per_slug = {}
    for row in augmented:
        slug = row["augmentation_slug"]
        per_slug[slug] = per_slug.get(slug, 0) + 1

    unique_source_ids = {row["source_id"] for row in rows}

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "manifest_version": MANIFEST_VERSION,
        "manifest_path": str(manifest_path),
        "input_dir": str(input_dir),
        "augmented_dir": str(augmented_dir),
        "pipeline_random_seed": RANDOM_SEED,
        "letterbox_target_size": list(WORKING_SIZE),
        "totals": {
            "rows": len(rows),
            "original_rows": len(originals),
            "augmented_rows": len(augmented),
            "unique_sources": len(unique_source_ids),
            "expected_originals": EXPECTED_TOTAL_IMAGES,
            "expected_augmented": EXPECTED_TOTAL_IMAGES * len(build_augmentation_specs()),
            "expected_rows": EXPECTED_TOTAL_IMAGES * (1 + len(build_augmentation_specs())),
        },
        "per_class_original": dict(sorted(per_class_original.items())),
        "per_class_augmented": dict(sorted(per_class_augmented.items())),
        "per_augmentation_slug": dict(sorted(per_slug.items())),
        "warnings_count": len(warnings),
        "warnings_sample": warnings[:25],
    }


def validate_counts(summary: Dict[str, object]) -> List[str]:
    issues: List[str] = []
    totals = summary["totals"]
    expected_classes = EXPECTED_CLASS_COUNTS

    if totals["original_rows"] != totals["expected_originals"]:
        issues.append(
            f"Original row count {totals['original_rows']} != expected {totals['expected_originals']}"
        )
    if totals["augmented_rows"] != totals["expected_augmented"]:
        issues.append(
            f"Augmented row count {totals['augmented_rows']} != expected {totals['expected_augmented']}"
        )
    if totals["rows"] != totals["expected_rows"]:
        issues.append(f"Total row count {totals['rows']} != expected {totals['expected_rows']}")

    for class_label, expected in expected_classes.items():
        actual = summary["per_class_original"].get(class_label, 0)
        if actual != expected:
            issues.append(
                f"Class {class_label}: {actual} originals indexed, expected {expected}"
            )
        aug_actual = summary["per_class_augmented"].get(class_label, 0)
        aug_expected = expected * len(build_augmentation_specs())
        if aug_actual != aug_expected:
            issues.append(
                f"Class {class_label}: {aug_actual} augmented rows, expected {aug_expected}"
            )

    return issues


def run(
    project_root: Path = PROJECT_ROOT,
    input_dir_name: str = INPUT_DIR,
    augmented_dir_name: str = OUTPUT_AUGMENTED_DIR,
    manifest_path: Optional[Path] = None,
    summary_path: Optional[Path] = None,
    skip_hashes: bool = False,
) -> Dict[str, object]:
    input_dir = project_root / input_dir_name
    augmented_dir = project_root / augmented_dir_name
    manifest_path = manifest_path or DEFAULT_MANIFEST_PATH
    summary_path = summary_path or DEFAULT_SUMMARY_PATH

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not augmented_dir.is_dir():
        raise FileNotFoundError(f"Augmented directory not found: {augmented_dir}")

    rows, warnings = collect_manifest_rows(
        project_root=project_root,
        input_dir=input_dir,
        augmented_dir=augmented_dir,
        compute_hashes=not skip_hashes,
    )

    write_manifest_csv(rows, manifest_path)
    summary = build_summary(
        rows,
        warnings,
        input_dir=input_dir,
        augmented_dir=augmented_dir,
        manifest_path=manifest_path,
    )
    summary["validation_issues"] = validate_counts(summary)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 60)
    print("MASTER MANIFEST COMPLETE")
    print(f"Manifest CSV: {manifest_path}")
    print(f"Summary JSON: {summary_path}")
    print(
        f"Rows: {summary['totals']['rows']} "
        f"({summary['totals']['original_rows']} originals + "
        f"{summary['totals']['augmented_rows']} augmented)"
    )
    print(f"Unique sources: {summary['totals']['unique_sources']}")
    print(f"Warnings: {summary['warnings_count']}")
    if summary["validation_issues"]:
        print("Validation issues:")
        for issue in summary["validation_issues"]:
            print(f"  - {issue}")
    else:
        print("Validation: all expected counts matched")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build master dataset manifest CSV")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Output CSV path",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Output summary JSON path",
    )
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="Skip SHA256 computation (faster debug mode)",
    )
    args = parser.parse_args()

    summary = run(
        manifest_path=args.manifest_path,
        summary_path=args.summary_path,
        skip_hashes=args.skip_hashes,
    )
    if summary["validation_issues"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
