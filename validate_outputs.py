#!/usr/bin/env python3
"""Validate dataset and clustering outputs for expected file counts and manifest consistency."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def load_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_summary(summary_path: Path) -> Dict[str, object]:
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_manifest_rows(rows: List[Dict[str, str]], summary: Dict[str, object]) -> List[str]:
    issues: List[str] = []
    row_count = len(rows)
    summary_rows = summary.get("totals", {}).get("rows")
    if summary_rows is not None and row_count != summary_rows:
        issues.append(f"Manifest row count {row_count} != summary totals.rows {summary_rows}")

    source_ids = {row["source_id"] for row in rows}
    unique_source_count = len(source_ids)
    summary_unique = summary.get("totals", {}).get("unique_sources")
    if summary_unique is not None and unique_source_count != summary_unique:
        issues.append(
            f"Unique source_id count {unique_source_count} != summary totals.unique_sources {summary_unique}"
        )

    run_id_values = {row.get("run_id") for row in rows}
    if len(run_id_values) != 1 or None in run_id_values or "" in run_id_values:
        issues.append("Manifest run_id values are missing or inconsistent")

    return issues


def validate_summary(summary: Dict[str, object]) -> List[str]:
    issues: List[str] = []
    expected_keys = [
        "generated_at_utc",
        "run_id",
        "manifest_version",
        "manifest_path",
        "input_dir",
        "augmented_dir",
        "pipeline_random_seed",
        "totals",
    ]
    for key in expected_keys:
        if key not in summary:
            issues.append(f"Missing summary key: {key}")

    if "validation_issues" in summary and summary["validation_issues"]:
        issues.append("Summary contains validation issues")

    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate manifest and pipeline output consistency.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("logs/master_manifest.csv"),
        help="Path to the master manifest CSV.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("logs/master_manifest_summary.json"),
        help="Path to the manifest summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_rows = load_manifest(args.manifest_path)
    summary = load_summary(args.summary_path)

    issues = []
    issues.extend(validate_manifest_rows(manifest_rows, summary))
    issues.extend(validate_summary(summary))

    if issues:
        print("VALIDATION FAILED")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    print("VALIDATION PASSED")
    print(f"Manifest rows: {len(manifest_rows)}")
    print(f"Summary run_id: {summary.get('run_id')}")


if __name__ == "__main__":
    main()
