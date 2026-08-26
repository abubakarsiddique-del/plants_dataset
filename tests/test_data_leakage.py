"""The dedup / no-leakage invariant on the *real* data stage (Part A, stage 1).

This is the one test that touches the actual dataset and the real perceptual-hash
dedup + stratified-split code — the spec calls the leakage check non-optional. It
runs with a small per-class cap so hashing stays fast (no torch involved), and
skips cleanly when ``Malabar_Dataset`` is not present.
"""

from __future__ import annotations

import pytest

from common import get_logger


def _dataset_present(root) -> bool:
    d = root / "Malabar_Dataset"
    return d.exists() and any(d.iterdir())


def test_data_stage_is_leakage_free(project_root, base_cfg, run_paths, tmp_path):
    if not _dataset_present(project_root):
        pytest.skip("Malabar_Dataset not present in this checkout")

    from data import run_data_stage

    # Cap to keep hashing quick but leave enough per class to populate every split.
    base_cfg["data"]["max_images_per_class"] = {"fast": 24, "full": None}
    logger = get_logger("test.data", run_paths.root / "data.log")

    out = run_data_stage(base_cfg, run_paths, "fast", logger)
    dedup = out["dedup_report"]
    sm = out["split_manifest"]

    # ---- the core invariant: no near-duplicate crosses a split boundary ----
    assert dedup["leakage_pairs_remaining"] == 0

    # split-leakage matrix (if surfaced) must be all-zero
    matrix = dedup.get("leakage_matrix") or dedup.get("cross_split_leakage")
    if isinstance(matrix, dict):
        flat = [v for row in matrix.values()
                for v in (row.values() if isinstance(row, dict) else [row])]
        assert all(int(v) == 0 for v in flat)

    # ---- the 5 active classes are discovered; the external holdout is non-empty ----
    assert len(sm["classes"]) == 5
    counts = sm["counts"]
    assert counts["external_domain_test"] > 0
    assert counts["train"] > 0

    # ---- the 2 declared-but-unpopulated canonical classes are reported, not invented ----
    assert set(sm["declared_unpopulated"]) == {"Dichotomophthora-Leaf-Spot", "Bipolaris-Leaf-Spot"}
