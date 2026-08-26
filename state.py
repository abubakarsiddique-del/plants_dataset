#!/usr/bin/env python3
"""Shared graph state threaded through all six workflow nodes (Part B.1).

Matches the reference ``PipelineState`` schema exactly and adds a small number
of clearly-marked orchestration bookkeeping keys used by the graph engine
(retry counters, interrupt payload, node/edge history). ``total=False`` so the
state is valid while nodes progressively fill it in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from common import load_json, save_json


class PipelineState(TypedDict, total=False):
    # --- resolved run context ---
    config: dict
    run_id: str
    output_root: str
    profile: str

    # --- stage 1: data ingestion & cleaning ---
    class_domain_counts: dict
    dedup_report: dict                 # hashes checked, leaks found/removed, split leakage matrix
    cleaning_log: List[str]

    # --- agent 1: explainable data audit ---
    data_audit: dict                   # DataAuditOutput.model_dump()

    # --- stage 3: preprocessing & augmentation ---
    split_manifest: dict               # train/val/test/external_domain_test file lists
    augmentation_summary: dict

    # --- stage 4: train & evaluate ---
    cv_metrics: dict
    test_metrics: dict
    external_domain_metrics: dict
    confusion_matrices: dict
    explainability_report: dict        # Grad-CAM++ sample paths, deletion/insertion AUC
    export_report: dict                # ONNX latency/memory benchmark

    # --- agent 2: limitations coverage audit ---
    limitations_audit: dict            # LimitationsAuditOutput.model_dump()

    # --- stage 6: final report & dataset card ---
    final_report_path: str
    dataset_card_path: str
    diagram_path: str

    # --- control flow ---
    halted: bool
    halt_reason: Optional[str]

    # --- orchestration bookkeeping (engine-internal) ---
    retries: dict                      # {node_name: count}
    pending_interrupt: Optional[dict]  # payload surfaced at await_human_review
    resumed: bool
    history: List[dict]                # ordered node/edge-decision log


def new_state(config: dict, run_id: str, output_root: str, profile: str) -> PipelineState:
    return PipelineState(
        config=config,
        run_id=run_id,
        output_root=output_root,
        profile=profile,
        cleaning_log=[],
        halted=False,
        halt_reason=None,
        retries={},
        pending_interrupt=None,
        resumed=False,
        history=[],
    )


def record_history(state: PipelineState, kind: str, name: str, detail: Optional[dict] = None) -> None:
    """Append a node-visit or edge-decision entry to the run history."""
    state.setdefault("history", []).append(
        {"kind": kind, "name": name, "detail": detail or {}}
    )


def save_state(state: PipelineState, path: str | Path) -> Path:
    return save_json(dict(state), path)


def load_state(path: str | Path) -> PipelineState:
    data = load_json(path)
    return PipelineState(**data)  # type: ignore[arg-type]
