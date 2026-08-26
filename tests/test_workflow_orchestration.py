"""End-to-end orchestration smoke tests for the agentic workflow (Part B).

Covers the plan's Verification step 3 without paying for real training: the four
heavy deterministic nodes (data / augment / train+eval+explain / export) are
replaced with fast fakes that write realistic ``PipelineState`` numbers, while
the **real** engine, routing, both agent nodes (via an injected offline mock
LLM), the deterministic reconciliation, and the stage-6 reporter all run for real.

Asserted invariants:
  * Agent 1 ``proceed=False`` fires the ``await_human_review`` interrupt and
    checkpoints *before* any training happens.
  * ``--resume`` continues from the checkpoint (on disk) straight past the gate
    and runs to completion.
  * A failing fixable gap loops back exactly ``max_retries`` times, applies the
    gap-2 reweight patch, then force-ships — it never loops forever.
  * ``final_report.md`` + ``dataset_card.md`` + the diagram are generated with the
    required sections/tables.
"""

from __future__ import annotations

import pytest

from run_pipeline import build_default_mock_responder
from state import load_state, new_state, record_history
from workflow_graph import GraphInterrupt, MiniGraph, Pipeline

CLASSES = ["Healthy-Leaf", "Anthracnose", "Pest-Damage", "Bacterial-Spot", "Downy-Mildew"]
_CM = [[4, 0, 0, 0, 1], [0, 3, 1, 0, 0], [0, 0, 4, 0, 1], [0, 1, 0, 3, 0], [0, 0, 0, 0, 4]]


# ---------------------------------------------------------------------------
# mock LLM responders
# ---------------------------------------------------------------------------
def _responder(data_audit_fn):
    """Reuse the shipped mock limitations audit; swap in a custom data-audit fn."""
    r = build_default_mock_responder()
    r["DataAuditOutput"] = data_audit_fn
    return r


def _clean_audit(system, user):
    return {
        "findings": [{"finding": "No cross-split leakage after cleaning.",
                      "evidence": "dedup_report.leakage_pairs_remaining=0.", "severity": "info"}],
        "risk_level": "low", "recommendation": "Proceed.", "proceed": True,
    }


def _blocking_audit(system, user):
    # proceed=True on purpose — the guardrail must still force the interrupt.
    return {
        "findings": [{"finding": "Residual cross-split near-duplicate leakage.",
                      "evidence": "dedup_report.leakage_pairs_remaining=4 (>0).", "severity": "blocking"}],
        "risk_level": "low", "recommendation": "Fix leakage first.", "proceed": True,
    }


# ---------------------------------------------------------------------------
# fake heavy-node metrics
# ---------------------------------------------------------------------------
def _metrics_block(*, min_f1):
    """Report-valid test/external/cv/explain/export blocks. ``min_f1`` sets the
    worst per-class F1 (drives the minority_class_gap)."""
    per_class = {c: {"precision": 0.82, "recall": 0.80,
                     "f1": (min_f1 if i == 1 else 0.85), "support": 4 + i}
                 for i, c in enumerate(CLASSES)}
    boot = {"accuracy": {"point": 0.85, "lo": 0.70, "hi": 0.95},
            "macro_f1": {"point": 0.83, "lo": 0.68, "hi": 0.93}}
    return {
        "cv": {"n_folds": 2, "mean_accuracy": 0.84, "std_accuracy": 0.02,
               "mean_macro_f1": 0.82, "std_macro_f1": 0.03},
        "test": {"n": 20, "accuracy": 0.85, "macro_f1": 0.83, "weighted_f1": 0.84,
                 "per_class": per_class, "bootstrap": boot, "confusion_matrix": _CM},
        "external": {"n": 35, "accuracy": 0.80, "macro_f1": 0.78, "weighted_f1": 0.79,
                     "mode": "real_domain_holdout", "domain": "M-tall-qC",
                     "per_class": per_class, "bootstrap": boot, "confusion_matrix": _CM},
        "confusion": {"labels": CLASSES, "in_domain_test": _CM, "external_domain_test": _CM},
        "explain": {"method": "gradcampp", "samples_per_class": 1, "coverage_complete": True,
                    "classes_covered": CLASSES, "num_active_classes": 5, "coverage_missing": [],
                    "faithfulness": {"faithfulness_score": 0.20, "deletion_auc_mean": 0.10,
                                     "insertion_auc_mean": 0.30, "n_images": 5},
                    "overlays": [{"class": c, "pred": c, "path": f"explain/cam_{c}_0.png"} for c in CLASSES]},
        "export": {"onnx_exported": False, "onnx_export_error": "Module onnx is not installed!",
                   "runtime": "torch-cpu", "cpu_threads": 1, "opset": 17, "num_params": 5817601,
                   "latency_ms": {"mean": 60.0, "p95": 70.0}, "peak_mem_mb": 22.0,
                   "within_budget": {"latency": True, "peak_mem": True, "overall": True}},
    }


def _install_stub_nodes(pipeline, metrics_seq):
    """Swap the four heavy nodes for fakes. Returns a dict of call counters.

    ``metrics_seq`` is indexed by train-call number, so a retry can present a
    different (e.g. still-failing) metric set on the second pass.
    """
    calls = {"data": 0, "augment": 0, "train": 0, "export": 0}

    def data_ingestion(state):
        calls["data"] += 1
        state["class_domain_counts"] = {
            "Healthy-Leaf": {"L-tall-qC": 30, "M-tall-qC": 10},
            "Anthracnose": {"L-tall-qC": 25, "M-tall-qC": 15},
            "Pest-Damage": {"L-tall-qC": 28, "M-tall-qC": 12},
            "Bacterial-Spot": {"L-tall-qC": 22, "M-tall-qC": 18},
            "Downy-Mildew": {"L-tall-qC": 26, "M-tall-qC": 14},
        }
        state["dedup_report"] = {
            "images_hashed": 200, "hash": "self_contained_phash", "hash_bits": 256,
            "exact_duplicate_pairs": 0, "exact_duplicates_removed": 0,
            "near_dup_pairs_detected": 0, "near_dup_hamming_threshold": 12,
            "leakage_pairs_remaining": 0,
        }
        state["cleaning_log"] = ["fixture: no cleaning needed"]
        state["split_manifest"] = {
            "classes": CLASSES,
            "declared_unpopulated": ["Dichotomophthora-Leaf-Spot", "Bipolaris-Leaf-Spot"],
            "domains": ["L-tall-qC", "M-tall-qC"],
            "counts": {"train": 121, "val": 24, "test": 20, "external_domain_test": 35},
            "external_mode": "real_domain_holdout", "external_domain": "M-tall-qC",
            "image_size": 224, "working_size": 640,
        }
        record_history(state, "node", "data_ingestion",
                       {"images": 200, "leakage_pairs_remaining": 0})
        return state

    def preprocess_augment(state):
        calls["augment"] += 1
        state["augmentation_summary"] = {"num_train_ops": 12,
                                         "domain_randomization": {"group": ["sensor_noise", "white_balance", "jpeg"]}}
        record_history(state, "node", "preprocess_augment", {"num_train_ops": 12})
        return state

    def train_and_evaluate(state):
        if state.pop("_apply_reweight", False):
            pipeline._apply_reweight_patch(state)     # exercise the real gap-2 patch
        idx = min(calls["train"], len(metrics_seq) - 1)
        calls["train"] += 1
        m = metrics_seq[idx]
        state["_fixture_idx"] = idx
        state["_checkpoint"] = str(pipeline.run_paths.checkpoints / "fake.pt")
        state["cv_metrics"] = m["cv"]
        state["test_metrics"] = m["test"]
        state["external_domain_metrics"] = m["external"]
        state["confusion_matrices"] = m["confusion"]
        state["explainability_report"] = m["explain"]
        record_history(state, "node", "train_and_evaluate",
                       {"test_accuracy": m["test"]["accuracy"]})
        return state

    def export_model(state):
        calls["export"] += 1
        idx = state.get("_fixture_idx", 0)
        state["export_report"] = metrics_seq[idx]["export"]
        record_history(state, "node", "export_model", {"within_budget": True})
        return state

    pipeline.data_ingestion = data_ingestion
    pipeline.preprocess_augment = preprocess_augment
    pipeline.train_and_evaluate = train_and_evaluate
    pipeline.export_model = export_model
    return calls


def _make(run_paths, base_cfg, data_audit_fn, metrics_seq):
    """Build a stubbed Pipeline + MiniGraph engine wired to the mock LLM."""
    from agents.llm_provider import get_chat_model
    llm = get_chat_model(base_cfg, mock_responder=_responder(data_audit_fn))
    pipeline = Pipeline(run_paths, device="cpu", profile="fast", chat_model=llm)
    calls = _install_stub_nodes(pipeline, metrics_seq)
    return pipeline, MiniGraph(pipeline), calls


# ---------------------------------------------------------------------------
# 1. happy path — clean audit, all gaps pass -> ready_to_ship
# ---------------------------------------------------------------------------
def test_happy_path_ships(run_paths, base_cfg):
    pipeline, engine, calls = _make(run_paths, base_cfg, _clean_audit, [_metrics_block(min_f1=0.85)])
    state = new_state(base_cfg, run_paths.run_id, str(run_paths.root), "fast")

    result = engine.invoke(state)

    assert result["halted"] is False
    assert result["data_audit"]["proceed"] is True
    assert result["limitations_audit"]["overall_status"] == "ready_to_ship"
    assert calls["train"] == 1  # no retry
    from pathlib import Path
    assert Path(result["final_report_path"]).exists()


# ---------------------------------------------------------------------------
# 2. interrupt + resume
# ---------------------------------------------------------------------------
def test_blocking_audit_interrupts_before_training(run_paths, base_cfg):
    pipeline, engine, calls = _make(run_paths, base_cfg, _blocking_audit, [_metrics_block(min_f1=0.85)])
    state = new_state(base_cfg, run_paths.run_id, str(run_paths.root), "fast")

    with pytest.raises(GraphInterrupt) as ei:
        engine.invoke(state)

    payload = ei.value.payload
    assert payload["node"] == "await_human_review"
    assert payload["blocking_findings"], "interrupt payload must carry the blocking finding"
    # training must NOT have started
    assert calls["train"] == 0
    # checkpoint on disk is resumable
    saved = load_state(run_paths.state_path)
    assert saved["pending_interrupt"] is not None
    assert saved["halted"] is True


def test_resume_from_checkpoint_completes(run_paths, base_cfg):
    # 1) run until it pauses at the human-review gate
    pipeline, engine, calls = _make(run_paths, base_cfg, _blocking_audit, [_metrics_block(min_f1=0.85)])
    with pytest.raises(GraphInterrupt):
        engine.invoke(new_state(base_cfg, run_paths.run_id, str(run_paths.root), "fast"))
    assert calls["train"] == 0

    # 2) rebuild the engine (fresh process would), reload the checkpoint, resume.
    pipeline2, engine2, calls2 = _make(run_paths, base_cfg, _clean_audit, [_metrics_block(min_f1=0.85)])
    resumed_state = load_state(run_paths.state_path)
    resumed_state["config"] = base_cfg

    result = engine2.resume(resumed_state)

    assert result.get("resumed") is True
    assert result["halted"] is False
    # resume skips the gate and re-runs from augment onward exactly once
    assert calls2["augment"] == 1
    assert calls2["train"] == 1
    assert result["limitations_audit"]["overall_status"] == "ready_to_ship"


# ---------------------------------------------------------------------------
# 3. bounded rework loop
# ---------------------------------------------------------------------------
def test_needs_rework_loops_once_then_force_ships(run_paths, base_cfg):
    # A persistently-failing minority class (F1=0.0) on BOTH passes.
    seq = [_metrics_block(min_f1=0.0), _metrics_block(min_f1=0.0)]
    pipeline, engine, calls = _make(run_paths, base_cfg, _clean_audit, seq)
    state = new_state(base_cfg, run_paths.run_id, str(run_paths.root), "fast")

    result = engine.invoke(state)

    # trained twice: initial + exactly one rework retry (max_retries defaults to 1)
    assert calls["train"] == 2
    assert result["retries"]["rework"] == 1
    # bounded-retry termination guarantee
    assert result["limitations_audit"]["overall_status"] == "ship_with_documented_limitations"
    # the gap-2 reweight patch was applied on the retry
    assert result["config"]["loss"]["class_balancing"] == "focal"
    assert any("gap-2 reweight" in entry for entry in result["cleaning_log"])


def test_retry_count_respects_config_max_retries(run_paths, base_cfg):
    base_cfg["retry"]["max_retries"] = 2
    seq = [_metrics_block(min_f1=0.0)] * 4
    pipeline, engine, calls = _make(run_paths, base_cfg, _clean_audit, seq)

    result = engine.invoke(new_state(base_cfg, run_paths.run_id, str(run_paths.root), "fast"))

    assert result["retries"]["rework"] == 2      # honoured the raised cap
    assert calls["train"] == 3                    # initial + 2 retries
    assert result["limitations_audit"]["overall_status"] == "ship_with_documented_limitations"


# ---------------------------------------------------------------------------
# 4. report / card / diagram generation + content
# ---------------------------------------------------------------------------
def test_reports_generated_with_required_sections(run_paths, base_cfg):
    # minority fails so the report must visibly show a failed Agent-2 gap.
    seq = [_metrics_block(min_f1=0.0), _metrics_block(min_f1=0.0)]
    pipeline, engine, _ = _make(run_paths, base_cfg, _clean_audit, seq)
    result = engine.invoke(new_state(base_cfg, run_paths.run_id, str(run_paths.root), "fast"))

    from pathlib import Path
    report = Path(result["final_report_path"]).read_text(encoding="utf-8")
    card = Path(result["dataset_card_path"]).read_text(encoding="utf-8")
    diagram = Path(result["diagram_path"])

    # final report — all 8 sections + a visible failed gap + no-leakage badge
    for heading in ["## 1. Configuration", "## 2. Data & leakage check",
                    "## 3. Agent 1", "## 4. Results", "## 5. Explainability",
                    "## 6. Deployment", "## 7. Agent 2", "## 8. Orchestration trace"]:
        assert heading in report, f"missing section: {heading}"
    assert "✅ 0 (no cross-split leakage)" in report
    assert "❌ fail" in report                     # the failed minority gap is shown
    assert "🛡️ **Deterministic reconciliation**" in report  # LLM all-pass was overridden

    # dataset card — declared-unpopulated classes documented
    assert "declared, unpopulated" in card
    assert "Dichotomophthora-Leaf-Spot" in card

    # diagram artifact exists (PNG when matplotlib present, else the .mmd source)
    assert diagram.exists()
    assert (run_paths.reports / "pipeline_diagram.mmd").exists()


def test_diagram_mermaid_source_has_agent_and_det_styles():
    from workflow_graph import mermaid_source
    src = mermaid_source()
    assert "classDef agent" in src and "#8e6fd1" in src   # purple agents
    assert "classDef det" in src and "#d9d9d9" in src      # gray deterministic
    assert "await_human_review" in src
