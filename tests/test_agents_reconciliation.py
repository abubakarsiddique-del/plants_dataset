"""Unit tests for the deterministic guardrails that make the two LLM agents safe.

These are the invariants the spec calls non-optional:

* Agent 1 — *any* ``blocking`` finding forces ``proceed=False`` no matter what
  the LLM returned, so the human-review gate can never be silently skipped.
* Agent 2 — every gap status is recomputed from the numbers (``compute_gap_statuses``)
  and ``decide_overall_status`` is the single routing truth the graph also uses,
  including the bounded-retry termination guarantee.

All pure functions — no graph, no LLM key, no torch.
"""

from __future__ import annotations

from agents.data_audit_agent import run_data_audit
from agents.limitations_audit_agent import compute_gap_statuses, decide_overall_status
from agents.llm_provider import get_chat_model
from agents.schemas import GAP_NAMES


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _metrics(*, acc_drop=5.0, min_f1=0.85, leak=0, faith=0.20, cov=True, lat=60.0, mem=100.0):
    """A build_metrics()-shaped dict with only the fields compute_gap_statuses reads."""
    return {
        "thresholds": {
            "domain_gap_pts": 10.0, "min_class_f1": 0.70, "faithfulness_min": 0.05,
            "latency_ms": 200.0, "peak_mem_mb": 512.0,
        },
        "comparison": {
            "accuracy_drop_pts": acc_drop,
            "min_class_f1_in_domain": {"class": "Anthracnose", "f1": min_f1},
        },
        "dedup": {"leakage_pairs_remaining": leak},
        "explainability": {"faithfulness_score": faith, "coverage_complete": cov},
        "export": {"latency_ms_mean": lat, "peak_mem_mb": mem},
    }


def _status(metrics):
    return {g: s["status"] for g, s in compute_gap_statuses(metrics).items()}


# ---------------------------------------------------------------------------
# Agent 2 — gap arithmetic
# ---------------------------------------------------------------------------
def test_all_gaps_pass_on_healthy_numbers():
    assert _status(_metrics()) == {g: "pass" for g in GAP_NAMES}


def test_each_gap_fails_on_its_own_trigger():
    assert _status(_metrics(acc_drop=15.0))["domain_generalization_gap"] == "fail"
    assert _status(_metrics(min_f1=0.0))["minority_class_gap"] == "fail"
    assert _status(_metrics(leak=3))["leakage_gap"] == "fail"
    assert _status(_metrics(faith=0.01))["explainability_faithfulness_gap"] == "fail"
    # coverage incomplete alone must also fail the explainability gap
    assert _status(_metrics(cov=False))["explainability_faithfulness_gap"] == "fail"
    assert _status(_metrics(lat=300.0))["deployment_feasibility_gap"] == "fail"
    assert _status(_metrics(mem=600.0))["deployment_feasibility_gap"] == "fail"


def test_thresholds_are_boundaries_not_inclusive_failures():
    # exactly at the floor/ceiling should NOT fail (strict comparisons).
    assert _status(_metrics(acc_drop=10.0))["domain_generalization_gap"] == "pass"
    assert _status(_metrics(min_f1=0.70))["minority_class_gap"] == "pass"
    assert _status(_metrics(faith=0.05))["explainability_faithfulness_gap"] == "pass"
    assert _status(_metrics(lat=200.0, mem=512.0))["deployment_feasibility_gap"] == "pass"


# ---------------------------------------------------------------------------
# Agent 2 — routing / bounded-retry termination
# ---------------------------------------------------------------------------
def test_ship_when_no_gap_fails():
    overall, loopback = decide_overall_status(compute_gap_statuses(_metrics()), retries_remaining=1)
    assert overall == "ready_to_ship"
    assert loopback is None


def test_fixable_gap_loops_back_when_retries_remain():
    truth = compute_gap_statuses(_metrics(min_f1=0.0))
    overall, loopback = decide_overall_status(truth, retries_remaining=1)
    assert overall == "needs_rework"
    assert loopback == "train_and_evaluate"


def test_deployment_gap_loops_back_to_export():
    truth = compute_gap_statuses(_metrics(lat=300.0))
    overall, loopback = decide_overall_status(truth, retries_remaining=1)
    assert overall == "needs_rework"
    assert loopback == "export_model"


def test_retries_exhausted_forces_ship_not_infinite_loop():
    """The termination guarantee: a still-failing fixable gap with 0 retries ships."""
    truth = compute_gap_statuses(_metrics(min_f1=0.0))
    overall, loopback = decide_overall_status(truth, retries_remaining=0)
    assert overall == "ship_with_documented_limitations"
    assert loopback is None


def test_non_fixable_gap_never_loops():
    """A domain-generalization failure is document-only; it must not trigger rework."""
    truth = compute_gap_statuses(_metrics(acc_drop=25.0))
    overall, loopback = decide_overall_status(truth, retries_remaining=1)
    assert overall == "ship_with_documented_limitations"
    assert loopback is None


# ---------------------------------------------------------------------------
# Agent 1 — the proceed=False guardrail
# ---------------------------------------------------------------------------
def test_blocking_finding_forces_halt_even_if_llm_says_proceed(base_cfg):
    """The LLM deliberately returns proceed=True with a blocking finding;
    the deterministic guardrail must override it to False + risk=high."""
    responder = {
        "DataAuditOutput": lambda system, user: {
            "findings": [{
                "finding": "Residual cross-split near-duplicate leakage.",
                "evidence": "dedup_report.leakage_pairs_remaining=3 (>0).",
                "severity": "blocking",
            }],
            "risk_level": "low",       # wrong on purpose
            "recommendation": "meh",
            "proceed": True,           # wrong on purpose
        }
    }
    llm = get_chat_model(base_cfg, mock_responder=responder)
    state = {
        "config": base_cfg, "profile": "fast",
        "split_manifest": {"classes": [], "declared_unpopulated": [], "domains": [], "counts": {}},
        "dedup_report": {"leakage_pairs_remaining": 3},
        "class_domain_counts": {}, "cleaning_log": [],
    }
    out = run_data_audit(state, llm)
    assert out["proceed"] is False
    assert out["risk_level"] == "high"
    assert out["_guardrail"]["proceed_forced_false"] is True


def test_clean_audit_proceeds(base_cfg):
    responder = {
        "DataAuditOutput": lambda system, user: {
            "findings": [{
                "finding": "No cross-split leakage after cleaning.",
                "evidence": "dedup_report.leakage_pairs_remaining=0.",
                "severity": "info",
            }],
            "risk_level": "low",
            "recommendation": "Proceed.",
            "proceed": True,
        }
    }
    llm = get_chat_model(base_cfg, mock_responder=responder)
    state = {
        "config": base_cfg, "profile": "fast",
        "split_manifest": {"classes": [], "declared_unpopulated": [], "domains": [], "counts": {}},
        "dedup_report": {"leakage_pairs_remaining": 0},
        "class_domain_counts": {}, "cleaning_log": [],
    }
    out = run_data_audit(state, llm)
    assert out["proceed"] is True
    assert out["_guardrail"]["proceed_forced_false"] is False
