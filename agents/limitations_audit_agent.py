#!/usr/bin/env python3
"""Agent 2 — Limitations Coverage Audit (Part B).

Walks the fixed five-gap checklist over the post-training metrics and returns a
``LimitationsAuditOutput``. The five gaps are pure threshold comparisons, so a
deterministic guardrail **recomputes every gap status from the numbers** and
overrides the LLM if it disagrees — the LLM's value is the evidence prose,
suggested fixes, and summary, not the arithmetic. ``decide_overall_status`` is
the single source of truth the workflow graph also uses to route
ready_to_ship / ship_with_documented_limitations / needs_rework.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from agents.schemas import GAP_NAMES, GapResult, LimitationsAuditOutput
from common import cfg_get

_PROMPT = Path(__file__).resolve().parent / "prompts" / "limitations_audit_system.md"

# Gaps the pipeline can act on automatically (loop back), vs. document-only gaps.
FIXABLE_GAPS = {"minority_class_gap", "deployment_feasibility_gap"}
GAP_TO_NODE = {"minority_class_gap": "train_and_evaluate", "deployment_feasibility_gap": "export_model"}


def _load_system_prompt() -> str:
    return _PROMPT.read_text(encoding="utf-8")


def _thresholds(config: Dict[str, Any]) -> Dict[str, float]:
    return {
        "domain_gap_pts": float(cfg_get(config, "thresholds.domain_gap_pts", 10.0)),
        "min_class_f1": float(cfg_get(config, "thresholds.min_class_f1", 0.70)),
        "faithfulness_min": float(cfg_get(config, "thresholds.faithfulness_min", 0.05)),
        "latency_ms": float(cfg_get(config, "thresholds.latency_ms", 200.0)),
        "peak_mem_mb": float(cfg_get(config, "thresholds.peak_mem_mb", 512.0)),
    }


def build_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compact, number-only view of every gap's inputs (+ the thresholds)."""
    config = state.get("config", {})
    test = state.get("test_metrics", {})
    ext = state.get("external_domain_metrics", {})
    cv = state.get("cv_metrics", {})
    expl = state.get("explainability_report", {})
    exp = state.get("export_report", {})
    dedup = state.get("dedup_report", {})

    faith = expl.get("faithfulness", {})
    latency = exp.get("latency_ms", {})
    return {
        "cv": {"n_folds": cv.get("n_folds"), "mean_accuracy": cv.get("mean_accuracy"),
               "mean_macro_f1": cv.get("mean_macro_f1"), "std_macro_f1": cv.get("std_macro_f1")},
        "in_domain_test": {"accuracy": test.get("accuracy"), "macro_f1": test.get("macro_f1"),
                           "bootstrap": test.get("bootstrap")},
        "external_domain_test": {"accuracy": ext.get("accuracy"), "macro_f1": ext.get("macro_f1"),
                                 "mode": ext.get("mode"), "domain": ext.get("domain")},
        "comparison": _comparison(state),
        "dedup": {"leakage_pairs_remaining": dedup.get("leakage_pairs_remaining")},
        "explainability": {"faithfulness_score": faith.get("faithfulness_score"),
                           "deletion_auc_mean": faith.get("deletion_auc_mean"),
                           "insertion_auc_mean": faith.get("insertion_auc_mean"),
                           "coverage_complete": expl.get("coverage_complete"),
                           "coverage_missing": expl.get("coverage_missing")},
        "export": {"latency_ms_mean": latency.get("mean"), "latency_ms_p95": latency.get("p95"),
                   "peak_mem_mb": exp.get("peak_mem_mb"), "within_budget": exp.get("within_budget"),
                   "runtime": exp.get("runtime")},
        "thresholds": _thresholds(config),
    }


def _comparison(state: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute the in-domain→external comparison from the metric blocks."""
    from evaluate import _min_class_f1  # local import to avoid a heavy import cycle

    test = state.get("test_metrics", {})
    ext = state.get("external_domain_metrics", {})
    t_acc, e_acc = test.get("accuracy"), ext.get("accuracy")
    t_f1, e_f1 = test.get("macro_f1"), ext.get("macro_f1")
    comp: Dict[str, Any] = {
        "accuracy_drop_pts": None if t_acc is None or e_acc is None else (t_acc - e_acc) * 100.0,
        "macro_f1_drop_pts": None if t_f1 is None or e_f1 is None else (t_f1 - e_f1) * 100.0,
        "min_class_f1_in_domain": _min_class_f1(test.get("per_class", {})) if test.get("per_class") else {"class": None, "f1": None},
        "min_class_f1_external": _min_class_f1(ext.get("per_class", {})) if ext.get("per_class") else {"class": None, "f1": None},
    }
    return comp


# ---------------------------------------------------------------------------
# Deterministic gap evaluation (the guardrail + routing truth)
# ---------------------------------------------------------------------------
def compute_gap_statuses(metrics: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Return {gap_name: {'status': pass|fail, 'evidence': str}} straight from the numbers."""
    th = metrics["thresholds"]
    comp = metrics["comparison"]
    out: Dict[str, Dict[str, str]] = {}

    def _mk(status: bool, evidence: str) -> Dict[str, str]:
        return {"status": "fail" if status else "pass", "evidence": evidence}

    drop = comp.get("accuracy_drop_pts")
    out["domain_generalization_gap"] = _mk(
        drop is not None and drop > th["domain_gap_pts"],
        f"accuracy drop in→external = {_r(drop)} pts vs threshold {th['domain_gap_pts']} pts",
    )

    mcf1 = (comp.get("min_class_f1_in_domain") or {}).get("f1")
    worst = (comp.get("min_class_f1_in_domain") or {}).get("class")
    out["minority_class_gap"] = _mk(
        mcf1 is not None and mcf1 < th["min_class_f1"],
        f"worst per-class F1 = {_r(mcf1)} ({worst}) vs threshold {th['min_class_f1']}",
    )

    leak = metrics["dedup"].get("leakage_pairs_remaining")
    out["leakage_gap"] = _mk(
        leak is not None and leak > 0,
        f"cross-split near-duplicate leakage pairs remaining = {leak} (threshold 0)",
    )

    faith = metrics["explainability"].get("faithfulness_score")
    cov = metrics["explainability"].get("coverage_complete")
    out["explainability_faithfulness_gap"] = _mk(
        (faith is not None and faith < th["faithfulness_min"]) or cov is False,
        f"faithfulness_score = {_r(faith)} vs min {th['faithfulness_min']}; "
        f"coverage_complete = {cov}",
    )

    lat = metrics["export"].get("latency_ms_mean")
    mem = metrics["export"].get("peak_mem_mb")
    out["deployment_feasibility_gap"] = _mk(
        (lat is not None and lat > th["latency_ms"]) or (mem is not None and mem > th["peak_mem_mb"]),
        f"latency {_r(lat)} ms vs {th['latency_ms']} ms; peak_mem {_r(mem)} MB vs {th['peak_mem_mb']} MB",
    )
    return out


def _r(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{float(v):.3f}".rstrip("0").rstrip(".")


def decide_overall_status(gap_statuses: Dict[str, Dict[str, str]], retries_remaining: int) -> Tuple[str, Optional[str]]:
    """Authoritative routing decision shared by the agent and the graph.

    Returns (overall_status, loopback_node_or_None).
    """
    failed = [g for g in GAP_NAMES if gap_statuses[g]["status"] == "fail"]
    if not failed:
        return "ready_to_ship", None
    fixable_failed = [g for g in failed if g in FIXABLE_GAPS]
    if retries_remaining > 0 and fixable_failed:
        # Prefer the earliest fixable gap in checklist order.
        return "needs_rework", GAP_TO_NODE[fixable_failed[0]]
    return "ship_with_documented_limitations", None


# ---------------------------------------------------------------------------
# Agent entry
# ---------------------------------------------------------------------------
def run_limitations_audit(
    state: Dict[str, Any],
    chat_model,
    retries_remaining: int = 0,
    logger=None,
    tracker=None,
) -> Dict[str, Any]:
    metrics = build_metrics(state)
    system = _load_system_prompt()
    user = (
        "Evaluate the five-gap limitations checklist over these metrics. Compare each gap to its "
        "threshold and cite the numbers.\n\n" + json.dumps(metrics, indent=2, default=str)
    )
    llm_result: LimitationsAuditOutput = chat_model.structured_output(system, user, LimitationsAuditOutput)

    # ---- deterministic reconciliation ----
    truth = compute_gap_statuses(metrics)
    llm_by_name = {g.gap_name: g for g in llm_result.checklist}
    overrides = []
    reconciled = []
    for name in GAP_NAMES:
        det = truth[name]
        llm_gap = llm_by_name.get(name)
        suggested = llm_gap.suggested_fix if llm_gap and llm_gap.suggested_fix else "See gap-specific remediation."
        llm_evidence = llm_gap.evidence if llm_gap else ""
        if llm_gap and llm_gap.status != det["status"]:
            overrides.append({"gap": name, "llm": llm_gap.status, "deterministic": det["status"]})
        evidence = det["evidence"] if not llm_evidence else f"{llm_evidence} | verified: {det['evidence']}"
        reconciled.append(GapResult(gap_name=name, status=det["status"], evidence=evidence, suggested_fix=suggested))

    overall, loopback = decide_overall_status(truth, retries_remaining)
    final = LimitationsAuditOutput(checklist=reconciled, overall_status=overall,
                                   summary=llm_result.summary)
    output = final.model_dump()
    output["_deterministic"] = {
        "gap_statuses": {g: truth[g]["status"] for g in GAP_NAMES},
        "overrides": overrides,
        "loopback_node": loopback,
        "retries_remaining": retries_remaining,
    }

    if logger is not None:
        logger.info("Agent 2 (limitations audit): overall=%s, failed=%s%s",
                    overall, [g for g in GAP_NAMES if truth[g]["status"] == "fail"],
                    f", {len(overrides)} LLM status override(s)" if overrides else "")
    if tracker is not None:
        tracker.log_agent_call(
            agent="limitations_audit",
            request={"provider": chat_model.describe(), "metrics": metrics},
            response=output,
            meta={"call_count": chat_model.call_count, "overrides": overrides, "loopback": loopback},
        )
    return output
