#!/usr/bin/env python3
"""Agent 1 — Explainable Data Audit (Part B).

Reasons over the stage-1 bookkeeping (``class_domain_counts``, ``dedup_report``,
``cleaning_log`` + planned split sizes) — **JSON only, never pixels** — and
returns a ``DataAuditOutput``. A deterministic guardrail enforces the spec
invariant *any blocking finding ⇒ proceed=False* regardless of what the LLM
returns, so the human-review gate can never be silently skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from agents.schemas import DataAuditOutput
from common import cfg_get

_PROMPT = Path(__file__).resolve().parent / "prompts" / "data_audit_system.md"


def _load_system_prompt() -> str:
    return _PROMPT.read_text(encoding="utf-8")


def build_evidence(state: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the compact JSON payload the agent reasons over (no pixels, no per-file lists)."""
    config = state.get("config", {})
    sm = state.get("split_manifest", {})
    dedup = dict(state.get("dedup_report", {}))
    # Drop the (possibly long) file-name list; keep every number.
    dedup.pop("removed_files", None)

    profile = state.get("profile", "fast")
    cv_folds = cfg_get(config, "eval.cv_folds", {})
    if isinstance(cv_folds, dict):
        cv_folds = cv_folds.get(profile, next(iter(cv_folds.values()), None))

    class_totals = {c: sum(d.values()) for c, d in state.get("class_domain_counts", {}).items()}
    return {
        "canonical_classes": cfg_get(config, "data.canonical_classes", []),
        "active_classes": sm.get("classes", []),
        "declared_unpopulated_classes": sm.get("declared_unpopulated", []),
        "class_totals": class_totals,
        "class_domain_counts": state.get("class_domain_counts", {}),
        "proxy_domains": sm.get("domains", []),
        "dedup_report": dedup,
        "cleaning_log": state.get("cleaning_log", []),
        "planned_split_counts": sm.get("counts", {}),
        "planned_split_fractions": cfg_get(config, "data.splits", {}),
        "external_domain_test": {"mode": sm.get("external_mode"), "domain": sm.get("external_domain"),
                                 "n": sm.get("counts", {}).get("external_domain_test")},
        "cv_folds_planned": cv_folds,
    }


def run_data_audit(
    state: Dict[str, Any],
    chat_model,
    logger=None,
    tracker=None,
) -> Dict[str, Any]:
    """Return ``DataAuditOutput.model_dump()`` with the proceed guardrail applied."""
    evidence = build_evidence(state)
    system = _load_system_prompt()
    user = (
        "Audit this dataset's structure and bookkeeping. Reason only over the numbers below.\n\n"
        + json.dumps(evidence, indent=2, default=str)
    )
    result: DataAuditOutput = chat_model.structured_output(system, user, DataAuditOutput)

    # Deterministic guardrail: a blocking finding must halt, whatever the LLM said.
    has_blocking = any(f.severity == "blocking" for f in result.findings)
    forced = False
    if has_blocking and result.proceed:
        result.proceed = False
        forced = True
    if has_blocking and result.risk_level != "high":
        result.risk_level = "high"

    output = result.model_dump()
    output["_guardrail"] = {"blocking_findings": has_blocking, "proceed_forced_false": forced}

    if logger is not None:
        logger.info("Agent 1 (data audit): %d findings, risk=%s, proceed=%s%s",
                    len(result.findings), result.risk_level, result.proceed,
                    " (forced by blocking finding)" if forced else "")
    if tracker is not None:
        tracker.log_agent_call(
            agent="data_audit",
            request={"provider": chat_model.describe(), "evidence": evidence},
            response=output,
            meta={"call_count": chat_model.call_count, "guardrail_forced": forced},
        )
    return output
