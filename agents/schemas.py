#!/usr/bin/env python3
"""Structured-output schemas for the two audit agents (Part B).

Pydantic v2 models. Every agent is forced to return JSON matching these; the
LLM provider validates against them and re-asks on mismatch. Written for
Python 3.9 (explicit ``Optional``/``List``/``Literal``, no PEP 604 unions).
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Agent 1 — Explainable Data Audit
# ---------------------------------------------------------------------------
Severity = Literal["info", "warning", "blocking"]
RiskLevel = Literal["low", "medium", "high"]


class DataAuditFinding(BaseModel):
    finding: str = Field(..., description="One concrete observation about the dataset.")
    evidence: str = Field(
        ...,
        description="Must cite a specific number from the provided JSON "
        "(a count, fraction, hamming distance, or split size).",
    )
    severity: Severity


class DataAuditOutput(BaseModel):
    findings: List[DataAuditFinding] = Field(..., min_length=1)
    risk_level: RiskLevel
    recommendation: str
    proceed: bool = Field(
        ...,
        description="False halts the pipeline for human review. Set False when "
        "any finding is 'blocking'.",
    )


# ---------------------------------------------------------------------------
# Agent 2 — Limitations Coverage Audit
# ---------------------------------------------------------------------------
GapStatus = Literal["pass", "fail"]
OverallStatus = Literal["ready_to_ship", "ship_with_documented_limitations", "needs_rework"]

# Fixed, ordered checklist. The audit must return exactly these five, in order.
GAP_NAMES: List[str] = [
    "domain_generalization_gap",
    "minority_class_gap",
    "leakage_gap",
    "explainability_faithfulness_gap",
    "deployment_feasibility_gap",
]


class GapResult(BaseModel):
    gap_name: str = Field(..., description=f"One of {GAP_NAMES}, in order.")
    status: GapStatus
    evidence: str = Field(..., description="Cite the metric value(s) and the threshold compared against.")
    suggested_fix: str

    @field_validator("gap_name")
    @classmethod
    def _known_gap(cls, value: str) -> str:
        if value not in GAP_NAMES:
            raise ValueError(f"gap_name must be one of {GAP_NAMES}, got {value!r}")
        return value


class LimitationsAuditOutput(BaseModel):
    checklist: List[GapResult] = Field(..., description="Exactly 5 gaps, in the fixed order.")
    overall_status: OverallStatus
    summary: str

    @field_validator("checklist")
    @classmethod
    def _exactly_five_ordered(cls, checklist: List[GapResult]) -> List[GapResult]:
        names = [g.gap_name for g in checklist]
        if names != GAP_NAMES:
            raise ValueError(
                "checklist must contain exactly the 5 gaps in order "
                f"{GAP_NAMES}; got {names}"
            )
        return checklist
