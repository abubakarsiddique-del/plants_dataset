#!/usr/bin/env python3
"""Part B — the agentic StateGraph wiring the whole pipeline together.

Six **logical** stages, threaded through a single typed ``PipelineState``:

    1. data_ingestion        (deterministic)   stage 1
    2. data_audit            (Agent 1, LLM)    -> await_human_review interrupt
    3. preprocess_augment    (deterministic)   stage 3
    4. train_and_evaluate    (deterministic)   train + evaluate + explain      ┐ stage 4
       export_model          (deterministic)   ONNX + CPU benchmark            ┘ (split for cheap re-export)
    5. limitations_audit     (Agent 2, LLM)    -> needs_rework loop-back
    6. final_report          (deterministic)   report + dataset card + diagram

The node functions are **engine-agnostic** (``PipelineState -> PipelineState``).
The primary engine is a real ``langgraph.StateGraph`` with a checkpointer,
conditional edges and an ``await_human_review`` interrupt; when langgraph is not
installed (this offline env) an identical-semantics dependency-free ``MiniGraph``
runs instead. Both are resumable and enforce the bounded retry loop so the graph
always terminates.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from agents.data_audit_agent import run_data_audit
from agents.limitations_audit_agent import (
    GAP_TO_NODE,
    compute_gap_statuses,
    decide_overall_status,
    run_limitations_audit,
)
from common import cfg_get, get_logger, save_json
from state import PipelineState, load_state, record_history, save_state

# Logical node names (used by both engines + the diagram)
N_DATA = "data_ingestion"
N_DATA_AUDIT = "data_audit"
N_AUGMENT = "preprocess_augment"
N_TRAIN = "train_and_evaluate"
N_EXPORT = "export_model"
N_LIMITS = "limitations_audit"
N_REPORT = "final_report"
N_INTERRUPT = "await_human_review"

LINEAR_ORDER: List[str] = [N_DATA, N_DATA_AUDIT, N_AUGMENT, N_TRAIN, N_EXPORT, N_LIMITS, N_REPORT]
AGENT_NODES = {N_DATA_AUDIT, N_LIMITS}
END = "__end__"


class GraphInterrupt(Exception):
    """Raised by the MiniGraph engine to pause at await_human_review."""

    def __init__(self, payload: Dict[str, Any]):
        super().__init__(payload.get("reason", "human review required"))
        self.payload = payload


# ---------------------------------------------------------------------------
# Pipeline: holds run dependencies; methods are the node functions
# ---------------------------------------------------------------------------
class Pipeline:
    def __init__(self, run_paths, device, profile, chat_model=None, tracker=None, logger=None):
        self.run_paths = run_paths
        self.device = device
        self.profile = profile
        self.chat_model = chat_model
        self.tracker = tracker
        self.logger = logger or get_logger("workflow", run_paths.root / "workflow.log")
        self._checkpoint: Optional[str] = None

    # -- helpers ----------------------------------------------------------
    def _require_agent(self, node: str):
        if self.chat_model is None:
            raise RuntimeError(
                f"Node {node!r} needs an LLM. Set agents.provider + API key (see README), "
                "or pass a mock_responder for offline runs."
            )
        return self.chat_model

    def _max_retries(self, state: PipelineState) -> int:
        return int(cfg_get(state["config"], "retry.max_retries", 1))

    def _retries_used(self, state: PipelineState) -> int:
        return int(state.get("retries", {}).get("rework", 0))

    def _retries_remaining(self, state: PipelineState) -> int:
        return max(0, self._max_retries(state) - self._retries_used(state))

    # -- node 1: data ingestion & cleaning (deterministic) ----------------
    def data_ingestion(self, state: PipelineState) -> PipelineState:
        from data import run_data_stage

        self.logger.info("[node] %s", N_DATA)
        out = run_data_stage(state["config"], self.run_paths, self.profile, self.logger)
        state["class_domain_counts"] = out["class_domain_counts"]
        state["dedup_report"] = out["dedup_report"]
        state["cleaning_log"] = out["cleaning_log"]
        state["split_manifest"] = out["split_manifest"]
        record_history(state, "node", N_DATA,
                       {"images": out["dedup_report"].get("images_hashed"),
                        "leakage_pairs_remaining": out["dedup_report"].get("leakage_pairs_remaining")})
        return state

    # -- node 2: Agent 1 explainable data audit (LLM) ---------------------
    def data_audit(self, state: PipelineState) -> PipelineState:
        self.logger.info("[node] %s (Agent 1)", N_DATA_AUDIT)
        llm = self._require_agent(N_DATA_AUDIT)
        audit = run_data_audit(state, llm, logger=self.logger, tracker=self.tracker)
        state["data_audit"] = audit
        record_history(state, "node", N_DATA_AUDIT,
                       {"proceed": audit["proceed"], "risk_level": audit["risk_level"]})
        return state

    # -- node 3: preprocessing & augmentation (deterministic) -------------
    def preprocess_augment(self, state: PipelineState) -> PipelineState:
        from augment import run_augment_stage

        self.logger.info("[node] %s", N_AUGMENT)
        out = run_augment_stage(state["config"], self.run_paths, self.profile, state["split_manifest"], self.logger)
        state["augmentation_summary"] = out["augmentation_summary"]
        record_history(state, "node", N_AUGMENT, {"num_train_ops": out["augmentation_summary"]["num_train_ops"]})
        return state

    # -- node 4: train + evaluate + explain (deterministic) ---------------
    def train_and_evaluate(self, state: PipelineState) -> PipelineState:
        from evaluate import run_evaluate_stage
        from explain import run_explain_stage
        from train import train_model

        self.logger.info("[node] %s", N_TRAIN)
        if state.pop("_apply_reweight", False):
            self._apply_reweight_patch(state)

        tr = train_model(state["config"], self.run_paths, self.profile, state["split_manifest"],
                          self.device, tracker=self.tracker)
        self._checkpoint = tr["checkpoint"]
        state["_checkpoint"] = tr["checkpoint"]

        ev = run_evaluate_stage(state["config"], self.run_paths, self.profile, state["split_manifest"],
                                self.device, tr["checkpoint"], tracker=self.tracker)
        state["cv_metrics"] = ev["cv_metrics"]
        state["test_metrics"] = ev["test_metrics"]
        state["external_domain_metrics"] = ev["external_domain_metrics"]
        state["confusion_matrices"] = ev["confusion_matrices"]

        expl = run_explain_stage(state["config"], self.run_paths, self.profile, state["split_manifest"],
                                 self.device, tr["checkpoint"], tracker=self.tracker)
        state["explainability_report"] = expl
        record_history(state, "node", N_TRAIN,
                       {"best_val_macro_f1": tr["best_val_macro_f1"],
                        "test_accuracy": ev["test_metrics"]["accuracy"],
                        "external_accuracy": ev["external_domain_metrics"]["accuracy"]})
        return state

    def _apply_reweight_patch(self, state: PipelineState) -> None:
        """Deterministic gap-2 remediation: switch to focal loss + heavier SupCon."""
        loss = state["config"].setdefault("loss", {})
        before = loss.get("class_balancing")
        loss["class_balancing"] = "focal"
        loss["supcon_weight"] = float(loss.get("supcon_weight", 0.5)) + 0.25
        state.setdefault("cleaning_log", []).append(
            f"[retry] gap-2 reweight: class_balancing {before}->focal, supcon_weight+=0.25")
        self.logger.info("Applied gap-2 reweight patch (class_balancing=focal, supcon_weight bumped)")

    # -- node 4b: export (deterministic; also the gap-5 loop-back target) --
    def export_model(self, state: PipelineState) -> PipelineState:
        from export import run_export_stage

        self.logger.info("[node] %s", N_EXPORT)
        ckpt = state.get("_checkpoint") or self._checkpoint
        out = run_export_stage(state["config"], self.run_paths, self.profile, state["split_manifest"],
                               self.device, ckpt, tracker=self.tracker)
        state["export_report"] = out
        record_history(state, "node", N_EXPORT,
                       {"latency_ms_mean": out["latency_ms"]["mean"], "peak_mem_mb": out["peak_mem_mb"],
                        "within_budget": out["within_budget"]["overall"]})
        return state

    # -- node 5: Agent 2 limitations coverage audit (LLM) -----------------
    def limitations_audit(self, state: PipelineState) -> PipelineState:
        self.logger.info("[node] %s (Agent 2)", N_LIMITS)
        llm = self._require_agent(N_LIMITS)
        audit = run_limitations_audit(state, llm, retries_remaining=self._retries_remaining(state),
                                      logger=self.logger, tracker=self.tracker)
        state["limitations_audit"] = audit
        record_history(state, "node", N_LIMITS,
                       {"overall_status": audit["overall_status"],
                        "loopback": audit["_deterministic"]["loopback_node"]})
        return state

    # -- node 6: final report & dataset card (deterministic) --------------
    def final_report(self, state: PipelineState) -> PipelineState:
        from report import run_report_stage

        self.logger.info("[node] %s", N_REPORT)
        out = run_report_stage(state, self.run_paths, self.logger)
        state["final_report_path"] = out["final_report_path"]
        state["dataset_card_path"] = out["dataset_card_path"]
        state["diagram_path"] = out["diagram_path"]
        record_history(state, "node", N_REPORT, {"report": out["final_report_path"]})
        return state

    def node_map(self) -> Dict[str, Callable[[PipelineState], PipelineState]]:
        return {
            N_DATA: self.data_ingestion,
            N_DATA_AUDIT: self.data_audit,
            N_AUGMENT: self.preprocess_augment,
            N_TRAIN: self.train_and_evaluate,
            N_EXPORT: self.export_model,
            N_LIMITS: self.limitations_audit,
            N_REPORT: self.final_report,
        }

    # -- routing (shared truth for both engines) --------------------------
    def route_after_data_audit(self, state: PipelineState) -> str:
        """proceed==False -> interrupt (unless already resumed past it)."""
        audit = state.get("data_audit", {})
        if not audit.get("proceed", True) and not state.get("resumed"):
            return N_INTERRUPT
        return N_AUGMENT

    def route_after_limits(self, state: PipelineState) -> str:
        """needs_rework -> loop back to a fixable node; else -> report."""
        audit = state.get("limitations_audit", {})
        det = audit.get("_deterministic", {})
        overall = audit.get("overall_status")
        loopback = det.get("loopback_node")
        if overall == "needs_rework" and loopback:
            return loopback
        return N_REPORT


# ---------------------------------------------------------------------------
# MiniGraph engine (offline fallback; the path exercised in this env)
# ---------------------------------------------------------------------------
class MiniGraph:
    """Dependency-free executor with langgraph-equivalent semantics.

    Sequential execution with conditional routing, an ``await_human_review``
    interrupt that checkpoints and stops, ``--resume`` from that checkpoint, and
    a bounded rework loop that always terminates.
    """

    def __init__(self, pipeline: Pipeline):
        self.p = pipeline
        self.nodes = pipeline.node_map()

    def _checkpoint(self, state: PipelineState) -> None:
        save_state(state, self.p.run_paths.state_path)

    def invoke(self, state: PipelineState, start_at: Optional[str] = None) -> PipelineState:
        order = LINEAR_ORDER
        current = start_at or order[0]
        guard = 0
        max_steps = 64  # generous upper bound; retries are separately capped
        while current not in (END, N_INTERRUPT):
            guard += 1
            if guard > max_steps:
                raise RuntimeError(f"MiniGraph exceeded {max_steps} steps — routing bug")
            self.nodes[current](state)
            self._checkpoint(state)

            nxt = self._route(current, state)
            if nxt == N_INTERRUPT:
                payload = self._interrupt_payload(state)
                state["pending_interrupt"] = payload
                state["halted"] = True
                state["halt_reason"] = payload["reason"]
                self._checkpoint(state)
                if self.p.tracker is not None:
                    self.p.tracker.log_edge_decision(current, "await_human_review", payload)
                self.p.logger.warning("PAUSED at %s — %s", N_INTERRUPT, payload["reason"])
                raise GraphInterrupt(payload)
            if self.p.tracker is not None and nxt != self._linear_next(current):
                self.p.tracker.log_edge_decision(current, nxt, {"conditional": True})
            record_history(state, "edge", f"{current}->{nxt}")
            current = nxt
        state["halted"] = False
        state["pending_interrupt"] = None
        self._checkpoint(state)
        return state

    def resume(self, state: PipelineState) -> PipelineState:
        """Continue after human review: clear interrupt, skip the gate, go to augment."""
        state["resumed"] = True
        state["halted"] = False
        state["pending_interrupt"] = None
        record_history(state, "edge", f"{N_INTERRUPT}->{N_AUGMENT}", )
        self.p.logger.info("RESUMED after human review -> %s", N_AUGMENT)
        return self.invoke(state, start_at=N_AUGMENT)

    def _linear_next(self, current: str) -> Optional[str]:
        idx = LINEAR_ORDER.index(current)
        return LINEAR_ORDER[idx + 1] if idx + 1 < len(LINEAR_ORDER) else END

    def _route(self, current: str, state: PipelineState) -> str:
        if current == N_DATA_AUDIT:
            return self.p.route_after_data_audit(state)
        if current == N_LIMITS:
            nxt = self.p.route_after_limits(state)
            if nxt != N_REPORT:
                # a rework loop-back: count it, and flag reweight if retraining
                state.setdefault("retries", {})
                state["retries"]["rework"] = state["retries"].get("rework", 0) + 1
                if nxt == N_TRAIN:
                    state["_apply_reweight"] = True
                self.p.logger.info("Agent 2 -> needs_rework: looping back to %s (retry %d/%d)",
                                   nxt, state["retries"]["rework"], self.p._max_retries(state))
            return nxt
        return self._linear_next(current)

    def _interrupt_payload(self, state: PipelineState) -> Dict[str, Any]:
        audit = state.get("data_audit", {})
        blocking = [f for f in audit.get("findings", []) if f.get("severity") == "blocking"]
        return {
            "node": N_INTERRUPT,
            "reason": "Agent 1 set proceed=False (blocking data finding).",
            "risk_level": audit.get("risk_level"),
            "blocking_findings": blocking,
            "recommendation": audit.get("recommendation"),
            "how_to_resume": "Address the findings (edit data/config), then re-run with --resume.",
        }


# ---------------------------------------------------------------------------
# Real langgraph engine (auto-selected when installed)
# ---------------------------------------------------------------------------
def _langgraph_available() -> bool:
    try:
        import langgraph  # noqa: F401
        return True
    except Exception:
        return False


def build_langgraph_app(pipeline: Pipeline, checkpointer=None):  # pragma: no cover - langgraph absent here
    """Wire the same node functions into a real langgraph StateGraph.

    Uses a dedicated ``await_human_review`` node with ``interrupt_before`` so the
    graph pauses for human input; conditional edges implement the audit gate and
    the bounded rework loop. Requires ``langgraph`` installed.
    """
    from langgraph.graph import END as LG_END
    from langgraph.graph import StateGraph

    try:
        from langgraph.checkpoint.memory import MemorySaver
    except Exception:  # older layout
        from langgraph.checkpoint import MemorySaver  # type: ignore

    g = StateGraph(PipelineState)
    g.add_node(N_DATA, pipeline.data_ingestion)
    g.add_node(N_DATA_AUDIT, pipeline.data_audit)
    g.add_node(N_INTERRUPT, lambda s: s)  # pause point (interrupt_before)
    g.add_node(N_AUGMENT, pipeline.preprocess_augment)
    g.add_node(N_TRAIN, pipeline.train_and_evaluate)
    g.add_node(N_EXPORT, pipeline.export_model)
    g.add_node(N_LIMITS, pipeline.limitations_audit)
    g.add_node(N_REPORT, pipeline.final_report)

    g.set_entry_point(N_DATA)
    g.add_edge(N_DATA, N_DATA_AUDIT)
    g.add_conditional_edges(N_DATA_AUDIT, pipeline.route_after_data_audit,
                            {N_INTERRUPT: N_INTERRUPT, N_AUGMENT: N_AUGMENT})
    g.add_edge(N_INTERRUPT, N_AUGMENT)
    g.add_edge(N_AUGMENT, N_TRAIN)
    g.add_edge(N_TRAIN, N_EXPORT)
    g.add_edge(N_EXPORT, N_LIMITS)

    def _lg_route_limits(state: PipelineState) -> str:
        nxt = pipeline.route_after_limits(state)
        if nxt != N_REPORT:
            state.setdefault("retries", {})
            state["retries"]["rework"] = state["retries"].get("rework", 0) + 1
            if nxt == N_TRAIN:
                state["_apply_reweight"] = True
        return nxt

    g.add_conditional_edges(N_LIMITS, _lg_route_limits,
                            {N_TRAIN: N_TRAIN, N_EXPORT: N_EXPORT, N_REPORT: N_REPORT})
    g.add_edge(N_REPORT, LG_END)

    checkpointer = checkpointer or MemorySaver()
    return g.compile(checkpointer=checkpointer, interrupt_before=[N_INTERRUPT])


def engine_name() -> str:
    return "langgraph" if _langgraph_available() else "minigraph"


# ---------------------------------------------------------------------------
# Mermaid diagram source (purple = agents, gray = deterministic)
# ---------------------------------------------------------------------------
def mermaid_source() -> str:
    lines = [
        "flowchart TD",
        f'    {N_DATA}["1 · Data ingestion & cleaning"]:::det',
        f'    {N_DATA_AUDIT}["2 · Agent 1 · Data Audit"]:::agent',
        f'    {N_INTERRUPT}(["await_human_review (interrupt)"]):::gate',
        f'    {N_AUGMENT}["3 · Preprocessing & augmentation"]:::det',
        f'    {N_TRAIN}["4 · Train · Evaluate · Explain"]:::det',
        f'    {N_EXPORT}["4b · Export ONNX + CPU benchmark"]:::det',
        f'    {N_LIMITS}["5 · Agent 2 · Limitations Audit"]:::agent',
        f'    {N_REPORT}["6 · Final report & dataset card"]:::det',
        f"    {N_DATA} --> {N_DATA_AUDIT}",
        f"    {N_DATA_AUDIT} -->|proceed=false| {N_INTERRUPT}",
        f"    {N_DATA_AUDIT} -->|proceed=true| {N_AUGMENT}",
        f"    {N_INTERRUPT} -->|--resume| {N_AUGMENT}",
        f"    {N_AUGMENT} --> {N_TRAIN}",
        f"    {N_TRAIN} --> {N_EXPORT}",
        f"    {N_EXPORT} --> {N_LIMITS}",
        f"    {N_LIMITS} -->|needs_rework · gap 2| {N_TRAIN}",
        f"    {N_LIMITS} -->|needs_rework · gap 5| {N_EXPORT}",
        f"    {N_LIMITS} -->|ship| {N_REPORT}",
        "    classDef agent fill:#8e6fd1,stroke:#5b3fa0,color:#ffffff;",
        "    classDef det fill:#d9d9d9,stroke:#8c8c8c,color:#000000;",
        "    classDef gate fill:#f2c14e,stroke:#b8860b,color:#000000;",
    ]
    return "\n".join(lines)
