#!/usr/bin/env python3
"""CLI entry point — Malabar plant-disease classifier + agentic audit workflow.

Usage
-----
Full agentic graph (needs an LLM key for the two agent gates, or ``--mock``)::

    GEMINI_API_KEY=... python run_pipeline.py --profile fast
    python run_pipeline.py --profile fast --mock          # offline, canned agents

Resume after the ``await_human_review`` interrupt (Agent 1 blocked)::

    python run_pipeline.py --resume --mock                # latest paused run
    python run_pipeline.py --resume --run-id <id>

Run a single Part-A stage standalone (deterministic, no key needed). Prerequisite
stages are auto-run in the same run directory and their artifacts reused::

    python run_pipeline.py --stage data --profile fast
    python run_pipeline.py --stage evaluate --profile fast

Common flags: ``--config``, ``--profile {fast,full}``, ``--override a.b=c`` (repeatable),
``--device {auto,cpu,cuda,mps}``, ``--simsiam``, ``--run-id``, ``--run-dir``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import (
    PROJECT_ROOT,
    RunPaths,
    apply_overrides,
    cfg_get,
    generate_run_id,
    get_logger,
    load_config,
    load_dotenv,
    profile_value,
    resolve_device,
    save_json,
    set_global_seed,
)
from state import PipelineState, load_state, new_state, save_state

STAGE_ORDER: List[str] = ["data", "augment", "train", "evaluate", "explain", "export"]


# ---------------------------------------------------------------------------
# default offline mock agents (also imported by tests)
# ---------------------------------------------------------------------------
def _safe_json(text: str) -> Dict[str, Any]:
    import json

    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return {}
    return {}


def _mock_data_audit(system: str, user: str) -> Dict[str, Any]:
    """Grounded, deterministic stand-in for Agent 1 (parses the evidence JSON)."""
    ev = _safe_json(user)
    findings: List[Dict[str, str]] = []
    leak = (ev.get("dedup_report") or {}).get("leakage_pairs_remaining", 0) or 0
    if leak > 0:
        findings.append({"finding": "Residual cross-split near-duplicate leakage.",
                         "evidence": f"dedup_report.leakage_pairs_remaining={leak} (>0).",
                         "severity": "blocking"})
    else:
        findings.append({"finding": "No cross-split near-duplicate leakage after cleaning.",
                         "evidence": "dedup_report.leakage_pairs_remaining=0.", "severity": "info"})
    for c in (ev.get("declared_unpopulated_classes") or []):
        findings.append({"finding": f"Canonical class '{c}' is declared but unpopulated.",
                         "evidence": f"class_totals['{c}']=0 (no images on disk).", "severity": "warning"})
    totals = {k: v for k, v in (ev.get("class_totals") or {}).items() if isinstance(v, (int, float))}
    populated = [v for v in totals.values() if v > 0]
    if populated:
        mx, mn = max(populated), min(populated)
        ratio = mx / mn if mn else 0.0
        findings.append({"finding": "Class-count imbalance across active classes.",
                         "evidence": f"max/min active class total = {mx}/{mn} = {ratio:.1f}x.",
                         "severity": "warning" if ratio > 5 else "info"})
    has_block = any(f["severity"] == "blocking" for f in findings)
    has_warn = any(f["severity"] == "warning" for f in findings)
    return {
        "findings": findings,
        "risk_level": "high" if has_block else ("medium" if has_warn else "low"),
        "recommendation": ("Resolve the blocking leakage before training."
                           if has_block else "Proceed; carry the warnings into the dataset card."),
        "proceed": not has_block,
    }


def _mock_limitations_audit(system: str, user: str) -> Dict[str, Any]:
    """Stand-in for Agent 2. Statuses are recomputed deterministically downstream,
    so returning all-pass here is safe — the reconciliation layer sets the truth."""
    from agents.schemas import GAP_NAMES

    fixes = {
        "domain_generalization_gap": "Increase domain-randomization strength or collect target-domain data.",
        "minority_class_gap": "Oversample/reweight the minority class and retrain.",
        "leakage_gap": "Re-run dedup and split; keep near-duplicate groups intra-split.",
        "explainability_faithfulness_gap": "Improve localization / add samples per class.",
        "deployment_feasibility_gap": "Quantize to int8 or reduce input resolution.",
    }
    checklist = [{"gap_name": g, "status": "pass",
                  "evidence": "mock: see verified metrics", "suggested_fix": fixes.get(g, "n/a")}
                 for g in GAP_NAMES]
    return {"checklist": checklist, "overall_status": "ready_to_ship",
            "summary": ("Mock audit; deterministic reconciliation assigns the authoritative "
                        "gap statuses and overall ship decision from the numbers.")}


def build_default_mock_responder() -> Dict[str, Any]:
    return {"DataAuditOutput": _mock_data_audit, "LimitationsAuditOutput": _mock_limitations_audit}


# ---------------------------------------------------------------------------
# config / run context
# ---------------------------------------------------------------------------
def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(args.config)
    overrides = list(args.override or [])
    if args.profile:
        cfg["run"]["profile"] = args.profile
    if args.simsiam:
        overrides.append("model.simsiam.enabled=true")
    cfg = apply_overrides(cfg, overrides)
    return cfg


def resolve_run_paths(args: argparse.Namespace, cfg: Dict[str, Any], *, fresh: bool,
                      standalone: bool) -> RunPaths:
    output_root = cfg_get(cfg, "run.output_root", "runs")
    if args.run_dir:
        root = Path(args.run_dir)
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        return RunPaths(root=root, run_id=root.name)
    if args.run_id:
        return RunPaths.create(output_root, args.run_id)
    if standalone:
        # stable id so successive --stage calls share one run directory
        return RunPaths.create(output_root, f"standalone-{cfg_get(cfg, 'run.profile', 'fast')}")
    if fresh:
        return RunPaths.create(output_root, generate_run_id())
    # resume without an explicit id -> discover the latest resumable run
    found = _find_resumable(output_root)
    if found is None:
        raise SystemExit("No resumable run found. Pass --run-id or --run-dir.")
    return RunPaths(root=found, run_id=found.name)


def _find_resumable(output_root: str) -> Optional[Path]:
    base = Path(output_root)
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    if not base.exists():
        return None
    candidates = []
    for d in base.iterdir():
        state_file = d / "pipeline_state.json"
        if d.is_dir() and state_file.exists():
            candidates.append(d)
    if not candidates:
        return None
    # Prefer runs that are actually paused at an interrupt; else most recent.
    def _paused(d: Path) -> bool:
        try:
            st = load_state(d / "pipeline_state.json")
            return bool(st.get("pending_interrupt"))
        except Exception:
            return False

    paused = [d for d in candidates if _paused(d)]
    pool = paused or candidates
    return sorted(pool, key=lambda p: p.name)[-1]


# ---------------------------------------------------------------------------
# standalone single-stage runners (finer-grained than the graph nodes)
# ---------------------------------------------------------------------------
def _stage_done(stage: str, state: PipelineState) -> bool:
    return {
        "data": bool(state.get("split_manifest")),
        "augment": bool(state.get("augmentation_summary")),
        "train": bool(state.get("_checkpoint")),
        "evaluate": bool(state.get("test_metrics")),
        "explain": bool(state.get("explainability_report")),
        "export": bool(state.get("export_report")),
    }[stage]


def _run_one_stage(stage: str, cfg, run_paths, profile, device, logger, tracker,
                   state: PipelineState) -> None:
    if stage == "data":
        from data import run_data_stage
        out = run_data_stage(cfg, run_paths, profile, logger)
        state["class_domain_counts"] = out["class_domain_counts"]
        state["dedup_report"] = out["dedup_report"]
        state["cleaning_log"] = out["cleaning_log"]
        state["split_manifest"] = out["split_manifest"]
    elif stage == "augment":
        from augment import run_augment_stage
        out = run_augment_stage(cfg, run_paths, profile, state["split_manifest"], logger)
        state["augmentation_summary"] = out["augmentation_summary"]
    elif stage == "train":
        from train import train_model
        tr = train_model(cfg, run_paths, profile, state["split_manifest"], device,
                         logger=logger, tracker=tracker)
        state["_checkpoint"] = tr["checkpoint"]
    elif stage == "evaluate":
        from evaluate import run_evaluate_stage
        ev = run_evaluate_stage(cfg, run_paths, profile, state["split_manifest"], device,
                                state["_checkpoint"], logger=logger, tracker=tracker)
        state["cv_metrics"] = ev["cv_metrics"]
        state["test_metrics"] = ev["test_metrics"]
        state["external_domain_metrics"] = ev["external_domain_metrics"]
        state["confusion_matrices"] = ev["confusion_matrices"]
    elif stage == "explain":
        from explain import run_explain_stage
        state["explainability_report"] = run_explain_stage(
            cfg, run_paths, profile, state["split_manifest"], device,
            state["_checkpoint"], logger=logger, tracker=tracker)
    elif stage == "export":
        from export import run_export_stage
        state["export_report"] = run_export_stage(
            cfg, run_paths, profile, state["split_manifest"], device,
            state["_checkpoint"], logger=logger, tracker=tracker)
    else:  # pragma: no cover
        raise ValueError(f"Unknown stage {stage!r}")
    save_state(state, run_paths.state_path)


def run_standalone(target: str, cfg, run_paths, profile, device, logger, tracker) -> PipelineState:
    """Run `target` plus any missing prerequisite stages, reusing prior artifacts."""
    state = _load_or_new_state(cfg, run_paths, profile)
    target_idx = STAGE_ORDER.index(target)
    for i, stage in enumerate(STAGE_ORDER):
        if i > target_idx:
            break
        if i < target_idx and _stage_done(stage, state):
            logger.info("[stage] %s already present — reusing.", stage)
            continue
        logger.info("[stage] running %s (%s)", stage, "target" if i == target_idx else "prerequisite")
        _run_one_stage(stage, cfg, run_paths, profile, device, logger, tracker, state)
    return state


def _load_or_new_state(cfg, run_paths, profile) -> PipelineState:
    if run_paths.state_path.exists():
        state = load_state(run_paths.state_path)
        state["config"] = cfg  # refresh in case overrides changed
        state["profile"] = profile
        return state
    return new_state(cfg, run_paths.run_id, cfg_get(cfg, "run.output_root", "runs"), profile)


# ---------------------------------------------------------------------------
# full-graph execution (langgraph if available, else MiniGraph)
# ---------------------------------------------------------------------------
def _make_chat_model(cfg, use_mock: bool, logger):
    from agents.llm_provider import StructuredLLMError, get_chat_model

    if use_mock:
        logger.info("Using offline MOCK LLM provider for the audit agents.")
        return get_chat_model(cfg, mock_responder=build_default_mock_responder())
    try:
        model = get_chat_model(cfg)
        logger.info("LLM provider: %s", model.describe())
        return model
    except StructuredLLMError as exc:
        raise SystemExit(
            f"\nCannot start the agent gates: {exc}\n"
            "Set the key in .env (e.g. GEMINI_API_KEY=...) or run with --mock for an offline demo.\n"
        )


def run_full_graph(cfg, run_paths, profile, device, logger, tracker, chat_model,
                   resume: bool) -> PipelineState:
    from workflow_graph import GraphInterrupt, MiniGraph, Pipeline, engine_name

    pipeline = Pipeline(run_paths, device, profile, chat_model=chat_model, tracker=tracker, logger=logger)

    if resume:
        if not run_paths.state_path.exists():
            raise SystemExit(f"No checkpoint at {run_paths.state_path} to resume.")
        state = load_state(run_paths.state_path)
        state["config"] = cfg
        logger.info("Resuming run %s from checkpoint.", run_paths.run_id)
    else:
        state = new_state(cfg, run_paths.run_id, cfg_get(cfg, "run.output_root", "runs"), profile)

    # Prefer real langgraph; fall back to MiniGraph on any driver/version issue.
    if engine_name() == "langgraph":
        try:
            return _drive_langgraph(pipeline, state, run_paths, logger, resume)
        except GraphInterrupt:
            raise
        except Exception as exc:  # pragma: no cover - langgraph absent here
            logger.warning("langgraph path failed (%s); falling back to MiniGraph.", exc)

    engine = MiniGraph(pipeline)
    if resume:
        return engine.resume(state)
    return engine.invoke(state)


def _drive_langgraph(pipeline, state, run_paths, logger, resume):  # pragma: no cover - langgraph absent here
    """Execute via a real compiled langgraph app with durable interrupt/resume."""
    from workflow_graph import N_INTERRUPT, GraphInterrupt, build_langgraph_app

    checkpointer = None
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        checkpointer = SqliteSaver.from_conn_string(str(run_paths.root / "langgraph.sqlite"))
    except Exception:
        checkpointer = None  # build_langgraph_app defaults to MemorySaver
    app = build_langgraph_app(pipeline, checkpointer=checkpointer)
    thread = {"configurable": {"thread_id": run_paths.run_id}}

    result = app.invoke(None if resume else state, thread)
    snap = app.get_state(thread)
    paused = bool(getattr(snap, "next", None)) and N_INTERRUPT in snap.next
    merged = dict(state)
    merged.update(result or {})
    if paused:
        payload = MiniGraph(pipeline)._interrupt_payload(merged)  # reuse payload builder
        merged["pending_interrupt"] = payload
        merged["halted"] = True
        merged["halt_reason"] = payload["reason"]
        save_state(merged, run_paths.state_path)
        raise GraphInterrupt(payload)
    merged["halted"] = False
    merged["pending_interrupt"] = None
    save_state(merged, run_paths.state_path)
    return merged


# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------
def _print_interrupt(payload: Dict[str, Any], run_paths: RunPaths) -> None:
    print("\n" + "=" * 74)
    print("⏸  PIPELINE PAUSED — await_human_review")
    print("=" * 74)
    print(f"Reason: {payload.get('reason')}")
    print(f"Risk level: {payload.get('risk_level')}")
    blocking = payload.get("blocking_findings") or []
    if blocking:
        print("\nBlocking findings:")
        for f in blocking:
            print(f"  • {f.get('finding')}  [{f.get('evidence')}]")
    print(f"\nRecommendation: {payload.get('recommendation')}")
    print(f"\nCheckpoint: {run_paths.state_path}")
    print("Resume after addressing the findings with:")
    print(f"    python run_pipeline.py --resume --run-id {run_paths.run_id}"
          " [--mock]")
    print("=" * 74 + "\n")


def _print_done(state: PipelineState, run_paths: RunPaths) -> None:
    print("\n" + "=" * 74)
    print("✅ PIPELINE COMPLETE")
    print("=" * 74)
    la = state.get("limitations_audit") or {}
    if la:
        print(f"Ship decision: {la.get('overall_status')}")
    if state.get("final_report_path"):
        print(f"Final report : {state['final_report_path']}")
    if state.get("dataset_card_path"):
        print(f"Dataset card : {state['dataset_card_path']}")
    if state.get("diagram_path"):
        print(f"Diagram      : {state['diagram_path']}")
    print(f"Run directory: {run_paths.root}")
    print("=" * 74 + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Malabar plant-disease classifier + agentic audit workflow.")
    p.add_argument("--config", default="config.yaml", help="Path to config YAML (default: config.yaml).")
    p.add_argument("--profile", choices=["fast", "full"], default=None, help="Override run.profile.")
    p.add_argument("--override", action="append", default=[], metavar="a.b=c",
                   help="Dotted config override (repeatable).")
    p.add_argument("--stage", choices=STAGE_ORDER, default=None,
                   help="Run a single Part-A stage standalone (prereqs auto-run).")
    p.add_argument("--resume", action="store_true", help="Resume a paused full-graph run.")
    p.add_argument("--mock", action="store_true", help="Use the offline mock LLM for agent gates.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                   help="Compute device (default: auto).")
    p.add_argument("--simsiam", action="store_true", help="Enable SimSiam SSL pretraining.")
    p.add_argument("--run-id", default=None, help="Explicit run id (for resume / stage sharing).")
    p.add_argument("--run-dir", default=None, help="Explicit run directory path.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    load_dotenv()  # pull GEMINI_API_KEY etc. from .env if present
    cfg = build_config(args)
    profile = cfg_get(cfg, "run.profile", "fast")
    set_global_seed(int(cfg.get("seed", 42)))
    device = resolve_device(args.device)

    # ---- standalone single stage (deterministic; no LLM) ----
    if args.stage:
        run_paths = resolve_run_paths(args, cfg, fresh=False, standalone=True)
        logger = get_logger("run_pipeline", run_paths.root / "run_pipeline.log")
        logger.info("Standalone stage=%s profile=%s device=%s run=%s",
                    args.stage, profile, device, run_paths.run_id)
        from tracking import get_tracker
        tracker = get_tracker(cfg, run_paths, run_paths.run_id, logger)
        tracker.start_run({"mode": f"stage:{args.stage}", "profile": profile})
        try:
            run_standalone(args.stage, cfg, run_paths, profile, device, logger, tracker)
            tracker.end_run("completed")
        except Exception:
            tracker.end_run("failed")
            raise
        print(f"\n✅ Stage '{args.stage}' complete. Artifacts under: {run_paths.root}\n")
        return 0

    # ---- full agentic graph (or resume) ----
    run_paths = resolve_run_paths(args, cfg, fresh=not args.resume, standalone=False)
    logger = get_logger("run_pipeline", run_paths.root / "run_pipeline.log")
    logger.info("Full graph run (resume=%s) profile=%s device=%s run=%s",
                args.resume, profile, device, run_paths.run_id)

    from tracking import get_tracker
    from workflow_graph import GraphInterrupt, engine_name

    tracker = get_tracker(cfg, run_paths, run_paths.run_id, logger)
    tracker.start_run({"mode": "resume" if args.resume else "full_graph", "profile": profile,
                       "engine": engine_name()})
    tracker.log_params({"profile": profile, "backbone": cfg_get(cfg, "model.backbone", ""),
                        "provider": cfg_get(cfg, "agents.provider", ""), "mock": args.mock,
                        "engine": engine_name()})
    chat_model = _make_chat_model(cfg, args.mock, logger)

    try:
        state = run_full_graph(cfg, run_paths, profile, device, logger, tracker, chat_model,
                               resume=args.resume)
    except GraphInterrupt as gi:
        tracker.end_run("paused")
        _print_interrupt(gi.payload, run_paths)
        return 0
    except Exception:
        tracker.end_run("failed")
        raise

    tracker.end_run("completed")
    _print_done(state, run_paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
