#!/usr/bin/env python3
"""Experiment tracking — offline-first, auto-upgrading.

Default backend is ``local``: everything (params, per-step metrics, artifacts,
**every LLM agent call, every graph edge decision**) is written to plain files
under ``runs/<run_id>/tracking/`` so an audit trail sits next to the metrics
with no network. ``mlflow`` / ``wandb`` are used *in addition* when the backend
is selected and the library is importable; otherwise we warn and stay local.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import get_logger, now_iso, save_json


def _truncate(value: Any, limit: int = 4000) -> Any:
    """Keep prompt/response logs bounded but human-readable."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"... <+{len(value) - limit} chars>"
    return value


class LocalTracker:
    """Files-only tracker (default). Everything is append-only and JSON/CSV."""

    backend = "local"

    def __init__(self, run_paths, run_id: str, project: str, logger=None):
        self.run_paths = run_paths
        self.run_id = run_id
        self.project = project
        self.dir = run_paths.tracking
        self.logger = logger or get_logger("tracking", self.dir / "tracking.log")
        self._metrics_csv = self.dir / "metrics.csv"
        self._metric_keys: List[str] = []
        self._params: Dict[str, Any] = {}
        self._artifacts: List[Dict[str, str]] = []
        self._agent_calls = self.dir / "agent_calls.jsonl"
        self._edges = self.dir / "edge_decisions.jsonl"

    # -- lifecycle ---------------------------------------------------------
    def start_run(self, meta: Optional[Dict[str, Any]] = None) -> None:
        save_json(
            {"run_id": self.run_id, "project": self.project, "backend": self.backend,
             "started": now_iso(), "meta": meta or {}},
            self.dir / "run.json",
        )
        self.logger.info("tracking run %s started (backend=%s)", self.run_id, self.backend)

    def log_params(self, params: Dict[str, Any]) -> None:
        self._params.update(params)
        save_json(self._params, self.dir / "params.json")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        row = {"step": step if step is not None else ""}
        row.update({k: v for k, v in metrics.items()})
        new_keys = [k for k in row if k not in self._metric_keys]
        if new_keys and self._metric_keys:
            # header grew — rewrite is overkill; extend header set and rely on DictWriter
            self._metric_keys.extend(new_keys)
        elif not self._metric_keys:
            self._metric_keys = list(row.keys())
        else:
            self._metric_keys.extend(new_keys)
        write_header = not self._metrics_csv.exists()
        with self._metrics_csv.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._metric_keys, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def log_artifact(self, path, name: Optional[str] = None) -> None:
        entry = {"name": name or Path(path).name, "path": str(path)}
        self._artifacts.append(entry)
        save_json(self._artifacts, self.dir / "artifacts.json")

    def _append_jsonl(self, target: Path, record: Dict[str, Any]) -> None:
        import json
        with target.open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def log_agent_call(self, agent: str, request: Dict[str, Any], response: Any,
                       meta: Optional[Dict[str, Any]] = None) -> None:
        self._append_jsonl(self._agent_calls, {
            "timestamp": now_iso(),
            "agent": agent,
            "request": {k: _truncate(v) for k, v in (request or {}).items()},
            "response": _truncate(response) if isinstance(response, str) else response,
            "meta": meta or {},
        })
        self.logger.info("agent call logged: %s", agent)

    def log_edge_decision(self, node: str, decision: str, detail: Optional[Dict[str, Any]] = None) -> None:
        self._append_jsonl(self._edges, {
            "timestamp": now_iso(), "node": node, "decision": decision, "detail": detail or {},
        })
        self.logger.info("edge decision: %s -> %s", node, decision)

    def end_run(self, status: str = "completed", meta: Optional[Dict[str, Any]] = None) -> None:
        save_json({"run_id": self.run_id, "status": status, "ended": now_iso(),
                   "meta": meta or {}}, self.dir / "run_end.json")
        self.logger.info("tracking run %s ended (%s)", self.run_id, status)


class _WrappedTracker(LocalTracker):
    """Local tracker + best-effort mirror to mlflow/wandb when available."""

    def __init__(self, backend: str, remote, run_paths, run_id, project, logger=None):
        super().__init__(run_paths, run_id, project, logger)
        self.backend = backend
        self._remote = remote

    def log_params(self, params):
        super().log_params(params)
        try:
            self._remote["log_params"](params)
        except Exception as exc:  # pragma: no cover - remote optional
            self.logger.warning("remote log_params failed: %s", exc)

    def log_metrics(self, metrics, step=None):
        super().log_metrics(metrics, step)
        try:
            self._remote["log_metrics"](metrics, step)
        except Exception as exc:  # pragma: no cover
            self.logger.warning("remote log_metrics failed: %s", exc)

    def end_run(self, status="completed", meta=None):
        super().end_run(status, meta)
        try:
            self._remote["end_run"]()
        except Exception as exc:  # pragma: no cover
            self.logger.warning("remote end_run failed: %s", exc)


def _try_mlflow(project, run_id):  # pragma: no cover - offline here
    import mlflow
    mlflow.set_experiment(project)
    mlflow.start_run(run_name=run_id)
    return {
        "log_params": lambda p: mlflow.log_params({k: str(v) for k, v in p.items()}),
        "log_metrics": lambda m, s: mlflow.log_metrics(
            {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}, step=s),
        "end_run": mlflow.end_run,
    }


def _try_wandb(project, run_id):  # pragma: no cover - offline here
    import wandb
    run = wandb.init(project=project, name=run_id, reinit=True)
    return {
        "log_params": lambda p: run.config.update(p, allow_val_change=True),
        "log_metrics": lambda m, s: run.log({k: v for k, v in m.items()}, step=s),
        "end_run": run.finish,
    }


def get_tracker(config: Dict[str, Any], run_paths, run_id: str, logger=None) -> LocalTracker:
    track_cfg = config.get("tracking", {})
    backend = str(track_cfg.get("backend", "local")).lower()
    project = str(track_cfg.get("project", "malabar-plant-disease"))
    logger = logger or get_logger("tracking", run_paths.tracking / "tracking.log")

    if backend in ("mlflow", "wandb"):
        try:
            remote = _try_mlflow(project, run_id) if backend == "mlflow" else _try_wandb(project, run_id)
            tracker = _WrappedTracker(backend, remote, run_paths, run_id, project, logger)
            tracker.start_run()
            return tracker
        except Exception as exc:
            logger.warning("%s unavailable (%s) — falling back to local tracker", backend, exc)

    tracker = LocalTracker(run_paths, run_id, project, logger)
    tracker.start_run()
    return tracker
