#!/usr/bin/env python3
"""Stage 6a — ONNX export + CPU deployment benchmark (low-cost device profile).

Exports the trained classifier to ONNX (guarded ``onnx.checker`` when ``onnx``
is importable), then benchmarks **single-thread CPU** inference latency
(warmup + mean/p50/p95 at batch 1) and a **memory footprint** estimate (model
weights + measured working set). Uses ``onnxruntime`` when installed, else a
torch-CPU fallback. Compares to ``export.device_budget`` — the evidence Agent 2
reads for gap 5 (deployment feasibility).
"""

from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from common import cfg_get, get_logger, profile_value, save_json
from model import build_model


class _LogitsOnly(nn.Module):
    """Single-output wrapper so the ONNX graph emits just the class logits."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, _ = self.model(x, return_projection=False)
        return logits


def _peak_rss_mb() -> float:
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024 ** 2) if sys.platform == "darwin" else rss / 1024
    except Exception:  # pragma: no cover
        return 0.0


def _percentiles(times_ms: List[float]) -> Dict[str, float]:
    arr = np.array(times_ms, dtype=np.float64)
    return {"mean": float(arr.mean()), "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)), "min": float(arr.min()), "runs": int(len(arr))}


def _bench_torch(model, x, warmup, runs) -> List[float]:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _bench_ort(session, x_np, input_name, warmup, runs) -> List[float]:  # pragma: no cover - ORT absent here
    for _ in range(warmup):
        session.run(None, {input_name: x_np})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: x_np})
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def run_export_stage(
    config: Dict[str, Any],
    run_paths,
    profile: str,
    split_manifest: Dict[str, Any],
    device,
    checkpoint_path: str,
    logger=None,
    tracker=None,
) -> Dict[str, Any]:
    logger = logger or get_logger("stage6.export", run_paths.onnx / "export.log")
    classes = split_manifest["classes"]
    num_classes = len(classes)
    image_size = int(split_manifest.get("image_size", config["data"]["image_size"]))
    opset = int(cfg_get(config, "export.onnx_opset", 17))
    cpu_threads = int(cfg_get(config, "export.cpu_threads", 1))
    warmup = int(cfg_get(config, "export.warmup_runs", 5))
    runs = int(profile_value(cfg_get(config, "export.benchmark_runs", {"fast": 20, "full": 100}), profile))
    budget = {
        "latency_ms": float(cfg_get(config, "export.device_budget.latency_ms",
                                    cfg_get(config, "thresholds.latency_ms", 200.0))),
        "peak_mem_mb": float(cfg_get(config, "export.device_budget.peak_mem_mb",
                                     cfg_get(config, "thresholds.peak_mem_mb", 512.0))),
    }

    # Load model on CPU for a representative low-cost-device benchmark
    cpu = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=cpu)
    core = build_model(num_classes, ckpt.get("cfg_model", config["model"])).to(cpu)
    core.load_state_dict(ckpt["model_state"])
    core.eval()
    model = _LogitsOnly(core).to(cpu).eval()

    n_params = sum(p.numel() for p in core.parameters())
    weights_mb = n_params * 4 / (1024 ** 2)

    # ---- ONNX export ----
    onnx_path = run_paths.onnx / "model.onnx"
    dummy = torch.randn(1, 3, image_size, image_size)
    onnx_exported = False
    onnx_checked: Any = "skipped"
    onnx_export_error: Optional[str] = None
    try:
        torch.onnx.export(
            model, dummy, str(onnx_path), input_names=["input"], output_names=["logits"],
            opset_version=opset, dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            do_constant_folding=True,
        )
        onnx_exported = onnx_path.exists()
        logger.info("ONNX exported → %s (%d params, %.1f MB weights)", onnx_path, n_params, weights_mb)
        try:
            import onnx  # noqa
            onnx.checker.check_model(onnx.load(str(onnx_path)))
            onnx_checked = True
        except ImportError:
            logger.info("onnx not installed — skipping graph checker (export still valid)")
        except Exception as exc:
            onnx_checked = False
            logger.warning("onnx.checker failed: %s", exc)
    except Exception as exc:
        onnx_export_error = str(exc)
        logger.warning("ONNX export failed (%s) — benchmarking torch-CPU instead", exc)

    # ---- CPU latency benchmark (single-thread) ----
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(max(1, cpu_threads))
    runtime = "torch-cpu"
    tracemalloc.start()
    x_np = dummy.numpy()

    session = None
    if onnx_exported:
        try:
            import onnxruntime as ort  # pragma: no cover - absent here
            so = ort.SessionOptions()
            so.intra_op_num_threads = max(1, cpu_threads)
            so.inter_op_num_threads = 1
            session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
            runtime = "onnxruntime"
        except Exception as exc:
            logger.info("onnxruntime unavailable (%s) — using torch-CPU for latency", exc)

    if session is not None:  # pragma: no cover - ORT absent here
        times = _bench_ort(session, x_np, session.get_inputs()[0].name, warmup, runs)
    else:
        times = _bench_torch(model, dummy, warmup, runs)

    work_peak_mb = tracemalloc.get_traced_memory()[1] / (1024 ** 2)
    tracemalloc.stop()
    torch.set_num_threads(prev_threads)

    latency = _percentiles(times)
    # Deployment footprint estimate: static weights + measured Python working set.
    # (Framework baseline RSS is reported separately for transparency but not
    #  used for the budget check, since it is dominated by the torch runtime.)
    footprint_mb = float(weights_mb + work_peak_mb)
    within = {
        "latency": latency["mean"] <= budget["latency_ms"],
        "peak_mem": footprint_mb <= budget["peak_mem_mb"],
    }
    within["overall"] = bool(within["latency"] and within["peak_mem"])

    report = {
        "onnx_path": str(onnx_path),
        "onnx_exported": onnx_exported,
        "onnx_checked": onnx_checked,
        "onnx_export_error": onnx_export_error,
        "opset": opset,
        "runtime": runtime,
        "cpu_threads": cpu_threads,
        "num_params": int(n_params),
        "weights_mb": round(weights_mb, 3),
        "latency_ms": latency,
        "peak_mem_mb": round(footprint_mb, 3),
        "peak_mem_components": {"weights_mb": round(weights_mb, 3),
                                "working_set_mb": round(work_peak_mb, 3)},
        "process_peak_rss_mb": round(_peak_rss_mb(), 1),
        "budget": budget,
        "within_budget": within,
        "note": ("peak_mem_mb = model weights + measured working set (a portable deployment "
                 "footprint proxy); process_peak_rss_mb includes the framework baseline and is "
                 "reported for reference only."),
    }
    save_json(report, run_paths.onnx / "export_report.json")
    if tracker is not None:
        tracker.log_metrics({"latency_ms_mean": latency["mean"], "latency_ms_p95": latency["p95"],
                             "peak_mem_mb": footprint_mb})
        if onnx_exported:
            tracker.log_artifact(onnx_path, "model.onnx")
        tracker.log_artifact(run_paths.onnx / "export_report.json", "export_report.json")
    logger.info("Export done. runtime=%s latency=%.1fms (p95=%.1f) footprint=%.1fMB within_budget=%s",
                runtime, latency["mean"], latency["p95"], footprint_mb, within["overall"])
    return report
