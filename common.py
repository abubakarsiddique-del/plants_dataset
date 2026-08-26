#!/usr/bin/env python3
"""Shared utilities for the Malabar plant-disease pipeline (Parts A & B).

Seeds, device resolution, config load/merge/override, run-id + run directory
layout, a zero-dependency ``.env`` loader, JSON/text IO, and small helpers.
Kept import-light (stdlib + numpy + optional torch) so every stage can depend
on it without pulling heavy libraries at import time.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import string
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:  # torch is present in this env, but keep common.py importable without it
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# .env loading (no python-dotenv dependency)
# ---------------------------------------------------------------------------
def load_dotenv(path: str | os.PathLike = ".env", *, override: bool = False) -> Dict[str, str]:
    """Parse a simple ``KEY=VALUE`` .env file into ``os.environ``.

    Existing environment variables win unless ``override=True``. Lines that are
    blank or start with ``#`` are ignored. Values may be optionally quoted.
    Returns the mapping that was parsed (whether or not it was applied).
    """
    env_path = Path(path)
    if not env_path.is_absolute():
        env_path = PROJECT_ROOT / env_path
    parsed: Dict[str, str] = {}
    if not env_path.exists():
        return parsed
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        parsed[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed


# ---------------------------------------------------------------------------
# Seeding & device
# ---------------------------------------------------------------------------
def set_global_seed(seed: int = 42) -> None:
    """Seed python/numpy/torch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(False)  # keep perf; seeds are enough
        except Exception:
            pass


def resolve_device(pref: str = "auto"):
    """Return a torch.device. ``auto`` -> cuda if available else cpu.

    MPS is only used when explicitly requested ("mps") because several ops used
    downstream (Grad-CAM hooks, ONNX export) are more reliable on CPU.
    """
    if not _HAS_TORCH:
        raise RuntimeError("torch is required for resolve_device()")
    pref = (pref or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Time / ids
# ---------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def generate_run_id(seed_hint: Optional[int] = None) -> str:
    """Timestamped run id with a short random suffix, e.g. 20260825-153000-a1b2."""
    rng = random.Random(seed_hint) if seed_hint is not None else random
    suffix = "".join(rng.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{utc_stamp()}-{suffix}"


# ---------------------------------------------------------------------------
# Config: load / merge / override / profile resolution
# ---------------------------------------------------------------------------
def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    import yaml

    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping, got {type(cfg)}")
    return cfg


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce_scalar(text: str) -> Any:
    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[Iterable[str]]) -> Dict[str, Any]:
    """Apply ``a.b.c=value`` dotted overrides (values coerced to bool/int/float)."""
    if not overrides:
        return cfg
    cfg = json.loads(json.dumps(cfg))  # deep copy via round-trip
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        dotted, _, value = item.partition("=")
        keys = dotted.strip().split(".")
        node = cfg
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot override into non-mapping at {key!r}")
        node[keys[-1]] = _coerce_scalar(value.strip())
    return cfg


def profile_value(value: Any, profile: str) -> Any:
    """Resolve a possibly profile-scoped value.

    ``{"fast": 3, "full": 40}`` with profile "fast" -> 3. Non-profiled values
    pass through unchanged.
    """
    if isinstance(value, dict) and value and set(value.keys()) <= {"fast", "full"}:
        if profile in value:
            return value[profile]
        return next(iter(value.values()))
    return value


def cfg_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


# ---------------------------------------------------------------------------
# Class-name helpers
# ---------------------------------------------------------------------------
_COUNT_SUFFIX = re.compile(r"\s*\(\d+\)\s*$")


def strip_class_count(folder_name: str) -> str:
    """``'Anthracnose(102)'`` -> ``'Anthracnose'`` (canonical class label)."""
    return _COUNT_SUFFIX.sub("", folder_name).strip()


# ---------------------------------------------------------------------------
# Run directory layout
# ---------------------------------------------------------------------------
@dataclass
class RunPaths:
    """Filesystem layout for a single pipeline run under ``<output_root>/<run_id>``."""

    root: Path
    run_id: str

    @classmethod
    def create(cls, output_root: str | os.PathLike, run_id: str) -> "RunPaths":
        base = Path(output_root)
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        rp = cls(root=base / run_id, run_id=run_id)
        for sub in (
            rp.root,
            rp.stage("stage1_data"),
            rp.stage("stage3_augment"),
            rp.stage("stage4_model"),
            rp.explain,
            rp.onnx,
            rp.reports,
            rp.tracking,
            rp.checkpoints,
        ):
            sub.mkdir(parents=True, exist_ok=True)
        return rp

    def stage(self, name: str) -> Path:
        return self.root / name

    @property
    def explain(self) -> Path:
        return self.root / "explain"

    @property
    def onnx(self) -> Path:
        return self.root / "onnx"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def tracking(self) -> Path:
        return self.root / "tracking"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def state_path(self) -> Path:
        return self.root / "pipeline_state.json"


# ---------------------------------------------------------------------------
# Logging & IO
# ---------------------------------------------------------------------------
def get_logger(name: str, log_file: Optional[str | os.PathLike] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


def save_json(obj: Any, path: str | os.PathLike, *, indent: int = 2) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=indent, default=_json_default), encoding="utf-8")
    return p


def load_json(path: str | os.PathLike) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_text(text: str, path: str | os.PathLike) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p
