"""Shared pytest fixtures for the Malabar pipeline test-suite.

The suite is layered so most of it runs in seconds with **no torch, no images
and no network** — the orchestration tests stub the heavy deterministic nodes
and inject an offline mock LLM (see ``tests/test_workflow_orchestration.py``).
Only ``test_data_leakage.py`` touches the real dataset, and it skips itself when
the dataset is absent.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import pytest

# Make the project root importable regardless of where pytest is launched from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Headless matplotlib for the diagram renderer.
os.environ.setdefault("MPLBACKEND", "Agg")

from common import RunPaths, load_config  # noqa: E402


@pytest.fixture
def base_cfg():
    """A fresh, deep-copied config so a test mutating it cannot leak into others."""
    return copy.deepcopy(load_config(ROOT / "config.yaml"))


@pytest.fixture
def run_paths(tmp_path):
    """An isolated run directory under pytest's tmp_path."""
    return RunPaths.create(tmp_path, "test-run")


@pytest.fixture
def project_root():
    return ROOT
