#!/usr/bin/env python3
"""Offline test runner — executes the pytest suite when pytest isn't installed.

PyPI is blocked in this sandbox, so ``pip install pytest`` isn't possible. This
harness injects a tiny ``pytest`` shim (only the handful of hooks the suite uses:
``fixture``, ``raises``, ``skip``) plus a minimal function-scoped fixture
resolver, so the *exact same* test files can be run and verified here.

On any normal machine, ignore this file and just run::

    pytest -q

Usage here::

    ./venv/bin/python tests/run_smoke.py
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import tempfile
import traceback
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
ROOT = TESTS_DIR.parent


# ---------------------------------------------------------------------------
# minimal pytest shim (only what this suite uses)
# ---------------------------------------------------------------------------
class Skipped(Exception):
    pass


class _RaisesCtx:
    def __init__(self, expected):
        self.expected = expected
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"DID NOT RAISE {self.expected!r}")
        if not issubclass(et, self.expected):
            return False  # unexpected exception -> propagate
        self.value = ev
        return True


class _PytestShim:
    Skipped = Skipped

    @staticmethod
    def fixture(func=None, **kwargs):
        def mark(f):
            f.__is_fixture__ = True
            return f
        return mark(func) if func else mark

    @staticmethod
    def raises(expected):
        return _RaisesCtx(expected)

    @staticmethod
    def skip(msg=""):
        raise Skipped(msg)


sys.modules["pytest"] = _PytestShim()  # must precede any test import
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# fixture resolution (function scope, recursive)
# ---------------------------------------------------------------------------
FIXTURES = {}


def _register_fixtures(module):
    for name, obj in vars(module).items():
        if callable(obj) and getattr(obj, "__is_fixture__", False):
            FIXTURES[name] = obj


def _builtin(name):
    if name == "tmp_path":
        return Path(tempfile.mkdtemp(prefix="smoke-"))
    raise KeyError(f"no fixture named {name!r}")


def _resolve(name, cache):
    if name in cache:
        return cache[name]
    if name in FIXTURES:
        fn = FIXTURES[name]
        args = [_resolve(p, cache) for p in inspect.signature(fn).parameters]
        val = fn(*args)
    else:
        val = _builtin(name)
    cache[name] = val
    return val


def _load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def main() -> int:
    conftest = _load_module(TESTS_DIR / "conftest.py")
    _register_fixtures(conftest)

    test_files = sorted(p for p in TESTS_DIR.glob("test_*.py"))
    passed = failed = skipped = 0
    failures = []

    for path in test_files:
        module = _load_module(path)
        tests = [(n, o) for n, o in vars(module).items()
                 if n.startswith("test_") and inspect.isfunction(o)]
        tests.sort(key=lambda t: inspect.getsourcelines(t[1])[1])  # source order
        print(f"\n{path.name}")
        for name, fn in tests:
            cache = {}
            try:
                kwargs = {p: _resolve(p, cache) for p in inspect.signature(fn).parameters}
                fn(**kwargs)
                print(f"  ok    {name}")
                passed += 1
            except Skipped as exc:
                print(f"  skip  {name}  ({exc})")
                skipped += 1
            except Exception:
                print(f"  FAIL  {name}")
                failures.append((path.name, name, traceback.format_exc()))
                failed += 1

    print("\n" + "=" * 70)
    for fname, tname, tb in failures:
        print(f"\n----- FAILURE: {fname}::{tname} -----\n{tb}")
    print("=" * 70)
    print(f"passed={passed}  failed={failed}  skipped={skipped}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
