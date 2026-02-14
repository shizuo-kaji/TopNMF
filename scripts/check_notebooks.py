#!/usr/bin/env python3
"""Execute notebooks under `notebook/` and report pass/fail status."""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

try:
    import nbformat
    from nbclient import NotebookClient
except ImportError as exc:  # pragma: no cover - import guard
    missing = getattr(exc, "name", "nbclient/nbformat")
    print(
        "Missing notebook execution dependency: "
        f"{missing}. Install with: python -m pip install nbclient nbformat",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Jupyter notebooks and fail on execution errors.",
    )
    parser.add_argument(
        "--notebook-dir",
        default="notebook",
        help="Directory containing notebooks (default: notebook).",
    )
    parser.add_argument(
        "--pattern",
        default="*.ipynb",
        help="Glob-style pattern to select notebooks (default: *.ipynb).",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob-style pattern to exclude (repeatable).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Per-cell timeout in seconds (default: 900).",
    )
    parser.add_argument(
        "--kernel",
        default=None,
        help="Kernel name override. Defaults to notebook metadata kernel.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first notebook failure.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List matching notebooks without executing them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full traceback for notebook execution failures.",
    )
    return parser.parse_args()


def _matches_any(path: Path, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)


def _discover_notebooks(notebook_dir: Path, pattern: str, exclude: list[str]) -> list[Path]:
    notebooks = sorted(p for p in notebook_dir.rglob("*.ipynb") if p.is_file())
    notebooks = [p for p in notebooks if fnmatch.fnmatch(p.name, pattern)]
    notebooks = [p for p in notebooks if not _matches_any(p, exclude)]
    return notebooks


def _execute_notebook(path: Path, timeout: int, kernel: str | None) -> tuple[bool, float, str, str]:
    start = time.perf_counter()
    try:
        notebook = nbformat.read(path, as_version=4)
        client_kwargs = {
            "nb": notebook,
            "timeout": timeout,
            "allow_errors": False,
            "resources": {"metadata": {"path": str(path.parent)}},
        }
        if kernel:
            client_kwargs["kernel_name"] = kernel
        client = NotebookClient(**client_kwargs)
        client.execute()
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return False, elapsed, f"{type(exc).__name__}: {exc}", traceback.format_exc()

    elapsed = time.perf_counter() - start
    return True, elapsed, "", ""


def _configure_runtime_dirs() -> Path:
    """
    Configure writable runtime/cache directories for notebook execution.

    This avoids failures in sandboxed environments where HOME-based locations
    are read-only.
    """
    runtime_root = Path(tempfile.mkdtemp(prefix="topnmf-notebook-runtime-"))
    dir_map = {
        "IPYTHONDIR": runtime_root / "ipython",
        "JUPYTER_RUNTIME_DIR": runtime_root / "jupyter_runtime",
        "MPLCONFIGDIR": runtime_root / "matplotlib",
        "XDG_CACHE_HOME": runtime_root / "cache",
    }
    for env_var, path in dir_map.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[env_var] = str(path)
    return runtime_root


def main() -> int:
    args = _parse_args()
    runtime_root = _configure_runtime_dirs()

    notebook_dir = Path(args.notebook_dir).resolve()
    if not notebook_dir.exists():
        print(f"Notebook directory does not exist: {notebook_dir}", file=sys.stderr)
        return 2

    notebooks = _discover_notebooks(notebook_dir, args.pattern, args.exclude)
    if not notebooks:
        print("No matching notebooks found.")
        return 0

    if args.list:
        print(f"Found {len(notebooks)} notebook(s):")
        for notebook in notebooks:
            print(f" - {notebook}")
        return 0

    print(f"Using runtime cache dir: {runtime_root}")
    print(f"Executing {len(notebooks)} notebook(s) from {notebook_dir}")
    failures: list[tuple[Path, str]] = []

    for notebook in notebooks:
        print(f"[RUN ] {notebook}")
        ok, elapsed, message, tb = _execute_notebook(
            path=notebook,
            timeout=args.timeout,
            kernel=args.kernel,
        )
        if ok:
            print(f"[PASS] {notebook} ({elapsed:.1f}s)")
            continue

        print(f"[FAIL] {notebook} ({elapsed:.1f}s)")
        print(f"       {message}")
        if args.verbose and tb:
            for line in tb.strip().splitlines():
                print(f"       {line}")
        failures.append((notebook, message))
        if args.fail_fast:
            break

    print("\nSummary")
    print(f" - Total:  {len(notebooks)}")
    print(f" - Passed: {len(notebooks) - len(failures)}")
    print(f" - Failed: {len(failures)}")

    if failures:
        print("\nFailures:")
        for notebook, message in failures:
            print(f" - {notebook}: {message}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
