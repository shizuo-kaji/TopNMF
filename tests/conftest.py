"""Shared pytest fixtures for TopNMF tests."""

from __future__ import annotations

import importlib.util
from importlib import import_module
from pathlib import Path
from types import ModuleType

import pytest


def _load_source_module(module_name: str, source_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def np():
    return pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


@pytest.fixture(scope="module")
def signal_generation_module(project_root: Path) -> ModuleType:
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    return _load_source_module(
        "topnmf_signal_generation_under_test",
        project_root / "TopNMF" / "signal_generation.py",
    )


@pytest.fixture(scope="module")
def nmf_utils_module(project_root: Path) -> ModuleType:
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    return _load_source_module(
        "topnmf_nmf_utils_under_test",
        project_root / "TopNMF" / "nmf_utils.py",
    )


@pytest.fixture(scope="module")
def topological_nmf_module() -> ModuleType:
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    pytest.importorskip("matplotlib")
    pytest.importorskip("tqdm")
    pytest.importorskip("torch_topological")
    pytest.importorskip("gudhi")
    pytest.importorskip("ripser")
    return import_module("TopNMF.topological_nmf")


@pytest.fixture
def time_array(np):
    return np.linspace(0.0, 2.0 * np.pi, 64)
