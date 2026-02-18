"""Shared pytest fixtures for TopNMF tests."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def np():
    return pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


@pytest.fixture
def time_array(np):
    return np.linspace(0.0, 2.0 * np.pi, 64)
