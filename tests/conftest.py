"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
import torch


@pytest.fixture
def time_array():
    """Create a standard time array for testing."""
    return np.linspace(0, 4 * np.pi, 200)


@pytest.fixture
def simple_signal(time_array):
    """Create a simple cosine signal."""
    return np.cos(time_array)


@pytest.fixture
def periodic_signal(time_array):
    """Create a periodic signal suitable for TDA."""
    return np.cos(2 * time_array)


@pytest.fixture
def random_matrix():
    """Create a random non-negative matrix."""
    np.random.seed(42)
    return np.abs(np.random.randn(50, 30))


@pytest.fixture
def small_image():
    """Create a small test image."""
    np.random.seed(42)
    return np.random.rand(8, 8)


@pytest.fixture
def torch_image(small_image):
    """Convert small image to torch tensor."""
    return torch.tensor(small_image, dtype=torch.float32)


@pytest.fixture
def simple_graph_edges():
    """Create simple graph edges for testing."""
    return [(0, 1), (1, 2), (2, 0), (0, 3)]


@pytest.fixture
def simple_edge_weights():
    """Create edge weights for testing."""
    return torch.tensor([0.1, 0.2, 0.3, 0.5], dtype=torch.float32)
