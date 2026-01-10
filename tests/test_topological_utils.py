"""Tests for topological_utils module."""

import pytest
import numpy as np
import torch
from TopNMF.topological_utils import (
    center_point_cloud,
    center_point_cloud_torch,
    TimeDelayEmbeddingTorch,
    compute_periodicity_score,
    compute_persistence_diagram,
)


class TestCenterPointCloud:
    """Tests for center_point_cloud function."""

    def test_output_shape(self):
        """Test that output shape matches input."""
        X = np.random.randn(100, 3)
        result = center_point_cloud(X)
        assert result.shape == X.shape

    def test_normalized_rows(self):
        """Test that output rows are unit normalized."""
        X = np.random.randn(50, 4)
        result = center_point_cloud(X)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-6)

    def test_centered(self):
        """Test that projection onto ones vector is zero."""
        X = np.random.randn(30, 5)
        result = center_point_cloud(X)
        ones = np.ones(5)
        projections = result @ ones
        np.testing.assert_allclose(projections, 0.0, atol=1e-10)


class TestCenterPointCloudTorch:
    """Tests for center_point_cloud_torch function."""

    def test_output_shape(self):
        """Test that output shape matches input."""
        X = torch.randn(100, 3)
        result = center_point_cloud_torch(X)
        assert result.shape == X.shape

    def test_normalized_rows(self):
        """Test that output rows are unit normalized."""
        X = torch.randn(50, 4)
        result = center_point_cloud_torch(X)
        norms = torch.norm(result, dim=1)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5)

    def test_gradient_flow(self):
        """Test that gradients flow through the operation."""
        X = torch.randn(20, 3, requires_grad=True)
        result = center_point_cloud_torch(X)
        loss = result.sum()
        loss.backward()
        assert X.grad is not None
        assert not torch.all(X.grad == 0)


class TestTimeDelayEmbeddingTorch:
    """Tests for TimeDelayEmbeddingTorch class."""

    def test_default_parameters(self):
        """Test embedding with default parameters."""
        embedder = TimeDelayEmbeddingTorch()
        assert embedder.dim == 3
        assert embedder.delay == 1

    def test_custom_parameters(self):
        """Test embedding with custom parameters."""
        embedder = TimeDelayEmbeddingTorch(dim=5, delay=2)
        assert embedder.dim == 5
        assert embedder.delay == 2

    def test_output_shape(self):
        """Test that output has correct shape."""
        embedder = TimeDelayEmbeddingTorch(dim=4, delay=2)
        x = torch.randn(100)
        result = embedder(x)
        # Expected: (N - (dim-1)*delay, dim) = (100 - 3*2, 4) = (94, 4)
        expected_length = 100 - (4 - 1) * 2
        assert result.shape == (expected_length, 4)

    def test_short_signal_raises(self):
        """Test that too short signal raises ValueError."""
        embedder = TimeDelayEmbeddingTorch(dim=10, delay=5)
        x = torch.randn(20)  # Too short for dim=10, delay=5 (needs > 45)
        with pytest.raises(ValueError, match="too short"):
            embedder(x)

    def test_gradient_flow(self):
        """Test that gradients flow through embedding."""
        embedder = TimeDelayEmbeddingTorch(dim=3, delay=1)
        x = torch.randn(50, requires_grad=True)
        result = embedder(x)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None

    def test_delay_structure(self):
        """Test that delay structure is correct."""
        embedder = TimeDelayEmbeddingTorch(dim=3, delay=2)
        x = torch.arange(10, dtype=torch.float32)
        result = embedder(x)
        # First row should be [0, 2, 4]
        expected_first = torch.tensor([0.0, 2.0, 4.0])
        torch.testing.assert_close(result[0], expected_first)


class TestComputePeriodicityScore:
    """Tests for compute_periodicity_score function."""

    def test_returns_float(self, periodic_signal):
        """Test that function returns a float."""
        score = compute_periodicity_score(periodic_signal, dim=10)
        assert isinstance(score, float)

    def test_score_bounds(self, periodic_signal):
        """Test that score is non-negative."""
        score = compute_periodicity_score(periodic_signal, dim=10)
        assert score >= 0

    def test_periodic_higher_than_noise(self):
        """Test that periodic signal has higher score than noise."""
        t = np.linspace(0, 4 * np.pi, 200)
        periodic = np.cos(t)
        np.random.seed(42)
        noise = np.random.randn(200)

        score_periodic = compute_periodicity_score(periodic, dim=15)
        score_noise = compute_periodicity_score(noise, dim=15)

        assert score_periodic > score_noise


class TestComputePersistenceDiagram:
    """Tests for compute_persistence_diagram function."""

    def test_returns_dict(self, periodic_signal):
        """Test that function returns a dictionary."""
        result = compute_persistence_diagram(periodic_signal, dim=10)
        assert isinstance(result, dict)

    def test_dict_keys(self, periodic_signal):
        """Test that result contains expected keys."""
        result = compute_persistence_diagram(periodic_signal, dim=10)
        assert 'dgms' in result
        assert 'embedded' in result
        assert 'centered' in result

    def test_embedded_shape(self):
        """Test that embedded point cloud has correct shape."""
        signal = np.cos(np.linspace(0, 4 * np.pi, 100))
        dim = 10
        result = compute_persistence_diagram(signal, dim=dim)
        # Embedded should have shape (N - dim + 1, dim) for delay=1
        expected_rows = len(signal) - dim + 1
        assert result['embedded'].shape[1] == dim

    def test_h0_diagram_exists(self, periodic_signal):
        """Test that H0 diagram is computed."""
        result = compute_persistence_diagram(periodic_signal, dim=10, max_dim=1)
        assert len(result['dgms']) >= 1
        assert len(result['dgms'][0]) > 0  # H0 should have points
