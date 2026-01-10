"""Tests for nmf_utils module."""

import pytest
import numpy as np
import torch
from TopNMF.nmf_utils import (
    sparse_opt,
    sparse_opt_hoyer,
    sparsity_score,
    svd_initialization,
    total_variation,
)


class TestSparseOpt:
    """Tests for sparse_opt function."""

    def test_returns_array(self):
        """Test that function returns numpy array."""
        b = np.array([1.0, 2.0, 3.0, 4.0])
        result = sparse_opt(b, k=1.5)
        assert isinstance(result, np.ndarray)

    def test_non_negative_k_zero(self):
        """Test that k<=0 returns input unchanged."""
        b = np.array([1.0, 2.0, 3.0])
        result = sparse_opt(b, k=0)
        np.testing.assert_array_equal(result, b)

    def test_output_shape(self):
        """Test that output has same shape as input."""
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sparse_opt(b, k=2.0)
        assert result.shape == b.shape

    def test_k_one_returns_unit_vector(self):
        """Test that k=1 returns unit vector at max position."""
        b = np.array([1.0, 5.0, 3.0])
        result = sparse_opt(b, k=1)
        assert result[1] == 1.0
        assert result[0] == 0.0
        assert result[2] == 0.0


class TestSparseOptHoyer:
    """Tests for sparse_opt_hoyer function."""

    def test_non_negative_output(self):
        """Test that output is non-negative."""
        x = np.array([1.0, -2.0, 3.0, -1.0])
        result = sparse_opt_hoyer(x, L1=2.0, L2=1.0)
        assert np.all(result >= 0)

    def test_output_shape(self):
        """Test that output has same shape as input."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = sparse_opt_hoyer(x, L1=2.0, L2=1.0)
        assert result.shape == x.shape

    def test_l2_constraint(self):
        """Test that L2 constraint is approximately satisfied."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        L2 = 2.0
        result = sparse_opt_hoyer(x, L1=3.0, L2=L2)
        # L2 norm should be close to target
        assert np.isclose(np.linalg.norm(result), L2, rtol=0.1)


class TestSparsityScore:
    """Tests for sparsity_score function."""

    def test_returns_float(self):
        """Test that function returns a float."""
        v = torch.tensor([1.0, 0.0, 0.0, 0.0])
        score = sparsity_score(v)
        assert isinstance(score, float)

    def test_sparse_vector_high_score(self):
        """Test that sparse vector has high sparsity score."""
        v_sparse = torch.tensor([1.0, 0.0, 0.0, 0.0])
        v_dense = torch.tensor([0.5, 0.5, 0.5, 0.5])
        score_sparse = sparsity_score(v_sparse)
        score_dense = sparsity_score(v_dense)
        assert score_sparse > score_dense

    def test_score_bounds(self):
        """Test that score is between 0 and 1."""
        v = torch.tensor([1.0, 2.0, 0.5, 0.1])
        score = sparsity_score(v)
        assert 0 <= score <= 1

    def test_uniform_vector_low_score(self):
        """Test that uniform vector has low sparsity score."""
        v = torch.ones(10)
        score = sparsity_score(v)
        assert score < 0.1  # Should be close to 0


class TestSvdInitialization:
    """Tests for svd_initialization function."""

    def test_output_shapes(self, random_matrix):
        """Test that output matrices have correct shapes."""
        n_components = 5
        W, V = svd_initialization(random_matrix, n_components)
        assert W.shape == (random_matrix.shape[0], n_components)
        assert V.shape == (n_components, random_matrix.shape[1])

    def test_non_negative_output(self, random_matrix):
        """Test that output matrices are non-negative."""
        W, V = svd_initialization(random_matrix, n_components=5)
        assert np.all(W >= 0)
        assert np.all(V >= 0)

    def test_reconstruction_quality(self, random_matrix):
        """Test that reconstruction is reasonable."""
        W, V = svd_initialization(random_matrix, n_components=10)
        reconstruction = W @ V
        # Reconstruction should have similar norm to original
        orig_norm = np.linalg.norm(random_matrix)
        recon_norm = np.linalg.norm(reconstruction)
        assert 0.5 * orig_norm < recon_norm < 2 * orig_norm


class TestTotalVariation:
    """Tests for total_variation function."""

    def test_constant_signal_zero_tv(self):
        """Test that constant signal has zero total variation."""
        v = torch.ones(10)
        tv = total_variation(v)
        assert tv.item() == 0.0

    def test_step_signal_positive_tv(self):
        """Test that step signal has positive total variation."""
        v = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
        tv = total_variation(v)
        assert tv.item() > 0

    def test_returns_tensor(self):
        """Test that function returns a tensor."""
        v = torch.tensor([1.0, 2.0, 3.0])
        tv = total_variation(v)
        assert isinstance(tv, torch.Tensor)

    def test_oscillating_high_tv(self):
        """Test that oscillating signal has high total variation."""
        v_smooth = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4])
        v_oscillating = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
        tv_smooth = total_variation(v_smooth)
        tv_oscillating = total_variation(v_oscillating)
        assert tv_oscillating > tv_smooth
