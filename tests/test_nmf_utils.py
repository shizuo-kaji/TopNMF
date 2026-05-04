"""Tests for optim, utils, and losses modules (previously nmf_utils)."""

from __future__ import annotations

import math
import pytest

from TopNMF.optim import sparse_opt, sparse_opt_hoyer, update_V
from TopNMF.utils import (
    center_point_cloud,
    center_point_cloud_torch,
    sparsity_score,
    svd_initialization,
)
from TopNMF.losses import total_variation, weighted_total_squared_persistence_loss
from TopNMF.persistence import PersistenceInfo


def test_sparse_opt_returns_input_when_k_nonpositive(np):
    values = np.array([0.2, 0.8, -0.1])
    optimized = sparse_opt(values, k=0.0)
    assert np.array_equal(optimized, values)


def test_sparse_opt_k_equal_one_selects_largest_entry(np):
    values = np.array([0.2, 2.0, 1.2])
    optimized = sparse_opt(values, k=1.0)
    np.testing.assert_allclose(optimized, np.array([0.0, 1.0, 0.0]))


def test_sparse_opt_hoyer_nonnegative_and_target_l1(np):
    projected = sparse_opt_hoyer(
        np.array([0.4, 0.3, 0.2, 0.1], dtype=float),
        L1=1.5,
        L2=1.0,
        max_iter=200,
    )

    assert np.all(projected >= -1e-12)
    assert projected.sum() == pytest.approx(1.5, abs=1e-4)


def test_sparsity_score_orders_dense_and_sparse_vectors(torch):
    dense = torch.ones(4, dtype=torch.float64)
    sparse_vec = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)

    assert sparsity_score(dense) == pytest.approx(0.0, abs=1e-7)
    assert sparsity_score(sparse_vec) == pytest.approx(1.0, abs=1e-7)


def test_svd_initialization_shapes_and_nonnegativity(np):
    matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.5],
            [0.3, 0.2, 0.1],
        ],
        dtype=float,
    )

    w_matrix, v_matrix = svd_initialization(matrix, n_components=2)
    assert w_matrix.shape == (3, 2)
    assert v_matrix.shape == (2, 3)
    assert np.all(w_matrix >= 0)
    assert np.all(v_matrix >= 0)


def test_center_point_cloud_handles_constant_rows(np) -> None:
    point_cloud = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=float,
    )

    centered = center_point_cloud(point_cloud)

    assert np.isfinite(centered).all()
    np.testing.assert_allclose(centered[0], np.zeros(3))


def test_center_point_cloud_torch_handles_constant_rows(torch) -> None:
    point_cloud = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    centered = center_point_cloud_torch(point_cloud)
    loss = centered.pow(2).sum()
    loss.backward()

    assert torch.isfinite(centered).all()
    assert torch.isfinite(point_cloud.grad).all()
    torch.testing.assert_close(centered[0], torch.zeros(3, dtype=torch.float64))


def test_total_variation_matches_manual_computation(torch):
    values = torch.tensor([1.0, 4.0, 2.0], dtype=torch.float64)
    total_var = total_variation(values)
    assert float(total_var) == pytest.approx(13.0)


def test_total_variation_2d_matches_manual_computation(torch):
    values = torch.tensor(
        [
            [1.0, 2.0, 4.0],
            [0.0, 3.0, 5.0],
        ],
        dtype=torch.float64,
    )
    total_var = total_variation(values)
    # Row diffs squared sum = 3, col diffs squared sum = 18
    # sqrt(3) + sqrt(18)
    expected = math.sqrt(3) + math.sqrt(18)
    assert float(total_var) == pytest.approx(expected, rel=1e-6)


def test_weighted_total_squared_persistence_loss_matches_formula(torch):
    diagram = torch.tensor(
        [
            [0.2, 0.5],
            [0.1, 0.9],
        ],
        dtype=torch.float64,
    )

    loss = weighted_total_squared_persistence_loss(
        [diagram],
        PH_dims=[0],
        device="cpu",
        p=2.0,
    )

    expected = (1.0 - 0.5) ** 2 * (0.5 - 0.2) ** 2
    expected += (1.0 - 0.9) ** 2 * (0.9 - 0.1) ** 2
    assert float(loss) == pytest.approx(expected)


def test_weighted_total_squared_persistence_loss_ignores_infinite_pairs(torch):
    diagram = torch.tensor(
        [
            [0.2, 0.5],
            [0.0, float("inf")],
        ],
        dtype=torch.float64,
    )
    persistence = PersistenceInfo(
        diagram=diagram,
        pairing=torch.empty((2, 2), dtype=torch.long),
        dimension=0,
    )

    loss = weighted_total_squared_persistence_loss(
        [persistence],
        PH_dims=[0],
        device="cpu",
        p=2.0,
    )

    expected = (1.0 - 0.5) ** 2 * (0.5 - 0.2) ** 2
    assert float(loss) == pytest.approx(expected)


def test_update_v_returns_finite_nonnegative_tensor(torch):
    torch.manual_seed(0)
    x_matrix = torch.rand(6, 5, dtype=torch.float64)
    w_matrix = torch.rand(6, 3, dtype=torch.float64)
    v_matrix = torch.rand(3, 5, dtype=torch.float64)

    updated = update_V(
        x_matrix,
        w_matrix,
        v_matrix.clone(),
        target_L1=1.5,
        device="cpu",
    )

    assert updated.shape == v_matrix.shape
    assert torch.isfinite(updated).all()
    assert float(updated.min()) >= -1e-8
