from __future__ import annotations

import pytest


def test_sparse_opt_returns_input_when_k_nonpositive(nmf_utils_module, np):
    values = np.array([0.2, 0.8, -0.1])
    optimized = nmf_utils_module.sparse_opt(values, k=0.0)
    assert np.array_equal(optimized, values)


def test_sparse_opt_k_equal_one_selects_largest_entry(nmf_utils_module, np):
    values = np.array([0.2, 2.0, 1.2])
    optimized = nmf_utils_module.sparse_opt(values, k=1.0)
    np.testing.assert_allclose(optimized, np.array([0.0, 1.0, 0.0]))


def test_sparse_opt_hoyer_nonnegative_and_target_l1(nmf_utils_module, np):
    projected = nmf_utils_module.sparse_opt_hoyer(
        np.array([0.4, 0.3, 0.2, 0.1], dtype=float),
        L1=1.5,
        L2=1.0,
        max_iter=200,
    )

    assert np.all(projected >= -1e-12)
    assert projected.sum() == pytest.approx(1.5, abs=1e-4)


def test_sparsity_score_orders_dense_and_sparse_vectors(nmf_utils_module, torch):
    dense = torch.ones(4, dtype=torch.float64)
    sparse = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)

    assert nmf_utils_module.sparsity_score(dense) == pytest.approx(0.0, abs=1e-7)
    assert nmf_utils_module.sparsity_score(sparse) == pytest.approx(1.0, abs=1e-7)


def test_svd_initialization_shapes_and_nonnegativity(nmf_utils_module, np):
    matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.5],
            [0.3, 0.2, 0.1],
        ],
        dtype=float,
    )

    w_matrix, v_matrix = nmf_utils_module.svd_initialization(matrix, n_components=2)
    assert w_matrix.shape == (3, 2)
    assert v_matrix.shape == (2, 3)
    assert np.all(w_matrix >= 0)
    assert np.all(v_matrix >= 0)


def test_total_variation_matches_manual_computation(nmf_utils_module, torch):
    values = torch.tensor([1.0, 4.0, 2.0], dtype=torch.float64)
    total_var = nmf_utils_module.total_variation(values)
    assert float(total_var) == pytest.approx(13.0)


def test_total_variation_2d_matches_manual_computation(nmf_utils_module, torch):
    values = torch.tensor(
        [
            [1.0, 2.0, 4.0],
            [0.0, 3.0, 5.0],
        ],
        dtype=torch.float64,
    )
    total_var = nmf_utils_module.total_variation(values)
    assert float(total_var) == pytest.approx(21.0)


def test_update_v_returns_finite_nonnegative_tensor(nmf_utils_module, torch):
    torch.manual_seed(0)
    x_matrix = torch.rand(6, 5, dtype=torch.float64)
    w_matrix = torch.rand(6, 3, dtype=torch.float64)
    v_matrix = torch.rand(3, 5, dtype=torch.float64)

    updated = nmf_utils_module.update_V(
        x_matrix,
        w_matrix,
        v_matrix.clone(),
        target_L1=1.5,
        device="cpu",
    )

    assert updated.shape == v_matrix.shape
    assert torch.isfinite(updated).all()
    assert float(updated.min()) >= -1e-8
