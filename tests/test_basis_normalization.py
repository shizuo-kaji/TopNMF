"""Tests for the maximum-normalisation constraint on the basis rows."""

from __future__ import annotations

import itertools

import pytest
import torch

from TopNMF.model import TopologicalNMF
from TopNMF.optim import project_rows_max_norm, scale_rows_max_norm


def _brute_force_projection(v: torch.Tensor) -> float:
    """Squared distance from *v* to S, by projecting onto every face of the cube."""
    best = float("inf")
    for k in range(v.numel()):
        x = v.clamp(min=0.0, max=1.0).clone()
        x[k] = 1.0
        best = min(best, float(torch.sum((x - v) ** 2)))
    return best


@pytest.mark.parametrize(
    "row",
    [
        [0.3, 0.5, 0.2],          # interior, max below one
        [0.3, 1.0, 0.7, 0.0],     # already in S
        [2.0, 5.0, 0.5],          # max above one
        [-1.0, -0.2, -3.0],       # entirely negative
        [0.0, 0.0, 0.0],          # zero row
        [0.4, 0.4, 0.4],          # tied maximum
    ],
)
def test_projection_attains_the_optimal_distance(row) -> None:
    v = torch.tensor(row, dtype=torch.double)
    projected = project_rows_max_norm(v.reshape(1, -1)).reshape(-1)
    achieved = float(torch.sum((projected - v) ** 2))
    assert achieved == pytest.approx(_brute_force_projection(v))


def test_projection_lands_in_S() -> None:
    generator = torch.Generator().manual_seed(0)
    V = (torch.rand((7, 32), generator=generator) - 0.3) * 4.0
    projected = project_rows_max_norm(V)
    assert float(projected.min()) >= 0.0
    assert torch.allclose(projected.max(dim=1).values, torch.ones(7))


def test_projection_is_idempotent_and_fixes_S() -> None:
    generator = torch.Generator().manual_seed(1)
    V = torch.rand((4, 16), generator=generator) * 3.0
    projected = project_rows_max_norm(V)
    assert torch.allclose(project_rows_max_norm(projected), projected)


def test_projection_differs_from_rescaling_above_one() -> None:
    """The two normalisations agree only up to the clipping of large entries."""
    V = torch.tensor([[0.5, 4.0, 2.0]])
    assert torch.allclose(project_rows_max_norm(V), torch.tensor([[0.5, 1.0, 1.0]]))
    assert torch.allclose(scale_rows_max_norm(V), torch.tensor([[0.125, 1.0, 0.5]]),
                          atol=1e-6)


def test_fit_maximum_normalises_the_basis_by_default(np) -> None:
    matrix = np.array(
        [
            [1.0, 0.8, 0.2, 0.1],
            [0.9, 0.7, 0.3, 0.2],
            [0.2, 0.1, 0.8, 1.0],
        ],
        dtype=float,
    )
    model = TopologicalNMF(n_components=2, random_state=0, use_embedding=False)
    model.fit(matrix, n_iterations=20, lambda_top=0.0, init_method="random",
              scheduler_cls=None, verbose=False)

    V = model.get_components()
    assert V.min() >= 0.0
    assert np.allclose(V.max(axis=1), 1.0)


def test_basis_normalization_none_leaves_row_scale_free(np) -> None:
    matrix = np.array([[1.0, 0.8, 0.2], [0.2, 0.1, 0.9]], dtype=float)
    model = TopologicalNMF(n_components=2, random_state=0, use_embedding=False)
    model.fit(matrix, n_iterations=20, lambda_top=0.0, init_method="random",
              basis_normalization="none", scheduler_cls=None, verbose=False)

    assert not np.allclose(model.get_components().max(axis=1), 1.0)


def test_deprecated_normalize_V_max_maps_onto_the_projection(np) -> None:
    matrix = np.array([[1.0, 0.8, 0.2], [0.2, 0.1, 0.9]], dtype=float)
    kwargs = dict(n_iterations=20, lambda_top=0.0, init_method="random",
                  scheduler_cls=None, verbose=False)

    legacy = TopologicalNMF(n_components=2, random_state=0, use_embedding=False)
    with pytest.deprecated_call():
        legacy.fit(matrix, normalize_V_max=True, **kwargs)

    current = TopologicalNMF(n_components=2, random_state=0, use_embedding=False)
    current.fit(matrix, basis_normalization="project", **kwargs)

    assert np.allclose(legacy.get_components(), current.get_components())

    off = TopologicalNMF(n_components=2, random_state=0, use_embedding=False)
    with pytest.deprecated_call():
        off.fit(matrix, normalize_V_max=False, **kwargs)
    assert not np.allclose(off.get_components().max(axis=1), 1.0)


def test_unknown_basis_normalization_raises(np) -> None:
    matrix = np.array([[1.0, 0.8], [0.2, 0.9]], dtype=float)
    model = TopologicalNMF(n_components=1, random_state=0, use_embedding=False)
    with pytest.raises(ValueError, match="basis_normalization"):
        model.fit(matrix, n_iterations=1, basis_normalization="maximum",
                  verbose=False)
