"""Tests for the Vietoris-Rips persistence backend."""

from __future__ import annotations

import pytest


def test_rips_returns_correct_dimensions_up_to_two(torch):
    from TopNMF.persistence import GudhiVietorisRipsComplex

    torch.manual_seed(0)
    point_cloud = torch.rand(20, 3, dtype=torch.float)

    complex_fn = GudhiVietorisRipsComplex(dim=2, p=2)
    pers_info = complex_fn(point_cloud)

    assert [info.dimension for info in pers_info] == [0, 1, 2]
    for info in pers_info:
        assert info.diagram.ndim == 2
        assert info.diagram.shape[1] == 2
        assert torch.isfinite(info.diagram).all()


def test_rips_h0_births_are_zero(torch):
    from TopNMF.persistence import GudhiVietorisRipsComplex

    torch.manual_seed(1)
    point_cloud = torch.rand(12, 2, dtype=torch.float)
    pers_info = GudhiVietorisRipsComplex(dim=1, p=2)(point_cloud)

    h0 = pers_info[0].diagram
    assert h0.shape[0] == point_cloud.shape[0] - 1  # finite H0 bars
    torch.testing.assert_close(h0[:, 0], torch.zeros(h0.shape[0]))


def test_rips_rejects_non_euclidean_p(torch):
    from TopNMF.persistence import GudhiVietorisRipsComplex

    with pytest.raises(ValueError, match="p=2"):
        GudhiVietorisRipsComplex(dim=1, p=1)
