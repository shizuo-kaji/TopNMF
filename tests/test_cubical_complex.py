from __future__ import annotations

from importlib import import_module

import pytest


def _load_cubical_complex_module():
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    pytest.importorskip("cripser")
    return import_module("TopNMF.cubical_complex")


def test_cubical_complex_vertices_mode_shapes():
    module = _load_cubical_complex_module()
    torch = pytest.importorskip("torch")

    x = torch.tensor(
        [
            [0.2, 0.8, 0.1],
            [0.9, 0.0, 0.6],
            [0.3, 0.7, 0.4],
        ],
        dtype=torch.float64,
    )

    cc = module.CubicalComplex(mode="V")
    persistence = cc(x)

    assert isinstance(persistence, list)
    assert len(persistence) == 2

    for expected_dim, info in enumerate(persistence):
        assert isinstance(info, module.PersistenceInfo)
        assert info.dimension == expected_dim
        assert info.diagram.ndim == 2
        assert info.diagram.shape[1] == 2
        assert info.pairing.ndim == 2
        assert info.pairing.shape[1] == 4
        assert torch.isfinite(info.diagram).all()


def test_cubical_complex_t_mode_shapes():
    module = _load_cubical_complex_module()
    torch = pytest.importorskip("torch")

    x = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )

    cc = module.CubicalComplex(mode="T")
    persistence = cc(x)

    assert isinstance(persistence, list)
    assert len(persistence) == 2
    for info in persistence:
        assert info.diagram.ndim == 2
        assert info.diagram.shape[1] == 2
        assert info.pairing.ndim == 2
        assert info.pairing.shape[1] == 4


def test_cubical_complex_batched_forward_structure():
    module = _load_cubical_complex_module()
    torch = pytest.importorskip("torch")

    x = torch.rand(2, 3, 5, 5, dtype=torch.float64)
    cc = module.CubicalComplex(mode="V")
    persistence = cc(x)

    assert len(persistence) == 2
    assert len(persistence[0]) == 3
    assert len(persistence[0][0]) == 2


def test_cubical_complex_superlevel_diagram_values_use_original_tensor():
    module = _load_cubical_complex_module()
    torch = pytest.importorskip("torch")

    x = torch.tensor(
        [
            [0.1, 0.4, 0.9],
            [0.2, 0.3, 0.8],
            [0.5, 0.6, 0.7],
        ],
        dtype=torch.float64,
    )

    cc = module.CubicalComplex(mode="V", superlevel=True)
    persistence = cc(x)
    x_min = float(x.min())
    x_max = float(x.max())

    for info in persistence:
        if info.diagram.numel() == 0:
            continue
        assert float(info.diagram.min()) >= x_min - 1e-12
        assert float(info.diagram.max()) <= x_max + 1e-12
