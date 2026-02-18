"""Tests for graph-based persistent homology."""

from __future__ import annotations

import os
import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import gudhi
from TopNMF.persistence.graph import GraphFiltrationPH


def test_graph_filtration_rejects_negative_max_dim():
    with pytest.raises(ValueError, match="max_dim must be non-negative"):
        GraphFiltrationPH(max_dim=-1)


def test_canonical_edge_and_lookup_treat_graph_as_undirected():
    assert GraphFiltrationPH._canonical_edge((5, 2)) == (2, 5)

    edge_lookup = GraphFiltrationPH._build_edge_lookup([(2, 0), (0, 2), (1, 3)])
    assert edge_lookup[(0, 2)] == 0
    assert edge_lookup[(1, 3)] == 2


def test_prepare_inputs_rejects_empty_edges_and_mismatched_weights(torch):
    model = GraphFiltrationPH()

    with pytest.raises(ValueError, match="at least one edge"):
        model._prepare_inputs(all_edges=[], edge_weight=torch.tensor([], dtype=torch.float32))

    with pytest.raises(ValueError, match="length must match all_edges"):
        model._prepare_inputs(
            all_edges=[(0, 1), (1, 2)],
            edge_weight=torch.tensor([0.3], dtype=torch.float32),
        )


def test_prepare_inputs_superlevel_flips_sign_and_enables_grad(torch):
    all_edges = [(0, 1), (2, 1)]
    edge_weight = torch.tensor([0.2, 0.5], dtype=torch.float32)

    sublevel_model = GraphFiltrationPH(superlevel=False)
    normalized_edges, edge_values, vertex_values, edge_lookup = sublevel_model._prepare_inputs(
        all_edges=all_edges,
        edge_weight=edge_weight,
    )

    assert normalized_edges == all_edges
    assert edge_values.requires_grad
    assert vertex_values.requires_grad
    assert vertex_values.shape == (3,)
    assert edge_lookup[(1, 2)] == 1
    torch.testing.assert_close(edge_values, torch.tensor([0.2, 0.5], dtype=torch.float32))
    torch.testing.assert_close(vertex_values, torch.full((3,), -1.0, dtype=torch.float32))

    superlevel_model = GraphFiltrationPH(superlevel=True)
    _, superlevel_edge_values, _, _ = superlevel_model._prepare_inputs(
        all_edges=all_edges,
        edge_weight=[0.2, 0.5],
    )
    torch.testing.assert_close(
        superlevel_edge_values,
        torch.tensor([-0.2, -0.5], dtype=torch.float32),
    )


def test_max_filtration_value_handles_empty_and_nonempty_simplex_tree(torch):
    empty_tree = gudhi.SimplexTree()
    empty_max = GraphFiltrationPH._max_filtration_value(empty_tree, device=torch.device("cpu"))
    assert float(empty_max) == pytest.approx(0.0)

    simplex_tree = gudhi.SimplexTree()
    simplex_tree.insert([0], filtration=-1.0)
    simplex_tree.insert([1], filtration=-1.0)
    simplex_tree.insert([0, 1], filtration=0.4)
    nonempty_max = GraphFiltrationPH._max_filtration_value(simplex_tree, device=torch.device("cpu"))
    assert float(nonempty_max) == pytest.approx(0.4)


def test_birth_value_raises_for_unknown_creator_edge(torch):
    model = GraphFiltrationPH()
    simplex_tree = gudhi.SimplexTree()
    edge_values = torch.tensor([0.3], dtype=torch.float32, requires_grad=True)
    vertex_values = torch.full((2,), -1.0, dtype=torch.float32, requires_grad=True)

    with pytest.raises(ValueError, match="Creator edge"):
        model._birth_value(
            simplex=(0, 1),
            st=simplex_tree,
            edge_lookup={},
            edge_values=edge_values,
            vertex_values=vertex_values,
        )


def test_death_value_falls_back_to_max_for_unknown_edge(torch):
    model = GraphFiltrationPH()
    edge_values = torch.tensor([0.3], dtype=torch.float32, requires_grad=True)
    vertex_values = torch.full((2,), -1.0, dtype=torch.float32, requires_grad=True)
    max_filtration = torch.tensor(0.9, dtype=torch.float32)

    death, death_simplex = model._death_value(
        simplex=(0, 1),
        edge_lookup={},
        edge_values=edge_values,
        vertex_values=vertex_values,
        max_filtration=max_filtration,
    )
    assert float(death) == pytest.approx(0.9)
    assert death_simplex == (0, 1)


def test_forward_path_graph_returns_expected_h0_and_empty_h1(torch):
    model = GraphFiltrationPH(max_dim=1, superlevel=False)
    persistence_info = model(
        all_edges=[(0, 1), (1, 2)],
        edge_weight=torch.tensor([0.2, 0.8], dtype=torch.float32),
    )

    assert len(persistence_info) == 2
    assert [info.dimension for info in persistence_info] == [0, 1]

    h0_info, h1_info = persistence_info
    assert h0_info.diagram.shape == (3, 2)
    assert h1_info.diagram.shape == (0, 2)
    assert sum(pair[1] is None for pair in h0_info.pairing) == 1
    torch.testing.assert_close(
        torch.sort(h0_info.diagram[:, 0]).values,
        torch.full((3,), -1.0, dtype=torch.float32),
    )
    torch.testing.assert_close(
        torch.sort(h0_info.diagram[:, 1]).values,
        torch.tensor([0.2, 0.8, 0.8], dtype=torch.float32),
        atol=1e-6,
        rtol=0.0,
    )


def test_forward_superlevel_negates_edge_filtration(torch):
    model = GraphFiltrationPH(max_dim=0, superlevel=True)
    persistence_info = model(
        all_edges=[(0, 1), (1, 2)],
        edge_weight=[0.2, 0.8],
    )

    assert len(persistence_info) == 1
    h0_info = persistence_info[0]
    assert h0_info.dimension == 0
    assert h0_info.diagram.shape == (3, 2)
    assert sum(pair[1] is None for pair in h0_info.pairing) == 1
    torch.testing.assert_close(
        torch.sort(h0_info.diagram[:, 1]).values,
        torch.tensor([-0.8, -0.2, -0.2], dtype=torch.float32),
        atol=1e-6,
        rtol=0.0,
    )
