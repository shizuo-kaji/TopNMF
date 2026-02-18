"""Graph-based persistent homology via simplex tree filtration."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import gudhi
import torch
from torch import nn

from . import PersistenceInfo

Edge = Tuple[int, int]
Simplex = Tuple[int, ...]

__all__ = ["GraphFiltrationPH"]


class GraphFiltrationPH(nn.Module):
    """Compute persistence diagrams from graph edge filtrations."""

    def __init__(self, max_dim: int = 1, superlevel: bool = False):
        super().__init__()
        if max_dim < 0:
            raise ValueError(f"max_dim must be non-negative, got {max_dim}")
        self.max_dim = max_dim
        self.superlevel = superlevel

    @staticmethod
    def _canonical_edge(edge: Edge) -> Edge:
        u, v = edge
        return (u, v) if u <= v else (v, u)

    @staticmethod
    def _build_edge_lookup(all_edges: Sequence[Edge]) -> Dict[Edge, int]:
        edge_lookup: Dict[Edge, int] = {}
        for idx, edge in enumerate(all_edges):
            edge_lookup.setdefault(GraphFiltrationPH._canonical_edge(edge), idx)
        return edge_lookup

    def _prepare_inputs(
        self,
        all_edges: Sequence[Edge],
        edge_weight: Union[torch.Tensor, List[float]],
    ) -> Tuple[List[Edge], torch.Tensor, torch.Tensor, Dict[Edge, int]]:
        if not all_edges:
            raise ValueError("all_edges must contain at least one edge.")

        normalized_edges = [(int(u), int(v)) for u, v in all_edges]
        edge_weight_t = torch.as_tensor(edge_weight, dtype=torch.float32).reshape(-1)
        if edge_weight_t.numel() != len(normalized_edges):
            raise ValueError(
                f"edge_weight length must match all_edges length: "
                f"{edge_weight_t.numel()} != {len(normalized_edges)}"
            )
        if not edge_weight_t.requires_grad:
            edge_weight_t.requires_grad_(True)

        edge_index = torch.tensor(normalized_edges, dtype=torch.long).T
        num_nodes = int(edge_index.max().item()) + 1

        vertex_values = torch.full(
            (num_nodes,), -1.0,
            dtype=torch.float32, device=edge_weight_t.device,
        )
        vertex_values.requires_grad_(True)

        filtered_edge_values = -edge_weight_t if self.superlevel else edge_weight_t
        edge_lookup = self._build_edge_lookup(normalized_edges)
        return normalized_edges, filtered_edge_values, vertex_values, edge_lookup

    @staticmethod
    def _build_simplex_tree(all_edges, edge_values, vertex_values):
        st = gudhi.SimplexTree()
        for node_idx, node_value in enumerate(vertex_values):
            st.insert([node_idx], filtration=float(node_value.detach().cpu().item()))
        for edge_idx, (u, v) in enumerate(all_edges):
            st.insert([u, v], filtration=float(edge_values[edge_idx].detach().cpu().item()))
        st.make_filtration_non_decreasing()
        st.compute_persistence(persistence_dim_max=True)
        return st

    @staticmethod
    def _max_filtration_value(st, device):
        filtration_values = [f for _, f in st.get_filtration()]
        if not filtration_values:
            return torch.tensor(0.0, dtype=torch.float32, device=device)
        return torch.tensor(filtration_values, dtype=torch.float32, device=device).max()

    def _birth_value(self, simplex, st, edge_lookup, edge_values, vertex_values):
        if len(simplex) == 1:
            return vertex_values[simplex[0]]
        if len(simplex) == 2:
            edge_idx = edge_lookup.get(self._canonical_edge((simplex[0], simplex[1])))
            if edge_idx is None:
                raise ValueError(f"Creator edge {simplex} was not found in all_edges.")
            return edge_values[edge_idx]
        return torch.tensor(
            st.filtration(list(simplex)),
            dtype=torch.float32, device=vertex_values.device,
        )

    def _death_value(self, simplex, edge_lookup, edge_values, vertex_values,
                     max_filtration):
        if simplex and len(simplex) == 1:
            return vertex_values[simplex[0]], simplex
        if simplex and len(simplex) == 2:
            edge_idx = edge_lookup.get(self._canonical_edge((simplex[0], simplex[1])))
            if edge_idx is None:
                return max_filtration, simplex
            return edge_values[edge_idx], simplex
        return max_filtration, None

    def _build_persistence_info_for_dim(self, dim, st, edge_lookup, edge_values,
                                        vertex_values, max_filtration):
        diagram_entries: List[torch.Tensor] = []
        pairing_entries: List[List[Optional[Simplex]]] = []

        for creator_raw, destroyer_raw in st.persistence_pairs():
            creator: Simplex = tuple(creator_raw)
            destroyer: Simplex = tuple(destroyer_raw)
            if len(creator) - 1 != dim:
                continue

            birth = self._birth_value(creator, st, edge_lookup, edge_values,
                                      vertex_values)
            death, death_simplex = self._death_value(
                destroyer, edge_lookup, edge_values, vertex_values, max_filtration)

            diagram_entries.append(torch.stack([birth, death]))
            pairing_entries.append([creator, death_simplex])

        if diagram_entries:
            diagram = torch.stack(diagram_entries, dim=0)
        else:
            diagram = torch.zeros((0, 2), dtype=torch.float32,
                                  device=vertex_values.device)

        return PersistenceInfo(
            diagram=diagram,
            pairing=pairing_entries,
            dimension=dim,
        )

    def forward(
        self,
        all_edges: List[Tuple[int, int]],
        edge_weight: Union[torch.Tensor, List[float]],
    ) -> List[PersistenceInfo]:
        normalized_edges, edge_values, vertex_values, edge_lookup = (
            self._prepare_inputs(all_edges=all_edges, edge_weight=edge_weight))

        st = self._build_simplex_tree(
            all_edges=normalized_edges, edge_values=edge_values,
            vertex_values=vertex_values)
        max_filtration = self._max_filtration_value(st, device=vertex_values.device)

        return [
            self._build_persistence_info_for_dim(
                dim=dim, st=st, edge_lookup=edge_lookup,
                edge_values=edge_values, vertex_values=vertex_values,
                max_filtration=max_filtration)
            for dim in range(self.max_dim + 1)
        ]
