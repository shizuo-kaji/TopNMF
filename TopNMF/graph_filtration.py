"""Graph-based persistent homology utilities."""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import gudhi
import torch
from torch import nn


Edge = Tuple[int, int]
Simplex = Tuple[int, ...]

__all__ = ["GraphFiltrationPH", "PersistenceInformation"]


class PersistenceInformation(NamedTuple):
    """Minimal persistence container for diagram + pairing metadata."""

    pairing: List[List[Optional[Simplex]]]
    diagram: torch.Tensor
    dimension: int


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
        """Return a canonical undirected representation for an edge."""
        u, v = edge
        return (u, v) if u <= v else (v, u)

    @staticmethod
    def _build_edge_lookup(all_edges: Sequence[Edge]) -> Dict[Edge, int]:
        """Map canonical edges to their first index in `all_edges`."""
        edge_lookup: Dict[Edge, int] = {}
        for idx, edge in enumerate(all_edges):
            edge_lookup.setdefault(GraphFiltrationPH._canonical_edge(edge), idx)
        return edge_lookup

    def _prepare_inputs(
        self,
        all_edges: Sequence[Edge],
        edge_weight: Union[torch.Tensor, List[float]],
    ) -> Tuple[List[Edge], torch.Tensor, torch.Tensor, Dict[Edge, int]]:
        """Normalize and validate graph inputs."""
        if not all_edges:
            raise ValueError("all_edges must contain at least one edge.")

        normalized_edges = [(int(u), int(v)) for u, v in all_edges]
        edge_weight_t = torch.as_tensor(edge_weight, dtype=torch.float32).reshape(-1)
        if edge_weight_t.numel() != len(normalized_edges):
            raise ValueError(
                "edge_weight length must match all_edges length: "
                f"{edge_weight_t.numel()} != {len(normalized_edges)}"
            )
        if not edge_weight_t.requires_grad:
            edge_weight_t.requires_grad_(True)

        edge_index = torch.tensor(normalized_edges, dtype=torch.long).T
        num_nodes = int(edge_index.max().item()) + 1

        vertex_values = torch.full(
            (num_nodes,),
            -1.0,
            dtype=torch.float32,
            device=edge_weight_t.device,
        )
        vertex_values.requires_grad_(True)

        filtered_edge_values = -edge_weight_t if self.superlevel else edge_weight_t
        edge_lookup = self._build_edge_lookup(normalized_edges)
        return normalized_edges, filtered_edge_values, vertex_values, edge_lookup

    @staticmethod
    def _build_simplex_tree(
        all_edges: Sequence[Edge],
        edge_values: torch.Tensor,
        vertex_values: torch.Tensor,
    ) -> gudhi.SimplexTree:
        """Create and populate a simplex tree for graph filtration."""
        st = gudhi.SimplexTree()

        for node_idx, node_value in enumerate(vertex_values):
            st.insert([node_idx], filtration=float(node_value.detach().cpu().item()))

        for edge_idx, (u, v) in enumerate(all_edges):
            st.insert([u, v], filtration=float(edge_values[edge_idx].detach().cpu().item()))

        st.make_filtration_non_decreasing()
        st.compute_persistence(persistence_dim_max=True)
        return st

    @staticmethod
    def _max_filtration_value(st: gudhi.SimplexTree, device: torch.device) -> torch.Tensor:
        """Get the maximum filtration value in the simplex tree as a tensor."""
        filtration_values = [filtration for _, filtration in st.get_filtration()]
        if not filtration_values:
            return torch.tensor(0.0, dtype=torch.float32, device=device)

        return torch.tensor(filtration_values, dtype=torch.float32, device=device).max()

    def _birth_value(
        self,
        simplex: Simplex,
        st: gudhi.SimplexTree,
        edge_lookup: Dict[Edge, int],
        edge_values: torch.Tensor,
        vertex_values: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve birth value for a creator simplex."""
        if len(simplex) == 1:
            return vertex_values[simplex[0]]
        if len(simplex) == 2:
            edge_idx = edge_lookup.get(self._canonical_edge((simplex[0], simplex[1])))
            if edge_idx is None:
                raise ValueError(f"Creator edge {simplex} was not found in all_edges.")
            return edge_values[edge_idx]

        return torch.tensor(
            st.filtration(list(simplex)),
            dtype=torch.float32,
            device=vertex_values.device,
        )

    def _death_value(
        self,
        simplex: Simplex,
        edge_lookup: Dict[Edge, int],
        edge_values: torch.Tensor,
        vertex_values: torch.Tensor,
        max_filtration: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Simplex]]:
        """Resolve death value and death simplex metadata."""
        if simplex and len(simplex) == 1:
            return vertex_values[simplex[0]], simplex

        if simplex and len(simplex) == 2:
            edge_idx = edge_lookup.get(self._canonical_edge((simplex[0], simplex[1])))
            if edge_idx is None:
                return max_filtration, simplex
            return edge_values[edge_idx], simplex

        return max_filtration, None

    def _build_persistence_info_for_dim(
        self,
        dim: int,
        st: gudhi.SimplexTree,
        edge_lookup: Dict[Edge, int],
        edge_values: torch.Tensor,
        vertex_values: torch.Tensor,
        max_filtration: torch.Tensor,
    ) -> PersistenceInformation:
        """Collect persistence pairings and diagram entries for one dimension."""
        diagram_entries: List[torch.Tensor] = []
        pairing_entries: List[List[Optional[Simplex]]] = []

        for creator_simplex_raw, destroyer_simplex_raw in st.persistence_pairs():
            creator_simplex: Simplex = tuple(creator_simplex_raw)
            destroyer_simplex: Simplex = tuple(destroyer_simplex_raw)
            feature_dim = len(creator_simplex) - 1
            if feature_dim != dim:
                continue

            birth = self._birth_value(
                simplex=creator_simplex,
                st=st,
                edge_lookup=edge_lookup,
                edge_values=edge_values,
                vertex_values=vertex_values,
            )
            death, death_simplex = self._death_value(
                simplex=destroyer_simplex,
                edge_lookup=edge_lookup,
                edge_values=edge_values,
                vertex_values=vertex_values,
                max_filtration=max_filtration,
            )

            diagram_entries.append(torch.stack([birth, death]))
            pairing_entries.append([creator_simplex, death_simplex])

        if diagram_entries:
            diagram = torch.stack(diagram_entries, dim=0)
        else:
            diagram = torch.zeros((0, 2), dtype=torch.float32, device=vertex_values.device)

        return PersistenceInformation(pairing=pairing_entries, diagram=diagram, dimension=dim)

    def forward(
        self,
        all_edges: List[Tuple[int, int]],
        edge_weight: Union[torch.Tensor, List[float]],
    ) -> List[PersistenceInformation]:
        """Compute persistence information for graph edge filtrations."""
        (
            normalized_edges,
            edge_values,
            vertex_values,
            edge_lookup,
        ) = self._prepare_inputs(all_edges=all_edges, edge_weight=edge_weight)

        st = self._build_simplex_tree(
            all_edges=normalized_edges,
            edge_values=edge_values,
            vertex_values=vertex_values,
        )
        max_filtration = self._max_filtration_value(st, device=vertex_values.device)

        return [
            self._build_persistence_info_for_dim(
                dim=dim,
                st=st,
                edge_lookup=edge_lookup,
                edge_values=edge_values,
                vertex_values=vertex_values,
                max_filtration=max_filtration,
            )
            for dim in range(self.max_dim + 1)
        ]
