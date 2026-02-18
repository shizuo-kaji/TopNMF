"""Vietoris-Rips persistence and time-delay embedding."""

import numpy as np
import torch
import gudhi
from typing import Optional, List

from . import PersistenceInfo


class TimeDelayEmbeddingTorch(torch.nn.Module):
    """
    PyTorch time-delay embedding for time series.

    Parameters
    ----------
    dim : int
        Embedding dimension (number of delayed copies).
    delay : int
        Time delay between copies.
    """

    def __init__(self, dim: int = 3, delay: int = 1):
        super().__init__()
        self.dim = dim
        self.delay = delay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        d = self.dim
        tau = self.delay
        if N - (d - 1) * tau <= 0:
            raise ValueError(
                f"Time series length {N} is too short for embedding "
                f"with dim={d} and delay={tau}"
            )
        return torch.stack(
            [x[i:N - (d - 1) * tau + i] for i in range(0, d * tau, tau)],
            dim=1,
        )


class GudhiVietorisRipsComplex:
    """
    Gudhi-backed Vietoris-Rips persistence wrapper.

    Returns a list of PersistenceInfo compatible with TopologicalNMF loss functions.

    Parameters
    ----------
    dim : int
        Maximum homology dimension.
    p : int
        Minkowski norm (must be 2).
    max_edge_length : float, optional
        Maximum edge length for the Rips complex.
    """

    def __init__(self, dim: int = 1, p: int = 2,
                 max_edge_length: Optional[float] = None):
        if dim < 0:
            raise ValueError(f"dim must be non-negative, got {dim}")
        if p != 2:
            raise ValueError("GudhiVietorisRipsComplex currently supports only p=2")
        self.dim = dim
        self.p = p
        self.max_edge_length = max_edge_length

    def __call__(self, point_cloud: torch.Tensor) -> List[PersistenceInfo]:
        points = np.asarray(point_cloud.detach().cpu().numpy(), dtype=np.float64)
        if points.ndim != 2:
            raise ValueError(
                f"point_cloud must have shape (n_points, n_dimensions), "
                f"got {points.shape}"
            )

        rips = (
            gudhi.RipsComplex(points=points)
            if self.max_edge_length is None
            else gudhi.RipsComplex(points=points, max_edge_length=float(self.max_edge_length))
        )
        simplex_tree = rips.create_simplex_tree(max_dimension=self.dim + 1)
        simplex_tree.compute_persistence()

        result: List[PersistenceInfo] = []
        for h_dim in range(self.dim + 1):
            intervals = np.asarray(
                simplex_tree.persistence_intervals_in_dimension(h_dim),
                dtype=np.float64,
            )
            if intervals.size == 0:
                result.append(PersistenceInfo(
                    diagram=torch.empty((0, 2), dtype=point_cloud.dtype,
                                        device=point_cloud.device),
                    pairing=torch.empty((0, 0), dtype=torch.long,
                                        device=point_cloud.device),
                    dimension=h_dim,
                ))
                continue

            deaths = intervals[:, 1].copy()
            finite_deaths = deaths[np.isfinite(deaths)]
            replacement = (
                float(np.max(finite_deaths))
                if finite_deaths.size > 0
                else float(np.max(intervals[:, 0]))
            )
            deaths[~np.isfinite(deaths)] = replacement
            diagram_np = np.column_stack([intervals[:, 0], deaths])

            result.append(PersistenceInfo(
                diagram=torch.as_tensor(diagram_np, dtype=point_cloud.dtype,
                                        device=point_cloud.device),
                pairing=torch.empty((diagram_np.shape[0], 0), dtype=torch.long,
                                    device=point_cloud.device),
                dimension=h_dim,
            ))

        return result
