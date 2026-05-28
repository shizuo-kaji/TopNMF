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

    def _extract_indices(self, gens, h_dim: int, device) -> torch.Tensor:
        """Pull the generator index array for homology dimension ``h_dim``.

        ``flag_persistence_generators()`` returns
        ``(dim0_finite (n,3), [dim>=1 arrays (n,4)], dim0_essential, [essential higher])``.
        Essential (infinite) H0 generators (``gens[2]``) are intentionally
        dropped here; infinite deaths in finite bars are handled below.
        """
        if h_dim == 0:
            arr = gens[0]
            cols = 3
        else:
            higher = gens[1] if len(gens) > 1 else []
            arr = higher[h_dim - 1] if h_dim - 1 < len(higher) else None
            cols = 4

        if arr is None or np.size(arr) == 0:
            return torch.empty((0, cols), dtype=torch.long, device=device)
        return torch.as_tensor(np.asarray(arr), dtype=torch.long, device=device)

    def __call__(self, point_cloud: torch.Tensor) -> List[PersistenceInfo]:
        points_np = point_cloud.detach().cpu().numpy()
        rips = (
            gudhi.RipsComplex(points=points_np)
            if self.max_edge_length is None
            else gudhi.RipsComplex(points=points_np, max_edge_length=float(self.max_edge_length))
        )
        simplex_tree = rips.create_simplex_tree(max_dimension=self.dim + 1)
        simplex_tree.compute_persistence()
        gens = simplex_tree.flag_persistence_generators()

        result: List[PersistenceInfo] = []
        for h_dim in range(self.dim + 1):
            indices = self._extract_indices(gens, h_dim, point_cloud.device)

            if indices.shape[0] == 0:
                result.append(PersistenceInfo(
                    diagram=torch.empty((0, 2), dtype=point_cloud.dtype,
                                        device=point_cloud.device),
                    pairing=torch.empty((0, indices.shape[1]), dtype=torch.long,
                                        device=point_cloud.device),
                    dimension=h_dim,
                ))
                continue

            # Compute persistence values from generator indices.
            if indices.shape[1] == 4:
                birth_death = torch.norm(
                    point_cloud[indices[:, (0, 2)]] - point_cloud[indices[:, (1, 3)]],
                    dim=-1
                )
            else:  # 3 columns: H0 finite bars born at filtration 0.
                deaths = torch.norm(
                    point_cloud[indices[:, 1]] - point_cloud[indices[:, 2]],
                    dim=-1
                )
                births = torch.zeros_like(deaths)
                birth_death = torch.stack([births, deaths], dim=1)

            deaths = birth_death[:, 1].clone()  # Clone to avoid in-place issues
            finite_deaths = deaths[torch.isfinite(deaths)]
            if len(finite_deaths) > 0:
                replacement_val = torch.max(finite_deaths)
            elif birth_death.shape[0] > 0:
                replacement_val = torch.max(birth_death[:, 0])
            else:
                replacement_val = 0.0 # Should not happen given check above

            deaths[~torch.isfinite(deaths)] = replacement_val

            # Reconstruct diagram with replaced infinite deaths
            diagram = torch.stack([birth_death[:, 0], deaths], dim=1)

            result.append(PersistenceInfo(
                diagram=diagram,
                pairing=indices, # Store the indices as pairing info
                dimension=h_dim,
            ))

        return result
