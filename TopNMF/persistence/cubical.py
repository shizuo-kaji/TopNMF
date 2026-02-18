"""Cubical complex for computing persistence diagrams from structured data."""

import numpy as np
import torch
from typing import List

import cripser

from . import PersistenceInfo

_INF_CUTOFF = np.finfo(np.float64).max / 2.0


class CubicalComplex(torch.nn.Module):
    """
    Compute persistence diagrams from structured data using cubical complexes.

    Parameters
    ----------
    superlevel : bool, default=False
        If True, use superlevel set filtration.
    mode : str, default='T'
        Filtration mode: 'T' or 'V'.
    dim : int, optional
        Maximum homology dimension.
    """

    def __init__(self, superlevel: bool = False, mode: str = 'T', dim: int = None):
        super().__init__()
        self.superlevel = superlevel
        if mode not in ['T', 'V']:
            raise ValueError(f"mode must be 'T' or 'V', got '{mode}'")
        self.mode = mode
        self.dim = dim

    def forward(self, x: torch.Tensor) -> List:
        ndim = x.ndim
        if ndim <= 2:
            return self._compute_single(x)
        elif ndim == 3:
            return [self._compute_single(x[i]) for i in range(x.shape[0])]
        elif ndim == 4:
            return [[self._compute_single(x[i, j])
                     for j in range(x.shape[1])]
                    for i in range(x.shape[0])]
        elif ndim == 5:
            return [[[self._compute_single(x[i, j, k])
                      for k in range(x.shape[2])]
                     for j in range(x.shape[1])]
                    for i in range(x.shape[0])]
        else:
            raise ValueError(f"Unsupported tensor dimension: {ndim}")

    def _compute_single(self, x: torch.Tensor) -> List[PersistenceInfo]:
        x_np = np.asarray(x.detach().cpu().numpy(), dtype=np.float64)
        if self.superlevel:
            x_np = -x_np

        spatial_dim = x_np.ndim
        max_dim = spatial_dim - 1
        if self.dim is not None:
            max_dim = min(max_dim, int(self.dim))

        if self.mode == 'T':
            if hasattr(cripser, "computePH_T"):
                ph = cripser.computePH_T(x_np, maxdim=max_dim, location='yes')
            else:
                ph = cripser.computePH(x_np, maxdim=max_dim, top_dim=True, location='yes')
        else:
            ph = cripser.computePH(x_np, maxdim=max_dim, location='yes')

        return [self._extract_persistence_info(x, ph, dim, spatial_dim)
                for dim in range(max_dim + 1)]

    def _extract_persistence_info(self, x, ph, dim, spatial_dim):
        device = x.device
        rows = np.asarray(ph)
        if rows.size == 0:
            return PersistenceInfo(
                diagram=torch.empty((0, 2), dtype=x.dtype, device=device),
                pairing=torch.empty((0, 2 * spatial_dim), dtype=torch.long, device=device),
                dimension=dim,
            )

        rows = rows[rows[:, 0].astype(int) == dim]
        if rows.size == 0:
            return PersistenceInfo(
                diagram=torch.empty((0, 2), dtype=x.dtype, device=device),
                pairing=torch.empty((0, 2 * spatial_dim), dtype=torch.long, device=device),
                dimension=dim,
            )

        creators_np = rows[:, 3:3 + spatial_dim].astype(np.int64, copy=False)
        destroyers_np = rows[:, 6:6 + spatial_dim].astype(np.int64, copy=False)
        infinite_mask = rows[:, 2] >= _INF_CUTOFF
        if infinite_mask.any():
            max_coord = np.array(
                np.unravel_index(int(torch.argmax(x).item()), x.shape), dtype=np.int64)
            destroyers_np = destroyers_np.copy()
            destroyers_np[infinite_mask] = max_coord

        for axis, axis_size in enumerate(x.shape):
            creators_np[:, axis] = np.clip(creators_np[:, axis], 0, axis_size - 1)
            destroyers_np[:, axis] = np.clip(destroyers_np[:, axis], 0, axis_size - 1)

        creator_flat = np.ravel_multi_index(creators_np.T, x.shape)
        destroyer_flat = np.ravel_multi_index(destroyers_np.T, x.shape)
        creator_idx = torch.as_tensor(creator_flat, dtype=torch.long, device=device)
        destroyer_idx = torch.as_tensor(destroyer_flat, dtype=torch.long, device=device)

        x_flat = x.reshape(-1)
        diagram = torch.stack([x_flat[creator_idx], x_flat[destroyer_idx]], dim=1)
        pairings = torch.as_tensor(
            np.concatenate([creators_np, destroyers_np], axis=1),
            dtype=torch.long, device=device)

        return PersistenceInfo(diagram=diagram, pairing=pairings, dimension=dim)
