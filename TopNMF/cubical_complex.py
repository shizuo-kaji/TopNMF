"""
Cubical Complex Module

This module provides functionality for computing persistence diagrams from
structured data (e.g., images, volumes) using cubical complexes.
Backed by CubicalRipser (`cripser`/`tcripser`).
"""

import numpy as np
import torch
from typing import List, NamedTuple

import cripser

_INF_CUTOFF = np.finfo(np.float64).max / 2.0


class PersistenceInfo(NamedTuple):
    """
    Container for persistence diagram information.

    Attributes
    ----------
    diagram : torch.Tensor
        Persistence diagram of shape (n_pairs, 2) with birth-death pairs
    pairing : torch.Tensor
        Coordinate pairings of shape (n_pairs, 2*dim) with creator/destroyer coordinates
    dimension : int
        Homology dimension (0, 1, 2, ...)
    """
    diagram: torch.Tensor
    pairing: torch.Tensor
    dimension: int


class CubicalComplex(torch.nn.Module):
    """
    Compute persistence diagrams from structured data using cubical complexes.

    Parameters
    ----------
    superlevel : bool, default=False
        If True, use superlevel set filtration. Otherwise, use sublevel sets.
    mode : str, default='T'
        Filtration mode: 'T' or 'vertices'

    Examples
    --------
    >>> # 2D image
    >>> cc = CubicalComplex()
    >>> image = torch.randn(28, 28)
    >>> persistence = cc(image)

    >>> # Batch of images with channels
    >>> images = torch.randn(8, 3, 28, 28)  # (batch, channel, height, width)
    >>> persistence = cc(images)

    References
    ----------
    Rieck et al., "Uncovering the Topology of Time-Varying fMRI Data Using
    Cubical Persistence", NeurIPS 2020.
    """

    def __init__(self, superlevel: bool = False,
                 mode: str = 'T', dim: int = None):
        super().__init__()
        self.superlevel = superlevel

        if mode not in ['T', 'V']:
            raise ValueError(
                f"mode must be 'T' or 'V', got '{mode}'"
            )
        self.mode = mode
        self.dim = dim

    def forward(self, x: torch.Tensor) -> List:
        """
        Compute persistence diagrams for input tensor.

        Handles various input shapes:
        - 2D (H, W): single image
        - 3D (C, H, W) or (D, H, W): single image with channels OR 3D volume
        - 4D (B, C, H, W): batch of images with channels
        - 5D (B, C, D, H, W): batch of 3D volumes with channels

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape described above

        Returns
        -------
        list or nested list of PersistenceInfo
            Persistence information for each homology dimension.
            Returns nested lists for batched/channeled inputs.
        """
        ndim = x.ndim

        if ndim <= 2:
            # Single image (1D or 2D)
            return self._compute_single(x)
        elif ndim == 3:
            # Either (C, H, W) or single 3D volume
            # Treat as channels
            return [self._compute_single(x[i]) for i in range(x.shape[0])]
        elif ndim == 4:
            # Batch with channels: (B, C, H, W)
            return [[self._compute_single(x[i, j])
                     for j in range(x.shape[1])]
                    for i in range(x.shape[0])]
        elif ndim == 5:
            # Batch of 3D volumes with channels: (B, C, D, H, W)
            return [[[self._compute_single(x[i, j, k])
                      for k in range(x.shape[2])]
                     for j in range(x.shape[1])]
                    for i in range(x.shape[0])]
        else:
            raise ValueError(f"Unsupported tensor dimension: {ndim}")

    def _compute_single(self, x: torch.Tensor) -> List[PersistenceInfo]:
        """
        Compute persistence for a single image/volume (no batch/channel dimensions).

        Parameters
        ----------
        x : torch.Tensor
            Single image/volume tensor

        Returns
        -------
        list of PersistenceInfo
            One entry per homology dimension
        """
        x_np = np.asarray(x.detach().cpu().numpy(), dtype=np.float64)

        # Flip values for superlevel filtration
        if self.superlevel:
            x_np = -x_np

        spatial_dim = x_np.ndim
        max_dim = spatial_dim - 1
        if self.dim is not None:
            max_dim = min(max_dim, int(self.dim))

        # Compute persistence with CubicalRipser backend.
        if self.mode == 'T':
            # T-construction if available, fallback to top_dim mode otherwise.
            if hasattr(cripser, "computePH_T"):
                ph = cripser.computePH_T(x_np, maxdim=max_dim, location='yes')
            else:
                ph = cripser.computePH(x_np, maxdim=max_dim, top_dim=True, location='yes')
        else:
            ph = cripser.computePH(x_np, maxdim=max_dim, location='yes')

        # Extract persistence information for each dimension
        result = []
        for dim in range(max_dim + 1):
            info = self._extract_persistence_info(x, ph, dim, spatial_dim)
            result.append(info)

        return result

    def _extract_persistence_info(
        self,
        x: torch.Tensor,
        ph: np.ndarray,
        dim: int,
        spatial_dim: int,
    ) -> PersistenceInfo:
        """
        Extract persistence diagram and pairings for a specific dimension.

        Parameters
        ----------
        x : torch.Tensor
            Original input tensor
        ph : np.ndarray
            CubicalRipser output with columns
            [dim, birth, death, b_x, b_y, b_z, d_x, d_y, d_z]
        dim : int
            Homology dimension to extract
        spatial_dim : int
            Number of spatial dimensions in the input

        Returns
        -------
        PersistenceInfo
            Persistence information for the specified dimension
        """
        device = x.device
        rows = np.asarray(ph)
        if rows.size == 0:
            diagram = torch.empty((0, 2), dtype=x.dtype, device=device)
            pairings = torch.empty((0, 2 * spatial_dim), dtype=torch.long, device=device)
            return PersistenceInfo(diagram=diagram, pairing=pairings, dimension=dim)

        rows = rows[rows[:, 0].astype(int) == dim]
        if rows.size == 0:
            diagram = torch.empty((0, 2), dtype=x.dtype, device=device)
            pairings = torch.empty((0, 2 * spatial_dim), dtype=torch.long, device=device)
            return PersistenceInfo(diagram=diagram, pairing=pairings, dimension=dim)

        creators_np = rows[:, 3:3 + spatial_dim].astype(np.int64, copy=False)
        destroyers_np = rows[:, 6:6 + spatial_dim].astype(np.int64, copy=False)
        infinite_mask = rows[:, 2] >= _INF_CUTOFF
        if infinite_mask.any():
            max_coord = np.array(np.unravel_index(int(torch.argmax(x).item()), x.shape), dtype=np.int64)
            destroyers_np = destroyers_np.copy()
            destroyers_np[infinite_mask] = max_coord

        # Keep coordinates in-bounds before indexing into x.
        for axis, axis_size in enumerate(x.shape):
            creators_np[:, axis] = np.clip(creators_np[:, axis], 0, axis_size - 1)
            destroyers_np[:, axis] = np.clip(destroyers_np[:, axis], 0, axis_size - 1)

        creator_flat_idx = np.ravel_multi_index(creators_np.T, x.shape)
        destroyer_flat_idx = np.ravel_multi_index(destroyers_np.T, x.shape)
        creator_flat_idx_t = torch.as_tensor(creator_flat_idx, dtype=torch.long, device=device)
        destroyer_flat_idx_t = torch.as_tensor(destroyer_flat_idx, dtype=torch.long, device=device)

        x_flat = x.reshape(-1)
        diagram = torch.stack([
            x_flat[creator_flat_idx_t],
            x_flat[destroyer_flat_idx_t],
        ], dim=1)

        pairings_np = np.concatenate([creators_np, destroyers_np], axis=1)
        pairings = torch.as_tensor(pairings_np, dtype=torch.long, device=device)

        return PersistenceInfo(
            diagram=diagram,
            pairing=pairings,
            dimension=dim
        )
