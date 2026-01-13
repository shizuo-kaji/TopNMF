"""
Cubical Complex Module

This module provides functionality for computing persistence diagrams from
structured data (e.g., images, volumes) using cubical complexes.
Based on the "torch_topological" package.
"""

import numpy as np
import torch
import gudhi
from typing import List, NamedTuple, Optional


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
        # Convert to numpy for gudhi
        x_np = x.detach().cpu().numpy()

        # Flip values for superlevel filtration
        if self.superlevel:
            x_np = -x_np

        # Build cubical complex
        if self.mode == 'T':
            cc = gudhi.CubicalComplex(
                dimensions=x_np.shape,
                T=x_np.flatten()
            )
        else:
            cc = gudhi.CubicalComplex(
                dimensions=x_np.shape,
                vertices=x_np.flatten()
            )

        # Compute persistence
        cc.persistence()

        # Get persistence pairs
        if self.mode == 'T':
            pairs = cc.cofaces_of_persistence_pairs()
        else:
            pairs = cc.vertices_of_persistence_pairs()

        # Extract persistence information for each dimension
        max_dim = len(x_np.shape)
        result = []

        for dim in range(max_dim):
            info = self._extract_persistence_info(x, x_np.shape, pairs, dim)
            result.append(info)

        return result

    def _extract_persistence_info(
        self,
        x: torch.Tensor,
        shape: tuple,
        pairs: tuple,
        dim: int
    ) -> PersistenceInfo:
        """
        Extract persistence diagram and pairings for a specific dimension.

        Parameters
        ----------
        x : torch.Tensor
            Original input tensor
        shape : tuple
            Shape of the input
        pairs : tuple
            Persistence pairs from gudhi (finite, infinite)
        dim : int
            Homology dimension to extract

        Returns
        -------
        PersistenceInfo
            Persistence information for the specified dimension
        """
        device = x.device
        all_pairs = []

        # Extract finite pairs
        try:
            finite_pairs = torch.as_tensor(
                pairs[0][dim], dtype=torch.long, device=device
            )
            all_pairs.append(finite_pairs)
        except (IndexError, KeyError):
            pass

        # Extract infinite pairs
        try:
            infinite_pairs = torch.as_tensor(
                pairs[1][dim], dtype=torch.long, device=device
            )
            # Pair infinite features with maximum value index
            max_idx = torch.argmax(x.flatten())
            fake_destroyers = torch.full_like(infinite_pairs, max_idx)
            infinite_pairs = torch.stack([infinite_pairs, fake_destroyers], dim=1)
            all_pairs.append(infinite_pairs)
        except (IndexError, KeyError):
            pass

        # Combine all pairs
        if all_pairs:
            combined_pairs = torch.cat(all_pairs, dim=0)
        else:
            # No pairs for this dimension
            combined_pairs = torch.empty((0, 2), dtype=torch.long, device=device)

        # Convert flat indices to coordinates
        creators = self._unravel_index(combined_pairs[:, 0], shape, device)
        destroyers = self._unravel_index(combined_pairs[:, 1], shape, device)
        pairings = torch.cat([creators, destroyers], dim=1)

        # Build persistence diagram (birth, death)
        x_flat = x.flatten()
        if combined_pairs.shape[0] > 0:
            diagram = torch.stack([
                x_flat[combined_pairs[:, 0]],
                x_flat[combined_pairs[:, 1]]
            ], dim=1)
        else:
            diagram = torch.empty((0, 2), dtype=x.dtype, device=device)

        return PersistenceInfo(
            diagram=diagram,
            pairing=pairings,
            dimension=dim
        )

    @staticmethod
    def _unravel_index(
        indices: torch.Tensor,
        shape: tuple,
        device: torch.device
    ) -> torch.Tensor:
        """
        Convert flat indices to multi-dimensional coordinates.

        Similar to numpy.unravel_index but for PyTorch tensors.

        Parameters
        ----------
        indices : torch.Tensor
            Flat indices of shape (n,)
        shape : tuple
            Shape to unravel into
        device : torch.device
            Device for output tensor

        Returns
        -------
        torch.Tensor
            Coordinates of shape (n, len(shape))
        """
        if indices.numel() == 0:
            return torch.empty((0, len(shape)), dtype=torch.long, device=device)

        # Convert to numpy for unraveling
        indices_np = indices.cpu().numpy()
        coords_np = np.column_stack(np.unravel_index(indices_np, shape))

        return torch.as_tensor(coords_np, dtype=torch.long, device=device)
