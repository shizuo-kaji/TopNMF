"""
Topological Data Analysis Utilities

This module provides functions for time delay embedding, persistent homology,
and topological feature extraction from time series data.
"""

import numpy as np
import torch
from ripser import ripser
from gudhi.point_cloud.timedelay import TimeDelayEmbedding
from typing import Optional


def center_point_cloud(X: np.ndarray) -> np.ndarray:
    """
    Center and normalize a point cloud by projecting onto perpendicular space.

    Parameters
    ----------
    X : np.ndarray
        Point cloud of shape (n_points, n_dimensions)

    Returns
    -------
    np.ndarray
        Centered and normalized point cloud
    """
    one = np.ones(X.shape[1])
    projection = (X @ one) / (one @ one)
    centered = X - np.outer(projection, one)
    normalized = centered / np.linalg.norm(centered, axis=1, keepdims=True)

    return normalized


class TimeDelayEmbeddingTorch(torch.nn.Module):
    """
    PyTorch-based time delay embedding for time series.

    This class creates a delayed embedding of a time series, suitable for
    topological data analysis and dynamical systems reconstruction.

    Parameters
    ----------
    dim : int, optional
        Embedding dimension (number of delayed copies)
    delay : int, optional
        Time delay between copies

    Attributes
    ----------
    dim : int
        Embedding dimension
    delay : int
        Time delay parameter
    """

    def __init__(self, dim: int = 3, delay: int = 1):
        super(TimeDelayEmbeddingTorch, self).__init__()
        self.dim = dim
        self.delay = delay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply time delay embedding to input time series.

        Parameters
        ----------
        x : torch.Tensor
            1D tensor of shape (T,) representing time series

        Returns
        -------
        torch.Tensor
            2D embedded tensor of shape (T - (dim-1)*delay, dim)

        Raises
        ------
        ValueError
            If time series is too short for the specified embedding
        """
        N = x.shape[0]
        d = self.dim
        τ = self.delay

        if N - (d - 1) * τ <= 0:
            raise ValueError(
                f"Time series length {N} is too short for embedding with "
                f"dim={d} and delay={τ}"
            )

        # Create delayed copies
        embedded = torch.stack(
            [x[i:N - (d - 1) * τ + i] for i in range(0, d * τ, τ)],
            dim=1
        )

        return embedded


def center_point_cloud_torch(X: torch.Tensor) -> torch.Tensor:
    """
    Center and normalize a point cloud using PyTorch tensors.

    This function projects the point cloud onto the space perpendicular
    to the all-ones vector and normalizes each point.

    Parameters
    ----------
    X : torch.Tensor
        Point cloud of shape (n_points, n_dimensions)

    Returns
    -------
    torch.Tensor
        Centered and normalized point cloud
    """
    device = X.device
    one = torch.ones(X.shape[1], device=device)  # (D,)
    projection = (X @ one) / (one @ one)  # (N,)
    outer = projection.unsqueeze(1) * one.unsqueeze(0)  # (N, D)
    centered = X - outer
    normalized = centered / torch.norm(centered, dim=1, keepdim=True)

    return normalized


def compute_periodicity_score(signal: np.ndarray, dim: int = 30,
                               tau: int = 1, max_dim: int = 1) -> float:
    """
    Compute periodicity score for a time series signal.

    The periodicity score measures the periodicity of a signal using
    topological data analysis. A higher score indicates stronger periodicity.

    Parameters
    ----------
    signal : np.ndarray
        1D time series signal
    dim : int, optional
        Embedding dimension for time delay embedding
    tau : int, optional
        Time delay for embedding
    max_dim : int, optional
        Maximum homology dimension to compute (0 or 1)

    Returns
    -------
    float
        Normalized periodicity score (0 to 1), where 1 indicates
        perfect periodicity
    """
    # Create time delay embedding
    embedder = TimeDelayEmbedding(dim=dim, delay=tau)
    X = embedder(signal)
    X = center_point_cloud(X)

    # Compute persistence diagrams
    diagrams = ripser(X, maxdim=max_dim)['dgms']

    # Extract H1 (1-dimensional homology) persistence
    if len(diagrams) > 1 and len(diagrams[1]) > 0:
        # Compute persistence (death - birth)
        persistence_values = diagrams[1][:, 1] - diagrams[1][:, 0]
        max_persistence = np.max(persistence_values)
        # Normalize by sqrt(3) (theoretical maximum for unit sphere)
        score = max_persistence / np.sqrt(3)
    else:
        score = 0.0

    return score


def compute_persistence_diagram(signal: np.ndarray, dim: int = 30,
                                 tau: int = 1, max_dim: int = 1) -> dict:
    """
    Compute full persistence diagram for a time series signal.

    Parameters
    ----------
    signal : np.ndarray
        1D time series signal
    dim : int, optional
        Embedding dimension for time delay embedding
    tau : int, optional
        Time delay for embedding
    max_dim : int, optional
        Maximum homology dimension to compute

    Returns
    -------
    dict
        Dictionary containing:
        - 'dgms': List of persistence diagrams for each dimension
        - 'embedded': The embedded point cloud
        - 'centered': The centered point cloud
    """
    # Create time delay embedding
    embedder = TimeDelayEmbedding(dim=dim, delay=tau)
    embedded = embedder(signal)
    centered = center_point_cloud(embedded)

    # Compute persistence diagrams
    result = ripser(centered, maxdim=max_dim)

    return {
        'dgms': result['dgms'],
        'embedded': embedded,
        'centered': centered
    }


def compute_bottleneck_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """
    Compute bottleneck distance between two persistence diagrams.

    Note: This is a placeholder. For full implementation, use gudhi or persim.

    Parameters
    ----------
    dgm1 : np.ndarray
        First persistence diagram (n x 2 array)
    dgm2 : np.ndarray
        Second persistence diagram (m x 2 array)

    Returns
    -------
    float
        Bottleneck distance between diagrams
    """
    # This is a simplified implementation
    # For production use, consider using gudhi.bottleneck_distance
    # or persim.bottleneck
    raise NotImplementedError(
        "Full bottleneck distance requires additional dependencies. "
        "Consider using gudhi.bottleneck_distance or persim.bottleneck"
    )
