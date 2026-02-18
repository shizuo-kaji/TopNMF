"""Utility functions: sparsity metrics, initialisation, point-cloud helpers."""

import numpy as np
import torch
from typing import Union

from gudhi.point_cloud.timedelay import TimeDelayEmbedding
from ripser import ripser


def sparsity_score(v: torch.Tensor) -> Union[float, torch.Tensor]:
    """
    Hoyer sparsity score in [0, 1] (0 = dense, 1 = maximally sparse).

    Parameters
    ----------
    v : torch.Tensor
        Input vector

    Returns
    -------
    float or torch.Tensor
        Sparsity score. Returns a tensor when ``v`` requires gradients.
    """
    n = len(v.ravel())
    l1_norm = v.abs().sum()
    l2_norm = (v ** 2).sum().sqrt()

    score = (np.sqrt(n) - l1_norm / l2_norm) / (np.sqrt(n) - 1)
    if score.requires_grad:
        return score
    return float(score.item())


def svd_initialization(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialise W and V for NMF via truncated SVD with absolute-value projection.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_components : int
        Number of components

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (W, V) initialised factor matrices
    """
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    U = U[:, :n_components]
    S = np.diag(S[:n_components])
    VT = VT[:n_components, :]
    W = np.abs(U @ np.sqrt(S))
    V = np.abs(np.sqrt(S) @ VT)
    return W, V


# ---------------------------------------------------------------------------
# Point-cloud centring
# ---------------------------------------------------------------------------

def center_point_cloud(X: np.ndarray) -> np.ndarray:
    """Centre and normalise a NumPy point cloud (project perpendicular to all-ones, then L2-normalise)."""
    one = np.ones(X.shape[1])
    projection = (X @ one) / (one @ one)
    centered = X - np.outer(projection, one)
    return centered / np.linalg.norm(centered, axis=1, keepdims=True)


def center_point_cloud_torch(X: torch.Tensor) -> torch.Tensor:
    """Centre and normalise a PyTorch point cloud (differentiable)."""
    device = X.device
    one = torch.ones(X.shape[1], device=device)
    projection = (X @ one) / (one @ one)
    centered = X - projection.unsqueeze(1) * one.unsqueeze(0)
    return centered / torch.norm(centered, dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Persistence helpers (NumPy / offline)
# ---------------------------------------------------------------------------

def compute_persistence_diagram(signal: np.ndarray, dim: int = 30,
                                tau: int = 1, max_dim: int = 1) -> dict:
    """
    Compute persistence diagram for a 1-D signal via time-delay embedding.

    Parameters
    ----------
    signal : np.ndarray
        1D time series signal
    dim : int
        Embedding dimension
    tau : int
        Time delay
    max_dim : int
        Maximum homology dimension

    Returns
    -------
    dict
        Keys: 'dgms' (list of diagrams), 'embedded', 'centered'.
    """
    embedder = TimeDelayEmbedding(dim=dim, delay=tau)
    embedded = embedder(signal)
    centered = center_point_cloud(embedded)
    result = ripser(centered, maxdim=max_dim)
    return {
        'dgms': result['dgms'],
        'embedded': embedded,
        'centered': centered,
    }


def compute_periodicity_score(signal: np.ndarray, dim: int = 30,
                              tau: int = 1, max_dim: int = 1) -> float:
    """
    Periodicity score in [0, 1] from max H1 persistence normalised by sqrt(3).

    Parameters
    ----------
    signal : np.ndarray
        1D time series signal
    dim : int
        Embedding dimension
    tau : int
        Time delay
    max_dim : int
        Maximum homology dimension

    Returns
    -------
    float
        Normalised periodicity score
    """
    result = compute_persistence_diagram(signal, dim=dim, tau=tau, max_dim=max_dim)
    diagrams = result['dgms']
    if len(diagrams) > 1 and len(diagrams[1]) > 0:
        persistence_values = diagrams[1][:, 1] - diagrams[1][:, 0]
        return float(np.max(persistence_values) / np.sqrt(3))
    return 0.0
