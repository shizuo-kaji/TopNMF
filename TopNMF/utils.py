"""Utility functions: sparsity metrics, initialisation, point-cloud helpers."""

import numpy as np
import torch
from typing import Optional, Union

from gudhi.point_cloud.timedelay import TimeDelayEmbedding

# Maximum H1 persistence of a perfectly circular point cloud after centring and
# L2-normalisation; used to map periodicity scores into [0, 1].
SQRT3 = float(np.sqrt(3.0))


def l1_l2_sq_ratio(x: torch.Tensor, dim: Optional[int] = None,
                   eps: float = 1e-10) -> torch.Tensor:
    """
    Ratio ``(sum |x|)^2 / (sum x^2 + eps)`` of a tensor.

    This quantity ranges from 1 (a single non-zero entry) to ``n`` (all entries
    equal) and underlies several sparsity-related losses in the package.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    dim : int, optional
        Reduction dimension. If None, reduce over all elements and return a
        scalar; otherwise reduce along ``dim`` and return one value per slice.
    eps : float
        Numerical stability constant added to the denominator.

    Returns
    -------
    torch.Tensor
        The L1^2 / L2^2 ratio.
    """
    l1_sq = x.abs().sum(dim=dim) ** 2
    l2_sq = (x ** 2).sum(dim=dim)
    return l1_sq / (l2_sq + eps)


def sparsity_score(v: torch.Tensor, eps: float = 1e-10) -> Union[float, torch.Tensor]:
    """
    Hoyer sparsity score in [0, 1] (0 = dense, 1 = maximally sparse).

    Parameters
    ----------
    v : torch.Tensor
        Input vector.
    eps : float
        Numerical stability constant (avoids 0/0 for all-zero vectors).

    Returns
    -------
    float or torch.Tensor
        Sparsity score. Returns a tensor when ``v`` requires gradients.
    """
    n = v.numel()
    l1_norm = v.abs().sum()
    l2_norm = (v ** 2).sum().sqrt()

    score = (np.sqrt(n) - l1_norm / (l2_norm + eps)) / (np.sqrt(n) - 1)
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

def center_point_cloud(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Centre and normalise a NumPy point cloud (project perpendicular to all-ones, then L2-normalise)."""
    one = np.ones(X.shape[1], dtype=X.dtype)
    projection = (X @ one) / (one @ one)
    centered = X - np.outer(projection, one)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return np.divide(centered, norms, out=np.zeros_like(centered), where=norms > eps)


def center_point_cloud_torch(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Centre and normalise a PyTorch point cloud (differentiable)."""
    device = X.device
    one = torch.ones(X.shape[1], dtype=X.dtype, device=device)
    projection = (X @ one) / (one @ one)
    centered = X - projection.unsqueeze(1) * one.unsqueeze(0)
    norms = torch.norm(centered, dim=1, keepdim=True).clamp_min(eps)
    return centered / norms


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def periodicity_from_diagram(diagram: Union[torch.Tensor, object],
                             eps: float = 1e-12) -> torch.Tensor:
    """
    Periodicity score: ``max(death - birth) / sqrt(3)`` over a persistence diagram.

    Parameters
    ----------
    diagram : torch.Tensor or PersistenceInfo
        Birth-death pairs of shape (n, 2), or an object exposing a ``diagram``
        attribute (e.g. :class:`~TopNMF.persistence.PersistenceInfo`).
    eps : float
        Unused placeholder kept for signature symmetry with other helpers.

    Returns
    -------
    torch.Tensor
        Scalar periodicity score (0 if the diagram is empty).
    """
    pd = diagram.diagram if hasattr(diagram, "diagram") else diagram
    if pd.shape[0] == 0:
        return torch.zeros((), dtype=pd.dtype, device=pd.device)
    persistence = (pd[:, 1] - pd[:, 0])
    return persistence.max() / SQRT3


def compute_persistence_diagram(signal: np.ndarray, embedding_dim: int = 30,
                                tau: int = 1, max_dim: int = 1) -> dict:
    """
    Compute persistence diagrams for a 1-D signal via time-delay embedding.

    Uses :class:`~TopNMF.persistence.GudhiVietorisRipsComplex` as the single
    persistence backend (shared with training and visualisation).

    Parameters
    ----------
    signal : np.ndarray
        1-D time-series signal.
    embedding_dim : int
        Number of delayed copies in the time-delay embedding.
    tau : int
        Time delay.
    max_dim : int
        Maximum homology dimension.

    Returns
    -------
    dict
        Keys: 'dgms' (list of (n, 2) NumPy birth-death arrays per dimension),
        'embedded', 'centered'.
    """
    from .persistence import GudhiVietorisRipsComplex

    embedder = TimeDelayEmbedding(dim=embedding_dim, delay=tau)
    embedded = np.asarray(embedder(signal))
    centered = center_point_cloud(embedded)

    complex_fn = GudhiVietorisRipsComplex(dim=max_dim, p=2)
    pers_info = complex_fn(torch.as_tensor(centered, dtype=torch.float))
    dgms = [info.diagram.detach().cpu().numpy() for info in pers_info]
    return {
        'dgms': dgms,
        'embedded': embedded,
        'centered': centered,
    }


def compute_periodicity_score(signal: np.ndarray, embedding_dim: int = 30,
                              tau: int = 1, max_dim: int = 1) -> float:
    """
    Periodicity score in [0, 1] from max H1 persistence normalised by sqrt(3).

    Parameters
    ----------
    signal : np.ndarray
        1-D time-series signal.
    embedding_dim : int
        Number of delayed copies in the time-delay embedding.
    tau : int
        Time delay.
    max_dim : int
        Maximum homology dimension.

    Returns
    -------
    float
        Normalised periodicity score.
    """
    result = compute_persistence_diagram(
        signal, embedding_dim=embedding_dim, tau=tau, max_dim=max_dim)
    dgms = result['dgms']
    if len(dgms) > 1 and len(dgms[1]) > 0:
        h1 = torch.as_tensor(dgms[1], dtype=torch.float)
        return float(periodicity_from_diagram(h1))
    return 0.0
