"""
Loss Functions for Topological NMF

This module provides various loss functions for persistent homology
and topological data analysis in the context of NMF optimization.
"""

import torch
from typing import List, Optional, Dict


def ph_sparsity_loss(diagrams: List, PH_dims: List[int], target_diagrams: Optional[List] = None,
                     device: str = 'cpu') -> torch.Tensor:
    """
    Persistent homology sparsity loss (L1^2 / L2^2 ratio).

    This loss encourages sparse persistence diagrams by minimizing the
    ratio of squared L1 norm to squared L2 norm of persistence values.

    Parameters
    ----------
    diagrams : List
        List of persistence diagrams from complex
    PH_dims : List[int]
        Homology dimensions to consider
    target_diagrams : Optional[List]
        Target persistence diagrams (not used in this loss)
    device : str
        Computing device

    Returns
    -------
    torch.Tensor
        Computed loss value
    """
    PH = torch.cat([diagrams[dim].diagram for dim in PH_dims])
    pers1 = torch.diff(PH, dim=1).reshape(-1)
    epsilon = 1e-10
    l1sq = pers1.sum()**2
    l2sq = pers1.pow(2).sum()
    return l1sq / (l2sq + epsilon)


def target_diagram_loss(diagrams: List, PH_dims: List[int], target_diagrams: Optional[List] = None,
                        device: str = 'cpu', power: int = 2,
                        remove_longest: Optional[Dict[int, bool]] = None) -> torch.Tensor:
    """
    Loss function comparing persistence diagrams to target diagrams.

    Computes the difference between computed persistence diagrams and
    target diagrams using sorted persistence values.

    Parameters
    ----------
    diagrams : List
        List of persistence diagrams from complex
    PH_dims : List[int]
        Homology dimensions to consider
    target_diagrams : Optional[List]
        Target persistence diagrams for each dimension
    device : str
        Computing device
    power : int
        Power for persistence difference (default: 2)
    remove_longest : Optional[Dict[int, bool]]
        Whether to remove longest persistence for each dimension

    Returns
    -------
    torch.Tensor
        Computed loss value
    """
    if target_diagrams is None:
        raise ValueError("target_diagrams must be provided for target_diagram_loss")

    if remove_longest is None:
        remove_longest = {0: True, 1: False}

    loss = torch.tensor(0., device=device)

    for dim in PH_dims:
        D1 = diagrams[dim].diagram
        D2 = target_diagrams[dim]

        pers1 = torch.diff(D1, dim=1).reshape(-1)
        pers1, _ = torch.sort(pers1, dim=0)
        if remove_longest.get(dim, False) and len(pers1) > 0:
            pers1 = pers1[:-1]

        pers2 = torch.diff(D2, dim=1).reshape(-1)
        pers2, _ = torch.sort(pers2, dim=0)

        if len(pers1) > len(pers2):
            p = len(pers1) - len(pers2)
            loss += (pers1[:p]).abs().pow(power).sum()
            if len(pers2) > 0:
                loss += (pers1[p:] - pers2).abs().pow(power).sum()
        else:
            p = len(pers2) - len(pers1)
            loss += (pers2[:p]).abs().pow(power).sum()
            if len(pers1) > 0:
                loss += (pers2[p:] - pers1).abs().pow(power).sum()

    return loss


def weighted_persistence_loss(D: torch.Tensor, pow: int = 2, eps: float = 1e-4,
                               remove_longest: bool = True) -> torch.Tensor:
    """
    Compute weighted persistence loss.

    L_top(v) = sum_i (w_i * |b_i - d_i|)^p

    where weights are computed as (1 - |death|)^p, penalizing features
    that persist close to the boundary.

    Parameters
    ----------
    D : torch.Tensor
        Persistence diagram of shape [N, 2], where each row is (birth, death)
    pow : int
        The power p in the loss definition
    eps : float
        Small constant to avoid division by zero when d_i is very small
    remove_longest : bool
        Whether to remove the interval with longest persistence

    Returns
    -------
    torch.Tensor
        The computed weighted topological loss
    """
    births = D[:, 0]
    deaths = D[:, 1]
    persistence = (deaths - births).abs()

    # Optional: remove the longest bar
    if remove_longest and len(persistence) > 0:
        longest_idx = torch.argmax(persistence)
        mask = torch.ones(len(D), dtype=torch.bool)
        mask[longest_idx] = False
        births = births[mask]
        deaths = deaths[mask]
        persistence = persistence[mask]

    # Compute the weighted loss
    weights = (1.0 - torch.abs(deaths)).pow(pow)
    weighted = (weights * persistence).pow(pow)
    loss = weighted.sum()

    return loss

def reconstruction_loss(X: torch.Tensor, W: torch.Tensor, V: torch.Tensor,
                        reduction: str = 'mean') -> torch.Tensor:
    """
    Compute reconstruction loss for NMF.

    Parameters
    ----------
    X : torch.Tensor
        Original data matrix of shape (n_samples, n_features)
    W : torch.Tensor
        Weight matrix of shape (n_samples, n_components)
    V : torch.Tensor
        Basis matrix of shape (n_components, n_features)
    reduction : str
        Reduction method: 'mean', 'sum', or 'none'

    Returns
    -------
    torch.Tensor
        Reconstruction loss
    """
    diff = X - W @ V
    if reduction == 'mean':
        return (diff ** 2).mean()
    elif reduction == 'sum':
        return (diff ** 2).sum()
    else:
        return diff ** 2


def sparsity_loss(v: torch.Tensor, target_sparsity: Optional[float] = None) -> torch.Tensor:
    """
    Compute sparsity loss for a vector.

    If target_sparsity is provided, computes squared difference from target.
    Otherwise, computes L1^2/L2^2 ratio.

    Parameters
    ----------
    v : torch.Tensor
        Input vector
    target_sparsity : Optional[float]
        Target sparsity level (0 to 1)

    Returns
    -------
    torch.Tensor
        Sparsity loss value
    """
    if target_sparsity is not None:
        # Hoyer sparsity score
        n = len(v.ravel())
        l1_norm = v.abs().sum()
        l2_norm = (v**2).sum().sqrt()
        score = (n**0.5 - l1_norm / l2_norm) / (n**0.5 - 1)
        return (score - target_sparsity) ** 2
    else:
        # L1^2 / L2^2 ratio
        return (v.abs().sum())**2 / ((v**2).sum() + 1e-10)
