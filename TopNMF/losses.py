"""
Loss Functions for Topological NMF

This module provides various loss functions for persistent homology
and topological data analysis in the context of NMF optimization.
"""

import torch
from typing import List, Optional, Dict


def total_variation(v: torch.Tensor) -> torch.Tensor:
    """
    Compute squared total variation for 1D or 2D tensors.

    For a 1D tensor, this is:
        sum_i (v[i+1] - v[i])^2

    For a 2D tensor, this is the sum of squared finite differences
    along both axes:
        sum_{i,j} (v[i+1, j] - v[i, j])^2 +
        sum_{i,j} (v[i, j+1] - v[i, j])^2

    Parameters
    ----------
    v : torch.Tensor
        Input tensor with 1 or 2 dimensions.

    Returns
    -------
    torch.Tensor
        Squared total variation as a scalar tensor.

    Raises
    ------
    ValueError
        If `v` is not 1D or 2D.
    """
    if v.ndim == 1:
        return torch.diff(v).pow(2).sum()

    if v.ndim == 2:
        tv_rows = torch.diff(v, dim=0).pow(2).sum()
        tv_cols = torch.diff(v, dim=1).pow(2).sum()
        return tv_rows.sqrt() + tv_cols.sqrt()

    raise ValueError(f"total_variation expects a 1D or 2D tensor, got {v.ndim}D")


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
        if hasattr(diagrams[dim], 'diagram'):
            D1 = diagrams[dim].diagram
        else:
            D1 = diagrams[dim]
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


def weighted_persistence_loss(diagrams: List, PH_dims: List[int],
                              target_diagrams: Optional[List] = None,
                              device: str = 'cpu', pow: int = 2, eps: float = 1e-4,
                              remove_longest: bool = True) -> torch.Tensor:
    """
    Compute weighted persistence loss for multiple dimensions.
    L_top(v) = sum_i (w_i * |b_i - d_i|)^p

    It iterates over specified homology dimensions and computes the weighted
    persistence loss for each.

    Parameters
    ----------
    diagrams : List
        List of persistence diagrams from complex
    PH_dims : List[int]
        Homology dimensions to consider
    target_diagrams : Optional[List]
        Target diagrams (not used in this loss, but kept for signature compatibility)
    device : str
        Computing device
    pow : int
        The power p in the loss definition
    eps : float
        Small constant
    remove_longest : bool
        Whether to remove the interval with longest persistence

    Returns
    -------
    torch.Tensor
        Total weighted topological loss
    """
    loss = torch.tensor(0., device=device)

    for dim in PH_dims:
        # Extract diagram tensor
        if dim < len(diagrams):
            # Handle both Diagram object and Tensor
            if hasattr(diagrams[dim], 'diagram'):
                D = diagrams[dim].diagram
            else:
                D = diagrams[dim]

            # Compute loss for this dimension
            if D is not None and D.shape[0] > 0:
                births = D[:, 0]
                deaths = D[:, 1]
                persistence = (deaths - births).abs()

                # Optional: remove the longest bar
                if remove_longest and len(persistence) > 0:
                    longest_idx = torch.argmax(persistence)
                    mask = torch.ones(len(D), dtype=torch.bool, device=D.device)
                    mask[longest_idx] = False
                    births = births[mask]
                    deaths = deaths[mask]
                    persistence = persistence[mask]

                # Compute the weighted loss
                weights = (1.0 - torch.abs(deaths)).clamp(min=0)
                weighted = (weights * persistence).pow(pow)
                loss += weighted.sum()

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




def clique_deviation_loss(diags: List, PH_dims: List[int], target_PH: List, device: str,
                          alpha: float = 1.0,
                          remove_longest: Optional[Dict[int, bool]] = None) -> torch.Tensor:
    """
    Clique Deviation Loss.

    Computes topological loss by subtracting persistence loss from the total loss.
    Designed to maximize deviation from target diagrams or persistence features.

    Parameters
    ----------
    diags : List
        List of persistence diagrams (or objects with .diagram attribute)
    PH_dims : List[int]
        Dimensions to consider
    target_PH : List
        Target persistence diagrams/values
    device : str
        Computing device
    alpha : float
        Weight for higher dimensions (dim > 0)
    remove_longest : Dict[int, bool], optional
        Dictionary specifying whether to remove longest persistence for each dimension.
        Key is dimension, Value is bool. Default is False for all.

    Returns
    -------
    torch.Tensor
        Computed loss value (typically negative due to subtraction logic)
    """
    # Initialize loss_PH
    loss_PH = torch.tensor(0., device=device)

    if remove_longest is None:
        remove_longest = {}

    for dim in PH_dims:
        term = target_diagram_loss(diags, [dim], target_PH, device, remove_longest=remove_longest)

        if dim == 0:
            loss_PH -= term
        else:
            loss_PH -= alpha * term


    return loss_PH
