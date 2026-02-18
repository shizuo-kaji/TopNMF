"""Loss functions for topological NMF optimisation."""

import torch
from typing import List, Optional, Dict


def total_variation(v: torch.Tensor) -> torch.Tensor:
    """
    Squared total variation for 1-D or 2-D tensors.

    For 1-D: sum_i (v[i+1] - v[i])^2
    For 2-D: sqrt(sum row-diffs^2) + sqrt(sum col-diffs^2)

    Parameters
    ----------
    v : torch.Tensor
        Input tensor with 1 or 2 dimensions.

    Returns
    -------
    torch.Tensor
        Total variation as a scalar tensor.
    """
    if v.ndim == 1:
        return torch.diff(v).pow(2).sum()

    if v.ndim == 2:
        tv_rows = torch.diff(v, dim=0).pow(2).sum()
        tv_cols = torch.diff(v, dim=1).pow(2).sum()
        return tv_rows.sqrt() + tv_cols.sqrt()

    raise ValueError(f"total_variation expects a 1D or 2D tensor, got {v.ndim}D")


def ph_sparsity_loss(diagrams: List, PH_dims: List[int],
                     target_diagrams: Optional[List] = None,
                     device: str = 'cpu') -> torch.Tensor:
    """
    Persistence sparsity loss (L1^2 / L2^2 ratio of persistence values).

    Parameters
    ----------
    diagrams : List
        List of persistence diagrams from complex
    PH_dims : List[int]
        Homology dimensions to consider
    target_diagrams : Optional[List]
        Not used (kept for signature compatibility)
    device : str
        Computing device

    Returns
    -------
    torch.Tensor
        Computed loss value
    """
    PH = torch.cat([diagrams[dim].diagram for dim in PH_dims])
    pers = torch.diff(PH, dim=1).reshape(-1)
    l1sq = pers.sum() ** 2
    l2sq = pers.pow(2).sum()
    return l1sq / (l2sq + 1e-10)


def target_diagram_loss(diagrams: List, PH_dims: List[int],
                        target_diagrams: Optional[List] = None,
                        device: str = 'cpu', power: int = 2,
                        remove_longest: Optional[Dict[int, bool]] = None) -> torch.Tensor:
    """
    Element-wise comparison of sorted persistence values against a target diagram.

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
        Power for persistence difference
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
        D1 = diagrams[dim].diagram if hasattr(diagrams[dim], 'diagram') else diagrams[dim]
        D2 = target_diagrams[dim]

        pers1 = torch.diff(D1, dim=1).reshape(-1)
        pers1, _ = torch.sort(pers1, dim=0)
        if remove_longest.get(dim, False) and len(pers1) > 0:
            pers1 = pers1[:-1]

        pers2 = torch.diff(D2, dim=1).reshape(-1)
        pers2, _ = torch.sort(pers2, dim=0)

        if len(pers1) > len(pers2):
            p = len(pers1) - len(pers2)
            loss += pers1[:p].abs().pow(power).sum()
            if len(pers2) > 0:
                loss += (pers1[p:] - pers2).abs().pow(power).sum()
        else:
            p = len(pers2) - len(pers1)
            loss += pers2[:p].abs().pow(power).sum()
            if len(pers1) > 0:
                loss += (pers2[p:] - pers1).abs().pow(power).sum()

    return loss


def weighted_persistence_loss(diagrams: List, PH_dims: List[int],
                              target_diagrams: Optional[List] = None,
                              device: str = 'cpu', pow: int = 2, eps: float = 1e-4,
                              remove_longest: bool = True) -> torch.Tensor:
    """
    Weighted persistence loss: weights births by (1 - |deaths|).

    Parameters
    ----------
    diagrams : List
        List of persistence diagrams from complex
    PH_dims : List[int]
        Homology dimensions to consider
    target_diagrams : Optional[List]
        Not used (kept for signature compatibility)
    device : str
        Computing device
    pow : int
        Power in the loss definition
    eps : float
        Small constant
    remove_longest : bool
        Whether to remove the longest persistence interval

    Returns
    -------
    torch.Tensor
        Total weighted topological loss
    """
    loss = torch.tensor(0., device=device)

    for dim in PH_dims:
        if dim >= len(diagrams):
            continue
        D = diagrams[dim].diagram if hasattr(diagrams[dim], 'diagram') else diagrams[dim]
        if D is None or D.shape[0] == 0:
            continue

        births = D[:, 0]
        deaths = D[:, 1]
        persistence = (deaths - births).abs()

        if remove_longest and len(persistence) > 0:
            longest_idx = torch.argmax(persistence)
            mask = torch.ones(len(D), dtype=torch.bool, device=D.device)
            mask[longest_idx] = False
            births, deaths, persistence = births[mask], deaths[mask], persistence[mask]

        weights = (1.0 - torch.abs(deaths)).clamp(min=0)
        loss += (weights * persistence).pow(pow).sum()

    return loss


def clique_deviation_loss(diags: List, PH_dims: List[int], target_PH: List,
                          device: str, alpha: float = 1.0,
                          remove_longest: Optional[Dict[int, bool]] = None) -> torch.Tensor:
    """
    Clique deviation loss: negated target-diagram loss (for maximisation).

    Parameters
    ----------
    diags : List
        List of persistence diagrams
    PH_dims : List[int]
        Dimensions to consider
    target_PH : List
        Target persistence diagrams
    device : str
        Computing device
    alpha : float
        Weight for higher dimensions (dim > 0)
    remove_longest : Dict[int, bool], optional
        Whether to remove longest persistence for each dimension

    Returns
    -------
    torch.Tensor
        Computed loss value (typically negative)
    """
    loss_PH = torch.tensor(0., device=device)
    if remove_longest is None:
        remove_longest = {}

    for dim in PH_dims:
        term = target_diagram_loss(diags, [dim], target_PH, device,
                                   remove_longest=remove_longest)
        loss_PH -= (alpha * term) if dim > 0 else term

    return loss_PH
