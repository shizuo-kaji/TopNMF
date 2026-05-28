"""Loss functions for topological NMF optimisation.

All persistence-based losses share the calling convention
``fn(diagrams, PH_dims, target_diagrams=None, device='cpu', **kwargs)`` so they
are interchangeable as the ``ph_loss_fn`` of :class:`~TopNMF.model.TopologicalNMF`.
The single tunable exponent of each loss is named ``power``.
"""

import torch
from typing import List, Optional, Dict

from .utils import l1_l2_sq_ratio


def _get_diagram(diagrams: List, dim: int) -> Optional[torch.Tensor]:
    """
    Return the birth-death tensor for homology dimension ``dim``.

    Accepts either a list of :class:`~TopNMF.persistence.PersistenceInfo`
    objects (with a ``.diagram`` attribute) or a list of raw (n, 2) tensors.
    Returns None when ``dim`` is out of range.

    Parameters
    ----------
    diagrams : List
        Persistence diagrams, indexed by homology dimension.
    dim : int
        Homology dimension to extract.

    Returns
    -------
    torch.Tensor or None
        The (n, 2) diagram tensor, or None if ``dim`` is out of range.
    """
    if dim < 0 or dim >= len(diagrams):
        return None
    entry = diagrams[dim]
    return entry.diagram if hasattr(entry, "diagram") else entry


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
        Persistence diagrams from a complex.
    PH_dims : List[int]
        Homology dimensions to consider.
    target_diagrams : Optional[List]
        Not used (kept for signature compatibility).
    device : str
        Computing device.

    Returns
    -------
    torch.Tensor
        Computed loss value.
    """
    pers_parts = []
    for dim in PH_dims:
        D = _get_diagram(diagrams, dim)
        if D is None or D.shape[0] == 0:
            continue
        pers_parts.append((D[:, 1] - D[:, 0]).reshape(-1))
    if not pers_parts:
        return torch.tensor(0., device=device)
    pers = torch.cat(pers_parts)
    return l1_l2_sq_ratio(pers)


def target_diagram_loss(diagrams: List, PH_dims: List[int],
                        target_diagrams: Optional[List] = None,
                        device: str = 'cpu', power: int = 2,
                        remove_longest: Optional[Dict[int, bool]] = None) -> torch.Tensor:
    """
    Element-wise comparison of sorted persistence values against a target diagram.

    Parameters
    ----------
    diagrams : List
        Persistence diagrams from a complex.
    PH_dims : List[int]
        Homology dimensions to consider.
    target_diagrams : Optional[List]
        Target persistence diagrams for each dimension.
    device : str
        Computing device.
    power : int
        Exponent applied to the persistence differences.
    remove_longest : Optional[Dict[int, bool]]
        Whether to remove the longest persistence for each dimension.

    Returns
    -------
    torch.Tensor
        Computed loss value.
    """
    if target_diagrams is None:
        raise ValueError("target_diagrams must be provided for target_diagram_loss")

    if remove_longest is None:
        remove_longest = {0: True, 1: False}

    loss = torch.tensor(0., device=device)

    for dim in PH_dims:
        D1 = _get_diagram(diagrams, dim)
        if D1 is None:
            continue
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
                              device: str = 'cpu', power: int = 2,
                              remove_longest: bool = True) -> torch.Tensor:
    """
    Weighted persistence loss: weights births by ``(1 - |deaths|)``.

    Parameters
    ----------
    diagrams : List
        Persistence diagrams from a complex.
    PH_dims : List[int]
        Homology dimensions to consider.
    target_diagrams : Optional[List]
        Not used (kept for signature compatibility).
    device : str
        Computing device.
    power : int
        Exponent applied to the weighted persistence.
    remove_longest : bool
        Whether to remove the longest persistence interval.

    Returns
    -------
    torch.Tensor
        Total weighted topological loss.
    """
    loss = torch.tensor(0., device=device)

    for dim in PH_dims:
        D = _get_diagram(diagrams, dim)
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
        loss += (weights * persistence).pow(power).sum()

    return loss


def weighted_total_squared_persistence_loss(
    diagrams: List,
    PH_dims: List[int],
    target_diagrams: Optional[List] = None,
    device: str = 'cpu',
    power: float = 2.0,
) -> torch.Tensor:
    """
    Weighted total squared persistence loss.

    Computes
    ``sum_{(b, d) in PH_k^fin} (1 - d)^power (d - b)^2``
    over the requested homology dimensions. The ``target_diagrams`` argument is
    ignored and kept for compatibility with :class:`TopologicalNMF`.

    Parameters
    ----------
    diagrams : List
        Persistence diagrams from a complex.
    PH_dims : List[int]
        Homology dimensions to consider.
    target_diagrams : Optional[List]
        Not used (kept for signature compatibility).
    device : str
        Computing device.
    power : float
        Death-value weight exponent. ``power=2`` gives :math:`WTP_2^{(k)}`.

    Returns
    -------
    torch.Tensor
        Total weighted squared persistence over finite intervals.
    """
    loss = torch.tensor(0., device=device)

    for dim in PH_dims:
        D = _get_diagram(diagrams, dim)
        if D is None or D.shape[0] == 0:
            continue

        finite_mask = torch.isfinite(D).all(dim=1)
        if not finite_mask.any():
            continue
        D = D[finite_mask]

        births = D[:, 0]
        deaths = D[:, 1]
        loss += ((1.0 - deaths).pow(power) * (deaths - births).pow(2)).sum()

    return loss


def clique_deviation_loss(diagrams: List, PH_dims: List[int],
                          target_diagrams: Optional[List] = None,
                          device: str = 'cpu', alpha: float = 1.0,
                          power: int = 2,
                          remove_longest: Optional[Dict[int, bool]] = None) -> torch.Tensor:
    """
    Clique deviation loss: negated target-diagram loss (for maximisation).

    Parameters
    ----------
    diagrams : List
        Persistence diagrams from a complex.
    PH_dims : List[int]
        Homology dimensions to consider.
    target_diagrams : List
        Target persistence diagrams.
    device : str
        Computing device.
    alpha : float
        Weight for higher dimensions (dim > 0).
    power : int
        Exponent forwarded to :func:`target_diagram_loss`.
    remove_longest : Dict[int, bool], optional
        Whether to remove the longest persistence for each dimension.

    Returns
    -------
    torch.Tensor
        Computed loss value (typically negative).
    """
    loss_PH = torch.tensor(0., device=device)
    if remove_longest is None:
        remove_longest = {}

    for dim in PH_dims:
        term = target_diagram_loss(diagrams, [dim], target_diagrams, device,
                                   power=power, remove_longest=remove_longest)
        loss_PH -= (alpha * term) if dim > 0 else term

    return loss_PH
