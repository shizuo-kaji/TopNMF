"""Persistence computation backends for topological data analysis."""

from typing import NamedTuple

import torch


class PersistenceInfo(NamedTuple):
    """
    Container for persistence diagram information.

    Attributes
    ----------
    diagram : torch.Tensor
        Persistence diagram of shape (n_pairs, 2) with birth-death pairs.
    pairing : torch.Tensor
        Coordinate pairings for creator/destroyer simplices.
    dimension : int
        Homology dimension (0, 1, 2, ...).
    """

    diagram: torch.Tensor
    pairing: torch.Tensor
    dimension: int


try:
    from .cubical import CubicalComplex
except ImportError:
    CubicalComplex = None

try:
    from .rips import GudhiVietorisRipsComplex, TimeDelayEmbeddingTorch
except ImportError:
    GudhiVietorisRipsComplex = None
    TimeDelayEmbeddingTorch = None

try:
    from .graph import GraphFiltrationPH
except ImportError:
    GraphFiltrationPH = None

__all__ = [
    "PersistenceInfo",
    "CubicalComplex",
    "GudhiVietorisRipsComplex",
    "TimeDelayEmbeddingTorch",
    "GraphFiltrationPH",
]
