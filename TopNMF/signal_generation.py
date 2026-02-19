"""
Signal Generation Utilities

This module provides functions for generating synthetic time series signals
with various characteristics for testing and experimentation.
"""

import numpy as np
from itertools import combinations
from typing import List, Optional, Tuple, Union
from scipy import signal
import random

def generate_ichimatsu_pattern(
    num_samples: int = 100,
    image_shape=(36, 36),
    pat: Optional[np.ndarray] = None,
    pat_step: int = 3,
    min_pat: int = 10,
    max_pat: int = 30,
    binarize: bool = False,
    seed: Optional[int] = 42,
) -> np.ndarray:
    if seed is not None:
        random.seed(seed)

    if pat is None:
        pat = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=float)
    pat = np.asarray(pat, dtype=float)
    ih, iw = image_shape
    ph, pw = pat.shape

    if ph > ih or pw > iw:
        raise ValueError("Pattern shape must be smaller than or equal to image_shape")

    if min_pat < 0 or max_pat < min_pat:
        raise ValueError("Invalid min_pat/max_pat values")

    max_x_steps = max((ih - ph) // pat_step, 0)
    max_y_steps = max((iw - pw) // pat_step, 0)

    X = np.zeros((num_samples, ih, iw), dtype=np.float64)

    for i in range(num_samples):
        n_patches = random.randint(min_pat, max_pat)

        for _ in range(n_patches):
            if max_x_steps > 0:
                cx = pat_step * random.randint(0, max_x_steps)
            else:
                cx = random.randint(0, ih - ph)

            if max_y_steps > 0:
                cy = pat_step * random.randint(0, max_y_steps)
            else:
                cy = random.randint(0, iw - pw)

            X[i, cx:cx + ph, cy:cy + pw] += pat

            if binarize:
                X[i] = (X[i] > 0).astype(np.float64)

    return X


def generate_signals(
    t: np.ndarray,
    kind: str = "cosine",
    num: int = 2,
    noise: float = 0.0,
) -> List[np.ndarray]:
    """
    Generate either cosine-based or triangle-like signals with linear trends.

    Parameters
    ----------
    t : np.ndarray
        Time points array
    kind : str, optional
        Type of signals to generate. Options: "cosine", "triangle".
        Default is "cosine".
    num : int, optional
        Number of signals to generate. Default is 2.
    noise : float, optional
        Standard deviation of additive Gaussian noise. Default is 0.0.

    Returns
    -------
    List[np.ndarray]
        List of signal arrays
    """
    kind = kind.lower()

    if kind == "cosine":
        base_fn = lambda a, b: a * np.cos(2 * t) + b * t + 1
    elif kind == "triangle":
        base_fn = lambda a, b: a * signal.sawtooth(2 * t - np.pi, 0.5) + b * t + 1
    else:
        raise ValueError("kind must be either 'cosine' or 'triangle'")

    amplitudes = np.linspace(2.0, 1.0, num)
    slopes = np.linspace(1.0, 2.0, num)

    signals = []
    for a, b in zip(amplitudes, slopes):
        s = base_fn(a, b)
        if noise > 0:
            s = s + noise * np.random.randn(len(t))
        signals.append(s / np.max(np.abs(s)) if np.max(np.abs(s)) > 0 else s)

    return signals


def generate_edge_weighted_graph(
    cliques: Tuple[Tuple[int, ...], ...] = ((1, 2, 3, 4, 5), (5, 6, 7), (6, 7, 8, 9)),
    weights: Tuple[Tuple[float, ...], ...] = ((2, 1, 2), (1, 2, 1), (1, 1, 2)),
) -> Tuple[np.ndarray, list]:
    """
    Build edge-weight vectors from overlapping cliques with given mixing coefficients.

    Parameters
    ----------
    cliques : tuple of tuples of int
        Each inner tuple defines the nodes of one clique.
    weights : tuple of tuples of float
        Each inner tuple gives the mixing coefficients over *cliques*
        for one observation (row of the returned matrix).

    Returns
    -------
    X : np.ndarray, shape (len(weights), n_edges)
        Edge-weight matrix.
    edge_list : list of (int, int)
        Sorted list of all node pairs in the complete graph over the nodes.
    """
    all_nodes = sorted({i for clique in cliques for i in clique})
    edge_list = [tuple(sorted(e)) for e in combinations(all_nodes, 2)]
    edge_index = {e: i for i, e in enumerate(edge_list)}
    X = []

    for alpha in weights:
        row = np.zeros(len(edge_list))
        for coeff, clique in zip(alpha, cliques):
            for u, v in combinations(clique, 2):
                row[edge_index[tuple(sorted((u, v)))]] += coeff
        X.append(row)

    return np.stack(X), edge_list


def normalize_signals(
    signals: Union[List[np.ndarray], np.ndarray],
) -> Union[List[np.ndarray], np.ndarray]:
    """
    Min-max normalize signals to [0, 1].

    Parameters
    ----------
    signals : list of np.ndarray or np.ndarray
        A list of 1-D arrays, or a single array (1-D or 2-D).
        For a 2-D array each row is normalized independently.

    Returns
    -------
    list of np.ndarray or np.ndarray
        Normalized signals in the same format as the input.
    """
    if isinstance(signals, np.ndarray):
        if signals.ndim == 1:
            mn, mx = signals.min(), signals.max()
            return (signals - mn) / (mx - mn) if mx != mn else np.zeros_like(signals)
        # 2-D: normalize each row
        mn = signals.min(axis=1, keepdims=True)
        mx = signals.max(axis=1, keepdims=True)
        denom = mx - mn
        denom[denom == 0] = 1.0
        return (signals - mn) / denom

    return [normalize_signals(s) for s in signals]
