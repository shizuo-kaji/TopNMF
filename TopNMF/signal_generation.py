"""
Signal Generation Utilities

This module provides functions for generating synthetic time series signals
with various characteristics for testing and experimentation.
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Dict, Optional
from scipy import signal
import random

def create_ichimatsu_pattern(
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


def generate_signals(t: np.ndarray, kind: str = "cosine") -> Dict[str, np.ndarray]:
    """
    Generate either cosine-based or triangle-like signals with linear trends.

    Parameters
    ----------
    t : np.ndarray
        Time points array
    kind : str, optional
        Type of signals to generate. Options: "cosine", "triangle".
        Default is "cosine".

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with signal names as keys and signal arrays as values
    """
    
    kind = kind.lower()
    
    if kind == "cosine":
        return {
            "cosine 1": np.cos(2 * t) + t + 1,
            "cosine 2": 0.5 * np.cos(2 * t) + 2 * t + 0.5,
        }

    elif kind == "triangle":
        return {
            "triangle 1": signal.sawtooth(2 * t - np.pi, 0.5) + t + 1,
            "triangle 2": 0.5 * signal.sawtooth(2 * t - np.pi, 0.5) + 2 * t + 0.5,
        }

    else:
        raise ValueError("kind must be either 'cosine' or 'triangle'")



def generate_mixed_periodic_nonperiodic(t: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generate signals mixing periodic and non-periodic components.

    Parameters
    ----------
    t : np.ndarray
        Time points array

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with signal names as keys and signal arrays as values
    """
    return {
        "linear_cosine": t / (4 * np.pi) + np.cos(t),
        "quadratic_cosine": t**2 / (4 * np.pi)**2 + np.cos(t),
        "gaussian_cosine": t / (4 * np.pi) + np.exp(-t**2),
    }


def generate_noisy_periodic(t: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generate noisy periodic signals with Gaussian components.

    Parameters
    ----------
    t : np.ndarray
        Time points array

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with signal names as keys and signal arrays as values
    """
    return {
        "cosine_gaussian_1": np.cos(1 * t) + 2 * np.exp(-t**2),
        "cosine_gaussian_2": np.cos(5 * t) + 2 * np.exp(-t**2),
        "cosine_gaussian_3": np.cos(3 * t) + np.exp(-t**2),
        "cosine_gaussian_4": np.cos(4 * t) + np.exp(-t**2),
    }


def generate_complex_signals(t: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generate complex mixed signals with various components.

    Parameters
    ----------
    t : np.ndarray
        Time points array

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with signal names as keys and signal arrays as values
    """
    signals = {
        "ramp_cosine": t / (4 * np.pi) + np.cos(t),
        "quadratic_cosine": (t**2) / (4 * np.pi)**2 + np.cos(t),
        "gaussian_bump": np.exp(-((t - 2 * np.pi)**2)) + np.cos(t),
        "chirp": 0.5 * np.cos(t) + np.sin(t**1.5),
        "step_cosine": (t > 2 * np.pi).astype(float) + np.cos(t),
        "sawtooth_cosine": (t % (2 * np.pi)) / (2 * np.pi) + np.cos(t),
    }
    return signals


def generate_noisy_signals(t: np.ndarray, n_samples: int = 5,
                            noise_scale: float = 0.1,
                            seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Generate multiple noisy signal realizations.

    Parameters
    ----------
    t : np.ndarray
        Time points array
    n_samples : int, optional
        Number of signal samples to generate
    noise_scale : float, optional
        Standard deviation of Gaussian noise
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with signal names as keys and signal arrays as values
    """
    signals = {}
    for i in range(n_samples):
        if seed is not None:
            np.random.seed(seed + i)
        periodic = np.cos(t)
        non_periodic = t / (4 * np.pi)
        noise = noise_scale * np.random.randn(len(t))
        mixed = periodic + non_periodic + noise
        signals[f"signal_seed_{i}"] = mixed
    return signals


def generate_step_signals(t: np.ndarray, n_samples: int = 5,
                           noise_scale: float = 0.01,
                           seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Generate signals with step discontinuities and periodic components.

    Parameters
    ----------
    t : np.ndarray
        Time points array
    n_samples : int, optional
        Number of signal samples to generate
    noise_scale : float, optional
        Standard deviation of Gaussian noise
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with signal names as keys and signal arrays as values
    """
    signals = {}
    for i in range(n_samples):
        if seed is not None:
            np.random.seed(seed + i)
        periodic = np.cos(t)
        non_periodic = (t > 2 * np.pi).astype(float)
        noise = noise_scale * np.random.randn(len(t))
        mixed = periodic + non_periodic + noise
        signals[f"signal_seed_{i}"] = mixed
    return signals


def normalize_signals(signals: Dict[str, np.ndarray],
                      method: str = 'max') -> Dict[str, np.ndarray]:
    """
    Normalize signals to a common scale.

    Parameters
    ----------
    signals : Dict[str, np.ndarray]
        Dictionary of signals to normalize
    method : str, optional
        Normalization method: 'max' (divide by max), 'std' (standardize),
        'minmax' (scale to [0,1])

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of normalized signals
    """
    normalized = {}

    for name, sig in signals.items():
        if method == 'max':
            normalized[name] = sig / np.max(np.abs(sig))
        elif method == 'std':
            normalized[name] = (sig - np.mean(sig)) / np.std(sig)
        elif method == 'minmax':
            min_val, max_val = np.min(sig), np.max(sig)
            normalized[name] = (sig - min_val) / (max_val - min_val)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    return normalized


def create_time_array(start: float = 0, stop: float = 2*np.pi,
                      n_points: int = 100) -> np.ndarray:
    """
    Create a linearly spaced time array.

    Parameters
    ----------
    start : float, optional
        Start time
    stop : float, optional
        End time
    n_points : int, optional
        Number of time points

    Returns
    -------
    np.ndarray
        Time array
    """
    return np.linspace(start, stop, n_points)
