"""
NMF Optimization Utilities

This module provides utility functions for Non-negative Matrix Factorization (NMF)
with sparsity constraints and optimization methods.
"""

import numpy as np
import torch
from typing import Optional


def update_V(X: torch.Tensor, W: torch.Tensor, V: torch.Tensor,
             target_L1: float, device: str = 'cpu') -> torch.Tensor:
    """
    Update matrix V using sparse optimization.

    Parameters
    ----------
    X : torch.Tensor
        Input data matrix of shape (n_samples, n_features)
    W : torch.Tensor
        Weight matrix of shape (n_samples, n_components)
    V : torch.Tensor
        Basis matrix of shape (n_components, n_features)
    target_L1 : float
        Target L1 norm for sparsity constraint
    device : str, optional
        Device for computation ('cpu' or 'cuda')

    Returns
    -------
    torch.Tensor
        Updated V matrix
    """
    WtW = W.T @ W
    cache = -W.T @ X + WtW @ V

    for i in range(V.shape[0]):
        C = cache[i] - WtW[i, i] * V[i]
        vi = torch.as_tensor(
            sparse_opt(-C.to('cpu').detach().numpy(), target_L1),
            device=device
        )
        cache = cache + torch.outer(WtW[:, i], vi - V[i])
        V[i] = vi

    return V


def sparse_opt(b: np.ndarray, k: float, epsilon: float = 1e-10) -> np.ndarray:
    """
    Sparse optimization solver for L1/L2 constrained problem.

    This function solves an optimization problem with sparsity constraints
    using a permutation-based algorithm.

    Parameters
    ----------
    b : np.ndarray
        Input vector to optimize
    k : float
        Sparsity parameter (target L1 norm)
    epsilon : float, optional
        Numerical stability constant

    Returns
    -------
    np.ndarray
        Optimized sparse vector
    """
    if k <= 0:
        return b

    permutation = np.argsort(b)[::-1]
    a = b[permutation]
    m = len(b)
    x = np.zeros(m, dtype=np.float64)
    y = np.zeros_like(x)

    bot = np.int64(np.ceil(k * k))
    if bot > m or k == 1:
        y[permutation[0]] = 1
        return y

    # Compute squared values and cumulative sums
    a_squared = a**2
    cumsum_a_squared = np.cumsum(a_squared)
    cumsum_a = np.cumsum(a)

    P = np.arange(1, m + 1, dtype=np.float64)

    # Calculate lambda and mu values
    numerator = P * cumsum_a_squared - cumsum_a**2
    denominator = P - k**2
    valid_mask = (denominator > 0) & (numerator > 0)

    lambda_values = np.zeros_like(P) + epsilon
    mu_values = np.zeros_like(P) + epsilon
    lambda_values[valid_mask] = -np.sqrt(numerator[valid_mask] / denominator[valid_mask])
    mu_values[valid_mask] = (-cumsum_a[valid_mask] / P[valid_mask] -
                              k * lambda_values[valid_mask] / P[valid_mask])

    # Find optimal p*
    p_candidates = np.arange(np.int64(np.ceil(k * k)), m + 1)
    a_p_candidates = a[p_candidates - 1]
    mu_p_candidates = mu_values[p_candidates - 1]
    valid_p = np.where(a_p_candidates < -mu_p_candidates)[0]

    pstar = m if len(valid_p) == 0 else p_candidates[valid_p[0]] - 1
    lam_pstar = lambda_values[pstar - 1]
    mu_pstar = mu_values[pstar - 1]

    # Compute solution
    x[:pstar] = -(a[:pstar] + mu_pstar) / lam_pstar
    y[permutation] = x

    return y


def sparse_opt_hoyer(x: np.ndarray, L1: float, L2: float = 1,
                     max_iter: int = 100) -> np.ndarray:
    """
    Hoyer's projection algorithm for sparse non-negative optimization.

    Projects a vector onto the non-negative simplex with L1 and L2 constraints.

    Parameters
    ----------
    x : np.ndarray
        Input vector
    L1 : float
        Target L1 norm
    L2 : float, optional
        Target L2 norm
    max_iter : int, optional
        Maximum number of iterations

    Returns
    -------
    np.ndarray
        Projected sparse non-negative vector
    """
    dim = len(x)
    s = x + (L1 - np.sum(x)) / dim
    Z = set()

    for _ in range(max_iter):
        # Compute m_i
        m = np.zeros(dim)
        for i in range(dim):
            if i not in Z:
                m[i] = L1 / (dim - len(Z))

        # Compute s with quadratic constraint
        sm = s - m
        a = np.sum(sm ** 2)
        b = 2 * np.dot(m, sm)
        c = np.sum(m ** 2) - L2 ** 2

        alpha = 0
        if a != 0:
            alpha = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        s = m + alpha * sm

        # Check non-negativity
        if np.all(s >= 0):
            return s

        # Update zero set
        for i in range(dim):
            if s[i] < 0:
                Z.add(i)
                s[i] = 0

        # Update non-zero elements
        c = (np.sum(s) - L1) / (dim - len(Z))
        for i in range(dim):
            if i not in Z:
                s[i] -= c

    return np.maximum(s, 0)


def sparsity_score(v: torch.Tensor) -> float:
    """
    Compute Hoyer sparsity score for a vector.

    The score ranges from 0 (dense) to 1 (sparse).

    Parameters
    ----------
    v : torch.Tensor
        Input vector

    Returns
    -------
    float
        Sparsity score between 0 and 1
    """
    n = len(v.ravel())
    l1_norm = v.abs().sum()
    l2_norm = (v**2).sum().sqrt()

    numerator = np.sqrt(n) - l1_norm / l2_norm
    denominator = np.sqrt(n) - 1

    return float(numerator / denominator)


def svd_initialization(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize W and V matrices for NMF using SVD decomposition.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_components : int
        Number of components (rank of factorization)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        W : Basis matrix of shape (n_samples, n_components)
        V : Coefficient matrix of shape (n_components, n_features)
    """
    U, S, VT = np.linalg.svd(X, full_matrices=False)

    # Select top n_components
    U = U[:, :n_components]
    S = np.diag(S[:n_components])
    VT = VT[:n_components, :]

    # Ensure non-negativity
    W = np.abs(U @ np.sqrt(S))
    V = np.abs(np.sqrt(S) @ VT)

    return W, V


def total_variation(v: torch.Tensor) -> torch.Tensor:
    """
    Compute total variation (TV) of a vector.

    TV measures the sum of squared differences between consecutive elements.

    Parameters
    ----------
    v : torch.Tensor
        Input vector

    Returns
    -------
    torch.Tensor
        Total variation value
    """
    return ((v[1:] - v[:-1])**2).sum()
