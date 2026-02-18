"""NMF optimization utilities: sparse projections and multiplicative updates."""

import numpy as np
import torch


def update_V(X: torch.Tensor, W: torch.Tensor, V: torch.Tensor,
             target_L1: float, device: str = 'cpu') -> torch.Tensor:
    """
    Update matrix V using sparse optimization with cached Gram matrix.

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
    device : str
        Device for computation

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
            device=device,
        )
        cache = cache + torch.outer(WtW[:, i], vi - V[i])
        V[i] = vi

    return V


def sparse_opt(b: np.ndarray, k: float, epsilon: float = 1e-10) -> np.ndarray:
    """
    L1/L2-constrained non-negative projection.

    Solves: min ||x - b||^2  s.t. ||x||_1 = k, x >= 0.

    Parameters
    ----------
    b : np.ndarray
        Input vector
    k : float
        Sparsity parameter (target L1 norm)
    epsilon : float
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

    a_squared = a ** 2
    cumsum_a_squared = np.cumsum(a_squared)
    cumsum_a = np.cumsum(a)

    P = np.arange(1, m + 1, dtype=np.float64)

    numerator = P * cumsum_a_squared - cumsum_a ** 2
    denominator = P - k ** 2
    valid_mask = (denominator > 0) & (numerator > 0)

    lambda_values = np.zeros_like(P) + epsilon
    mu_values = np.zeros_like(P) + epsilon
    lambda_values[valid_mask] = -np.sqrt(numerator[valid_mask] / denominator[valid_mask])
    mu_values[valid_mask] = (
        -cumsum_a[valid_mask] / P[valid_mask]
        - k * lambda_values[valid_mask] / P[valid_mask]
    )

    p_candidates = np.arange(np.int64(np.ceil(k * k)), m + 1)
    a_p_candidates = a[p_candidates - 1]
    mu_p_candidates = mu_values[p_candidates - 1]
    valid_p = np.where(a_p_candidates < -mu_p_candidates)[0]

    pstar = m if len(valid_p) == 0 else p_candidates[valid_p[0]] - 1
    lam_pstar = lambda_values[pstar - 1]
    mu_pstar = mu_values[pstar - 1]

    x[:pstar] = -(a[:pstar] + mu_pstar) / lam_pstar
    y[permutation] = x

    return y


def sparse_opt_hoyer(x: np.ndarray, L1: float, L2: float = 1,
                     max_iter: int = 100) -> np.ndarray:
    """
    Hoyer's projection onto the non-negative simplex with L1/L2 constraints.

    Parameters
    ----------
    x : np.ndarray
        Input vector
    L1 : float
        Target L1 norm
    L2 : float
        Target L2 norm
    max_iter : int
        Maximum iterations

    Returns
    -------
    np.ndarray
        Projected sparse non-negative vector
    """
    dim = len(x)
    s = x + (L1 - np.sum(x)) / dim
    Z = set()

    for _ in range(max_iter):
        m = np.zeros(dim)
        for i in range(dim):
            if i not in Z:
                m[i] = L1 / (dim - len(Z))

        sm = s - m
        a = np.sum(sm ** 2)
        b_val = 2 * np.dot(m, sm)
        c = np.sum(m ** 2) - L2 ** 2

        alpha = 0
        if a != 0:
            alpha = (-b_val + np.sqrt(b_val ** 2 - 4 * a * c)) / (2 * a)
        s = m + alpha * sm

        if np.all(s >= 0):
            return s

        for i in range(dim):
            if s[i] < 0:
                Z.add(i)
                s[i] = 0

        c_adj = (np.sum(s) - L1) / (dim - len(Z))
        for i in range(dim):
            if i not in Z:
                s[i] -= c_adj

    return np.maximum(s, 0)
