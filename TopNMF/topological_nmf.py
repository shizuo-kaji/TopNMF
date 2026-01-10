"""
Topological NMF Optimizer

This module provides the main TopologicalNMF class for performing
Non-negative Matrix Factorization with topological constraints.
"""

import numpy as np
import torch
import torch.optim as optim
from torch_topological.nn import VietorisRipsComplex
from tqdm.auto import tqdm
from typing import Optional, Dict, List, Tuple, Callable, Union
from sklearn.decomposition._nmf import _initialize_nmf

from .nmf_utils import (
    update_V, sparsity_score, total_variation
)
from .topological_utils import (
    TimeDelayEmbeddingTorch, center_point_cloud_torch
)
from .losses import ph_sparsity_loss, target_diagram_loss


class TopologicalNMF:
    """
    Non-negative Matrix Factorization with topological constraints.

    This class implements NMF with optional sparsity constraints and
    topological regularization based on persistent homology.

    Parameters
    ----------
    n_components : int
        Number of components/basis vectors
    device : str, optional
        Computing device ('cpu' or 'cuda')
    random_state : int, optional
        Random seed for reproducibility
    """

    def __init__(self, n_components: int, device: str = 'cpu',
                 random_state: Optional[int] = None,
                 complex: Optional[object] = None,
                 ph_loss_fn: Optional[Callable] = None,
                 data_shape: Optional[Tuple] = None,
                 use_embedding: bool = True):
        """
        Initialize TopologicalNMF.

        Parameters
        ----------
        n_components : int
            Number of components/basis vectors
        device : str, optional
            Computing device ('cpu' or 'cuda')
        random_state : int, optional
            Random seed for reproducibility
        complex : object, optional
            Persistent homology complex (e.g., VietorisRipsComplex, CubicalComplex).
            If None, uses VietorisRipsComplex with time delay embedding.
        ph_loss_fn : Callable, optional
            Loss function for persistent homology. Should accept (diagrams, PH_dims, target_diagrams, device).
            If None, uses ph_sparsity_loss.
        data_shape : Tuple, optional
            Shape of each data sample for cubical complexes (e.g., (height, width) for images).
            Only needed when using CubicalComplex or similar.
        use_embedding : bool, optional
            Whether to use time delay embedding (for 1D time series).
            Set to False when using cubical complexes on images.
        """
        self.n_components = n_components
        self.device = device
        self.random_state = random_state
        self.complex = complex
        self.ph_loss_fn = ph_loss_fn if ph_loss_fn is not None else ph_sparsity_loss
        self.data_shape = data_shape
        self.use_embedding = use_embedding

        # Will be set during fit
        self.W = None
        self.V = None
        self.losses = {
            'PH': [], 'approx': [], 'sparse_W': [],
            'sparse_V': [], 'lr': []
        }

    def initialize_factors(self, X: np.ndarray, method: str = 'nndsvda'):
        """
        Initialize W and V matrices.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix
        method : str, optional
            Initialization method ('random', 'nndsvda', 'nndsvd', 'nndsvdar')
        """
        n_samples, n_features = X.shape

        if method == 'random':
            W = np.abs(np.random.normal(size=(n_samples, self.n_components)))
            V = np.abs(np.random.normal(size=(self.n_components, n_features)))
        else:
            W, V = _initialize_nmf(
                X, n_components=self.n_components,
                init=method, random_state=self.random_state
            )

        return W, V

    def fit(self, X: np.ndarray,
            n_iterations: int = 1000,
            lr: float = 0.005,
            lambda_apx: float = 1.0,
            lambda_spa_V: float = 0.0,
            lambda_spa_W: float = 0.0,
            lambda_top: float = 0.001,
            lambda_tv: float = 0.0,
            weight_decay: float = 0.0,
            target_sparsity: Optional[float] = None,
            target_diagrams: Optional[List] = None,
            target_periodicity: Optional[float] = None,
            gd_iter: int = 1,
            mu_iter: int = 0,
            W_iter: int = 0,
            M: int = 4,
            tau: Optional[int] = None,
            PH_dims: List[int] = [1],
            tol: float = 1e-4,
            tol_count: int = 50000,
            init_method: str = 'nndsvda',
            normalize: bool = False,
            normalize_V_max: bool = False,
            start_epoch_topological: int = 0,
            complex_inputs: Optional[Dict[str, object]] = None,
            verbose: bool = True) -> 'TopologicalNMF':
        """
        Fit the TopologicalNMF model to data.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix of shape (n_samples, n_features)
        n_iterations : int, optional
            Maximum number of iterations
        lr : float, optional
            Learning rate for gradient descent
        lambda_apx : float, optional
            Weight for approximation loss
        lambda_spa_V : float, optional
            Weight for V sparsity loss
        lambda_spa_W : float, optional
            Weight for W sparsity loss
        lambda_top : float, optional
            Weight for topological loss
        lambda_tv : float, optional
            Weight for total variation loss
        weight_decay : float, optional
            Weight decay for optimizer
        target_sparsity : float, optional
            Target sparsity level (0 to 1)
        target_diagrams : List, optional
            Target persistence diagrams for each dimension. Used with target_diagram_loss.
        target_periodicity : float, optional
            Target periodicity score (0 to 1). Used with periodicity loss.
        gd_iter : int, optional
            Number of gradient descent iterations per epoch
        mu_iter : int, optional
            Number of multiplicative update iterations per epoch
        W_iter : int, optional
            Number of W update iterations per multiplicative update
        M : int, optional
            Embedding dimension minus 1 for time delay (only used with use_embedding=True)
        tau : int, optional
            Time delay parameter (only used with use_embedding=True)
        PH_dims : List[int], optional
            Homology dimensions to consider
        tol : float, optional
            Tolerance for early stopping
        tol_count : int, optional
            Number of iterations below tolerance before stopping
        init_method : str, optional
            Initialization method
        normalize : bool, optional
            Whether to normalize W during training
        normalize_V_max : bool, optional
            Whether to normalize V by its row-wise maxima during training
        start_epoch_topological : int, optional
            Epoch to start applying topological loss (useful for warm start)
        complex_inputs : Optional[Dict[str, object]], optional
            Extra inputs required by custom complexes (e.g., graph edge lists)
        verbose : bool, optional
            Whether to show progress bar

        Returns
        -------
        self
            Fitted model
        """
        n_samples, n_features = X.shape
        epsilon = 1e-10

        # Initialize factors
        W, V = self.initialize_factors(X, method=init_method)

        # Convert to torch tensors
        X_t = torch.as_tensor(X, dtype=torch.float, device=self.device)
        self.W = torch.tensor(W, dtype=torch.float, device=self.device,
                              requires_grad=True)
        self.V = torch.tensor(V, dtype=torch.float, device=self.device,
                              requires_grad=True)
        if normalize_V_max:
            with torch.no_grad():
                normal_value = torch.max(self.V, dim=1).values.unsqueeze(1)
                self.V /= (normal_value + epsilon)

        # Set up time delay embedding and complex
        if self.use_embedding:
            if tau is None:
                tau = int(n_features / (2 * (M + 1)))
            embedder = TimeDelayEmbeddingTorch(dim=M+1, delay=tau)
        else:
            embedder = None

        # Use provided complex or default to VietorisRips
        if self.complex is not None:
            ph_complex = self.complex
        else:
            ph_complex = VietorisRipsComplex(dim=1, p=2)

        # Set up optimizer
        opt = optim.AdamW([self.W, self.V], lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.9, patience=10000
        )

        # Compute target L1 for sparsity
        if target_sparsity is not None and target_sparsity > 0:
            target_L1 = (np.sqrt(n_features) -
                        target_sparsity * (np.sqrt(n_features) - 1))
        else:
            target_L1 = 0

        # Loss function
        loss_fn = torch.nn.MSELoss()

        # Training loop
        progress = tqdm(range(n_iterations), disable=not verbose)
        prev_loss = np.inf
        count = 0

        for epoch in progress:
            # Multiplicative update
            with torch.no_grad():
                for _ in range(mu_iter):
                    # Update V
                    if target_sparsity is None:
                        W_TX = self.W.T @ X_t
                        W_TWV = self.W.T @ self.W @ self.V + epsilon
                        self.V *= W_TX / W_TWV
                    else:
                        update_V(X_t, self.W, self.V, target_L1, self.device)

                    # Update W
                    for _ in range(W_iter):
                        XV_T = X_t @ self.V.T
                        WVV_T = self.W @ self.V @ self.V.T + epsilon
                        self.W *= XV_T / WVV_T

            # Gradient descent update
            for _ in range(gd_iter):
                loss_PH = torch.tensor(0., device=self.device)
                loss_spa_V = torch.tensor(0., device=self.device)
                loss_spa_W = torch.tensor(0., device=self.device)
                sp_score = torch.tensor(0., device=self.device)
                loss_tv_V = torch.tensor(0., device=self.device)

                # Compute topological loss for each component
                for j in range(self.n_components - 1):
                    v = self.V[j]

                    # Topological loss
                    if lambda_top > 0 and epoch >= start_epoch_topological:
                        # Prepare input for complex
                        if self.use_embedding:
                            # For time series: use time delay embedding
                            point_cloud = embedder(v)
                            point_cloud = center_point_cloud_torch(point_cloud)
                            diags = ph_complex(point_cloud)
                        elif complex_inputs is not None and "all_edges" in complex_inputs:
                            diags = ph_complex(complex_inputs["all_edges"], v)
                        else:
                            # For images/grids: reshape and pass directly to complex
                            if self.data_shape is not None:
                                v_shaped = v.reshape(self.data_shape)
                            else:
                                v_shaped = v
                            diags = ph_complex(v_shaped)

                        # Compute loss using configured loss function
                        if target_diagrams is not None:
                            # Use target diagram loss
                            loss_PH += self.ph_loss_fn(diags, PH_dims, target_diagrams, self.device)

                        if target_periodicity is not None:
                            PH = torch.cat([diags[dim].diagram for dim in PH_dims])
                            pers1 = torch.diff(PH, dim=1).reshape(-1)
                            if len(pers1) > 0:
                                periodicity_score = max(pers1) / np.sqrt(3)
                            else:
                                periodicity_score = 0
                            loss_PH += (periodicity_score - target_periodicity[j])**2

                    # Total variation loss
                    loss_tv_V += total_variation(v)

                    # Sparsity loss
                    if target_sparsity is not None:
                        loss_spa_V += (sparsity_score(v) - target_sparsity)**2
                    else:
                        loss_spa_V += (v.abs().sum())**2 / (v**2).sum()

                    sp_score += sparsity_score(v)

                # W sparsity loss
                loss_spa_W = torch.sum(
                    torch.sum(torch.abs(self.W), dim=1)**2 /
                    torch.sum(self.W**2, dim=1)
                )

                # Normalize losses
                loss_spa_V /= self.n_components
                loss_spa_W /= self.n_components
                loss_PH /= self.n_components
                sp_score /= self.n_components

                # Approximation loss
                loss_apx = loss_fn(torch.mm(self.W, self.V), X_t)

                # Total loss
                loss = (lambda_top * loss_PH +
                       lambda_spa_V * loss_spa_V +
                       lambda_spa_W * loss_spa_W +
                       lambda_apx * loss_apx +
                       lambda_tv * loss_tv_V)

                # Optimization step
                opt.zero_grad()
                loss.backward()
                opt.step()
                scheduler.step(loss)

                # Record losses
                self.losses['PH'].append(loss_PH.item())
                self.losses['approx'].append(loss_apx.item())
                self.losses['sparse_V'].append(loss_spa_V.item())
                self.losses['sparse_W'].append(loss_spa_W.item())
                self.losses['lr'].append(scheduler.get_last_lr()[0])

            # Enforce non-negativity
            with torch.no_grad():
                self.V.clamp_(min=0)
                self.W.clamp_(min=0)
                if normalize:
                    self.W /= (torch.norm(self.W, p=1, dim=1, keepdim=True) +
                              epsilon)
                if normalize_V_max:
                    normal_value = torch.max(self.V, dim=1).values.unsqueeze(1)
                    self.V /= (normal_value + epsilon)

            # Progress display
            if verbose:
                progress.set_postfix(
                    loss=f'PH: {loss_PH.item():.6f}, '
                         f'sparsity: {sp_score.item():.6f}, '
                         f'approx: {loss_apx.item():.6f}'
                )

            # Early stopping
            current = (lambda_top * loss_PH + lambda_spa_V * loss_spa_V +
                      lambda_spa_W * loss_spa_W + loss_apx)
            if (prev_loss - current) < tol:
                count += 1
                if count > tol_count:
                    break
            else:
                prev_loss = current.item()
                count = 0

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted basis.

        Parameters
        ----------
        X : np.ndarray
            Input data

        Returns
        -------
        np.ndarray
            Transformed coefficients
        """
        if self.W is None or self.V is None:
            raise ValueError("Model must be fitted before transform")

        return self.W.detach().cpu().numpy()

    def inverse_transform(self, W: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reconstruct data from coefficients.

        Parameters
        ----------
        W : np.ndarray, optional
            Coefficient matrix. If None, uses fitted W.

        Returns
        -------
        np.ndarray
            Reconstructed data
        """
        if self.V is None:
            raise ValueError("Model must be fitted before inverse_transform")

        if W is None:
            W_t = self.W
        else:
            W_t = torch.as_tensor(W, dtype=torch.float, device=self.device)

        return (W_t @ self.V).detach().cpu().numpy()

    def get_components(self) -> np.ndarray:
        """
        Get learned basis vectors.

        Returns
        -------
        np.ndarray
            Basis matrix V
        """
        if self.V is None:
            raise ValueError("Model must be fitted")

        return self.V.detach().cpu().numpy()

    def get_losses(self) -> Dict[str, List[float]]:
        """
        Get training loss history.

        Returns
        -------
        Dict[str, List[float]]
            Dictionary of loss curves
        """
        return self.losses
