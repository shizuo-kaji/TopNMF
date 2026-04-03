"""TopologicalNMF: NMF with topological constraints via persistent homology."""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from sklearn.decomposition._nmf import _initialize_nmf
from tqdm.auto import tqdm

from .losses import ph_sparsity_loss, total_variation
from .optim import update_V
from .utils import sparsity_score, center_point_cloud_torch
from .persistence import TimeDelayEmbeddingTorch, GudhiVietorisRipsComplex


class TopologicalNMF:
    """
    Non-negative Matrix Factorization with topological constraints.

    Parameters
    ----------
    n_components : int
        Number of basis vectors.
    device : str
        Computing device ('cpu' or 'cuda').
    random_state : int, optional
        Random seed for reproducibility.
    complex : object, optional
        Persistence complex (GudhiVietorisRipsComplex, CubicalComplex, GraphFiltrationPH).
        Defaults to GudhiVietorisRipsComplex with time-delay embedding.
    ph_loss_fn : callable, optional
        PH loss function with signature ``(diagrams, PH_dims, target_diagrams, device, **kwargs)``.
        Defaults to ph_sparsity_loss.
    ph_loss_params : dict, optional
        Extra keyword arguments forwarded to *ph_loss_fn*.
    data_shape : tuple, optional
        Shape of each sample for cubical complexes (e.g. ``(H, W)`` for images).
    use_embedding : bool
        Whether to use time-delay embedding (for 1-D time series).
    """

    def __init__(
        self,
        n_components: int,
        device: str = "cpu",
        random_state: Optional[int] = None,
        complex: Optional[object] = None,
        ph_loss_fn: Optional[Callable] = None,
        ph_loss_params: Optional[Dict] = None,
        data_shape: Optional[Tuple] = None,
        use_embedding: bool = False,
    ):
        self.n_components = n_components
        self.device = device
        self.random_state = random_state
        self.complex = complex
        self.ph_loss_fn = ph_loss_fn if ph_loss_fn is not None else ph_sparsity_loss
        self.ph_loss_params = ph_loss_params if ph_loss_params is not None else {}
        self.data_shape = data_shape
        self.use_embedding = use_embedding

        self.W = None
        self.V = None
        self.losses = self._empty_losses()

    @staticmethod
    def _empty_losses() -> Dict[str, List[float]]:
        return {"PH": [], "approx": [], "sparse_W": [], "sparse_V": [], "lr": []}

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_factors(self, X: np.ndarray, method: str = "nndsvda"):
        """
        Initialize W and V matrices.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix
        method : str
            Initialization method ('random', 'nndsvda', 'nndsvd', 'nndsvdar')
        """
        n_samples, n_features = X.shape
        if method == "random":
            np.random.seed(self.random_state)
            W = np.random.rand(n_samples, self.n_components)
            V = np.random.rand(self.n_components, n_features)
        else:
            W, V = _initialize_nmf(
                X, n_components=self.n_components,
                init=method, random_state=self.random_state,
            )
        return W, V

    def _initialize_model_tensors(self, X, init_method, normalize_v_max, epsilon):
        W, V = self.initialize_factors(X, method=init_method)
        X_t = torch.as_tensor(X, dtype=torch.float, device=self.device)
        self.W = torch.tensor(W, dtype=torch.float, device=self.device, requires_grad=True)
        self.V = torch.tensor(V, dtype=torch.float, device=self.device, requires_grad=True)
        if normalize_v_max:
            with torch.no_grad():
                self._normalize_v_rows_max(epsilon)
        return X_t

    # ------------------------------------------------------------------
    # Optimizer / scheduler helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_optimizer(optimizer_cls, parameters, lr, weight_decay, optimizer_kwargs):
        kw = dict(optimizer_kwargs or {})
        kw.setdefault("lr", lr)
        kw.setdefault("weight_decay", weight_decay)
        return optimizer_cls(parameters, **kw)

    @staticmethod
    def _build_scheduler(optimizer, scheduler_cls, scheduler_kwargs):
        if scheduler_cls is None:
            return None
        kw = dict(scheduler_kwargs or {})
        if scheduler_cls == optim.lr_scheduler.ReduceLROnPlateau:
            kw.setdefault("factor", 0.9)
            kw.setdefault("patience", 10000)
        return scheduler_cls(optimizer, **kw)

    @staticmethod
    def _compute_target_l1(n_features, target_sparsity):
        if target_sparsity is not None and target_sparsity > 0:
            return np.sqrt(n_features) - target_sparsity * (np.sqrt(n_features) - 1)
        return 0.0

    @staticmethod
    def _step_scheduler(scheduler, loss):
        if scheduler is None:
            return
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss.detach())
        else:
            scheduler.step()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _resolve_embedder(self, n_features, embedding_dim, tau, n_periods=None):
        if not self.use_embedding:
            return None, tau
        if tau is None:
            if n_periods is None:
                n_periods = int(np.round(embedding_dim / 2))
            resolved_tau = int(n_features / (n_periods * (embedding_dim + 1)))
        else:
            resolved_tau = tau
        return TimeDelayEmbeddingTorch(dim=embedding_dim + 1, delay=resolved_tau), resolved_tau

    def _resolve_complex(self):
        if self.complex is not None:
            return self.complex
        return GudhiVietorisRipsComplex(dim=1, p=2)

    def _compute_component_diagrams(self, component, embedder, ph_complex,
                                    complex_inputs):
        if self.use_embedding:
            if embedder is None:
                raise ValueError("Embedder required when use_embedding is True.")
            point_cloud = center_point_cloud_torch(embedder(component))
            return ph_complex(point_cloud)

        if complex_inputs is not None and "all_edges" in complex_inputs:
            return ph_complex(complex_inputs["all_edges"], component)

        if self.data_shape is not None:
            component_input = component.reshape(self.data_shape)
        elif component.ndim == 1:
            component_input = component.unsqueeze(1)
        else:
            component_input = component
        return ph_complex(component_input)

    @staticmethod
    def _resolve_periodicity_target(target_periodicity, component_idx):
        if target_periodicity is None:
            return None
        if hasattr(target_periodicity, "__getitem__") and not isinstance(target_periodicity, str):
            try:
                return float(target_periodicity[component_idx])
            except (IndexError, TypeError, ValueError):
                return None
        return float(target_periodicity)

    @staticmethod
    def _compute_periodicity_loss(diagrams, PH_dims, target_value, device):
        pd = torch.cat([
            diagrams[dim].diagram if hasattr(diagrams[dim], "diagram") else diagrams[dim]
            for dim in PH_dims
        ])
        persistence = torch.diff(pd, dim=1).reshape(-1)
        if len(persistence) > 0:
            score = torch.max(persistence) / np.sqrt(3)
        else:
            score = torch.tensor(0.0, device=device)
        return (score - target_value) ** 2

    # ------------------------------------------------------------------
    # Per-epoch computation
    # ------------------------------------------------------------------

    def _run_multiplicative_updates(self, X_t, target_sparsity, target_l1,
                                    mu_iter, W_iter, epsilon):
        if mu_iter <= 0:
            return
        with torch.no_grad():
            for _ in range(mu_iter):
                if target_sparsity is None:
                    W_TX = self.W.T @ X_t
                    W_TWV = self.W.T @ self.W @ self.V + epsilon
                    self.V *= W_TX / W_TWV
                else:
                    update_V(X_t, self.W, self.V, target_l1, self.device)

                for _ in range(W_iter):
                    XV_T = X_t @ self.V.T
                    WVV_T = self.W @ self.V @ self.V.T + epsilon
                    self.W *= XV_T / WVV_T

    def _compute_component_losses(self, epoch, lambda_top, start_epoch_topological,
                                  ph_complex, embedder, complex_inputs, PH_dims,
                                  target_diagrams, target_periodicity, target_sparsity):
        loss_ph = torch.tensor(0.0, device=self.device)
        loss_spa_v = torch.tensor(0.0, device=self.device)
        sp_score = torch.tensor(0.0, device=self.device)
        loss_tv_v = torch.tensor(0.0, device=self.device)

        apply_topology = lambda_top > 0 and epoch >= start_epoch_topological
        for idx in range(self.n_components):
            component = self.V[idx]

            if apply_topology:
                diagrams = self._compute_component_diagrams(
                    component, embedder, ph_complex, complex_inputs)
                if target_diagrams is not None:
                    loss_ph += self.ph_loss_fn(
                        diagrams, PH_dims, target_diagrams, self.device,
                        **self.ph_loss_params)
                target_val = self._resolve_periodicity_target(target_periodicity, idx)
                if target_val is not None:
                    loss_ph += self._compute_periodicity_loss(
                        diagrams, PH_dims, target_val, self.device)

            loss_tv_v += total_variation(component)
            if target_sparsity is not None:
                loss_spa_v += (sparsity_score(component) - target_sparsity) ** 2
            else:
                loss_spa_v += component.abs().sum() ** 2 / ((component ** 2).sum() + 1e-10)

            sp_score += sparsity_score(component)

        norm = float(self.n_components)
        return loss_ph / norm, loss_spa_v / norm, sp_score / norm, loss_tv_v

    def _compute_w_sparsity_loss(self):
        loss = torch.sum(
            torch.sum(torch.abs(self.W), dim=1) ** 2 / (torch.sum(self.W ** 2, dim=1) + 1e-10)
)
        return loss / float(self.n_components)

    def _record_losses(self, loss_ph, loss_apx, loss_spa_v, loss_spa_w, optimizer):
        self.losses["PH"].append(loss_ph.item())
        self.losses["approx"].append(loss_apx.item())
        self.losses["sparse_V"].append(loss_spa_v.item())
        self.losses["sparse_W"].append(loss_spa_w.item())
        self.losses["lr"].append(optimizer.param_groups[0]["lr"])

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def _normalize_v_rows_max(self, epsilon):
        normal_value = torch.max(self.V, dim=1).values.unsqueeze(1)
        self.V /= normal_value + epsilon

    def _apply_constraints(self, normalize, normalize_v_max, epsilon):
        with torch.no_grad():
            self.V.clamp_(min=0)
            self.W.clamp_(min=0)
            if normalize:
                self.W /= torch.norm(self.W, p=1, dim=1, keepdim=True) + epsilon
            if normalize_v_max:
                self._normalize_v_rows_max(epsilon)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
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
        embedding_dim: int = 4,
        tau: Optional[int] = None,
        n_periods: Optional[int] = None,
        PH_dims: List[int] = [1],
        tol: float = 1e-4,
        tol_count: int = 50000,
        init_method: str = "nndsvda",
        normalize: bool = False,
        normalize_V_max: bool = False,
        start_epoch_topological: int = 0,
        complex_inputs: Optional[Dict[str, object]] = None,
        optimizer_cls: Callable = optim.AdamW,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_cls: Optional[Callable] = optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_kwargs: Optional[Dict] = None,
        verbose: bool = True,
        monitor: Optional[object] = None,
    ) -> "TopologicalNMF":
        """
        Fit the model to data.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features).
        n_iterations : int
            Maximum training epochs.
        lr : float
            Learning rate.
        lambda_apx, lambda_spa_V, lambda_spa_W, lambda_top, lambda_tv : float
            Loss-term weights.
        weight_decay : float
            Optimizer weight decay.
        target_sparsity : float, optional
            Target Hoyer sparsity for V.
        target_diagrams : list, optional
            Target persistence diagrams per dimension.
        target_periodicity : float, optional
            Target periodicity score.
        gd_iter : int
            Gradient-descent steps per epoch.
        mu_iter : int
            Multiplicative-update steps per epoch.
        W_iter : int
            W-update sub-steps per multiplicative update.
        embedding_dim : int
            Time-delay embedding dimension minus 1.
        tau : int, optional
            Time delay (auto-computed if None and use_embedding is True).
        n_periods : int, optional
            Number of expected periods (used only if tau is None).
        PH_dims : list of int
            Homology dimensions to optimise.
        tol, tol_count : float, int
            Early-stopping tolerance and patience.
        init_method : str
            NMF initialisation ('random', 'nndsvda', 'nndsvd', 'nndsvdar').
        normalize : bool
            L1-normalise W rows each epoch.
        normalize_V_max : bool
            Normalise V rows by their max each epoch.
        start_epoch_topological : int
            Epoch at which topological loss activates.
        complex_inputs : dict, optional
            Extra inputs for custom complexes (e.g. graph edge lists).
        optimizer_cls, optimizer_kwargs : callable, dict
            Optimizer class and extra kwargs.
        scheduler_cls, scheduler_kwargs : callable, dict
            LR scheduler class and extra kwargs (None to disable).
        verbose : bool
            Show tqdm progress bar.
        monitor : object, optional
            Live visualisation monitor (see ``TopNMF.visualization.FitMonitor``).

        Returns
        -------
        self
        """
        _, n_features = X.shape
        epsilon = 1e-10

        X_t = self._initialize_model_tensors(X, init_method, normalize_V_max, epsilon)
        embedder, tau = self._resolve_embedder(n_features, embedding_dim, tau, n_periods)
        ph_complex = self._resolve_complex()

        optimizer = self._build_optimizer(
            optimizer_cls, [self.W, self.V], lr, weight_decay, optimizer_kwargs)
        scheduler = self._build_scheduler(optimizer, scheduler_cls, scheduler_kwargs)
        target_l1 = self._compute_target_l1(n_features, target_sparsity)
        loss_fn = torch.nn.MSELoss()

        if monitor is not None:
            monitor.setup(
                n_features=n_features, embedding_dim=embedding_dim, tau=tau,
                complex_inputs=complex_inputs,
                data_shape=self.data_shape,
                use_embedding=self.use_embedding,
            )

        progress = tqdm(range(n_iterations), disable=not verbose)
        prev_loss = np.inf
        count = 0

        for epoch in progress:
            self._run_multiplicative_updates(
                X_t, target_sparsity, target_l1, mu_iter, W_iter, epsilon)

            # Defaults keep progress/early-stop logic valid when gd_iter=0.
            loss_ph = torch.tensor(0.0, device=self.device)
            loss_spa_v = torch.tensor(0.0, device=self.device)
            loss_spa_w = torch.tensor(0.0, device=self.device)
            sp_score = torch.tensor(0.0, device=self.device)
            loss_tv_v = torch.tensor(0.0, device=self.device)
            loss_apx = loss_fn(torch.mm(self.W, self.V), X_t)

            for _ in range(gd_iter):
                loss_ph, loss_spa_v, sp_score, loss_tv_v = (
                    self._compute_component_losses(
                        epoch, lambda_top, start_epoch_topological,
                        ph_complex, embedder, complex_inputs,
                        PH_dims, target_diagrams, target_periodicity,
                        target_sparsity))
                loss_spa_w = self._compute_w_sparsity_loss()
                loss_apx = loss_fn(torch.mm(self.W, self.V), X_t)

                loss = (
                    lambda_top * loss_ph
                    + lambda_spa_V * loss_spa_v
                    + lambda_spa_W * loss_spa_w
                    + lambda_apx * loss_apx
                    + lambda_tv * loss_tv_v
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self._step_scheduler(scheduler, loss)
                self._record_losses(loss_ph, loss_apx, loss_spa_v, loss_spa_w,
                                    optimizer)

            self._apply_constraints(normalize, normalize_V_max, epsilon)

            if verbose:
                progress.set_postfix(
                    loss=f"PH: {loss_ph.item():.6f}, "
                         f"sparsity: {sp_score.item():.6f}, "
                         f"approx: {loss_apx.item():.6f}")

            if monitor is not None:
                monitor.update(epoch, self)

            current = (
                lambda_top * loss_ph
                + lambda_spa_V * loss_spa_v
                + lambda_spa_W * loss_spa_w
                + loss_apx
            )
            if (prev_loss - current) < tol:
                count += 1
                if count > tol_count:
                    break
            else:
                prev_loss = current.item()
                count = 0

        self.W = self.W.detach()
        self.V = self.V.detach()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Get coefficient matrix W (X is ignored; uses fitted W)."""
        if self.W is None or self.V is None:
            raise ValueError("Model must be fitted before transform")
        return self.W.detach().cpu().numpy()

    def inverse_transform(self, W: Optional[np.ndarray] = None) -> np.ndarray:
        """Reconstruct data from coefficients."""
        if self.V is None:
            raise ValueError("Model must be fitted before inverse_transform")
        if W is None:
            W_t = self.W
        else:
            W_t = torch.as_tensor(W, dtype=torch.float, device=self.device)
        return (W_t @ self.V).detach().cpu().numpy()

    def get_components(self) -> np.ndarray:
        """Get learned basis vectors V."""
        if self.V is None:
            raise ValueError("Model must be fitted")
        return self.V.detach().cpu().numpy()

    def get_losses(self) -> Dict[str, List[float]]:
        """Get training loss history."""
        return self.losses
