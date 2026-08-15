"""TopologicalNMF: NMF with topological constraints via persistent homology."""

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.optim as optim
from sklearn.decomposition._nmf import _initialize_nmf
from tqdm.auto import tqdm

from .losses import (
    total_variation, _get_diagram, wasserstein_reconstruction_loss
)
from .optim import update_V
from .utils import (
    sparsity_score,
    center_point_cloud_torch,
    l1_l2_sq_ratio,
    periodicity_from_diagram,
)
from .persistence import TimeDelayEmbeddingTorch, GudhiVietorisRipsComplex

PeriodicityTarget = Optional[Union[float, Sequence[Optional[float]]]]


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
        When None (the default) no diagram-shape loss is applied, and the
        topological term of :meth:`fit` consists only of the
        ``target_periodicity`` penalty (if requested). Set it explicitly, e.g.
        to :func:`~TopNMF.losses.ph_sparsity_loss` or
        :func:`~TopNMF.losses.target_diagram_loss`, to add a diagram loss.
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
        recon_loss: str = "mse",
        wasserstein_blur: float = 0.05,
        data_shape: Optional[Tuple] = None,
        use_embedding: bool = False,
    ):
        self.n_components = n_components
        self.device = device
        self.random_state = random_state
        self.complex = complex
        self.ph_loss_fn = ph_loss_fn
        self.ph_loss_params = ph_loss_params if ph_loss_params is not None else {}
        self.recon_loss = recon_loss
        self.wasserstein_blur = wasserstein_blur
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
    def _coerce_periodicity_targets(
        target_periodicity: PeriodicityTarget,
        n_components: int,
    ) -> Optional[List[Optional[float]]]:
        if target_periodicity is None:
            return None

        if isinstance(target_periodicity, str):
            return [float(target_periodicity)] * n_components

        try:
            raw_targets = list(target_periodicity)
        except TypeError:
            return [float(target_periodicity)] * n_components

        targets: List[Optional[float]] = []
        for value in raw_targets[:n_components]:
            if value is None:
                targets.append(None)
                continue
            try:
                targets.append(float(value))
            except (TypeError, ValueError):
                targets.append(None)

        targets.extend([None] * (n_components - len(targets)))
        return targets

    @staticmethod
    def _match_periodicity_targets_by_rank(
        periodicity_scores: List[Optional[torch.Tensor]],
        targets: List[Optional[float]],
    ) -> List[Optional[float]]:
        """Assign sorted targets to components sorted by current periodicity.

        ``periodicity_scores`` is indexed by component and may contain None for
        components without a score; those components receive no target.
        """
        score_order = sorted(
            (idx for idx, score in enumerate(periodicity_scores) if score is not None),
            key=lambda idx: float(periodicity_scores[idx].detach().cpu()),
        )
        target_order = sorted(
            range(len(targets)),
            key=lambda idx: 0.5 if targets[idx] is None else targets[idx],
        )

        matched_targets: List[Optional[float]] = [None] * len(targets)
        for rank, target_idx in enumerate(target_order):
            if rank >= len(score_order):
                break
            target_value = targets[target_idx]
            if target_value is not None:
                matched_targets[score_order[rank]] = target_value
        return matched_targets

    @staticmethod
    def _compute_periodicity_score(
        diagrams: List[object],
        PH_dims: List[int],
        device: str,
    ) -> torch.Tensor:
        parts = [d for d in (_get_diagram(diagrams, dim) for dim in PH_dims)
                 if d is not None and d.shape[0] > 0]
        if not parts:
            return torch.tensor(0.0, device=device)
        return periodicity_from_diagram(torch.cat(parts))

    @staticmethod
    def _compute_periodicity_loss(
        periodicity_score: torch.Tensor,
        target_value: float,
    ) -> torch.Tensor:
        return (periodicity_score - target_value) ** 2

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
        periodicity_targets = self._coerce_periodicity_targets(
            target_periodicity, self.n_components)
        compute_periodicity = (
            apply_topology
            and periodicity_targets is not None
            and any(target is not None for target in periodicity_targets)
        )
        periodicity_scores: List[Optional[torch.Tensor]] = [None] * self.n_components

        # A diagram is needed for the diagram loss and for periodicity targets.
        needs_diagram = apply_topology and (
            self.ph_loss_fn is not None or compute_periodicity)

        for idx in range(self.n_components):
            component = self.V[idx]

            if needs_diagram:
                diagrams = self._compute_component_diagrams(
                    component, embedder, ph_complex, complex_inputs)
                if self.ph_loss_fn is not None:
                    loss_ph += self.ph_loss_fn(
                        diagrams, PH_dims, target_diagrams, self.device,
                        **self.ph_loss_params)
                if compute_periodicity:
                    periodicity_scores[idx] = self._compute_periodicity_score(
                        diagrams, PH_dims, self.device)

            # Structured samples get their 2-D total variation, not the
            # flattened one, which would wrap around row boundaries.
            loss_tv_v += total_variation(
                component.reshape(self.data_shape)
                if self.data_shape is not None and len(self.data_shape) == 2
                else component)
            if target_sparsity is not None:
                loss_spa_v += (sparsity_score(component) - target_sparsity) ** 2
            else:
                loss_spa_v += l1_l2_sq_ratio(component)

            sp_score += sparsity_score(component)

        if compute_periodicity:
            matched_targets = self._match_periodicity_targets_by_rank(
                periodicity_scores, periodicity_targets)
            for idx, target_value in enumerate(matched_targets):
                if target_value is not None:
                    loss_ph += self._compute_periodicity_loss(
                        periodicity_scores[idx], target_value)

        norm = float(self.n_components)
        return loss_ph / norm, loss_spa_v / norm, sp_score / norm, loss_tv_v / norm

    def _reconstruction_loss(self, X_t, loss_fn):
        """Reconstruction loss of ``W V`` against ``X`` for the configured metric."""
        recon_t = torch.mm(self.W, self.V)
        if self.recon_loss == "wasserstein":
            return wasserstein_reconstruction_loss(
                recon_t, X_t, self.data_shape, blur=self.wasserstein_blur)
        return loss_fn(recon_t, X_t)

    def _compute_w_sparsity_loss(self):
        return l1_l2_sq_ratio(self.W, dim=1).sum() / float(self.n_components)

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
        target_periodicity: PeriodicityTarget = None,
        gd_iter: int = 1,
        mu_iter: int = 0,
        W_iter: int = 0,
        embedding_dim: int = 4,
        tau: Optional[int] = None,
        n_periods: Optional[int] = None,
        PH_dims: Optional[List[int]] = None,
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
        target_periodicity : float or sequence of float, optional
            Target periodicity score. Sequence targets are sorted and matched to
            basis components sorted by their current periodicity score.
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
        if PH_dims is None:
            PH_dims = [1]

        X_t = self._initialize_model_tensors(X, init_method, normalize_V_max, epsilon)
        embedder, tau = self._resolve_embedder(n_features, embedding_dim, tau, n_periods)
        ph_complex = self._resolve_complex()

        optimizer = self._build_optimizer(
            optimizer_cls, [self.W, self.V], lr, weight_decay, optimizer_kwargs)
        scheduler = self._build_scheduler(optimizer, scheduler_cls, scheduler_kwargs)
        target_l1 = self._compute_target_l1(n_features, target_sparsity)
        
        if self.recon_loss == "wasserstein" and self.data_shape is None:
            raise ValueError("data_shape must be provided for wasserstein_reconstruction_loss")
        if target_diagrams is not None and self.ph_loss_fn is None:
            raise ValueError(
                "target_diagrams was given but ph_loss_fn is None, so the targets "
                "would be ignored. Pass e.g. ph_loss_fn=target_diagram_loss to "
                "TopologicalNMF(...).")

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
            if gd_iter <= 0:
                loss_apx = self._reconstruction_loss(X_t, loss_fn)

            for _ in range(gd_iter):
                loss_ph, loss_spa_v, sp_score, loss_tv_v = (
                    self._compute_component_losses(
                        epoch, lambda_top, start_epoch_topological,
                        ph_complex, embedder, complex_inputs,
                        PH_dims, target_diagrams, target_periodicity,
                        target_sparsity))
                loss_spa_w = self._compute_w_sparsity_loss()
                loss_apx = self._reconstruction_loss(X_t, loss_fn)

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

            # Early-stopping uses the same composite objective that is optimised.
            current = float(
                lambda_top * loss_ph.item()
                + lambda_spa_V * loss_spa_v.item()
                + lambda_spa_W * loss_spa_w.item()
                + lambda_apx * loss_apx.item()
                + lambda_tv * loss_tv_v.item()
            )
            if (prev_loss - current) < tol:
                count += 1
                if count > tol_count:
                    break
            else:
                prev_loss = current
                count = 0

        self.W = self.W.detach()
        self.V = self.V.detach()
        return self

    def transform(self, X: np.ndarray, n_iterations: int = 200,
                  epsilon: float = 1e-10) -> np.ndarray:
        """
        Compute the coefficient matrix W for new data X under the fitted basis V.

        Solves ``min_{W >= 0} || X - W V ||_F^2`` with V held fixed, using
        non-negative multiplicative updates.

        .. note::
           Breaking change from earlier versions, where ``transform`` ignored
           ``X`` and returned the W learned during ``fit``. To recover the
           training coefficients, use ``model.W`` directly.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features).
        n_iterations : int
            Number of multiplicative-update iterations.
        epsilon : float
            Numerical stability constant.

        Returns
        -------
        np.ndarray
            Coefficient matrix W of shape (n_samples, n_components).

        Notes
        -----
        The multiplicative updates start from a random W. When ``random_state``
        was set on the estimator the draw is seeded, so repeated calls return
        the same coefficients.
        """
        if self.V is None:
            raise ValueError("Model must be fitted before transform")
        X_t = torch.as_tensor(X, dtype=torch.float, device=self.device)
        if X_t.shape[1] != self.V.shape[1]:
            raise ValueError(
                f"X has {X_t.shape[1]} features but V expects {self.V.shape[1]}")

        V = self.V.detach()
        VVt = V @ V.T
        XVt = X_t @ V.T
        generator = None
        if self.random_state is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(self.random_state))
        W = torch.rand(X_t.shape[0], self.n_components, device=self.device,
                       generator=generator) + epsilon
        for _ in range(n_iterations):
            W *= XVt / (W @ VVt + epsilon)
        return W.detach().cpu().numpy()

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
