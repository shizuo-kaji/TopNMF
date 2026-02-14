"""
Topological NMF

This module provides the main TopologicalNMF class for performing
Non-negative Matrix Factorization with topological constraints.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from sklearn.decomposition._nmf import _initialize_nmf
from tqdm.auto import tqdm

from .losses import ph_sparsity_loss
from .nmf_utils import sparsity_score, total_variation, update_V
from .topological_utils import (
    TimeDelayEmbeddingTorch,
    center_point_cloud_torch,
    GudhiVietorisRipsComplex,
)
from .visualization import (
    plot_PD_graph,
    plot_gallery,
    plot_gallery_graph,
    plot_loss,
    plot_persistence_diagrams,
)

try:
    from IPython.display import display

    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False


@dataclass
class _PlotHandles:
    """Container for optional in-notebook visualization handles."""

    show_plots: List[str]
    n_row: int
    n_col: int
    current_interval: int
    fig_loss: Optional[Any] = None
    ax_loss: Optional[Any] = None
    disp_loss: Optional[Any] = None
    fig_basis: Optional[Any] = None
    ax_basis: Optional[Any] = None
    disp_basis: Optional[Any] = None
    fig_ph: Optional[Any] = None
    ax_ph: Optional[Any] = None
    disp_ph: Optional[Any] = None


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
            Persistent homology complex (e.g., GudhiVietorisRipsComplex, CubicalComplex).
            If None, uses GudhiVietorisRipsComplex with time delay embedding.
        ph_loss_fn : Callable, optional
            Loss function for persistent homology. Should accept (diagrams, PH_dims, target_diagrams, device).
            If None, uses ph_sparsity_loss.
        ph_loss_params : Dict, optional
            Additional keyword arguments for the ph_loss_fn.
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
        self.ph_loss_params = ph_loss_params if ph_loss_params is not None else {}
        self.data_shape = data_shape
        self.use_embedding = use_embedding

        # Will be set during fit
        self.W = None
        self.V = None
        self.losses = self._empty_losses()

    @staticmethod
    def _empty_losses() -> Dict[str, List[float]]:
        return {"PH": [], "approx": [], "sparse_W": [], "sparse_V": [], "lr": []}

    def initialize_factors(self, X: np.ndarray, method: str = "nndsvda"):
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

        if method == "random":
            np.random.seed(self.random_state)
            W = np.random.rand(n_samples, self.n_components)
            V = np.random.rand(self.n_components, n_features)
        else:
            W, V = _initialize_nmf(
                X,
                n_components=self.n_components,
                init=method,
                random_state=self.random_state,
            )

        return W, V

    def _initialize_model_tensors(
        self,
        X: np.ndarray,
        init_method: str,
        normalize_v_max: bool,
        epsilon: float,
    ) -> torch.Tensor:
        W, V = self.initialize_factors(X, method=init_method)
        X_t = torch.as_tensor(X, dtype=torch.float, device=self.device)
        self.W = torch.tensor(W, dtype=torch.float, device=self.device, requires_grad=True)
        self.V = torch.tensor(V, dtype=torch.float, device=self.device, requires_grad=True)
        if normalize_v_max:
            with torch.no_grad():
                self._normalize_v_rows_max(epsilon)
        return X_t

    @staticmethod
    def _build_optimizer(
        optimizer_cls: Callable,
        parameters: List[torch.Tensor],
        lr: float,
        weight_decay: float,
        optimizer_kwargs: Optional[Dict],
    ) -> optim.Optimizer:
        opt_kwargs = dict(optimizer_kwargs or {})
        opt_kwargs.setdefault("lr", lr)
        opt_kwargs.setdefault("weight_decay", weight_decay)
        return optimizer_cls(parameters, **opt_kwargs)

    @staticmethod
    def _build_scheduler(
        optimizer: optim.Optimizer,
        scheduler_cls: Optional[Callable],
        scheduler_kwargs: Optional[Dict],
    ) -> Optional[object]:
        if scheduler_cls is None:
            return None

        sch_kwargs = dict(scheduler_kwargs or {})
        if scheduler_cls == optim.lr_scheduler.ReduceLROnPlateau:
            sch_kwargs.setdefault("factor", 0.9)
            sch_kwargs.setdefault("patience", 10000)
        return scheduler_cls(optimizer, **sch_kwargs)

    @staticmethod
    def _compute_target_l1(n_features: int, target_sparsity: Optional[float]) -> float:
        if target_sparsity is not None and target_sparsity > 0:
            return np.sqrt(n_features) - target_sparsity * (np.sqrt(n_features) - 1)
        return 0.0

    def _resolve_embedder(
        self, n_features: int, M: int, tau: Optional[int]
    ) -> Tuple[Optional[TimeDelayEmbeddingTorch], Optional[int]]:
        if not self.use_embedding:
            return None, tau

        resolved_tau = tau if tau is not None else int(n_features / (2 * (M + 1)))
        embedder = TimeDelayEmbeddingTorch(dim=M + 1, delay=resolved_tau)
        return embedder, resolved_tau

    def _resolve_complex(self) -> Callable:
        if self.complex is not None:
            return self.complex
        return GudhiVietorisRipsComplex(dim=1, p=2)

    def _run_multiplicative_updates(
        self,
        X_t: torch.Tensor,
        target_sparsity: Optional[float],
        target_l1: float,
        mu_iter: int,
        W_iter: int,
        epsilon: float,
    ) -> None:
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

    def _compute_component_diagrams(
        self,
        component: torch.Tensor,
        embedder: Optional[TimeDelayEmbeddingTorch],
        ph_complex: Callable,
        complex_inputs: Optional[Dict[str, object]],
    ):
        if self.use_embedding:
            if embedder is None:
                raise ValueError("Embedder must be provided when use_embedding is True.")
            point_cloud = embedder(component)
            point_cloud = center_point_cloud_torch(point_cloud)
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
    def _resolve_periodicity_target(
        target_periodicity: Optional[object],
        component_idx: int,
    ) -> Optional[float]:
        if target_periodicity is None:
            return None

        if hasattr(target_periodicity, "__getitem__") and not isinstance(target_periodicity, str):
            try:
                return float(target_periodicity[component_idx])
            except (IndexError, TypeError, ValueError):
                return None

        return float(target_periodicity)

    @staticmethod
    def _compute_periodicity_loss(
        diagrams,
        PH_dims: List[int],
        target_value: float,
        device: str,
    ) -> torch.Tensor:
        persistence_diagram = torch.cat(
            [
                diagrams[dim].diagram if hasattr(diagrams[dim], "diagram") else diagrams[dim]
                for dim in PH_dims
            ]
        )
        persistence = torch.diff(persistence_diagram, dim=1).reshape(-1)
        if len(persistence) > 0:
            periodicity_score = torch.max(persistence) / np.sqrt(3)
        else:
            periodicity_score = torch.tensor(0.0, device=device)
        return (periodicity_score - target_value) ** 2

    def _compute_component_losses(
        self,
        epoch: int,
        lambda_top: float,
        start_epoch_topological: int,
        ph_complex: Callable,
        embedder: Optional[TimeDelayEmbeddingTorch],
        complex_inputs: Optional[Dict[str, object]],
        PH_dims: List[int],
        target_diagrams: Optional[List],
        target_periodicity: Optional[object],
        target_sparsity: Optional[float],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_ph = torch.tensor(0.0, device=self.device)
        loss_spa_v = torch.tensor(0.0, device=self.device)
        sp_score = torch.tensor(0.0, device=self.device)
        loss_tv_v = torch.tensor(0.0, device=self.device)

        apply_topology = lambda_top > 0 and epoch >= start_epoch_topological
        for component_idx in range(self.n_components):
            component = self.V[component_idx]

            if apply_topology:
                diagrams = self._compute_component_diagrams(
                    component=component,
                    embedder=embedder,
                    ph_complex=ph_complex,
                    complex_inputs=complex_inputs,
                )

                if target_diagrams is not None:
                    loss_ph += self.ph_loss_fn(
                        diagrams,
                        PH_dims,
                        target_diagrams,
                        self.device,
                        **self.ph_loss_params,
                    )

                target_value = self._resolve_periodicity_target(target_periodicity, component_idx)
                if target_value is not None:
                    loss_ph += self._compute_periodicity_loss(
                        diagrams=diagrams,
                        PH_dims=PH_dims,
                        target_value=target_value,
                        device=self.device,
                    )

            loss_tv_v += total_variation(component)
            if target_sparsity is not None:
                loss_spa_v += (sparsity_score(component) - target_sparsity) ** 2
            else:
                loss_spa_v += (component.abs().sum()) ** 2 / (component**2).sum()

            sp_score += sparsity_score(component)

        norm = float(self.n_components)
        return loss_ph / norm, loss_spa_v / norm, sp_score / norm, loss_tv_v

    def _compute_w_sparsity_loss(self) -> torch.Tensor:
        loss_spa_w = torch.sum(
            torch.sum(torch.abs(self.W), dim=1) ** 2 / torch.sum(self.W**2, dim=1)
        )
        return loss_spa_w / float(self.n_components)

    @staticmethod
    def _step_scheduler(scheduler: Optional[object], loss: torch.Tensor) -> None:
        if scheduler is None:
            return
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss)
        else:
            scheduler.step()

    def _record_losses(
        self,
        loss_ph: torch.Tensor,
        loss_apx: torch.Tensor,
        loss_spa_v: torch.Tensor,
        loss_spa_w: torch.Tensor,
        optimizer: optim.Optimizer,
    ) -> None:
        self.losses["PH"].append(loss_ph.item())
        self.losses["approx"].append(loss_apx.item())
        self.losses["sparse_V"].append(loss_spa_v.item())
        self.losses["sparse_W"].append(loss_spa_w.item())
        self.losses["lr"].append(optimizer.param_groups[0]["lr"])

    def _normalize_v_rows_max(self, epsilon: float) -> None:
        normal_value = torch.max(self.V, dim=1).values.unsqueeze(1)
        self.V /= normal_value + epsilon

    def _apply_constraints(
        self,
        normalize: bool,
        normalize_v_max: bool,
        epsilon: float,
    ) -> None:
        with torch.no_grad():
            self.V.clamp_(min=0)
            self.W.clamp_(min=0)
            if normalize:
                self.W /= torch.norm(self.W, p=1, dim=1, keepdim=True) + epsilon
            if normalize_v_max:
                self._normalize_v_rows_max(epsilon)

    def _reshape_for_visualization(self, V_np: np.ndarray) -> np.ndarray:
        if self.data_shape is not None:
            return V_np.reshape(-1, *self.data_shape)
        return V_np

    @staticmethod
    def _setup_plot_handles(
        show_plots: Optional[List[str]],
        plot_grid: Tuple[int, int],
        disp_interval: int,
    ) -> _PlotHandles:
        requested = list(show_plots or [])
        n_row, n_col = plot_grid
        handles = _PlotHandles(
            show_plots=requested,
            n_row=n_row,
            n_col=n_col,
            current_interval=disp_interval,
        )

        if not requested:
            return handles

        if not IPYTHON_AVAILABLE:
            print(
                "Warning: show_plots requires IPython/Jupyter environment. "
                "Plots will not be displayed."
            )
            handles.show_plots = []
            return handles

        if "loss" in requested:
            handles.fig_loss, handles.ax_loss = plt.subplots(1, 1, figsize=(8, 5))
            handles.disp_loss = display(handles.fig_loss, display_id=True)

        if "basis" in requested:
            handles.fig_basis, handles.ax_basis = plt.subplots(
                n_row, n_col, figsize=(2.0 * n_col, 2.26 * n_row)
            )
            handles.disp_basis = display(handles.fig_basis, display_id=True)

        if "PH" in requested:
            handles.fig_ph, handles.ax_ph = plt.subplots(
                n_row,
                n_col,
                figsize=(2.0 * n_col, 2.26 * n_row),
                squeeze=False,
            )
            handles.disp_ph = display(handles.fig_ph, display_id=True)

        return handles

    def _update_plots(
        self,
        epoch: int,
        plot_handles: _PlotHandles,
        complex_inputs: Optional[Dict[str, object]],
        superlevel: bool,
        PHmode: str,
        M: int,
        tau: Optional[int],
        n_features: int,
    ) -> None:
        if not plot_handles.show_plots:
            return

        interval = max(1, int(plot_handles.current_interval))
        if epoch % interval != 0:
            return

        V_np = self.V.detach().cpu().numpy().copy()
        n_row = plot_handles.n_row
        n_col = plot_handles.n_col
        is_graph_case = complex_inputs is not None and "all_edges" in complex_inputs

        if "loss" in plot_handles.show_plots and plot_handles.disp_loss is not None:
            plot_loss(self.losses, ax=plot_handles.ax_loss)
            plot_handles.disp_loss.update(plot_handles.fig_loss)

        if "basis" in plot_handles.show_plots and plot_handles.disp_basis is not None:
            if is_graph_case:
                edge_list = complex_inputs["all_edges"]
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                node_pos = complex_inputs.get("node_pos", None)
                plot_gallery_graph(
                    V_np,
                    edge_index,
                    title="Basis",
                    n_col=n_col,
                    n_row=n_row,
                    axs=plot_handles.ax_basis,
                    pos=node_pos,
                )
            else:
                plot_gallery(
                    self._reshape_for_visualization(V_np),
                    title="basis",
                    n_row=n_row,
                    n_col=n_col,
                    axs=plot_handles.ax_basis,
                )
            plot_handles.disp_basis.update(plot_handles.fig_basis)

        if "PH" in plot_handles.show_plots and plot_handles.disp_ph is not None:
            if is_graph_case:
                plot_PD_graph(
                    [V_np[i] for i in range(min(len(V_np), n_row * n_col))],
                    complex_inputs["all_edges"],
                    n_col=n_col,
                    n_row=n_row,
                    axs=plot_handles.ax_ph,
                    superlevel=superlevel,
                )
            else:
                plot_persistence_diagrams(
                    self._reshape_for_visualization(V_np),
                    n_col=n_col,
                    n_row=n_row,
                    superlevel=superlevel,
                    PHmode=PHmode,
                    M=M,
                    tau=tau if tau is not None else int(n_features / (2 * (M + 1))),
                    axs=plot_handles.ax_ph,
                    use_embedding=self.use_embedding,
                )
            plot_handles.disp_ph.update(plot_handles.fig_ph)

        plot_handles.current_interval = max(1, int(1.2 * plot_handles.current_interval))

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
        M: int = 4,
        tau: Optional[int] = None,
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
        show_plots: Optional[List[str]] = None,
        disp_interval: int = 100,
        plot_grid: Tuple[int, int] = (2, 3),
        PHmode: str = "T",
        superlevel: bool = False,
    ) -> "TopologicalNMF":
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
        optimizer_cls : Callable, optional
            Optimizer class to use (default: torch.optim.AdamW)
        optimizer_kwargs : Dict, optional
            Additional keyword arguments for the optimizer
        scheduler_cls : Callable, optional
            Scheduler class to use (default: torch.optim.lr_scheduler.ReduceLROnPlateau).
            Set to None to disable scheduler.
        scheduler_kwargs : Dict, optional
            Additional keyword arguments for the scheduler
        verbose : bool, optional
            Whether to show progress bar
        show_plots : Optional[List[str]], optional
            List of plots to display during training. Options: 'loss', 'basis', 'PH'.
            Requires IPython/Jupyter environment. If None, no plots are shown.
        disp_interval : int, optional
            Initial interval (in epochs) for updating displays. Increases by 1.2x after each update.
        plot_grid : Tuple[int, int], optional
            (n_row, n_col) for basis and PH plot grids
        PHmode : str, optional
            'T' for top-dimensional cells, 'V' for vertices (used for PH plots)
        superlevel : bool, optional
            Whether to use superlevel set filtration for PH plots

        Returns
        -------
        self
            Fitted model
        """
        _, n_features = X.shape
        epsilon = 1e-10

        X_t = self._initialize_model_tensors(
            X=X,
            init_method=init_method,
            normalize_v_max=normalize_V_max,
            epsilon=epsilon,
        )
        embedder, tau = self._resolve_embedder(n_features=n_features, M=M, tau=tau)
        ph_complex = self._resolve_complex()

        optimizer = self._build_optimizer(
            optimizer_cls=optimizer_cls,
            parameters=[self.W, self.V],
            lr=lr,
            weight_decay=weight_decay,
            optimizer_kwargs=optimizer_kwargs,
        )
        scheduler = self._build_scheduler(
            optimizer=optimizer,
            scheduler_cls=scheduler_cls,
            scheduler_kwargs=scheduler_kwargs,
        )
        target_l1 = self._compute_target_l1(
            n_features=n_features, target_sparsity=target_sparsity
        )
        loss_fn = torch.nn.MSELoss()
        plot_handles = self._setup_plot_handles(
            show_plots=show_plots, plot_grid=plot_grid, disp_interval=disp_interval
        )

        progress = tqdm(range(n_iterations), disable=not verbose)
        prev_loss = np.inf
        count = 0

        for epoch in progress:
            self._run_multiplicative_updates(
                X_t=X_t,
                target_sparsity=target_sparsity,
                target_l1=target_l1,
                mu_iter=mu_iter,
                W_iter=W_iter,
                epsilon=epsilon,
            )

            # Default values keep progress/early-stop logic valid even when gd_iter=0.
            loss_ph = torch.tensor(0.0, device=self.device)
            loss_spa_v = torch.tensor(0.0, device=self.device)
            loss_spa_w = torch.tensor(0.0, device=self.device)
            sp_score = torch.tensor(0.0, device=self.device)
            loss_tv_v = torch.tensor(0.0, device=self.device)
            loss_apx = loss_fn(torch.mm(self.W, self.V), X_t)

            for _ in range(gd_iter):
                loss_ph, loss_spa_v, sp_score, loss_tv_v = self._compute_component_losses(
                    epoch=epoch,
                    lambda_top=lambda_top,
                    start_epoch_topological=start_epoch_topological,
                    ph_complex=ph_complex,
                    embedder=embedder,
                    complex_inputs=complex_inputs,
                    PH_dims=PH_dims,
                    target_diagrams=target_diagrams,
                    target_periodicity=target_periodicity,
                    target_sparsity=target_sparsity,
                )
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

                self._step_scheduler(scheduler=scheduler, loss=loss)
                self._record_losses(
                    loss_ph=loss_ph,
                    loss_apx=loss_apx,
                    loss_spa_v=loss_spa_v,
                    loss_spa_w=loss_spa_w,
                    optimizer=optimizer,
                )

            self._apply_constraints(
                normalize=normalize,
                normalize_v_max=normalize_V_max,
                epsilon=epsilon,
            )

            if verbose:
                progress.set_postfix(
                    loss=f"PH: {loss_ph.item():.6f}, "
                    f"sparsity: {sp_score.item():.6f}, "
                    f"approx: {loss_apx.item():.6f}"
                )

            self._update_plots(
                epoch=epoch,
                plot_handles=plot_handles,
                complex_inputs=complex_inputs,
                superlevel=superlevel,
                PHmode=PHmode,
                M=M,
                tau=tau,
                n_features=n_features,
            )

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

        # Expose learned factors as plain tensors after optimization completes.
        self.W = self.W.detach()
        self.V = self.V.detach()
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
