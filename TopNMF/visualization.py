"""Visualisation utilities for TopNMF."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import gudhi
import networkx as nx
from gudhi.point_cloud.timedelay import TimeDelayEmbedding
from typing import Optional, List, Tuple, Union

from .persistence.graph import GraphFiltrationPH

try:
    from IPython.display import display
    _IPYTHON_AVAILABLE = True
except ImportError:
    _IPYTHON_AVAILABLE = False


# ------------------------------------------------------------------
# Plot functions
# ------------------------------------------------------------------

def plot_gallery(images, title: str = "", n_col: int = 5, n_row: int = 5,
                 cmap=plt.cm.gray, axs=None):
    """
    Plot a gallery of time series or images in a grid.

    Parameters
    ----------
    images : array-like
        Array of 1D vectors (line plots) or 2D arrays (heatmaps).
    title : str
        Suptitle for the figure.
    n_col, n_row : int
        Grid dimensions.
    cmap : matplotlib colormap
        Colormap for 2D data.
    axs : matplotlib axes array, optional
        Pre-existing axes.

    Returns
    -------
    matplotlib axes array
    """
    if axs is None:
        fig, axs = plt.subplots(n_row, n_col, figsize=(2. * n_col, 2.26 * n_row))
        plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
        plt.suptitle(title, size=16)

    for i, comp in enumerate(images[:n_col * n_row]):
        ax = axs[i // n_col, i % n_col] if n_row != 1 else axs[i % n_col]
        ax.cla()
        if len(comp.shape) == 1:
            ax.plot(comp)
        else:
            vmax = max(comp.max(), -comp.min())
            ax.imshow(comp, cmap=cmap, interpolation='nearest',
                      vmin=-vmax, vmax=vmax)
        ax.set_xticks(())
        ax.set_yticks(())

    return axs


def plot_persistence_diagrams(data, n_col: int = 5, n_row: int = 5,
                              superlevel: bool = False, PHmode: str = "V",
                              embedding_dim: int = 1, tau: int = 1, axs=None,
                              center_func=None, use_embedding: bool = True):
    """
    Compute and plot persistence diagrams for multiple signals/images.

    Parameters
    ----------
    data : array-like
        Array of time series or images.
    n_col, n_row : int
        Grid dimensions.
    superlevel : bool
        Whether to compute superlevel set persistence.
    PHmode : str
        'V' for vertex-based, 'T' for top-dimensional cells.
    embedding_dim : int
        Embedding dimension minus 1.
    tau : int
        Time delay.
    axs : matplotlib axes array, optional
        Pre-existing axes.
    center_func : callable, optional
        Point cloud centering function.
    use_embedding : bool
        Whether to use time-delay embedding.

    Returns
    -------
    matplotlib axes array
    """
    if axs is None:
        fig, axs = plt.subplots(n_row, n_col, figsize=(2. * n_col, 2.26 * n_row))

    for i, comp in enumerate(data[:n_col * n_row]):
        if not use_embedding or len(comp.shape) > 1:
            sign = -1 if superlevel else 1
            if PHmode == "V":
                cc = gudhi.CubicalComplex(vertices=sign * comp)
            else:
                cc = gudhi.CubicalComplex(top_dimensional_cells=sign * comp)
            pd = cc.persistence()
        else:
            embedder = TimeDelayEmbedding(dim=embedding_dim + 1, delay=tau)
            embedded = embedder(comp)
            centered = center_func(embedded) if center_func is not None else embedded
            rips = gudhi.RipsComplex(points=centered).create_simplex_tree(max_dimension=2)
            pd = rips.persistence()

        ax = axs[i // n_col, i % n_col]
        ax.clear()
        gudhi.plot_persistence_diagram(pd, axes=ax, legend=False, fontsize=4)
        ax.set_xticks(())
        ax.set_yticks(())

    return axs


def plot_loss(losses, ax=None):
    """
    Plot training losses.

    If PH loss has negative values, uses dual y-axes
    (left log-scale for approx/lr, right linear for PH).

    Parameters
    ----------
    losses : dict
        Loss history from TopologicalNMF.get_losses().
    ax : matplotlib axes, optional
        Pre-existing axes.

    Returns
    -------
    matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.clear()
    colors = {'approx': 'C1', 'lr': 'C2', 'PH': 'C0'}

    ph_values = np.asarray(losses.get('PH', []))
    has_negative_ph = np.any(ph_values < 0) if len(ph_values) > 0 else False

    ax2 = None
    if len(fig.axes) > 1 and fig.axes[0] == ax:
        ax2 = fig.axes[1]

    if has_negative_ph:
        if ax2 is None:
            ax2 = ax.twinx()
        else:
            ax2.set_visible(True)
            ax2.clear()
            ax2.patch.set_visible(False)
    elif ax2 is not None:
        ax2.set_visible(False)
        ax2.clear()

    for key in ['approx', 'lr']:
        if key in losses:
            ax.plot(np.asarray(losses[key]), label=key, color=colors.get(key, 'k'))

    if not has_negative_ph and 'PH' in losses and len(ph_values) > 0:
        ax.plot(ph_values, label='PH', color=colors['PH'])

    ax.set_xlabel("Epoch")
    ax.set_ylabel(
        "Loss (log scale)" if (not has_negative_ph and 'PH' in losses)
        else "Loss (approx, lr)", size=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    if has_negative_ph and 'PH' in losses:
        ax2.plot(ph_values, label='PH', color=colors['PH'])
        ax2.set_ylabel("PH Loss (linear)", size=12)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        if len(ph_values) > 0:
            ph_min, ph_max = float(np.min(ph_values)), float(np.max(ph_values))
            pad = 0.1 * ((ph_max - ph_min) if ph_min != ph_max else (abs(ph_min) + 1.0))
            ax2.set_ylim(ph_min - pad, ph_max + pad)
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))

    lines1, labels1 = ax.get_legend_handles_labels()
    if has_negative_ph and ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax.legend(lines1, labels1, loc='upper right')

    return ax


def plot_gallery_graph(edge_values: Union[np.ndarray, torch.Tensor],
                       edge_index: torch.Tensor, title: str,
                       n_col: int = 3, n_row: int = None, axs=None, pos=None,
                       threshold: float = 0.1, use_labels: bool = False):
    """
    Network visualisation for graph basis vectors.

    Parameters
    ----------
    edge_values : array-like
        Edge weight matrix of shape (n_basis, n_edges).
    edge_index : torch.Tensor
        Edge indices of shape (2, n_edges).
    title : str
        Title prefix for subplots.
    n_col, n_row : int
        Grid dimensions.
    axs : matplotlib axes array, optional
        Pre-existing axes.
    pos : dict, optional
        Node positions {node_id: (x, y)}.
    threshold : float
        Minimum edge weight to display.
    use_labels : bool
        Whether to draw node labels.

    Returns
    -------
    matplotlib axes array
    """
    edge_index_pairs = edge_index.transpose(0, 1) if edge_index.dim() == 2 else edge_index
    num_basis = edge_values.shape[0]

    if pos is None:
        nodes = (set(int(u.item()) for u, v in edge_index_pairs)
                 | set(int(v.item()) for u, v in edge_index_pairs))
        G_temp = nx.Graph()
        G_temp.add_nodes_from(nodes)
        G_temp.add_edges_from([(int(u.item()), int(v.item())) for u, v in edge_index_pairs])
        pos = nx.spring_layout(G_temp, seed=42)

    if n_row is None:
        n_row = int(np.ceil(num_basis / n_col))

    if axs is None:
        fig, axs = plt.subplots(n_row, n_col, figsize=(6. * n_col, 5. * n_row))
    axs = np.atleast_2d(axs)

    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    global_xlim = (min(all_x) - 0.5, max(all_x) + 0.5) if all_x else (-1, 1)
    global_ylim = (min(all_y) - 0.5, max(all_y) + 0.5) if all_y else (-1, 1)

    for idx in range(n_row * n_col):
        ax = axs[idx // n_col, idx % n_col]
        ax.clear()

        if idx >= num_basis:
            ax.axis('off')
            continue

        weights = edge_values[idx]
        edge_weights = {}

        for j, (u, v) in enumerate(edge_index_pairs):
            w = weights[j].item() if isinstance(weights[j], torch.Tensor) else float(weights[j])
            if w > threshold:
                e = tuple(sorted([int(u.item()), int(v.item())]))
                edge_weights[e] = edge_weights.get(e, 0.0) + w

        G = nx.Graph()
        G.add_edges_from(edge_weights.keys())

        nx.draw_networkx_nodes(G, pos, node_color='lightcoral', node_size=200,
                               edgecolors='black', ax=ax)
        if use_labels:
            nx.draw_networkx_labels(G, pos, font_size=10, font_color='black', ax=ax)

        edge_widths = [1.5 + edge_weights[tuple(sorted([u, v]))] * 2 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, edge_color='black', width=edge_widths, ax=ax)

        edge_labels = {(u, v): f"{edge_weights[tuple(sorted([u, v]))]:.2f}" for u, v in G.edges()}
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8,
                                         bbox=dict(facecolor='white', edgecolor='none',
                                                   alpha=0.8, pad=0.3), ax=ax)

        ax.set_title(f"{title} {idx + 1}", fontsize=12)
        ax.set_xlim(global_xlim)
        ax.set_ylim(global_ylim)
        ax.axis('off')

    return axs


def plot_time_series_comparison(original, reconstructed, basis_vectors,
                                time_axis=None, save_path: Optional[str] = None):
    """Side-by-side original vs. reconstructed signals with basis vectors."""
    n_basis = len(basis_vectors)
    fig, axes = plt.subplots(2 + n_basis, 1, figsize=(12, 4 * (2 + n_basis)))

    if time_axis is None:
        time_axis = np.arange(len(original))

    axes[0].plot(time_axis, original, 'b-', linewidth=2)
    axes[0].set_title('Original Signal', fontsize=14)
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_axis, reconstructed, 'r-', linewidth=2)
    axes[1].set_title('Reconstructed Signal', fontsize=14)
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)

    for i, basis in enumerate(basis_vectors):
        axes[2 + i].plot(time_axis, basis, 'g-', linewidth=2)
        axes[2 + i].set_title(f'Basis Vector {i + 1}', fontsize=14)
        axes[2 + i].set_ylabel('Amplitude')
        axes[2 + i].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, axes


def plot_fourier_spectrum(signals, labels=None, n_components: int = 50,
                          figsize=(12, 6), save_path: Optional[str] = None):
    """Fourier magnitude spectrum for one or more signals."""
    n_signals = len(signals)
    fig, axes = plt.subplots(1, n_signals, figsize=figsize)

    if n_signals == 1:
        axes = [axes]
    if labels is None:
        labels = [f'Signal {i + 1}' for i in range(n_signals)]

    for i, (sig, label) in enumerate(zip(signals, labels)):
        spectrum = np.abs(np.fft.fft(sig))[:n_components] / len(sig)
        axes[i].plot(spectrum, linewidth=2)
        axes[i].set_title(f'Fourier Spectrum: {label}', fontsize=14)
        axes[i].set_xlabel('Frequency Index')
        axes[i].set_ylabel('Amplitude')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(bottom=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, axes


def plot_PD_graph(graphs: List, edge_list: List[Tuple[int, int]],
                  n_col: int = 5, n_row: int = 5, axs=None,
                  max_dim: int = 1, superlevel: bool = False):
    """
    Plot persistence diagrams for graph filtrations.

    Parameters
    ----------
    graphs : List
        List of edge weight vectors.
    edge_list : List[Tuple[int, int]]
        List of all edges as (source, target) pairs.
    n_col, n_row : int
        Grid dimensions.
    axs : matplotlib axes array, optional
        Pre-existing axes.
    max_dim : int
        Maximum homology dimension.
    superlevel : bool
        If True, use superlevel set filtration.

    Returns
    -------
    matplotlib axes array
    """
    if axs is None:
        fig, axs = plt.subplots(n_row, n_col, figsize=(2. * n_col, 2.26 * n_row))
    axs = np.atleast_2d(axs)

    ph_model = GraphFiltrationPH(max_dim=max_dim, superlevel=superlevel)

    for i, edge_attr in enumerate(graphs[:(n_col * n_row)]):
        pers_info = ph_model(edge_list, edge_attr)

        ax = axs[i // n_col, i % n_col]
        ax.clear()
        for pi in pers_info:
            dim = int(getattr(pi, "dimension", 0))
            pts = pi.diagram.detach().cpu().numpy()

            tol = 1e-6
            m = np.isfinite(pts).all(axis=1) & (abs(pts[:, 1] - pts[:, 0]) > tol)
            arr = pts[m]
            gd_list = [(dim, (np.abs(d), np.abs(b))) for b, d in arr]
            gudhi.plot_persistence_diagram(gd_list, axes=ax, legend=False,
                                           fontsize=4, alpha=0.1)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Graph {i}", fontsize=8)

    return axs


# ------------------------------------------------------------------
# FitMonitor: live visualisation callback for TopologicalNMF.fit()
# ------------------------------------------------------------------

class FitMonitor:
    """
    Live visualisation monitor for ``TopologicalNMF.fit()``.

    Parameters
    ----------
    show : list of str, optional
        Which plots to display: 'loss', 'basis', 'PH'.
    interval : int
        Initial display interval in epochs (grows by 1.2x each update).
    grid : tuple of (int, int)
        ``(n_row, n_col)`` for basis and PH grids.
    PHmode : str
        'T' or 'V' for persistence diagram computation.
    superlevel : bool
        Whether to use superlevel set filtration for PH plots.

    Examples
    --------
    >>> from TopNMF.visualization import FitMonitor
    >>> monitor = FitMonitor(show=['loss', 'basis'], interval=50, grid=(2, 3))
    >>> model.fit(X, ..., monitor=monitor)
    """

    def __init__(self, show=None, interval=100, grid=(2, 3),
                 PHmode='T', superlevel=False):
        self.show = list(show or [])
        self.interval = interval
        self.grid = grid
        self.PHmode = PHmode
        self.superlevel = superlevel

        # Set during setup()
        self._n_features = None
        self._M = None
        self._tau = None
        self._complex_inputs = None
        self._data_shape = None
        self._use_embedding = False
        self._current_interval = interval

        # Display handles
        self._fig_loss = None
        self._ax_loss = None
        self._disp_loss = None
        self._fig_basis = None
        self._ax_basis = None
        self._disp_basis = None
        self._fig_ph = None
        self._ax_ph = None
        self._disp_ph = None

    def setup(self, *, n_features, embedding_dim, tau, complex_inputs=None,
              data_shape=None, use_embedding=False):
        """Initialise figures and display handles. Called once by ``fit()``."""
        self._n_features = n_features
        self._embedding_dim = embedding_dim
        self._tau = tau
        self._complex_inputs = complex_inputs
        self._data_shape = data_shape
        self._use_embedding = use_embedding
        self._current_interval = self.interval

        if not self.show:
            return

        if not _IPYTHON_AVAILABLE:
            print("Warning: show requires IPython/Jupyter. Plots disabled.")
            self.show = []
            return

        n_row, n_col = self.grid

        if 'loss' in self.show:
            self._fig_loss, self._ax_loss = plt.subplots(1, 1, figsize=(8, 5))
            self._disp_loss = display(self._fig_loss, display_id=True)

        if 'basis' in self.show:
            self._fig_basis, self._ax_basis = plt.subplots(
                n_row, n_col, figsize=(2.0 * n_col, 2.26 * n_row))
            self._disp_basis = display(self._fig_basis, display_id=True)

        if 'PH' in self.show:
            self._fig_ph, self._ax_ph = plt.subplots(
                n_row, n_col, figsize=(2.0 * n_col, 2.26 * n_row), squeeze=False)
            self._disp_ph = display(self._fig_ph, display_id=True)

    def update(self, epoch, model):
        """Update plots for the current epoch. Called each epoch by ``fit()``."""
        if not self.show:
            return

        interval = max(1, int(self._current_interval))
        if epoch % interval != 0:
            return

        V_np = model.V.detach().cpu().numpy().copy()
        n_row, n_col = self.grid
        is_graph = (self._complex_inputs is not None
                    and 'all_edges' in self._complex_inputs)

        if self._data_shape is not None:
            V_display = V_np.reshape(-1, *self._data_shape)
        else:
            V_display = V_np

        if 'loss' in self.show and self._disp_loss is not None:
            plot_loss(model.losses, ax=self._ax_loss)
            self._disp_loss.update(self._fig_loss)

        if 'basis' in self.show and self._disp_basis is not None:
            if is_graph:
                edge_list = self._complex_inputs['all_edges']
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                node_pos = self._complex_inputs.get('node_pos', None)
                plot_gallery_graph(V_np, edge_index, title='Basis',
                                   n_col=n_col, n_row=n_row,
                                   axs=self._ax_basis, pos=node_pos)
            else:
                plot_gallery(V_display, title='basis',
                             n_row=n_row, n_col=n_col,
                             axs=self._ax_basis)
            self._disp_basis.update(self._fig_basis)

        if 'PH' in self.show and self._disp_ph is not None:
            if is_graph:
                plot_PD_graph(
                    [V_np[i] for i in range(min(len(V_np), n_row * n_col))],
                    self._complex_inputs['all_edges'],
                    n_col=n_col, n_row=n_row,
                    axs=self._ax_ph, superlevel=self.superlevel)
            else:
                tau = (self._tau if self._tau is not None
                       else int(self._n_features / (2 * (self._embedding_dim + 1))))
                plot_persistence_diagrams(
                    V_display, n_col=n_col, n_row=n_row,
                    superlevel=self.superlevel, PHmode=self.PHmode,
                    embedding_dim=self._embedding_dim, tau=tau, axs=self._ax_ph,
                    use_embedding=self._use_embedding)
            self._disp_ph.update(self._fig_ph)

        self._current_interval = max(1, int(1.2 * self._current_interval))
