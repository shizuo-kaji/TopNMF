"""
Visualization Utilities

This module provides plotting functions for time series, persistence diagrams,
and NMF basis visualization.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import gudhi
import networkx as nx
from gudhi.point_cloud.timedelay import TimeDelayEmbedding
from typing import Optional, List, Tuple, Union

from .graph_filtration import GraphFiltrationPH


def plot_gallery(images, title: str = "", n_col: int = 5, n_row: int = 5,
                 cmap=plt.cm.gray, axs=None):
    """
    Plot a gallery of time series or basis vectors.

    Parameters
    ----------
    images : array-like
        Array of 1D vectors to plot
    title : str, optional
        Suptitle for the figure
    n_col : int, optional
        Number of columns in the grid
    n_row : int, optional
        Number of rows in the grid
    cmap : matplotlib colormap, optional
        Colormap (not used for line plots)
    axs : matplotlib axes array, optional
        Pre-existing axes to plot on

    Returns
    -------
    matplotlib axes array
        Array of axes used for plotting
    """
    if axs is None:
        fig, axs = plt.subplots(n_row, n_col, figsize=(2. * n_col, 2.26 * n_row))
        plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
        plt.suptitle(title, size=16)

    for i, comp in enumerate(images[:n_col * n_row]):
        if n_row != 1:
            ax = axs[i // n_col, i % n_col]
        else:
            ax = axs[i % n_col]

        ax.cla()  # Clear existing lines
        if len(comp.shape) == 1:
            ax.plot(comp)
        else:
            vmax = max(comp.max(), -comp.min())
            ax.imshow(comp, cmap=cmap,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)

        ax.set_xticks(())
        ax.set_yticks(())

    return axs



def plot_persistence_diagrams(data, n_col: int = 5, n_row: int = 5, superlevel: bool = False, PHmode: str ="V",
                               M: int = 1, tau: int = 1, axs=None,
                               center_func=None):
    """
    Plot persistence diagrams for multiple time series.

    Parameters
    ----------
    data : array-like
        Array of time series to compute persistence diagrams for
    n_col : int, optional
        Number of columns in the grid
    n_row : int, optional
        Number of rows in the grid
    superlevel : bool, optional
        Whether to compute superlevel set persistence
    PHmode : str, optional
        "V" for vertex-based cubical complex, otherwise top-dimensional cells
    M : int, optional
        Embedding dimension minus 1
    tau : int, optional
        Time delay for embedding
    axs : matplotlib axes array, optional
        Pre-existing axes to plot on
    center_func : callable, optional
        Function to center point clouds

    Returns
    -------
    matplotlib axes array
        Array of axes used for plotting
    """
    if axs is None:
        fig, axs = plt.subplots(n_row, n_col, figsize=(2. * n_col, 2.26 * n_row))

    for i, comp in enumerate(data[:n_col * n_row]):
        if len(comp.shape) > 1:
            sign = -1 if superlevel else 1
            if PHmode=="V":
                cubical_complex = gudhi.CubicalComplex(vertices=sign*comp)
            else:
                cubical_complex = gudhi.CubicalComplex(top_dimensional_cells=sign*comp)
            pd = cubical_complex.persistence()
        else:
            embedder = TimeDelayEmbedding(dim=M+1, delay=tau)
            embedded = embedder(comp)

            if center_func is not None:
                centered = center_func(embedded)
            else:
                centered = embedded

            rips_complex = gudhi.RipsComplex(points=centered).create_simplex_tree(max_dimension=2)
            pd = rips_complex.persistence()

        ax = axs[i // n_col, i % n_col]
        ax.clear()
        gudhi.plot_persistence_diagram(pd, axes=ax, legend=False, fontsize=4)
        ax.set_xticks(())
        ax.set_yticks(())

    return axs


def plot_loss(losses, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.clear()
    for key in losses.keys():
        ax.plot(losses[key], label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)
    return ax


def plot_time_series_comparison(original, reconstructed, basis_vectors,
                                 time_axis=None, save_path: Optional[str] = None):
    """
    Create comprehensive visualization comparing original and reconstructed signals.

    Parameters
    ----------
    original : np.ndarray
        Original time series
    reconstructed : np.ndarray
        Reconstructed time series from NMF
    basis_vectors : np.ndarray
        Learned basis vectors
    time_axis : np.ndarray, optional
        Time points for x-axis
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    tuple
        Figure and axes objects
    """
    n_basis = len(basis_vectors)
    fig, axes = plt.subplots(2 + n_basis, 1, figsize=(12, 4 * (2 + n_basis)))

    if time_axis is None:
        time_axis = np.arange(len(original))

    # Plot original
    axes[0].plot(time_axis, original, 'b-', linewidth=2)
    axes[0].set_title('Original Signal', fontsize=14)
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    # Plot reconstruction
    axes[1].plot(time_axis, reconstructed, 'r-', linewidth=2)
    axes[1].set_title('Reconstructed Signal', fontsize=14)
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)

    # Plot basis vectors
    for i, basis in enumerate(basis_vectors):
        axes[2 + i].plot(time_axis, basis, 'g-', linewidth=2)
        axes[2 + i].set_title(f'Basis Vector {i+1}', fontsize=14)
        axes[2 + i].set_ylabel('Amplitude')
        axes[2 + i].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes


def plot_fourier_spectrum(signals, labels=None, n_components: int = 50,
                          figsize=(12, 6), save_path: Optional[str] = None):
    """
    Plot Fourier spectrum of multiple signals.

    Parameters
    ----------
    signals : list of np.ndarray
        List of signals to analyze
    labels : list of str, optional
        Labels for each signal
    n_components : int, optional
        Number of frequency components to display
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    tuple
        Figure and axes objects
    """
    n_signals = len(signals)
    fig, axes = plt.subplots(1, n_signals, figsize=figsize)

    if n_signals == 1:
        axes = [axes]

    if labels is None:
        labels = [f'Signal {i+1}' for i in range(n_signals)]

    for i, (signal, label) in enumerate(zip(signals, labels)):
        spectrum = np.abs(np.fft.fft(signal))[:n_components]
        # Normalize by signal length
        spectrum = spectrum / len(signal)

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


def plot_time_series(images, title: str = "", n_col: int = 5, n_row: int = 5,
                     cmap=plt.cm.gray, axs=None):
    """
    Plot multiple time series in a grid layout.

    Parameters
    ----------
    images : array-like
        Array of 1D time series to plot
    title : str, optional
        Suptitle for the figure
    n_col : int, optional
        Number of columns in the grid
    n_row : int, optional
        Number of rows in the grid
    cmap : matplotlib colormap, optional
        Colormap (not used for line plots)
    axs : matplotlib axes array, optional
        Pre-existing axes to plot on

    Returns
    -------
    matplotlib axes array
        Array of axes used for plotting
    """
    if axs is None:
        fig, axs = plt.subplots(n_row, n_col, figsize=(2. * n_col, 2.26 * n_row))
        plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
        plt.suptitle(title, size=16)

    for i, comp in enumerate(images):
        if n_row != 1:
            ax = axs[i // n_col, i % n_col]
        else:
            ax = axs[i % n_col]

        ax.cla()  # Clear existing lines
        ax.plot(comp)
        ax.set_xticks(())
        ax.set_yticks(())

    return axs


def plot_gallery_graph(edge_values: Union[np.ndarray, torch.Tensor],
                       edge_index: torch.Tensor, title: str,
                       n_col: int, n_row: int, axs):
    """
    Plot graph visualizations for multiple basis vectors.

    Each subplot shows a graph where edge thickness represents edge weight
    from the corresponding basis vector.

    Parameters
    ----------
    edge_values : np.ndarray or torch.Tensor
        Edge weights matrix of shape (n_components, n_edges).
        Each row contains edge weights for one basis.
    edge_index : torch.LongTensor
        Edge index tensor of shape (2, n_edges).
        Each column is [source, target] for an edge.
    title : str
        Title prefix for each subplot
    n_col : int
        Number of columns in the subplot grid
    n_row : int
        Number of rows in the subplot grid
    axs : matplotlib axes array
        Pre-created axes array of shape (n_row, n_col)

    Returns
    -------
    matplotlib axes array
        Array of axes used for plotting
    """
    # Transpose edge_index from (2, n_edges) to (n_edges, 2)
    edge_index_pairs = edge_index.transpose(0, 1)

    # Compute node positions using spring layout on a temporary graph
    nodes = set(int(u.item()) for u, v in edge_index_pairs) | set(int(v.item()) for u, v in edge_index_pairs)
    G_temp = nx.Graph()
    G_temp.add_nodes_from(nodes)
    G_temp.add_edges_from([(int(u.item()), int(v.item())) for u, v in edge_index_pairs])
    pos = nx.spring_layout(G_temp, seed=42)

    # Number of basis vectors to plot
    num_basis = edge_values.shape[0]
    max_plots = n_row * n_col
    n_plots = min(num_basis, max_plots)

    # Initialize all subplot axes
    for idx in range(max_plots):
        r = idx // n_col
        c = idx % n_col
        ax = axs[r, c]
        ax.clear()

        if idx < n_plots:
            # Plot the idx-th basis
            weights = edge_values[idx]

            # Draw edges with weight > 0
            for j, (u, v) in enumerate(edge_index_pairs):
                w = weights[j]
                if isinstance(w, torch.Tensor):
                    w = w.item()
                if w <= 0:
                    continue
                ux, uy = pos[int(u.item())]
                vx, vy = pos[int(v.item())]
                ax.plot(
                    [ux, vx],
                    [uy, vy],
                    color='tab:gray',
                    linewidth=w * 5
                )
            ax.set_title(f"{title} {idx}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Empty subplot - hide frame
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

    return axs


def plot_PD_graph(graphs: List, edge_list: List[Tuple[int, int]],
                  n_col: int = 5, n_row: int = 5, axs=None,
                  max_dim: int = 1, superlevel: bool = False):
    """
    Plot persistence diagrams for graph filtrations.

    Computes persistent homology for each graph defined by edge weights
    and plots the resulting persistence diagrams.

    Parameters
    ----------
    graphs : List
        List of edge weight vectors. Each item has length = len(edge_list).
    edge_list : List[Tuple[int, int]]
        List of all edges as (source, target) pairs.
    n_col : int, optional
        Number of columns in the subplot grid
    n_row : int, optional
        Number of rows in the subplot grid
    axs : matplotlib axes array, optional
        Pre-created axes array. If None, creates new figure.
    max_dim : int, optional
        Maximum homology dimension to compute
    superlevel : bool, optional
        If True, use superlevel set filtration (negate values)

    Returns
    -------
    matplotlib axes array
        Array of axes used for plotting
    """
    if axs is None:
        fig, axs = plt.subplots(n_row, n_col, figsize=(2. * n_col, 2.26 * n_row))
    axs = np.atleast_2d(axs)

    ph_model = GraphFiltrationPH(max_dim=max_dim, superlevel=superlevel)

    for i, edge_attr in enumerate(graphs[: (n_col * n_row)]):
        # Compute persistence
        pers_info = ph_model(edge_list, edge_attr)

        # Plot persistence diagram
        ax = axs[i // n_col, i % n_col]
        ax.clear()
        for pi in pers_info:
            dim = int(getattr(pi, "dimension", 0))
            pts = pi.diagram.detach().cpu().numpy()

            # Filter out trivial points
            tol = 1e-6
            m = np.isfinite(pts).all(axis=1) & (abs(pts[:, 1] - pts[:, 0]) > tol)
            arr = pts[m]
            gd_list = [(dim, (np.abs(d), np.abs(b))) for b, d in arr]

            gudhi.plot_persistence_diagram(gd_list, axes=ax, legend=False, fontsize=4, alpha=0.1)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Graph {i}", fontsize=8)

    return axs