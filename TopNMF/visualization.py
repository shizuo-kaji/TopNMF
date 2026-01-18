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
                               center_func=None, use_embedding: bool = True):
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
        if not use_embedding or len(comp.shape) > 1:
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
    """
    Plot losses.
    If PH loss has negative values:
        - Left y-axis (log): approx, lr
        - Right y-axis (linear): PH
    If PH loss is all non-negative:
        - Single y-axis (log): approx, lr, PH
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.clear()

    # Fixed colors
    colors = {
        'approx': 'C1',  # Orange
        'lr':     'C2',  # Green
        'PH':     'C0',  # Blue
    }

    # Check PH values
    ph_values = np.asarray(losses.get('PH', []))
    has_negative_ph = np.any(ph_values < 0) if len(ph_values) > 0 else False

    # Handle secondary axis
    ax2 = None
    if len(fig.axes) > 1 and fig.axes[0] == ax:
         ax2 = fig.axes[1]

    if has_negative_ph:
        # Dual axis mode
        if ax2 is None:
            ax2 = ax.twinx()
        else:
            ax2.set_visible(True)
            ax2.clear()
            ax2.patch.set_visible(False) # Make sure it's transparent? twinx usually is.

    elif ax2 is not None:
        # Single axis mode: hide secondary axis if it exists
        ax2.set_visible(False)
        ax2.clear()

    # ---------- Main Axis (Left) ----------
    # Plot approx and lr
    for key in ['approx', 'lr']:
        if key in losses:
            y = np.asarray(losses[key])
            ax.plot(y, label=key, color=colors.get(key, 'k'))

    # If single axis mode, plot PH here too
    if not has_negative_ph and 'PH' in losses and len(ph_values) > 0:
        ax.plot(ph_values, label='PH', color=colors['PH'])

    ax.set_xlabel("Epoch")
    if not has_negative_ph and 'PH' in losses:
        ax.set_ylabel("Loss (log scale)", size=12)
    else:
        ax.set_ylabel("Loss (approx, lr)", size=12)
    
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # ---------- Secondary Axis (Right) - Only if negative PH ----------
    if has_negative_ph and 'PH' in losses:
        # Plot on ax2
        ax2.plot(ph_values, label='PH', color=colors['PH'])

        ax2.set_ylabel("PH Loss (linear)", size=12)
        
        # Ensure label and ticks are on the right
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")

        # Adjust y-limits for PH with padding
        if len(ph_values) > 0:
            ph_min, ph_max = float(np.min(ph_values)), float(np.max(ph_values))
            if ph_min == ph_max:
                pad = 0.1 * (abs(ph_min) + 1.0)
            else:
                pad = 0.1 * (ph_max - ph_min)
            ax2.set_ylim(ph_min - pad, ph_max + pad)

        # Scientific notation
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))

    # ---------- Legend ----------
    lines1, labels1 = ax.get_legend_handles_labels()
    if has_negative_ph and ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax.legend(lines1, labels1, loc='upper right')

    return ax

def plot_gallery_graph(edge_values: Union[np.ndarray, torch.Tensor],
                       edge_index: torch.Tensor, title: str,
                       n_col: int, n_row: int, axs, pos=None):
    """
    Plot graph visualizations for multiple basis vectors.
    """
    # Transpose edge_index from (2, n_edges) to (n_edges, 2)
    edge_index_pairs = edge_index.transpose(0, 1)

    # Compute node positions if not provided
    if pos is None:
        nodes = set(int(u.item()) for u, v in edge_index_pairs) | set(int(v.item()) for u, v in edge_index_pairs)
        G_temp = nx.Graph()
        G_temp.add_nodes_from(nodes)
        G_temp.add_edges_from([(int(u.item()), int(v.item())) for u, v in edge_index_pairs])
        pos = nx.spring_layout(G_temp, seed=42) # Seed for reproducibility

    # Number of basis vectors to plot
    num_basis = edge_values.shape[0]
    max_plots = n_row * n_col
    n_plots = min(num_basis, max_plots)

    # Initialize all subplot axes
    for idx in range(max_plots):
        if n_row > 1:
            ax = axs[idx // n_col, idx % n_col]
        else:
            ax = axs[idx % n_col]
            
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
            # Draw nodes
            for node, (x, y) in pos.items():
                 ax.plot(x, y, 'o', color='black', markersize=2)
                 
            ax.set_title(f"{title} {idx}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')

    return axs


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