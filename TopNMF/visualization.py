"""
Visualization Utilities

This module provides plotting functions for time series, persistence diagrams,
and NMF basis visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import gudhi
from gudhi.point_cloud.timedelay import TimeDelayEmbedding
from typing import Optional


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


def plot_loss(losses: dict, ax=None):
    """
    Plot training loss curves.

    Parameters
    ----------
    losses : dict
        Dictionary with loss names as keys and loss values as lists
    ax : matplotlib axis, optional
        Axis to plot on

    Returns
    -------
    matplotlib axis
        Axis used for plotting
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.clear()

    for key, values in losses.items():
        if key != 'lr':  # Don't plot learning rate with losses
            ax.plot(values, label=key)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

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
