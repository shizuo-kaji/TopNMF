"""
Topological NMF Package

A refactored implementation of Non-negative Matrix Factorization with
topological constraints using persistent homology.
"""

from .topological_nmf import (
    TopologicalNMF,
    ph_sparsity_loss
)
from .nmf_utils import (
    update_V,
    sparse_opt,
    sparse_opt_hoyer,
    sparsity_score,
    svd_initialization,
    total_variation
)
from .topological_utils import (
    center_point_cloud,
    center_point_cloud_torch,
    TimeDelayEmbeddingTorch,
    compute_periodicity_score,
    compute_persistence_diagram
)
from .visualization import (
    plot_gallery,
    plot_persistence_diagrams,
    plot_loss,
    plot_time_series_comparison,
    plot_fourier_spectrum
)
from .signal_generation import (
    generate_triangle_signals,
    generate_mixed_periodic_nonperiodic,
    generate_noisy_periodic,
    generate_complex_signals,
    generate_noisy_signals,
    generate_step_signals,
    normalize_signals,
    create_time_array
)

__version__ = '1.0.0'
__all__ = [
    'TopologicalNMF',
    'ph_sparsity_loss',
    'update_V',
    'sparse_opt',
    'sparse_opt_hoyer',
    'sparsity_score',
    'svd_initialization',
    'total_variation',
    'center_point_cloud',
    'center_point_cloud_torch',
    'TimeDelayEmbeddingTorch',
    'compute_periodicity_score',
    'compute_persistence_diagram',
    'plot_gallery',
    'plot_persistence_diagrams',
    'plot_loss',
    'plot_time_series_comparison',
    'plot_fourier_spectrum',
    'generate_triangle_signals',
    'generate_mixed_periodic_nonperiodic',
    'generate_noisy_periodic',
    'generate_complex_signals',
    'generate_noisy_signals',
    'generate_step_signals',
    'normalize_signals',
    'create_time_array',
]
