"""
Topological NMF Package

A refactored implementation of Non-negative Matrix Factorization with
topological constraints using persistent homology.
"""

from .topological_nmf import TopologicalNMF
from .losses import (
    ph_sparsity_loss,
    target_diagram_loss,
    weighted_persistence_loss,
    reconstruction_loss,
    sparsity_loss,
    total_variation,
    clique_deviation_loss,
)
from .nmf_utils import (
    update_V,
    sparse_opt,
    sparse_opt_hoyer,
    sparsity_score,
    svd_initialization,
)
from .topological_utils import (
    center_point_cloud,
    center_point_cloud_torch,
    TimeDelayEmbeddingTorch,
    GudhiVietorisRipsComplex,
    compute_periodicity_score,
    compute_persistence_diagram
)
from .visualization import (
    plot_gallery,
    plot_persistence_diagrams,
    plot_loss,
    plot_time_series_comparison,
    plot_fourier_spectrum,
    plot_time_series,
    plot_gallery_graph,
    plot_PD_graph,
)
from .signal_generation import (
    create_ichimatsu_pattern,
    generate_signals,
    generate_mixed_periodic_nonperiodic,
    generate_noisy_periodic,
    generate_complex_signals,
    generate_noisy_signals,
    generate_step_signals,
    normalize_signals,
    create_time_array
)

from .graph_filtration import GraphFiltrationPH
from .cubical_complex import CubicalComplex, PersistenceInfo

__version__ = '1.0.0'
__all__ = [
    # Main class
    'TopologicalNMF',
    # Loss functions
    'ph_sparsity_loss',
    'target_diagram_loss',
    'weighted_persistence_loss',
    'reconstruction_loss',
    'sparsity_loss',
    'clique_deviation_loss',
    # NMF utilities
    'update_V',
    'sparse_opt',
    'sparse_opt_hoyer',
    'sparsity_score',
    'svd_initialization',
    'total_variation',
    'center_point_cloud',
    'center_point_cloud_torch',
    'TimeDelayEmbeddingTorch',
    'GudhiVietorisRipsComplex',
    'compute_periodicity_score',
    'compute_persistence_diagram',
    'plot_gallery',
    'plot_persistence_diagrams',
    'plot_loss',
    'plot_time_series_comparison',
    'plot_fourier_spectrum',
    'plot_time_series',
    'plot_gallery_graph',
    'plot_PD_graph',
    'create_ichimatsu_pattern',
    'generate_signals',
    'generate_mixed_periodic_nonperiodic',
    'generate_noisy_periodic',
    'generate_complex_signals',
    'generate_noisy_signals',
    'generate_step_signals',
    'normalize_signals',
    'create_time_array',
    'GraphFiltrationPH',
    'CubicalComplex',
    'PersistenceInfo',
]
