"""
Topological NMF Package

Non-negative Matrix Factorization with topological regularisation
via persistent homology.
"""

from .model import TopologicalNMF

from .losses import (
    ph_sparsity_loss,
    target_diagram_loss,
    weighted_persistence_loss,
    weighted_total_squared_persistence_loss,
    total_variation,
    clique_deviation_loss,
)

from .optim import (
    update_V,
    sparse_opt,
    sparse_opt_hoyer,
    project_rows_max_norm,
    scale_rows_max_norm,
)

from .utils import (
    sparsity_score,
    l1_l2_sq_ratio,
    svd_initialization,
    center_point_cloud,
    center_point_cloud_torch,
    periodicity_from_diagram,
    compute_periodicity_score,
    compute_persistence_diagram,
)

from .persistence import PersistenceInfo
from .persistence import CubicalComplex
from .persistence import GudhiVietorisRipsComplex
from .persistence import TimeDelayEmbeddingTorch
from .persistence import GraphFiltrationPH

from .visualization import (
    FitMonitor,
    plot_gallery,
    plot_persistence_diagrams,
    plot_loss,
    plot_time_series_comparison,
    plot_fourier_spectrum,
    plot_gallery_graph,
    plot_PD_graph,
)

from .signal_generation import (
    generate_ichimatsu_pattern,
    generate_signals,
    generate_edge_weighted_graph,
    normalize_signals,
)

__version__ = '1.0.0'  # Keep in sync with pyproject.toml::version
__all__ = [
    # Main class
    'TopologicalNMF',
    # Persistence
    'PersistenceInfo',
    'CubicalComplex',
    'GudhiVietorisRipsComplex',
    'TimeDelayEmbeddingTorch',
    'GraphFiltrationPH',
    # Loss functions
    'ph_sparsity_loss',
    'target_diagram_loss',
    'weighted_persistence_loss',
    'weighted_total_squared_persistence_loss',
    'clique_deviation_loss',
    'total_variation',
    # Optimization
    'update_V',
    'sparse_opt',
    'sparse_opt_hoyer',
    'project_rows_max_norm',
    'scale_rows_max_norm',
    # Utilities
    'sparsity_score',
    'l1_l2_sq_ratio',
    'svd_initialization',
    'center_point_cloud',
    'center_point_cloud_torch',
    'periodicity_from_diagram',
    'compute_periodicity_score',
    'compute_persistence_diagram',
    # Visualization
    'FitMonitor',
    'plot_gallery',
    'plot_persistence_diagrams',
    'plot_loss',
    'plot_time_series_comparison',
    'plot_fourier_spectrum',
    'plot_gallery_graph',
    'plot_PD_graph',
    # Signal generation
    'generate_ichimatsu_pattern',
    'generate_signals',
    'generate_edge_weighted_graph',
    'normalize_signals',
]
