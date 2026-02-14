# TopNMF: Topological Non-negative Matrix Factorization

TopNMF is a NumPy/PyTorch implementation of Non-negative Matrix Factorization (NMF) with topological regularization. The core `TopologicalNMF` class augments standard NMF loss with persistent homology penalties to capture periodic, structured, or sparse behaviors in time-series and image signals.

Written by Shizuo Kaji and Keunsu Kim

## Repository Layout

```
.
├── TopNMF/                      # Installable Python package
│   ├── __init__.py              # Re-exports public API (__version__ = 1.0.0)
│   ├── topological_nmf.py       # TopologicalNMF class
│   ├── losses.py                # PH loss functions (sparsity, target diagram, etc.)
│   ├── nmf_utils.py             # Sparse NMF updates, SVD init, total variation
│   ├── topological_utils.py     # Time-delay embedding + persistence helpers
│   ├── cubical_complex.py       # CubicalComplex for image/grid data
│   ├── graph_filtration.py      # GraphFiltrationPH for graph-based PH
│   ├── signal_generation.py     # Synthetic datasets for demos/tests
│   └── visualization.py         # Plotting utilities for NMF + TDA outputs
├── tests/                       # Pytest unit tests
├── notebook/                    # Example and appendix notebooks
│   ├── example.ipynb            # Hands-on walkthrough of the pipeline
│   ├── Example 1.ipynb
│   ├── Example 2.ipynb
│   ├── Example 3.ipynb
│   ├── Appendix 1.ipynb
│   └── Appendix 2.ipynb
├── pyproject.toml               # Package configuration
├── AGENTS.md                    # AI agent development guide
└── README.md                    # This document
```

## Installation

```bash
# Install from source with pip
pip install -e .

# Or install with dev dependencies for testing
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from TopNMF import TopologicalNMF, generate_signals, create_time_array

# Generate synthetic signals
t = create_time_array(0, 4 * np.pi, 200)
signals = generate_signals(t, kind="cosine")
X = np.stack(list(signals.values()))

# Fit TopologicalNMF
model = TopologicalNMF(n_components=2)
model.fit(X, n_iterations=500, lambda_top=0.01)

# Get results
V = model.get_components()
losses = model.get_losses()
```

See the [example notebooks](notebook/) for a complete walkthrough.

## Module Documentation

### 1. `topological_nmf.py`

Main class implementing topological NMF.

**Key Class**: `TopologicalNMF`

**Key Methods**:
- `fit(X, ...)`: Fit model to data with topological constraints
- `transform(X)`: Get coefficient matrix
- `inverse_transform(W)`: Reconstruct data from coefficients
- `get_components()`: Get learned basis vectors
- `get_losses()`: Get training loss history

### 2. `losses.py`

Loss functions for topological NMF optimization.

**Key Functions**:
- `ph_sparsity_loss()`: L1²/L2² ratio for persistence diagrams
- `target_diagram_loss()`: Compare diagrams to target diagrams
- `weighted_persistence_loss()`: Weighted persistence with boundary penalty
- `reconstruction_loss()`: NMF reconstruction error
- `sparsity_loss()`: Hoyer sparsity or L1²/L2² loss

### 3. `nmf_utils.py`

Core NMF optimization utilities.

**Key Functions**:
- `update_V()`: Update basis matrix with sparsity constraints
- `sparse_opt()`: L1/L2 constrained optimization
- `sparse_opt_hoyer()`: Hoyer's projection algorithm
- `sparsity_score()`: Compute Hoyer sparsity measure
- `svd_initialization()`: SVD-based initialization
- `total_variation()`: Total variation regularization

### 4. `topological_utils.py`

Topological data analysis functions.

**Key Functions**:
- `center_point_cloud()`: Center and normalize point clouds
- `center_point_cloud_torch()`: PyTorch version with gradient support
- `compute_periodicity_score()`: Compute periodicity scores
- `compute_persistence_diagram()`: Full persistence diagram computation

**Key Classes**:
- `TimeDelayEmbeddingTorch`: PyTorch-based time delay embedding

### 5. `cubical_complex.py`

Cubical complex for structured data (images, grids).

**Key Class**: `CubicalComplex`
- Differentiable persistence diagrams for 2D/3D data
- Supports sublevel and superlevel filtrations

### 6. `graph_filtration.py`

Graph-based persistent homology.

**Key Class**: `GraphFiltrationPH`
- Computes persistence on graph filtrations
- Edge weights define filtration values

### 7. `visualization.py`

Plotting and visualization utilities.

**Key Functions**:
- `plot_gallery()`: Plot multiple signals/images in grid
- `plot_persistence_diagrams()`: Plot persistence diagrams
- `plot_loss()`: Plot training loss curves
- `plot_time_series_comparison()`: Compare original vs reconstructed
- `plot_fourier_spectrum()`: Plot Fourier spectra
- `plot_time_series()`: Plot time series in grid layout
- `plot_gallery_graph()`: Visualize graph basis vectors
- `plot_PD_graph()`: Plot persistence diagrams for graphs

### 8. `signal_generation.py`

Synthetic signal generation for testing.

**Key Functions**:
- `generate_signals()`: Cosine or triangle signals
- `generate_mixed_periodic_nonperiodic()`: Mixed components
- `generate_noisy_periodic()`: Noisy periodic signals
- `generate_complex_signals()`: Various complex patterns
- `normalize_signals()`: Signal normalization
- `create_time_array()`: Time point generation

## Key Parameters

### TopologicalNMF.fit() Parameters

**Loss Weights**:
- `lambda_apx`: Reconstruction loss weight (default: 1.0)
- `lambda_top`: Topological loss weight (default: 0.001)
- `lambda_spa_V`: Basis sparsity weight (default: 0.0)
- `lambda_spa_W`: Coefficient sparsity weight (default: 0.0)
- `lambda_tv`: Total variation weight (default: 0.0)

**Optimization**:
- `lr`: Learning rate (default: 0.005)
- `n_iterations`: Maximum iterations (default: 1000)
- `gd_iter`: Gradient descent steps per epoch (default: 1)
- `mu_iter`: Multiplicative update steps per epoch (default: 0)

**Topological**:
- `M`: Embedding dimension parameter (default: 4)
- `tau`: Time delay (default: auto-computed)
- `PH_dims`: Homology dimensions to use (default: [1])
- `target_diagrams`: Target persistence diagrams
- `target_periodicity`: Target periodicity scores

## License

MIT
