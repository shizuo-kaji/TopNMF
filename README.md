# TopNMF: Topological Non-negative Matrix Factorization

TopNMF is a NumPy/PyTorch implementation of Non-negative Matrix Factorization (NMF) with topological regularisation. The core `TopologicalNMF` class augments standard NMF loss with persistent homology penalties to capture periodic, structured, or sparse behaviours in time-series, image, and graph data.

Written by Shizuo Kaji and Keunsu Kim

## Repository Layout

```
.
├── TopNMF/                        # Installable Python package
│   ├── __init__.py                # Re-exports public API (__version__ = 1.0.0)
│   ├── model.py                   # TopologicalNMF class
│   ├── losses.py                  # PH loss functions (sparsity, target diagram, etc.)
│   ├── optim.py                   # Sparse NMF updates (update_V, sparse_opt, sparse_opt_hoyer)
│   ├── utils.py                   # Sparsity score, SVD init, periodicity helpers
│   ├── persistence/               # Persistence computation backends
│   │   ├── __init__.py            # Unified PersistenceInfo type
│   │   ├── cubical.py             # CubicalComplex (requires cripser)
│   │   ├── rips.py                # GudhiVietorisRipsComplex + TimeDelayEmbeddingTorch
│   │   └── graph.py               # GraphFiltrationPH for edge-weighted graphs
│   ├── signal_generation.py       # Synthetic datasets for demos/tests
│   └── visualization.py           # Plotting utilities + FitMonitor callback
├── tests/                         # Pytest unit tests
├── notebook/                      # Example notebooks
│   ├── 1D Signal.ipynb            # Time-series decomposition (Rips + cubical)
│   ├── 2D Image.ipynb             # Image decomposition (Hangul + ichimatsu)
│   └── Edge-weighted Graph.ipynb  # Graph decomposition (overlapping cliques)
├── pyproject.toml                 # Package configuration
├── AGENTS.md                      # AI agent development guide
└── README.md                      # This document
```

## Installation

```bash
# Install from source with pip
pip install -e .

# Or install with dev dependencies for testing
pip install -e ".[dev]"
```

### Optional dependencies

- **cripser** -- required for `CubicalComplex` (image/grid persistence). Install separately; the rest of the package works without it.

## Quick Start

### Basic Usage

```python
import numpy as np
from TopNMF import TopologicalNMF, generate_signals, create_time_array

# Generate synthetic signals
t = create_time_array(0, 4 * np.pi, 200)
signals = generate_signals(t, kind="cosine")
X = np.stack(list(signals.values()))

# Fit TopologicalNMF
model = TopologicalNMF(n_components=2, use_embedding=True)
model.fit(X, n_iterations=500, lambda_top=0.01)

# Get results
V = model.get_components()
losses = model.get_losses()
```

### Live visualisation with FitMonitor

```python
from TopNMF.visualization import FitMonitor

monitor = FitMonitor(
    show=["loss", "basis", "PH"],
    interval=50,
    grid=(1, 3),
)

model.fit(X, n_iterations=5000, lambda_top=0.01, monitor=monitor)
```

See the [example notebooks](notebook/) for complete walkthroughs.

## Module Documentation

### `model.py`

Main class implementing topological NMF.

**Key Class**: `TopologicalNMF`

**Key Methods**:
- `fit(X, ...)`: Fit model to data with topological constraints
- `transform(X)`: Get coefficient matrix
- `inverse_transform(W)`: Reconstruct data from coefficients
- `get_components()`: Get learned basis vectors
- `get_losses()`: Get training loss history

### `losses.py`

Loss functions for topological NMF optimisation.

**Key Functions**:
- `ph_sparsity_loss()`: L1²/L2² ratio for persistence diagrams
- `target_diagram_loss()`: Compare diagrams to target diagrams
- `weighted_persistence_loss()`: Weighted persistence with boundary penalty
- `clique_deviation_loss()`: Penalise clique deviation in graph bases
- `total_variation()`: Total variation regularisation (1-D and 2-D)

### `optim.py`

Core NMF optimisation utilities.

**Key Functions**:
- `update_V()`: Update basis matrix with sparsity constraints
- `sparse_opt()`: L1/L2 constrained optimisation
- `sparse_opt_hoyer()`: Hoyer's projection algorithm

### `utils.py`

General-purpose helpers.

**Key Functions**:
- `sparsity_score()`: Compute Hoyer sparsity measure
- `svd_initialization()`: SVD-based NMF initialisation
- `center_point_cloud()` / `center_point_cloud_torch()`: Center and normalise point clouds
- `compute_periodicity_score()`: Compute periodicity from persistence
- `compute_persistence_diagram()`: Full persistence diagram computation

### `persistence/`

Persistence computation backends sharing a unified `PersistenceInfo` named-tuple.

- **`cubical.py`** -- `CubicalComplex`: differentiable persistence for 2-D/3-D grids (requires cripser)
- **`rips.py`** -- `GudhiVietorisRipsComplex`: Vietoris-Rips complex; `TimeDelayEmbeddingTorch`: differentiable time-delay embedding
- **`graph.py`** -- `GraphFiltrationPH`: persistence on edge-weighted graph filtrations

### `visualization.py`

Plotting utilities and live training callback.

**Key Functions**:
- `plot_gallery()`: Plot multiple signals/images in a grid
- `plot_persistence_diagrams()`: Plot persistence diagrams
- `plot_loss()`: Plot training loss curves
- `plot_time_series_comparison()`: Compare original vs reconstructed
- `plot_fourier_spectrum()`: Plot Fourier spectra
- `plot_gallery_graph()`: Visualise graph basis vectors
- `plot_PD_graph()`: Plot persistence diagrams for graphs

**Key Class**: `FitMonitor` -- live visualisation callback for `TopologicalNMF.fit()`

### `signal_generation.py`

Synthetic signal generation for testing.

**Key Functions**:
- `generate_signals()`: Cosine or triangle signals
- `generate_mixed_periodic_nonperiodic()`: Mixed components
- `generate_noisy_periodic()`: Noisy periodic signals
- `generate_complex_signals()`: Various complex patterns
- `normalize_signals()`: Signal normalisation
- `create_time_array()`: Time point generation
- `create_ichimatsu_pattern()`: Checkerboard pattern images

## Key Parameters

### TopologicalNMF.fit()

**Loss Weights**:
- `lambda_apx`: Reconstruction loss weight (default: 1.0)
- `lambda_top`: Topological loss weight (default: 0.001)
- `lambda_spa_V`: Basis sparsity weight (default: 0.0)
- `lambda_spa_W`: Coefficient sparsity weight (default: 0.0)
- `lambda_tv`: Total variation weight (default: 0.0)

**Optimisation**:
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

**Visualisation**:
- `monitor`: A `FitMonitor` instance for live plots (default: None)

## License

MIT
