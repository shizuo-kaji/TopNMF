# TopNMF: Topological Non-negative Matrix Factorization

TopNMF is a NumPy/PyTorch implementation of Non-negative Matrix Factorization (NMF) with
topological regularisation. The core `TopologicalNMF` class augments standard NMF loss
with persistent-homology penalties (Vietoris–Rips or Cubical complexes) so that the
learned basis captures periodic, structured, or sparse behaviours in time-series or image
signals. This repository also ships utilities for signal generation, topological feature
engineering, and visualization notebooks.

## Repository Layout

```
.
├── TopNMF/                   # Installable Python package
│   ├── __init__.py           # Re-exports public API (__version__ = 1.0.0)
│   ├── topological_nmf.py    # TopologicalNMF class + PH loss helpers
│   ├── nmf_utils.py          # Sparse NMF updates, SVD init, total variation
│   ├── topological_utils.py  # Time-delay embedding + persistence helpers
│   ├── signal_generation.py  # Synthetic datasets for demos/tests
│   └── visualization.py      # Plotting utilities for NMF + TDA outputs
├── example.ipynb             # Hands-on walkthrough of the pipeline
└── README.md                 # This document
```

## Installation

```bash
# 1. Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# 2. Install required libraries
uv pip install numpy scipy torch matplotlib tqdm scikit-learn ripser gudhi torch-topological

# 3. Make the package importable (run from repo root)
export PYTHONPATH="$PWD:$PYTHONPATH"  # or add the path permanently in your shell profile
```

## Quick Start

### Basic Usage

Look at the example [Jupyter notebook](example.ipynb) _example.ipynb_.

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

### 2. `nmf_utils.py`

Core NMF optimization utilities.

**Key Functions**:
- `update_V()`: Update basis matrix with sparsity constraints
- `sparse_opt()`: L1/L2 constrained optimization
- `sparse_opt_hoyer()`: Hoyer's projection algorithm
- `sparsity_score()`: Compute Hoyer sparsity measure
- `svd_initialization()`: SVD-based initialization
- `total_variation()`: Total variation regularization

### 3. `topological_utils.py`

Topological data analysis functions.

**Key Functions**:
- `center_point_cloud()`: Center and normalize point clouds
- `compute_persistence_score()`: Compute periodicity scores
- `compute_persistence_diagram()`: Full persistence diagram computation

**Key Classes**:
- `TimeDelayEmbeddingTorch`: PyTorch-based time delay embedding

### 4. `visualization.py`

Plotting and visualization utilities.

**Key Functions**:
- `plot_gallery()`: Plot multiple data in grid
- `plot_persistence_diagrams()`: Plot persistence diagrams
- `plot_loss()`: Plot training loss curves
- `plot_time_series_comparison()`: Compare original vs reconstructed
- `plot_fourier_spectrum()`: Plot Fourier spectra

### 5. `signal_generation.py`

Synthetic signal generation for testing.

**Key Functions**:
- `generate_triangle_signals()`: Triangle-like signals
- `generate_mixed_periodic_nonperiodic()`: Mixed components
- `generate_noisy_periodic()`: Noisy periodic signals
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
- `target_score`: Target persistence scores per component


## License

MIT

---

**Version**: 0.0.1
**Last Updated**: 2025-10-26
