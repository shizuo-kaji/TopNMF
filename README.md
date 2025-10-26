# Topological NMF - Refactored Implementation

Non-negative Matrix Factorization (NMF) with topological constraints using persistent homology

## Project Structure

```
refactored/
├── __init__.py                 # Package initialization
├── topological_nmf.py          # Main TopologicalNMF class
├── nmf_utils.py                # NMF optimization utilities
├── topological_utils.py        # TDA and embedding functions
├── visualization.py            # Plotting functions
├── signal_generation.py        # Signal generation utilities
├── example_notebook.py         # Clean usage example
└── README.md                   # This file
```

## Quick Start

### Basic Usage

```python
import numpy as np
from topological_nmf import TopologicalNMF
from signal_generation import generate_triangle_signals, create_time_array

# Generate signals
t = create_time_array(start=0, stop=2*np.pi, n_points=100)
signals = generate_triangle_signals(t)
X = np.array([sig / np.max(sig) for sig in signals.values()])

# Initialize and fit model
model = TopologicalNMF(n_components=2, device='cpu')
model.fit(
    X,
    n_iterations=10000,
    lr=0.005,
    lambda_top=0.001,
    target_score=[1.0, 0.0]
)

# Get results
basis_vectors = model.get_components()
coefficients = model.transform(X)
reconstructed = model.inverse_transform()
```

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
- `plot_gallery()`: Plot multiple time series in grid
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

## Installation Requirements

```bash
pip install numpy scipy torch matplotlib seaborn
pip install scikit-learn gudhi ripser torch-topological tqdm
```

## Example Workflow

See `example_notebook.py` for a complete example. Basic workflow:

1. **Generate or load data**
```python
from signal_generation import generate_triangle_signals, create_time_array
t = create_time_array()
signals = generate_triangle_signals(t)
```

2. **Prepare data**
```python
X = np.array([sig / np.max(sig) for sig in signals.values()])
```

3. **Fit model**
```python
model = TopologicalNMF(n_components=2)
model.fit(X, n_iterations=10000, lambda_top=0.001)
```

4. **Analyze results**
```python
from visualization import plot_fourier_spectrum
basis = model.get_components()
plot_fourier_spectrum(basis)
```

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
