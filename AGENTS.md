# AGENTS.md

This file provides guidance for AI agents working with the TopNMF codebase.

## Project Overview

TopNMF is a Python package implementing Non-negative Matrix Factorization (NMF) with topological regularization via persistent homology. It combines classical NMF with Topological Data Analysis (TDA) to learn basis vectors that capture periodic and structured patterns in time-series and image data.

## Repository Structure

```
TopNMF/
├── TopNMF/                      # Main package
│   ├── __init__.py              # Public API exports
│   ├── topological_nmf.py       # Core TopologicalNMF class
│   ├── losses.py                # PH loss functions
│   ├── nmf_utils.py             # NMF optimization utilities
│   ├── topological_utils.py     # TDA utilities (embedding, PH computation)
│   ├── cubical_complex.py       # Cubical complex for images/grids
│   ├── graph_filtration.py      # Graph-based persistent homology
│   ├── signal_generation.py     # Synthetic signal generation
│   └── visualization.py         # Plotting utilities
├── tests/                       # Pytest unit tests
├── example.ipynb                # Usage walkthrough
├── pyproject.toml               # Package configuration
└── README.md                    # Documentation
```

## Key Files

| File | Purpose |
|------|---------|
| `TopNMF/topological_nmf.py` | Main `TopologicalNMF` class with fit/transform methods |
| `TopNMF/losses.py` | Loss functions: `ph_sparsity_loss`, `target_diagram_loss`, `weighted_persistence_loss` |
| `TopNMF/nmf_utils.py` | Core NMF: `update_V`, `sparse_opt`, `sparse_opt_hoyer`, `svd_initialization` |
| `TopNMF/topological_utils.py` | TDA: `TimeDelayEmbeddingTorch`, `center_point_cloud`, `compute_periodicity_score` |
| `TopNMF/cubical_complex.py` | `CubicalComplex` class for image persistence |
| `TopNMF/graph_filtration.py` | `GraphFiltrationPH` class for graph persistence |

## Development Guidelines

### Code Style

- Use type hints for function signatures
- Follow NumPy docstring conventions
- Keep functions focused and single-purpose
- Use PyTorch tensors for gradient-enabled operations

### Key Dependencies

- `torch`: Tensor operations and automatic differentiation
- `gudhi`: Persistence computation (Vietoris-Rips, cubical complexes)
- `ripser`: Fast Vietoris-Rips persistence
- `torch-topological`: Differentiable topology operations

### Adding New Loss Functions

Add to `TopNMF/losses.py` and export in `TopNMF/__init__.py`:

```python
def new_loss(diagrams, **kwargs):
    """
    Docstring with parameters and returns.
    """
    # Implementation using PyTorch for gradient support
    return loss_value
```

### Adding New Complex Types

Inherit from `nn.Module` and implement `forward()` method that returns persistence information compatible with existing loss functions.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=TopNMF

# Run specific test file
pytest tests/test_nmf_utils.py -v
```

### Test Files

- `tests/test_nmf_utils.py`: NMF utility function tests
- `tests/test_signal_generation.py`: Signal generation tests
- `tests/test_topological_utils.py`: TDA utility tests
- `tests/conftest.py`: Shared pytest fixtures

### Writing Tests

Use fixtures from `conftest.py`:

```python
def test_example(time_array, sample_signals):
    # time_array and sample_signals are fixtures
    result = some_function(sample_signals)
    assert result.shape == expected_shape
```

## Common Tasks

### Running the Example

```bash
jupyter notebook example.ipynb
```

### Installing for Development

```bash
pip install -e ".[dev]"
```

### Checking Public API

All public exports are listed in `TopNMF/__init__.py` under `__all__`.

## Architecture Notes

### Optimization Strategy

The `TopologicalNMF.fit()` method uses a hybrid approach:
1. **Multiplicative updates** (via `mu_iter`): Standard NMF updates preserving non-negativity
2. **Gradient descent** (via `gd_iter`): AdamW optimizer for topological losses

### Time Delay Embedding

For 1D time series, `TimeDelayEmbeddingTorch` creates point clouds from scalar signals:
- Input: `(T,)` tensor
- Output: `(T - (d-1)*tau, d)` tensor
- Used to reconstruct attractor manifold for TDA

### Persistence Computation

Three complex types supported:
1. **Vietoris-Rips**: Default for time series (via `gudhi` or `ripser`)
2. **Cubical**: For images/grids (via `CubicalComplex`)
3. **Graph**: For graph-structured data (via `GraphFiltrationPH`)

## Debugging Tips

### Loss Not Decreasing

- Check `lambda_top` weight (start with small values like 0.001)
- Verify embedding parameters `M` and `tau` are appropriate for signal length
- Ensure input data is normalized

### CUDA Errors

- Check tensor device consistency with `model.device`
- Some gudhi operations require CPU tensors

### Persistence Computation Issues

- Verify point cloud is not degenerate (all same point)
- Check `PH_dims` matches expected homology dimensions
