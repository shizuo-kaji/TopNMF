# AGENTS.md

**Current Status**: Active Development
**Last Updated**: 2026-02-18

This document serves as the primary source of truth for AI agents working on the `TopNMF` repository. Read this before proposing or making changes.

## 1. Mission & Context

**TopNMF** is a Python library that combines Non-negative Matrix Factorization (NMF) with Topological Data Analysis (TDA).
- **Goal**: Extract meaningful patterns (basis vectors) from data while enforcing topological constraints (e.g., sparsity, periodicity) using persistent homology.
- **Core Mechanism**: We use a hybrid optimization approach.
    - **NMF**: Standard multiplicative updates or projected gradient descent for non-negativity.
    - **Topology**: PyTorch-based gradient descent on the persistence diagram to minimize topological losses.

## 2. Environment Setup

- **Installation**:
  ```bash
  pip install -e ".[dev]"
  ```
- **Dependencies**: `numpy`, `torch`, `gudhi`, `scikit-learn`, `matplotlib`.
- **Optional**: `cripser` (for `CubicalComplex`).

## 3. Codebase Map

The package is structured as follows. Use this map to locate logic.

```text
TopNMF/
├── TopNMF/                      # Main Package
│   ├── __init__.py              # Public API
│   ├── model.py                 # [CORE] TopologicalNMF class. The main estimator.
│   ├── losses.py                # [CORE] Topological loss functions (ph_sparsity, etc.)
│   ├── optim.py                 # [CORE] NMF optimization (update_V, sparse_opt)
│   ├── visualization.py         # Plotting & FitMonitor callback
│   ├── utils.py                 # General helpers (sparsity_score, etc.)
│   ├── signal_generation.py     # Synthetic data generators
│   └── persistence/             # [COMPLEX] TDA Backends
│       ├── __init__.py
│       ├── rips.py              # GudhiVietorisRipsComplex & TimeDelayEmbeddingTorch
│       ├── cubical.py           # CubicalComplex (images)
│       └── graph.py             # GraphFiltrationPH (graphs)
├── tests/                       # Pytest suite
├── notebook/                    # Example notebooks
├── pyproject.toml               # Config
└── README.md                    # User facing documentation
```

### Key Files
- **`model.py`**: Contains `TopologicalNMF.fit()`. This is the loop where NMF updates and Topological gradient steps are interleaved.
- **`persistence/rips.py`**: Handles differentiation through the persistence calculation using `torch.autograd`. **Critical for topological loss.**
- **`losses.py`**: Contains `ph_sparsity_loss`, `target_diagram_loss`, etc.

## 4. Development Guidelines

### 4.1. Coding Standards
- **Type Hints**: STRICTLY REQUIRED for all function signatures.
- **Docstrings**: NumPy style. Must include Parameters and Returns.
- **Imports**: Absolute imports within the package preferred (or relative `from . import`).
- **Tensors**:
    - Use `torch.Tensor` for anything that needs gradients.
    - Ensure `device` (cpu/cuda) is consistent.

### 4.2. Mathematical Integrity
- **Non-negativity**: NMF requires $W, V \ge 0$. Ensure projections (`clamp(min=0)`) are preserved after gradient steps.
- **Differentiability**: The link between the data and the persistence diagram is delicate. We use `torch-topological` concepts or custom `autograd.Function` wrappers around GUDHI. **Do not replace these with non-differentiable standard library calls.**

### 4.3. Working with TDA
- **point_cloud**: Usually shape `(n_points, dimensionality)`.
- **diagrams**: List of tensor pairs `(birth, death)`.
- **losses**: Usually operate on the `persistence = death - birth`.

## 5. Workflows

### 5.1. Running Tests
Current test suite uses `pytest`.
```bash
pytest
```
Always run specific relevant tests if modifying core logic:
```bash
pytest tests/test_model.py  # (example)
```

### 5.2. Verifying Convergence
Since TDA optimization is stochastic and complex, unit tests might pass while the model fails to learn.
- **Check**: Use `notebook/1D Signal.ipynb` or a simple script to fit `TopologicalNMF` on a simple synthetic signal (e.g., sine wave).
- **Metric**: Watch the `PH` loss and `approx` loss. `PH` should decrease or stabilize.

### 5.3. Debugging
- **"Loss is nan"**: Check for division by zero in `losses.py` (e.g., `l1 / l2`). Check `epsilon` in `optim.py`.
- **"No Gradient"**: Ensure the `complex` being used (e.g., `GudhiVietorisRipsComplex`) is correctly attached to the computation graph.

## 6. Do's and Don'ts

| DO | DON'T |
|----|-------|
| Use `torch` functions for math to preserve gradients. | Use `numpy` functions on tensors inside the `fit` loop (breaks graph). |
| Check `device` before creating new tensors. | Hardcode `.cuda()` or assume CPU. |
| Add type hints to every new function. | Leave arguments untyped. |
| Run `pytest` before submitting changes. | Assume "it looks correct". |
