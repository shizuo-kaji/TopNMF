# Notebook Usage

Notebooks in this folder assume `TopNMF` is importable as an installed package.

From the repository root, install once in editable mode:

```bash
python3 -m pip install -e ".[dev]"
```

Then run Jupyter with the same Python environment and open notebooks in this
directory. This keeps imports such as `from TopNMF import *` working regardless
of notebook file location.

To execute all notebooks non-interactively and verify they run without errors:

```bash
python scripts/check_notebooks.py
```
