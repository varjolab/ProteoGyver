# -- Path setup --------------------------------------------------------------
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

# -- Project information -----------------------------------------------------
project = "ProteoGyver"
author = "ProteoGyver Developers"

# Read version from VERSION file (avoids importing the app on RTD)
version_file = ROOT / "VERSION"
release = version = version_file.read_text().strip() if version_file.exists() else "0.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_autodoc_typehints",
]

# Mock heavy deps to keep RTD builds fast/stable
autodoc_mock_imports = [
    "dash", "pandas", "numpy", "scipy", "plotly", "celery",
    "sqlalchemy", "numba", "matplotlib", "sklearn", "torch",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]