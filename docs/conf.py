"""Sphinx configuration for isitfair documentation."""

import importlib.metadata

# -- Project information -----------------------------------------------------

project = "isitfair"
copyright = "2026, Hugo Guillen-Ramirez"
author = "Hugo Guillen-Ramirez"
# `__init__.py` is the single source of truth for the version -- it is what
# hatch reads to build the wheel. Prefer it over the installed distribution
# metadata, which goes stale whenever an editable install predates a version
# bump and would otherwise title these docs with the wrong release.
try:
    from isitfair import __version__ as release
except ImportError:  # not importable (e.g. autodoc-less build); fall back
    release = importlib.metadata.version("isitfair")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST configuration ------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "fieldlist",
]

# -- Autodoc configuration ---------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autosummary_generate = True

# Napoleon settings for numpy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# -- Intersphinx mapping -----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/HugoGuillen/isitfair",
    "show_toc_level": 2,
    "navigation_with_keys": False,
}
html_static_path = []
html_title = f"isitfair {release}"
