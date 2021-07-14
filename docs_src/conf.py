# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from importlib import import_module


sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))

# -- Project information -----------------------------------------------------

project = "fractopo".replace("_", "-")
copyright = "2020, Nikolas Ovaskainen"
author = "Nikolas Ovaskainen"

# The full version, including alpha/beta/rc tags
imported_package = import_module("fractopo")  # noqa

release = imported_package.__version__  # type: ignore


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "nbsphinx",
]

# Add .md markdown files as sources.
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
master_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options for nbsphinx output ---------------------------------------------

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

if "READTHEDOCS" not in os.environ:
    # Always execute notebooks locally (to test that they work!)
    nbsphinx_execute = "always"
else:
    # Do not always execute notebooks on ReadtheDocs
    nbsphinx_execute = "auto"
