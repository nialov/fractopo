"""
Configuration file for Sphinx.
"""

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
import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, Path("..").resolve().as_posix())
sys.path.insert(0, Path().resolve().as_posix())

# -- Project information -----------------------------------------------------

project = "fractopo"
copyright = "2020-%Y"
author = "Nikolas Ovaskainen"

# The full version, including alpha/beta/rc tags
imported_package = import_module("fractopo")  # noqa
version = imported_package.__version__  # type: ignore
release = version


# -- General configuration ---------------------------------------------------

needs_sphinx = "8.0.0"

extensions = [
    "sphinx.ext.apidoc",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx_gallery.gen_gallery",
    "sphinx_rtd_theme",
    "nbsphinx",
    "sphinx_design",
    "sphinx_sitemap",
]

# Add .md markdown files as sources.
source_suffix = {
    ".rst": "restructuredtext",
}
master_doc = "index"

# Sphinx-gallery config
sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
}

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # auto_examples needs to be added due to nbsphinx executing the ipynb
    # files inside otherwise
    "auto_examples/*.ipynb",
]


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

nbsphinx_execute = "always"

# -- Options for apidoc ---------------------------------------------

apidoc_modules = [
    {
        "path": "../fractopo",
        "destination": "./apidoc/",
        "max_depth": 4,
    }
]

# -- Options for sphinx-sitemap -------------------------------------

html_baseurl = "https://nialov.github.io/fractopo/"
sitemap_url_scheme = "{link}"
html_extra_path = ["robots.txt"]
