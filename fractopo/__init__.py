"""
Main package with general utilites.

Analysis and validation tools in subpackages.
"""
import logging

from fractopo.analysis.network import Network
from fractopo.tval.trace_validation import Validation

# Versioneer import
from ._version import get_versions

# Versioneer handles
__version__ = get_versions()["version"]
del get_versions

logging.info(f"Main imports available from fractopo/__init__.py: {Network, Validation}")
