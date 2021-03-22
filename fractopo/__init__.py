"""
Main package with general utilites.

Analysis and validation tools in subpackages.
"""
from typing import Tuple

# Versioneer import
from ._version import get_versions

SetRangeTuple = Tuple[Tuple[float, float], ...]
BoundsTuple = Tuple[float, float, float, float]
PointTuple = Tuple[float, float]

# Versioneer handles
__version__ = get_versions()["version"]
del get_versions
