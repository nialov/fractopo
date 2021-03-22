"""
Main package with general utilites.

Analysis and validation tools in subpackages.
"""
from typing import Tuple

SetRangeTuple = Tuple[Tuple[float, float], ...]
BoundsTuple = Tuple[float, float, float, float]
PointTuple = Tuple[float, float]

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
