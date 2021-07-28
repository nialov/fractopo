"""
fractopo.

Fracture Network Analysis
"""
import logging

from fractopo.analysis.multi_network import MultiNetwork
from fractopo.analysis.network import Network
from fractopo.tval.trace_validation import Validation

__version__ = "0.0.1.post372.dev0+d65be7b"


logging.info(
    "Main imports available from fractopo/__init__.py:"
    f" {Network, Validation, MultiNetwork}"
)
