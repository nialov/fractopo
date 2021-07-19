"""
fractopo.

Fracture Network Analysis
"""
import logging

from fractopo.analysis.network import Network
from fractopo.analysis.multi_network import MultiNetwork
from fractopo.tval.trace_validation import Validation

__version__ = "0.0.1.post283.dev0+f03dfab"


logging.info(
    "Main imports available from fractopo/__init__.py:"
    f" {Network, Validation, MultiNetwork}"
)
