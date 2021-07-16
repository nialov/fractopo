"""
fractopo.

Fracture Network Analysis
"""
import logging

from fractopo.analysis.network import Network
from fractopo.tval.trace_validation import Validation

__version__ = "0.0.1.post283.dev0+f03dfab"


logging.info(f"Main imports available from fractopo/__init__.py: {Network, Validation}")
