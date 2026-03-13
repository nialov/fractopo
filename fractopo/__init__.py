"""
fractopo.

Fracture Network Analysis
"""

import logging

from fractopo.analysis.multi_network import MultiNetwork  # noqa: E402
from fractopo.analysis.network import Network  # noqa: E402
from fractopo.tval.trace_validation import Validation  # noqa: E402

log = logging.getLogger(__name__)

__version__ = "0.9.1"


log.info(
    "Main imports available from fractopo/__init__.py:"
    f" {Network, Validation, MultiNetwork}"
)
