"""
fractopo.

Fracture Network Analysis
"""

import logging
import os

# Make GeoPandas use shapely 2.0 instead of pygeos
os.environ["USE_PYGEOS"] = "0"


from fractopo.analysis.multi_network import MultiNetwork  # noqa: E402,C0413
from fractopo.analysis.network import Network  # noqa: E402,C0413
from fractopo.tval.trace_validation import Validation  # noqa: E402,C0413

log = logging.getLogger(__name__)

__version__ = "0.7.0"


log.info(
    "Main imports available from fractopo/__init__.py:"
    f" {Network, Validation, MultiNetwork}"
)
