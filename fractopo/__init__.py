"""
fractopo.

Fracture Network Analysis
"""
import logging
import warnings

warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message=r".*The Shapely GEOS version.*is incompatible with the GEOS",
)

from fractopo.analysis.multi_network import MultiNetwork  # noqa: E402
from fractopo.analysis.network import Network  # noqa: E402
from fractopo.tval.trace_validation import Validation  # noqa: E402

__version__ = "0.2.0.post5.dev0+27f32ad"


logging.info(
    "Main imports available from fractopo/__init__.py:"
    f" {Network, Validation, MultiNetwork}"
)
