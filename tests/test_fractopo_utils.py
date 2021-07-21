"""
Tests for fractopo_utils.py.
"""

import geopandas as gpd
from shapely.geometry import Point

import fractopo.fractopo_utils as fractopo_utils
from tests import Helpers


def test_remove_identical_sindex():
    """
    Test remove_identical_sindex.
    """
    geosrs = Helpers.get_geosrs_identicals()
    geosrs_orig_length = len(geosrs)
    result = fractopo_utils.remove_identical_sindex(geosrs, Helpers.snap_threshold)
    result_length = len(result)
    assert geosrs_orig_length > result_length
    assert geosrs_orig_length - result_length == 2


def test_remove_snapping_in_remove_identicals():
    """
    Test remove_identical_sindex with identicals.
    """
    snap_thresholds = [
        Helpers.snap_threshold,
        Helpers.snap_threshold * 2,
        Helpers.snap_threshold * 4,
        Helpers.snap_threshold * 8,
    ]
    for sn in snap_thresholds:
        # Snap distances for Points should always be within snapping
        # threshold
        geosrs = gpd.GeoSeries([Point(1, 1), Point(1 + sn * 0.99, 1)])
        result = fractopo_utils.remove_identical_sindex(geosrs, sn)
        assert len(geosrs) - len(result) == 1
