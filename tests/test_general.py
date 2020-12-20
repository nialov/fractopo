from hypothesis.strategies._internal.core import booleans, floats
import fractopo.general as general
from shapely.geometry import Point, LineString, Polygon
import geopandas as gpd
import numpy as np

from hypothesis import given
from tests import Helpers
import pytest


@pytest.mark.parametrize(
    "first,second,same,from_first,is_none", Helpers.test_match_crs_params
)
def test_match_crs(first, second, same: bool, from_first: bool, is_none: bool):
    if from_first:
        original_crs = first.crs
    else:
        original_crs = second.crs
    first_matched, second_matched = general.match_crs(first, second)
    if same:
        assert first_matched.crs == original_crs
        assert second_matched.crs == original_crs
    else:
        assert first_matched.crs != second_matched.crs
    if is_none:
        assert first_matched.crs is None
        assert second_matched.crs is None


@given(
    floats(min_value=0, max_value=180, allow_nan=False),
    floats(min_value=0, max_value=180, allow_nan=False),
    floats(min_value=0, max_value=180, allow_nan=False),
    booleans(),
)
def test_is_azimuth_close(first, second, tolerance, halved):
    result = general.is_azimuth_close(first, second, tolerance, halved)
    assert isinstance(result, bool)


@given(Helpers.nice_polyline)
def test_determine_regression_azimuth(line: LineString):
    trace = LineString(line)
    result = general.determine_regression_azimuth(trace)
    assert isinstance(result, float)
    assert not np.isnan(result)  # type: ignore


# @given()
# def test_determine_set(value, value_range, loop_around):
#     result = general.determine_set(value, value_range, loop_around)
#     assert isinstance(result, bool)


@pytest.mark.parametrize(
    "nodes,snap_threshold,snap_threshold_error_multiplier,error_threshold",
    Helpers.test_determine_node_junctions_params,
)
def test_determine_node_junctions(
    nodes, snap_threshold, snap_threshold_error_multiplier, error_threshold
):
    result = general.determine_node_junctions(
        nodes, snap_threshold, snap_threshold_error_multiplier, error_threshold
    )
    assert isinstance(result, set)
    return result


@pytest.mark.parametrize("geoseries", Helpers.test_bounding_polygon_params)
def test_bounding_polygon(geoseries):
    result = general.bounding_polygon(geoseries)
    assert isinstance(result, Polygon)
    for geom in geoseries:
        assert not geom.intersects(result.boundary)
        assert geom.within(result)
