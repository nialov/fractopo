"""
Tests for general utilites.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies._internal.core import booleans, floats
from shapely.geometry import Polygon

from fractopo import general
from tests import Helpers


@pytest.mark.parametrize(
    "first,second,same,from_first,is_none", Helpers.test_match_crs_params
)
def test_match_crs(first, second, same: bool, from_first: bool, is_none: bool):
    """
    Test match_crs with pytest params.
    """
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
    """
    Test is_azimuth_close with hypothesis.
    """
    result = general.is_azimuth_close(first, second, tolerance, halved)
    assert isinstance(result, bool)


@pytest.mark.parametrize(
    "nodes,snap_threshold,snap_threshold_error_multiplier,error_threshold",
    Helpers.test_determine_node_junctions_params,
)
def test_determine_node_junctions(
    nodes, snap_threshold, snap_threshold_error_multiplier, error_threshold
):
    """
    Test determine_node_junctions with pytest params.
    """
    result = general.determine_node_junctions(
        nodes, snap_threshold, snap_threshold_error_multiplier, error_threshold
    )
    assert isinstance(result, set)
    return result


@pytest.mark.parametrize("geoseries", Helpers.test_bounding_polygon_params)
def test_bounding_polygon(geoseries):
    """
    Test bounding_polygon with pytest params.
    """
    result = general.bounding_polygon(geoseries)
    assert isinstance(result, Polygon)
    for geom in geoseries:
        assert not geom.intersects(result.boundary)
        assert geom.within(result)


def test_crop_to_target_areas(file_regression):
    """
    Test cropping traces to target area with known right example data results.

    Also does regression testing with known right data.
    """
    trace_data = gpd.read_file(Path("tests/sample_data/mls_crop_samples/traces.gpkg"))
    area_data = gpd.read_file(Path("tests/sample_data/mls_crop_samples/mls_area.gpkg"))
    cropped_traces = general.crop_to_target_areas(
        traces=trace_data, areas=area_data, snap_threshold=0.01
    )
    assert isinstance(cropped_traces, gpd.GeoDataFrame)
    file_regression.check(cropped_traces.sort_index().to_json())


def test_dissolve_multi_part_traces(file_regression):
    """
    Test dissolving MultiLineString containing GeoDataFrame.

    Dissolve should copy all attribute data to new dissolved LineStrings.
    """
    trace_data = gpd.read_file(
        Path("tests/sample_data/mls_crop_samples/mls_traces.gpkg")
    )
    dissolved_traces = general.dissolve_multi_part_traces(trace_data)
    assert isinstance(dissolved_traces, gpd.GeoDataFrame)
    file_regression.check(dissolved_traces.sort_index().to_json())


@pytest.mark.parametrize(
    "line_gdf,area_gdf,snap_threshold,assumed_result_inter,assumed_result_cuts",
    Helpers.test_determine_boundary_intersecting_lines_params,
)
def test_determine_boundary_intersecting_lines(
    line_gdf, area_gdf, snap_threshold, assumed_result_inter, assumed_result_cuts
):
    """
    Test determining boundary intersecting lines.
    """
    (
        intersecting_lines,
        cuts_through_lines,
    ) = general.determine_boundary_intersecting_lines(
        line_gdf, area_gdf, snap_threshold
    )
    assert isinstance(intersecting_lines, np.ndarray)
    assert isinstance(cuts_through_lines, np.ndarray)
    assert all(intersecting_lines == assumed_result_inter)
    assert all(cuts_through_lines == assumed_result_cuts)
