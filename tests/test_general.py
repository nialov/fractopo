"""
Tests for general utilities.
"""
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis.strategies import booleans, floats
from shapely.geometry import LineString, Point, Polygon

import tests
from fractopo import general


@pytest.mark.parametrize(
    "first,second,same,from_first,is_none", tests.test_match_crs_params
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
    tests.test_determine_node_junctions_params,
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


@pytest.mark.parametrize("geoseries", tests.test_bounding_polygon_params)
def test_bounding_polygon(geoseries):
    """
    Test bounding_polygon with pytest params.
    """
    result = general.bounding_polygon(geoseries)
    assert isinstance(result, Polygon)
    for geom in geoseries:
        assert not geom.intersects(result.boundary)
        assert geom.within(result)


@pytest.mark.parametrize("keep_column_data", [True, False])
def test_crop_to_target_areas(keep_column_data: bool, file_regression):
    """
    Test cropping traces to target area with known right example data results.

    Also does regression testing with known right data.
    """
    trace_data = general.read_geofile(
        Path("tests/sample_data/mls_crop_samples/traces.geojson")
    )
    area_data = general.read_geofile(
        Path("tests/sample_data/mls_crop_samples/mls_area.geojson")
    )
    cropped_traces = general.crop_to_target_areas(
        traces=trace_data,
        areas=area_data,
        keep_column_data=keep_column_data,
    )
    assert isinstance(cropped_traces, gpd.GeoDataFrame)
    cropped_traces.sort_index(inplace=True)
    tests.geodataframe_regression_check(
        file_regression=file_regression, gdf=cropped_traces
    )
    # file_regression.check(cropped_traces.to_json(indent=1, sort_keys=True))


def test_dissolve_multi_part_traces(file_regression):
    """
    Test dissolving MultiLineString containing GeoDataFrame.

    Dissolve should copy all attribute data to new dissolved LineStrings.
    """
    trace_data = general.read_geofile(
        Path("tests/sample_data/mls_crop_samples/mls_traces.geojson")
    )
    dissolved_traces = general.dissolve_multi_part_traces(trace_data)
    assert isinstance(dissolved_traces, gpd.GeoDataFrame)
    dissolved_traces.sort_index(inplace=True)
    # file_regression.check(dissolved_traces.to_json(indent=1, sort_keys=True))
    tests.geodataframe_regression_check(
        file_regression=file_regression, gdf=dissolved_traces
    )


@pytest.mark.parametrize(
    "line_gdf,area_gdf,snap_threshold,assumed_result_inter,assumed_result_cuts",
    tests.test_determine_boundary_intersecting_lines_params,
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


@example(45.0)
@given(floats(allow_infinity=False))
def test_azimuth_to_unit_vector(azimuth: float):
    """
    Test azimuth_to_unit_vector.
    """
    result = general.azimuth_to_unit_vector(azimuth=azimuth)

    assert len(result) == 2
    assert isinstance(result, np.ndarray)


@pytest.mark.parametrize(
    "intersection_geoms,raises",
    [
        # MultiLineString
        (gpd.GeoSeries([tests.invalid_geom_multilinestring]), does_not_raise()),
        # Empty geometry
        (gpd.GeoSeries([tests.invalid_geom_empty]), does_not_raise()),
        # Both above
        (
            gpd.GeoSeries(
                [tests.invalid_geom_empty, tests.invalid_geom_multilinestring]
            ),
            does_not_raise(),
        ),
        # Polygon
        (gpd.GeoSeries([(tests.area_1)]), pytest.raises(TypeError)),
    ],
)
def test_determine_valid_intersection_points(intersection_geoms: gpd.GeoSeries, raises):
    """
    Test determine_valid_intersection_points.
    """
    with raises:
        result = general.determine_valid_intersection_points(
            intersection_geoms=intersection_geoms
        )
        assert isinstance(result, list)
        assert all(isinstance(val, Point) for val in result)


@pytest.mark.parametrize(
    "geodata",
    [
        gpd.GeoSeries([LineString([(0, 0, 0), (1, 1, 1)])]),
        gpd.GeoSeries(
            [
                LineString([(0, 0, 0), (1, 1, 1)]),
                LineString([(10, 10, 10), (20, 20, 20)]),
            ]
        ),
        gpd.GeoSeries([LineString([(0, 0), (1, 1)])]),
    ],
)
def test_total_bounds(geodata):
    """
    Test total_bounds.
    """
    result = general.total_bounds(geodata=geodata)
    assert isinstance(result, tuple)
    assert len(result) == 4
