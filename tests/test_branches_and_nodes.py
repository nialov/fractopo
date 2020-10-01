import geopandas as gpd
import shapely
from shapely.geometry import LineString, Point, MultiPoint, Polygon
from shapely.ops import snap, split
import shapely
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from hypothesis.strategies import tuples, floats, integers
from hypothesis import given, assume
from hypothesis_geometry import planar
from pathlib import Path
import pytest

# Import trace_validator
from tval import trace_validator, trace_builder
from fractopo import branches_and_nodes
from fractopo.branches_and_nodes import I_node, X_node, Y_node, E_node


class Helpers:

    snap_threshold = 0.001
    geosrs_identicals = gpd.GeoSeries(
        [Point(1, 1), Point(1, 1), Point(2, 1), Point(2, 1), Point(3, 1), Point(2, 3)]
    )

    traces_geosrs = gpd.GeoSeries(
        [
            LineString([(-1, 0), (1, 0)]),
            LineString([(0, -1), (0, 1)]),
            LineString(
                [(-1.0 - snap_threshold * 0.99, -1), (-1.0 - snap_threshold * 0.99, 1)]
            ),
        ]
    )
    areas_geosrs = gpd.GeoSeries([Polygon([(5, 5), (-5, 5), (-5, -5), (5, -5)])])

    nice_traces = gpd.GeoSeries(
        [
            # Horizontal
            LineString([(-10, 0), (10, 0)]),
            # Underlapping
            LineString([(-5, 2), (-5, 0 + snap_threshold * 0.01)]),
            LineString([(-4, 2), (-4, 0 + snap_threshold * 0.5)]),
            LineString([(-3, 2), (-3, 0 + snap_threshold * 0.7)]),
            LineString([(-2, 2), (-2, 0 + snap_threshold * 0.9)]),
            LineString([(-1, 2), (-1, 0 + snap_threshold * 1.1)]),
            # Overlapping
            LineString([(1, 2), (1, 0 - snap_threshold * 1.1)]),
            LineString([(2, 2), (2, 0 - snap_threshold * 0.9)]),
            LineString([(3, 2), (3, 0 - snap_threshold * 0.7)]),
            LineString([(4, 2), (4, 0 - snap_threshold * 0.5)]),
            LineString([(5, 2), (5, 0 - snap_threshold * 0.01)]),
        ]
    )
    nice_integer_coordinates = integers(-10, 10)
    nice_float = floats(
        allow_nan=False, allow_infinity=False, min_value=-1e5, max_value=1e5
    )
    nice_tuple = tuples(nice_float, nice_float,)
    triple_tuples = tuples(nice_tuple, nice_tuple, nice_tuple,)
    nice_point = planar.points(nice_integer_coordinates)
    nice_polyline = planar.polylines(nice_integer_coordinates)

    @classmethod
    def get_multi_polyline_strategy(cls):
        multi_polyline_strategy = tuples(
            *tuple(
                [
                    planar.polylines(
                        x_coordinates=cls.nice_integer_coordinates,
                        min_size=2,
                        max_size=5,
                    )
                    for _ in range(5)
                ]
            )
        )
        return multi_polyline_strategy

    @classmethod
    def get_nice_traces(cls):
        return cls.nice_traces.copy()

    @classmethod
    def get_traces_geosrs(cls):
        return cls.traces_geosrs.copy()

    @classmethod
    def get_areas_geosrs(cls):
        return cls.areas_geosrs.copy()

    @classmethod
    def get_geosrs_identicals(cls):
        return cls.geosrs_identicals.copy()


def test_remove_identical_sindex():
    geosrs = Helpers.get_geosrs_identicals()
    geosrs_orig_length = len(geosrs)
    result = branches_and_nodes.remove_identical_sindex(geosrs, Helpers.snap_threshold)
    result_length = len(result)
    assert geosrs_orig_length > result_length
    assert geosrs_orig_length - result_length == 2


def test_remove_snapping_in_remove_identicals():
    snap_thresholds = [
        0,
        Helpers.snap_threshold,
        Helpers.snap_threshold * 2,
        Helpers.snap_threshold * 4,
        Helpers.snap_threshold * 8,
    ]
    for sn in snap_thresholds:
        # Snap distances for Points should always be within snapping
        # threshold
        geosrs = gpd.GeoSeries([Point(1, 1), Point(1 + sn * 0.99, 1)])
        result = branches_and_nodes.remove_identical_sindex(geosrs, sn)
        assert len(geosrs) - len(result) == 1


def test_get_node_identities():
    traces_geosrs = Helpers.get_traces_geosrs()
    areas_geosrs = Helpers.get_areas_geosrs()
    traces_geosrs, any_changed_applied = branches_and_nodes.snap_traces(
        traces_geosrs, Helpers.snap_threshold
    )
    det_nodes, _ = trace_validator.BaseValidator.determine_nodes(
        gpd.GeoDataFrame({"geometry": traces_geosrs})
    )
    nodes_geosrs = branches_and_nodes.remove_identical_sindex(
        gpd.GeoSeries(det_nodes), Helpers.snap_threshold
    )
    result = branches_and_nodes.get_node_identities(
        traces_geosrs, nodes_geosrs, areas_geosrs, Helpers.snap_threshold
    )
    assert "X" in result and "Y" in result and "I" in result
    assert len([r for r in result if r == "X"]) == 1
    assert len([r for r in result if r == "Y"]) == 1
    assert len([r for r in result if r == "I"]) == 5


# number_of_I_nodes: int, number_of_XY_nodes: int, number_of_E_nodes: int
@pytest.mark.parametrize(
    "number_of_I_nodes, number_of_XY_nodes, number_of_E_nodes, should_pass",
    [
        (2, 0, 0, True),
        (2, 1, 0, False),
        (1, 1, 0, True),
        (0, 1, 1, True),
        (1, 0, 1, True),
        (1, 1, 1, False),
    ],
)
def test_determine_branch_identity(
    number_of_I_nodes, number_of_XY_nodes, number_of_E_nodes, should_pass
):
    result = branches_and_nodes.determine_branch_identity(
        number_of_I_nodes, number_of_XY_nodes, number_of_E_nodes
    )
    if not should_pass:
        assert result == branches_and_nodes.EE_branch


def check_gdf_contents(obtained_filename, expected_filename):
    """
    Checks that two GeoDataFrames have exactly matching contents.
    """
    assert Path(obtained_filename).exists() and Path(expected_filename).exists()
    obtained_gdf = gpd.read_file(Path(obtained_filename))
    expected_gdf = gpd.read_file(Path(expected_filename))
    for idxrow1, idxrow2 in zip(obtained_gdf.iterrows(), expected_gdf.iterrows()):
        assert obtained_gdf.crs == expected_gdf.crs
        assert all(idxrow1 == idxrow2)


def test_branches_and_nodes(file_regression):
    (
        valid_geoseries,
        invalid_geoseries,
        valid_areas_geoseries,
        invalid_areas_geoseries,
    ) = trace_builder.main(snap_threshold=Helpers.snap_threshold)
    branch_geodataframe, node_geodataframe = branches_and_nodes.branches_and_nodes(
        valid_geoseries, valid_areas_geoseries, Helpers.snap_threshold
    )
    # Use --force-regen to remake if fails after trace_builder changes.
    file_regression.check(str(branch_geodataframe) + str(node_geodataframe))

    for node_id in node_geodataframe["Type"]:
        assert node_id in [I_node, X_node, Y_node, E_node]
    assert len([node_id for node_id in node_geodataframe["Type"] if node_id == "X"]) > 0
    assert len([node_id for node_id in node_geodataframe["Type"] if node_id == "Y"]) > 0
    assert len([node_id for node_id in node_geodataframe["Type"] if node_id == "I"]) > 1


@pytest.mark.parametrize(
    "traces_geosrs, areas_geosrs",
    [(Helpers.get_traces_geosrs(), Helpers.get_areas_geosrs()),],
)
def test_get_branch_identities(traces_geosrs, areas_geosrs):
    traces_geosrs, any_changed_applied = branches_and_nodes.snap_traces(
        traces_geosrs, Helpers.snap_threshold
    )
    det_nodes, _ = trace_validator.BaseValidator.determine_nodes(
        gpd.GeoDataFrame({"geometry": traces_geosrs})
    )
    nodes_geosrs = branches_and_nodes.remove_identical_sindex(
        gpd.GeoSeries(det_nodes), Helpers.snap_threshold
    )
    node_identities = branches_and_nodes.get_node_identities(
        traces_geosrs, nodes_geosrs, areas_geosrs, Helpers.snap_threshold
    )
    branches_geosrs = branches_and_nodes.split_traces_to_branches_with_traces(
        traces_geosrs, nodes_geosrs, node_identities, Helpers.snap_threshold
    )
    result = branches_and_nodes.get_branch_identities(
        branches_geosrs, nodes_geosrs, node_identities, Helpers.snap_threshold
    )
    assert all(
        [
            branch in result
            for branch in (branches_and_nodes.CC_branch, branches_and_nodes.CI_branch,)
        ]
    )


def test_snap_traces():
    simple_traces = gpd.GeoSeries(
        [LineString([(0, 0), (0.99, 0)]), LineString([(1, -1), (1, 1)])]
    )
    simple_snap_threshold = 0.02
    simple_snapped_traces, any_changed_applied = branches_and_nodes.snap_traces(
        simple_traces, simple_snap_threshold
    )
    first_coords = simple_snapped_traces.iloc[0].coords
    first_coords_points = [Point(c) for c in first_coords]
    assert any(
        [p.intersects(simple_snapped_traces.iloc[1]) for p in first_coords_points]
    )


@given(Helpers.get_multi_polyline_strategy())
def test_snap_traces_hypothesis(multi_polyline_strategy):
    geosrs = gpd.GeoSeries(
        [LineString(segments) for segments in multi_polyline_strategy]
    )
    snapped_traces, any_changed_applied = branches_and_nodes.snap_traces(
        geosrs, Helpers.snap_threshold
    )
    while any_changed_applied:
        snapped_traces, any_changed_applied = branches_and_nodes.snap_traces(
            snapped_traces, Helpers.snap_threshold
        )


def assert_result_insert_point_to_linestring(result, point):
    # Assert it is in list
    assert tuple(*point.coords) in list(result.coords)
    # Assert index is correct
    assert list(result.coords).index(tuple(*point.coords)) == 1


@pytest.mark.parametrize(
    "linestring, point, assert_result",
    [
        (
            LineString([(0, 0), (1, 1), (2, 2)]),
            Point(0.5, 0.5),
            assert_result_insert_point_to_linestring,
        ),
        (
            LineString([(0, 0), (-1, -1), (-2, -2)]),
            Point(-0.5, -0.5),
            assert_result_insert_point_to_linestring,
        ),
    ],
)
def test_insert_point_to_linestring(linestring, point, assert_result):
    result = branches_and_nodes.insert_point_to_linestring(linestring, point)
    assert_result(result, point)


@given(Helpers.nice_polyline, Helpers.nice_point)
def test_insert_point_to_linestring_hypothesis(linestring, point):
    linestring = LineString(linestring)
    point = Point(point)
    assume(not any([point.intersects(Point(xy)) for xy in linestring.coords]))
    result = branches_and_nodes.insert_point_to_linestring(linestring, point)


def test_additional_snapping_func():
    ls = LineString([(0, 0), (1, 1), (2, 2)])
    idx = 0
    p = Point(0.5, 0.5)
    additional_snapping = [(0, p)]
    result = branches_and_nodes.additional_snapping_func(ls, idx, additional_snapping)
    # Assert it is in list
    assert tuple(*p.coords) in list(result.coords)
    # Assert index is correct
    assert list(result.coords).index(tuple(*p.coords)) == 1
    unchanged_result = branches_and_nodes.additional_snapping_func(
        ls, 1, additional_snapping
    )
    assert unchanged_result == ls


def test_nice_traces():
    nice_traces = Helpers.get_nice_traces()
    snapped_traces, any_changed_applied = branches_and_nodes.snap_traces(
        nice_traces, Helpers.snap_threshold
    )
    assert len(nice_traces) == len(snapped_traces)
    for geom in snapped_traces:
        geom: LineString
        assert isinstance(geom, LineString)
        assert geom.is_valid
        assert geom.is_simple
    return nice_traces, snapped_traces


def test_crop_to_target_area():

    (
        valid_geoseries,
        invalid_geoseries,
        valid_areas_geoseries,
        invalid_areas_geoseries,
    ) = trace_builder.main(snap_threshold=Helpers.snap_threshold)
    valid_result = branches_and_nodes.crop_to_target_areas(
        valid_geoseries, valid_areas_geoseries
    )
    invalid_result = branches_and_nodes.crop_to_target_areas(
        invalid_geoseries, invalid_areas_geoseries
    )
    assert isinstance(valid_result, gpd.GeoSeries)
    assert isinstance(invalid_result, gpd.GeoSeries)
    assert valid_geoseries.geometry.length.mean() > valid_result.geometry.length.mean()
    assert (
        invalid_geoseries.geometry.length.mean() > invalid_result.geometry.length.mean()
    )


@given(Helpers.triple_tuples)
def test_angle_to_point(triple_tuples):
    # assume(not all(np.isclose(triple_tuples[0], triple_tuples[1])))
    # assume(not all(np.isclose(triple_tuples[0], triple_tuples[2])))
    # assume(not all(np.isclose(triple_tuples[1], triple_tuples[2])))
    triple_points = tuple((Point(*t) for t in triple_tuples))
    try:
        rad_angle = branches_and_nodes.angle_to_point(*triple_points)
    except ValueError:
        pass

