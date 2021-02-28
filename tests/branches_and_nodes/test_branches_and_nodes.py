"""
Tests for branch and node determination.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from hypothesis import given
from shapely import wkt
from shapely.geometry import LineString, MultiLineString, Point, box

from fractopo import branches_and_nodes, general
from fractopo.branches_and_nodes import (
    CLASS_COLUMN,
    CONNECTION_COLUMN,
    E_node,
    EE_branch,
    I_node,
    X_node,
    Y_node,
)
from fractopo.tval import trace_builder
from tests import Helpers
from tests.sample_data.py_samples import samples

# Import trace_validator


def test_remove_identical_sindex():
    geosrs = Helpers.get_geosrs_identicals()
    geosrs_orig_length = len(geosrs)
    result = branches_and_nodes.remove_identical_sindex(geosrs, Helpers.snap_threshold)
    result_length = len(result)
    assert geosrs_orig_length > result_length
    assert geosrs_orig_length - result_length == 2


def test_remove_snapping_in_remove_identicals():
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
        result = branches_and_nodes.remove_identical_sindex(geosrs, sn)
        assert len(geosrs) - len(result) == 1


def test_get_node_identities():
    traces_geosrs = Helpers.get_traces_geosrs()
    areas_geosrs = Helpers.get_areas_geosrs()
    traces_geosrs, any_changed_applied = branches_and_nodes.snap_traces(
        traces_geosrs, Helpers.snap_threshold
    )
    det_nodes, _ = branches_and_nodes.determine_nodes(
        gpd.GeoDataFrame({"geometry": traces_geosrs}), snap_threshold=0.01
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
        assert result == branches_and_nodes.Error_branch


def check_gdf_contents(obtained_filename, expected_filename):
    """
    Check that two GeoDataFrames have exactly matching contents.
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
    branch_gdf, node_gdf = branches_and_nodes.branches_and_nodes(
        valid_geoseries, valid_areas_geoseries, Helpers.snap_threshold
    )
    # Use --force-regen to remake if fails after trace_builder changes.
    file_regression.check(str(branch_gdf) + str(node_gdf))

    for node_id in node_gdf[CLASS_COLUMN].values:
        assert node_id in [I_node, X_node, Y_node, E_node]
    assert (
        len([node_id for node_id in node_gdf[CLASS_COLUMN].values if node_id == "X"])
        > 0
    )
    assert (
        len([node_id for node_id in node_gdf[CLASS_COLUMN].values if node_id == "Y"])
        > 0
    )
    assert (
        len([node_id for node_id in node_gdf[CLASS_COLUMN].values if node_id == "I"])
        > 1
    )


@pytest.mark.parametrize(
    "traces_geosrs, areas_geosrs",
    [
        (Helpers.get_traces_geosrs(), Helpers.get_areas_geosrs()),
    ],
)
def test_get_branch_identities(traces_geosrs, areas_geosrs):
    traces_geosrs, any_changed_applied = branches_and_nodes.snap_traces(
        traces_geosrs, Helpers.snap_threshold
    )
    det_nodes, _ = branches_and_nodes.determine_nodes(
        gpd.GeoDataFrame({"geometry": traces_geosrs}), snap_threshold=0.01
    )
    nodes_geosrs = branches_and_nodes.remove_identical_sindex(
        gpd.GeoSeries(det_nodes), Helpers.snap_threshold
    )
    node_identities = branches_and_nodes.get_node_identities(
        traces_geosrs, nodes_geosrs, areas_geosrs, Helpers.snap_threshold
    )
    branches_geosrs = gpd.GeoSeries([b for b in traces_geosrs.unary_union])
    result = branches_and_nodes.get_branch_identities(
        branches_geosrs, nodes_geosrs, node_identities, Helpers.snap_threshold
    )
    assert all(
        [
            branch in result
            for branch in (
                branches_and_nodes.CC_branch,
                branches_and_nodes.CI_branch,
            )
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
    # assert Point(0.99, 0).intersects(
    #     gpd.GeoSeries([Point(xy) for xy in simple_snapped_traces.iloc[1].coords])
    # )
    # is_in_ls = False
    assert all(
        [isinstance(ls, LineString) for ls in simple_snapped_traces.geometry.values]
    )
    for xy in simple_snapped_traces.iloc[1].coords:
        p = Point(xy)
        if Point(0.99, 0).intersects(p):
            pass
            # is_in_ls = True


@pytest.mark.parametrize(
    "linestring, point, snap_threshold, assumed_result",
    Helpers.test_insert_point_to_linestring_params,
)
def test_insert_point_to_linestring_v2(
    linestring, point, snap_threshold, assumed_result
):
    """
    Test insert_point_to_linestring.
    """
    result = branches_and_nodes.insert_point_to_linestring_v2(
        linestring, point, snap_threshold
    )
    if assumed_result is None:
        # Assert it is in list
        assert tuple(*point.coords) in list(result.coords)
        # Assert index is correct
        assert list(result.coords).index(tuple(*point.coords)) == 1
    else:
        assert result.wkt == assumed_result.wkt


# @settings(suppress_health_check=(HealthCheck.filter_too_much,))
# @given(Helpers.nice_polyline, Helpers.nice_point)
# def test_insert_point_to_linestring_hypothesis(linestring, point):
#     linestring = LineString(linestring)
#     assume(linestring.is_valid)
#     assume(linestring.is_simple)
#     point = Point(point)
#     assume(not any([point.intersects(Point(xy)) for xy in linestring.coords]))
#     result = branches_and_nodes.insert_point_to_linestring(linestring, point)


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
    for geom in snapped_traces.geometry.values:
        geom: LineString
        assert isinstance(geom, LineString)
        assert geom.is_valid
        assert geom.is_simple
    return nice_traces, snapped_traces


def test_crop_to_target_area():
    """
    Test crop to target area.
    """
    (
        valid_geoseries,
        invalid_geoseries,
        valid_areas_geoseries,
        invalid_areas_geoseries,
    ) = trace_builder.main(snap_threshold=Helpers.snap_threshold)
    valid_result = general.crop_to_target_areas(
        valid_geoseries, valid_areas_geoseries, snap_threshold=0.01
    )
    try:
        _ = general.crop_to_target_areas(
            invalid_geoseries, invalid_areas_geoseries, snap_threshold=0.01
        )
        assert False
    except TypeError:
        pass
    assert isinstance(valid_result, gpd.GeoSeries)
    assert valid_geoseries.geometry.length.mean() > valid_result.geometry.length.mean()


@given(Helpers.triple_tuples)
def test_angle_to_point(triple_tuples):
    # assume(not all(np.isclose(triple_tuples[0], triple_tuples[1])))
    # assume(not all(np.isclose(triple_tuples[0], triple_tuples[2])))
    # assume(not all(np.isclose(triple_tuples[1], triple_tuples[2])))
    triple_points = tuple((Point(*t) for t in triple_tuples))
    try:
        rad_angle = branches_and_nodes.angle_to_point(*triple_points)
        assert isinstance(rad_angle, float)
    except ValueError:
        pass


def test_angle_to_point_known_err():
    points_wkt = [
        "POINT (286975.148 6677657.7042)",
        "POINT (284919.7632999998 6677522.154200001)",
        "POINT (280099.6969999997 6677204.276900001)",
    ]
    points = [wkt.loads(p) for p in points_wkt]
    result = branches_and_nodes.angle_to_point(*points)
    assert np.isclose(result, 180.0)


def test_with_known_snapping_error_data():
    linestrings = samples.results_in_non_simple_from_branches_and_nodes_linestring_list
    result, any_changed_applied = branches_and_nodes.snap_traces(
        gpd.GeoSeries(linestrings), Helpers.snap_threshold
    )
    count = 0
    while any_changed_applied:
        result, any_changed_applied = branches_and_nodes.snap_traces(
            result, Helpers.snap_threshold
        )
        count += 1
        if count > 10:
            raise RecursionError()

    # while any_changed_applied:
    #     result, any_changed_applied = branches_and_nodes.snap_traces(
    #         gpd.GeoSeries(linestrings), Helpers.snap_threshold
    #     )
    for ls in result:
        assert ls.is_simple
        assert isinstance(ls, LineString)


def test_with_known_mls_error():
    linestrings = samples.mls_from_these_linestrings_list
    target_area = [box(*MultiLineString(linestrings).bounds)]
    branches, nodes = branches_and_nodes.branches_and_nodes(
        gpd.GeoSeries(linestrings), gpd.GeoSeries(target_area), Helpers.snap_threshold
    )
    for branch in branches.geometry:
        assert EE_branch not in str(branches[CONNECTION_COLUMN])
        assert isinstance(branch, LineString)
        assert branch.is_simple
        assert not branch.is_empty
    for node in nodes.geometry:
        assert isinstance(node, Point)
        assert not node.is_empty


# def test_with_known_snapping_error_data_alt():
#     linestrings = samples.results_in_non_simple_from_branches_and_nodes_linestring_list
#     result, any_changed_applied = branches_and_nodes.snap_traces_alternative(
#         gpd.GeoSeries(linestrings), Helpers.snap_threshold
#     )
#     count = 0
#     while any_changed_applied:
#         result, any_changed_applied = branches_and_nodes.snap_traces_alternative(
#             result, Helpers.snap_threshold
#         )
#         count += 1
#         if count > 10:
#             raise RecursionError()

#     # while any_changed_applied:
#     #     result, any_changed_applied = branches_and_nodes.snap_traces(
#     #         gpd.GeoSeries(linestrings), Helpers.snap_threshold
#     #     )
#     for ls in result:
#         assert ls.is_simple


@pytest.mark.parametrize(
    "trace,another,snap_threshold,assumed_result",
    Helpers.test_snap_trace_to_another_params,
)
def test_snap_trace_to_another(
    trace: LineString,
    another: LineString,
    snap_threshold: float,
    assumed_result: LineString,
):
    result, _ = branches_and_nodes.snap_trace_to_another(
        trace, another, snap_threshold=snap_threshold
    )
    assert result.wkt == assumed_result.wkt


# def test_snap_traces_alternative_v2()


@pytest.mark.parametrize(
    "traces,areas,snap_threshold,allowed_loops,already_clipped",
    Helpers.test_branches_and_nodes_regression_params,
)
def test_branches_and_nodes_regression(
    traces, areas, snap_threshold, allowed_loops, already_clipped, file_regression
):
    branches, nodes = branches_and_nodes.branches_and_nodes(
        traces, areas, snap_threshold, allowed_loops, already_clipped
    )

    branches.to_file("/mnt/f/Users/nikke/Documents/projects/Misc/dump/branches.gpkg", driver="GPKG")
    nodes.to_file("/mnt/f/Users/nikke/Documents/projects/Misc/dump/nodes.gpkg", driver="GPKG")

    file_regression.check(branches.to_json() + nodes.to_json())

