"""
Tests for branch and node determination.
"""
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import pytest
from hypothesis import given
from shapely import wkt
from shapely.geometry import LineString, MultiLineString, Point, box

from fractopo import branches_and_nodes, general
from fractopo.branches_and_nodes import CONNECTION_COLUMN, EE_branch
from tests import Helpers, trace_builder
from tests.sample_data.py_samples import samples

# Import trace_validator


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
    """
    Test determine_branch_identity.
    """
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


def test_snap_traces_simple():
    """
    Test snap_traces with simple data.
    """
    simple_traces = gpd.GeoSeries(
        [LineString([(0, 0), (0.99, 0)]), LineString([(1, -1), (1, 1)])]
    )
    simple_snap_threshold = 0.02
    simple_snapped_traces, _ = branches_and_nodes.snap_traces(
        [
            line
            for line in simple_traces.geometry.values
            if isinstance(line, LineString)
        ],
        simple_snap_threshold,
    )
    first_coords = simple_snapped_traces[0].coords
    first_coords_points = [Point(c) for c in first_coords]
    assert any(p.intersects(simple_snapped_traces[1]) for p in first_coords_points)
    assert all([isinstance(ls, LineString) for ls in simple_snapped_traces])
    for xy in simple_snapped_traces[1].coords:
        p = Point(xy)
        if Point(0.99, 0).intersects(p):
            pass
            # is_in_ls = True


@pytest.mark.parametrize(
    "linestring, point, snap_threshold, assumed_result",
    Helpers.test_insert_point_to_linestring_params,
)
def test_insert_point_to_linestring(linestring, point, snap_threshold, assumed_result):
    """
    Test insert_point_to_linestring.
    """
    result = branches_and_nodes.insert_point_to_linestring(
        linestring, point, snap_threshold
    )
    if assumed_result is None:
        # Assert it is in list
        assert tuple(*point.coords) in list(result.coords)
        # Assert index is correct
        assert list(result.coords).index(tuple(*point.coords)) == 1
    else:
        assert result.wkt == assumed_result.wkt


def test_nice_traces():
    """
    Test snap_traces with nice trace.
    """
    nice_traces = Helpers.get_nice_traces()
    nice_traces_list = [
        trace for trace in nice_traces.geometry.values if isinstance(trace, LineString)
    ]
    assert nice_traces.shape[0] == len(nice_traces_list)
    snapped_traces, _ = branches_and_nodes.snap_traces(
        nice_traces_list, Helpers.snap_threshold
    )
    assert len(nice_traces) == len(snapped_traces)
    for geom in snapped_traces:
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
    assert isinstance(valid_result, (gpd.GeoDataFrame, gpd.GeoSeries))
    assert valid_geoseries.geometry.length.mean() > valid_result.geometry.length.mean()


@given(Helpers.triple_tuples)
def test_angle_to_point(triple_tuples):
    """
    Test angle_to_point.
    """
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
    """
    Test angle_to_point with known error.
    """
    points_wkt = [
        "POINT (286975.148 6677657.7042)",
        "POINT (284919.7632999998 6677522.154200001)",
        "POINT (280099.6969999997 6677204.276900001)",
    ]
    points_loaded = [wkt.loads(p) for p in points_wkt]
    points = [p for p in points_loaded if isinstance(p, Point)]
    assert len(points) == len(points_loaded)
    result = branches_and_nodes.angle_to_point(*points)
    assert np.isclose(result, 180.0)


def test_with_known_snapping_error_data():
    """
    Test snap_traces with known snapping error data.
    """
    linestrings = samples.results_in_non_simple_from_branches_and_nodes_linestring_list
    result, any_changed_applied = branches_and_nodes.snap_traces(
        linestrings, Helpers.snap_threshold
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
    """
    Test branches_and_nodes with known mls error.
    """
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
    "trace_endpoints,another,snap_threshold,assumed_result",
    Helpers.test_snap_trace_to_another_params,
)
def test_snap_trace_to_another(
    trace_endpoints: List[Point],
    another: LineString,
    snap_threshold: float,
    assumed_result: LineString,
):
    result, _ = branches_and_nodes.snap_trace_to_another(
        trace_endpoints, another, snap_threshold=snap_threshold
    )
    assert result.wkt == assumed_result.wkt


@pytest.mark.parametrize(
    "traces,areas,snap_threshold,allowed_loops,already_clipped",
    Helpers.test_branches_and_nodes_regression_params,
)
def test_branches_and_nodes_regression(
    traces, areas, snap_threshold, allowed_loops, already_clipped, data_regression
):
    """
    Test branches and nodes with regression.
    """
    branches, nodes = branches_and_nodes.branches_and_nodes(
        traces, areas, snap_threshold, allowed_loops, already_clipped
    )

    branches_value_counts = branches[general.CONNECTION_COLUMN].value_counts().to_dict()
    nodes_value_counts = nodes[general.CLASS_COLUMN].value_counts().to_dict()

    data_regression.check({**branches_value_counts, **nodes_value_counts})


def test_branches_and_nodes_troubling():
    """
    Test branches and nodes with known troubling data.
    """
    traces = Helpers.troubling_traces
    areas = Helpers.sample_areas
    snap_threshold = 0.001
    branches, nodes = branches_and_nodes.branches_and_nodes(
        traces, areas, snap_threshold, allowed_loops=10, already_clipped=False
    )
    assert isinstance(branches, gpd.GeoDataFrame)
    assert isinstance(nodes, gpd.GeoDataFrame)


@pytest.mark.parametrize(
    "trace,trace_candidates,snap_threshold,intersects_idx",
    Helpers.test_simple_snap_params,
)
def test_simple_snap(trace, trace_candidates, snap_threshold, intersects_idx):
    """
    Test branches_and_nodes.simple_snap.
    """
    result, was_simple_snapped = branches_and_nodes.simple_snap(
        trace, trace_candidates, snap_threshold
    )
    if intersects_idx is not None:
        assert was_simple_snapped
    assert result.intersects(trace_candidates.geometry.values[intersects_idx])


@pytest.mark.parametrize(
    "idx,trace,snap_threshold,traces,intersects_idx",
    Helpers.test_snap_trace_simple_params,
)
def test_snap_trace_simple(
    idx,
    trace,
    snap_threshold,
    traces,
    intersects_idx,
):
    """
    Test snap_trace_simple.
    """
    traces_spatial_index = general.pygeos_spatial_index(gpd.GeoSeries(traces))
    result, was_simple_snapped = branches_and_nodes.snap_trace_simple(
        idx, trace, snap_threshold, traces, traces_spatial_index
    )
    if intersects_idx is not None:
        assert was_simple_snapped
    assert result.intersects(traces[intersects_idx])


@pytest.mark.parametrize(
    "traces_geosrs,snap_threshold,size_threshold", Helpers.test_safer_unary_union_params
)
def test_safer_unary_union(traces_geosrs, snap_threshold, size_threshold):
    """
    Test safer_unary_union.
    """
    try:
        result = branches_and_nodes.safer_unary_union(
            traces_geosrs, snap_threshold, size_threshold
        )
    except ValueError:
        if size_threshold < branches_and_nodes.UNARY_ERROR_SIZE_THRESHOLD:
            return
        raise
    assert len(list(result.geoms)) >= traces_geosrs.shape[0]


@pytest.mark.parametrize(
    "loops,allowed_loops,will_error", Helpers.test_report_snapping_loop_params
)
def test_report_snapping_loop(loops, allowed_loops, will_error):
    """
    Test report_snapping_loop.
    """
    try:
        result = branches_and_nodes.report_snapping_loop(loops, allowed_loops)
    except RecursionError:
        if will_error:
            return
        raise
    assert result is None
