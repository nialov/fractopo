import geopandas as gpd
import shapely
from shapely.geometry import LineString, Point, MultiPoint
from shapely.ops import snap, split
import shapely
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# Import trace_validator
from tval import trace_validator, trace_builder
from fractopo import branches_and_nodes


class Helpers:

    snap_threshold = 0.001
    geosrs_identicals = gpd.GeoSeries(
        [Point(1, 1), Point(1, 1), Point(2, 1), Point(2, 1), Point(3, 1), Point(2, 3)]
    )

    traces_geosrs = gpd.GeoSeries(
        [
            LineString([(-1, 0), (1, 0)]),
            LineString([(0, -1), (0, 1)]),
            LineString([(-1.0 - snap_threshold, -1), (-1.0 - snap_threshold, 1)]),
        ]
    )
    nodes_geosrs = gpd.GeoSeries(
        [
            Point(-1, 0),
            Point(1, 0),
            Point(0, -1),
            Point(0, 1),
            Point(-1.0 - snap_threshold, -1),
            Point(-1.0 - snap_threshold, 1),
            Point(0, 0),
        ]
    )

    @classmethod
    def get_geosrs_identicals(cls):
        return cls.geosrs_identicals.copy()

    @classmethod
    def get_matching_traces_and_nodes_geosrs(cls):
        return cls.traces_geosrs.copy(), cls.nodes_geosrs.copy()


def test_remove_identical():
    geosrs = Helpers.get_geosrs_identicals()
    geosrs_orig_length = len(geosrs)
    result = branches_and_nodes.remove_identical(geosrs, Helpers.snap_threshold)
    result_length = len(result)
    assert geosrs_orig_length > result_length
    assert geosrs_orig_length - result_length == 2


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
    traces_geosrs, nodes_geosrs = Helpers.get_matching_traces_and_nodes_geosrs()
    result = branches_and_nodes.get_node_identities(
        traces_geosrs, nodes_geosrs, Helpers.snap_threshold
    )
    print(nodes_geosrs, result)
    assert "X" in result and "Y" in result and "I" in result
    assert len([r for r in result if r == "X"]) == 1
    assert len([r for r in result if r == "Y"]) == 1
    assert len([r for r in result if r == "I"]) == 5


def test_find_y_nodes_and_snap_em():
    traces_geosrs, nodes_geosrs = Helpers.get_matching_traces_and_nodes_geosrs()
    node_identities = branches_and_nodes.get_node_identities(
        traces_geosrs, nodes_geosrs, Helpers.snap_threshold
    )
    result = branches_and_nodes.find_y_nodes_and_snap_em(
        traces_geosrs, nodes_geosrs, node_identities, Helpers.snap_threshold
    )
    assert len(nodes_geosrs) == len(node_identities) == len(result)
    assert any([Point(-1 - Helpers.snap_threshold, 0).intersects(p) for p in result])
    assert (
        len([p for p in result if Point(-1 - Helpers.snap_threshold, 0).intersects(p)])
        == 1
    )


def test_branches_and_nodes():
    traces = gpd.GeoSeries(trace_builder.make_valid_traces())
    branch_geodataframe, node_geodataframe = branches_and_nodes.branches_and_nodes(
        traces, Helpers.snap_threshold
    )

    for node_id in node_geodataframe["Type"]:
        assert node_id in ["X", "Y", "I"]
    assert len([node_id for node_id in node_geodataframe["Type"] if node_id == "X"]) > 0
    assert len([node_id for node_id in node_geodataframe["Type"] if node_id == "Y"]) > 0
    assert len([node_id for node_id in node_geodataframe["Type"] if node_id == "I"]) > 1


def test_get_branch_identities():
    traces_geosrs, nodes_geosrs = Helpers.get_matching_traces_and_nodes_geosrs()
    node_identities = branches_and_nodes.get_node_identities(
        traces_geosrs, nodes_geosrs, Helpers.snap_threshold
    )
    nodes_geosrs = branches_and_nodes.find_y_nodes_and_snap_em(
        traces_geosrs, nodes_geosrs, node_identities, Helpers.snap_threshold
    )
    branches_geosrs = branches_and_nodes.split_traces_to_branches(
        traces_geosrs, nodes_geosrs, node_identities
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
    simple_snapped_traces = branches_and_nodes.snap_traces(
        simple_traces, simple_snap_threshold
    )
    first_coords = simple_snapped_traces.iloc[0].coords
    first_coords_points = [Point(c) for c in first_coords]
    assert any(
        [p.intersects(simple_snapped_traces.iloc[1]) for p in first_coords_points]
    )
