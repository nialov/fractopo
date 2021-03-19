"""
Tests for contour_grid.
"""
import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon

from fractopo import contour_grid
from fractopo.analysis.network import Network
from fractopo.general import CC_branch, CI_branch, II_branch
from tests import Helpers

cell_width = 0.10
branches = gpd.GeoDataFrame(
    {
        "geometry": [
            LineString(((0, 0), (1, 1))),
            LineString(((0, 0), (-1, 1))),
            LineString(((0, 0), (1, -1))),
            LineString(((0, 0), (0.5, 1))),
            LineString(((0, 0), (0.5, 0.5))),
            LineString(((0, 0), (1.5, 1.5))),
        ]
    }
)

nodes = gpd.GeoDataFrame(
    {
        "geometry": [
            Point((0, 0)),
            Point((0, 1)),
            Point((0.5, 1)),
            Point((-0.5, -1)),
            Point((5, -1)),
            Point((1, 1)),
        ],
        "Class": [
            CC_branch,
            II_branch,
            CI_branch,
            CC_branch,
            CI_branch,
            II_branch,
        ],
    }
)


def test_create_grid():
    """
    Test create_grid.
    """
    grid = contour_grid.create_grid(cell_width, branches)
    assert isinstance(grid, gpd.GeoDataFrame)
    some_intersect = False
    for cell in grid.geometry:
        if any(branches.intersects(cell)):
            some_intersect = True
    assert some_intersect
    return grid


def test_sample_grid():
    """
    Test sampling a grid.
    """
    grid = test_create_grid()
    grid_with_topo = contour_grid.sample_grid(
        grid, branches, nodes, snap_threshold=0.01
    )
    assert isinstance(grid_with_topo, gpd.GeoDataFrame)


@pytest.mark.parametrize(
    "traces,areas,snap_threshold", Helpers.test_run_grid_sampling_params
)
def test_run_grid_sampling(traces, areas, snap_threshold):
    """
    Test sample_grid with network determined b and n.
    """
    traces = traces.iloc[0:150]
    network = Network(
        trace_gdf=traces,
        area_gdf=areas,
        determine_branches_nodes=True,
        snap_threshold=snap_threshold,
        truncate_traces=True,
        circular_target_area=False,
    )
    branches, nodes = network.branch_gdf, network.node_gdf
    assert isinstance(branches, gpd.GeoDataFrame)
    assert isinstance(nodes, gpd.GeoDataFrame)
    result = contour_grid.run_grid_sampling(
        traces=network.trace_gdf,
        branches=branches,
        nodes=nodes,
        snap_threshold=snap_threshold,
        cell_width=5,
    )

    assert isinstance(result, gpd.GeoDataFrame)
    assert result.shape[0] > 0
    assert isinstance(result.geometry.values[0], Polygon)
