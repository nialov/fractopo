"""
Tests for contour_grid.
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from shapely.geometry import LineString, Point, Polygon

import tests
from fractopo.analysis import contour_grid
from fractopo.analysis.network import Network
from fractopo.general import (
    CC_branch,
    CI_branch,
    II_branch,
    Param,
    pygeos_spatial_index,
)
from tests import Helpers

CELL_WIDTH = 0.10
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


@pytest.mark.parametrize("cell_width", [CELL_WIDTH, 0.15, 0.25, 0.5])
def test_create_grid(cell_width: float):
    """
    Test create_grid.
    """
    assert isinstance(branches, gpd.GeoDataFrame)
    grid = contour_grid.create_grid(cell_width, branches)
    assert isinstance(grid, gpd.GeoDataFrame)
    some_intersect = False
    for cell in grid.geometry:
        # if any(branches.intersects(cell)):
        if any(branch.intersects(cell) for branch in branches.geometry.values):
            some_intersect = True
    assert some_intersect
    return grid


@pytest.mark.parametrize("snap_threshold", [0.01, 0.001])
def test_sample_grid(snap_threshold: float):
    """
    Test sampling a grid.
    """
    grid = test_create_grid(cell_width=CELL_WIDTH)
    grid_with_topo = contour_grid.sample_grid(
        grid, branches, nodes, snap_threshold=snap_threshold
    )
    assert isinstance(grid_with_topo, gpd.GeoDataFrame)


@pytest.mark.parametrize(
    "traces,areas,snap_threshold", Helpers.test_network_contour_grid_params
)
def test_network_contour_grid(traces, areas, snap_threshold, dataframe_regression):
    """
    Test contour_grid with network determined b and n.
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
    sampled_grid = network.contour_grid()
    fig, ax = network.plot_contour(
        parameter=Param.FRACTURE_INTENSITY_P21.value, sampled_grid=sampled_grid
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig=fig)

    assert isinstance(sampled_grid, gpd.GeoDataFrame)
    assert sampled_grid.shape[0] > 0
    assert isinstance(sampled_grid.geometry.values[0], Polygon)

    sampled_grid.sort_index(inplace=True)
    dataframe_regression.check(pd.DataFrame(sampled_grid).drop(columns=["geometry"]))


@pytest.mark.parametrize(
    "sample_cell,traces,snap_threshold", tests.test_populate_sample_cell_new_params()
)
def test_populate_sample_cell(sample_cell, traces, snap_threshold):
    """
    Test populate_sample_cell.
    """
    result = contour_grid.populate_sample_cell(
        sample_cell,
        sample_cell.area,
        pygeos_spatial_index(traces),
        nodes=gpd.GeoDataFrame(),
        traces=traces,
        snap_threshold=snap_threshold,
        resolve_branches_and_nodes=True,
    )

    assert isinstance(result, dict)