from pathlib import Path
from typing import Dict, List, Tuple

import fiona
import geopandas as gpd
import hypothesis
import numpy as np
import pytest
import shapely
from matplotlib import pyplot as plt
from shapely.geometry import LineString, Point, Polygon

from fractopo import branches_and_nodes, contour_grid
from fractopo.general import CC_branch, CI_branch, II_branch


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
    grid = contour_grid.create_grid(cell_width, branches)
    assert isinstance(grid, gpd.GeoDataFrame)
    some_intersect = False
    for cell in grid.geometry:
        if any(branches.intersects(cell)):
            some_intersect = True
    assert some_intersect
    return grid


def test_sample_grid():
    grid = test_create_grid()
    grid_with_topo = contour_grid.sample_grid(
        grid, branches, nodes, snap_threshold=0.01
    )
    assert isinstance(grid_with_topo, gpd.GeoDataFrame)


# def test_sample_grid_with_regressions(file_regression):
# TODO: REDO. KB7 probably not valid.
#     trace_data = "tests/sample_data/KB7/KB7_tulkinta.shp"
#     area_data = "tests/sample_data/KB7/KB7_tulkinta_alue.shp"
#     trace_data_path = Path(trace_data)
#     area_data_path = Path(area_data)
#     assert trace_data_path.exists()
#     assert area_data_path.exists()
#     trace_gdf = gpd.GeoDataFrame.from_file(trace_data_path)
#     area_gdf = gpd.GeoDataFrame.from_file(area_data_path)
#     branches, nodes = branches_and_nodes.branches_and_nodes(
#         trace_gdf.geometry, area_gdf.geometry, 0.0001
#     )
#     sample_grid = contour_grid.create_grid(10000, trace_gdf)
#     populated_grid = contour_grid.sample_grid(
#         sample_grid, trace_gdf, nodes, snap_threshold=0.01
#     )
#     file_regression.check(populated_grid.to_json())
