"""
File: contour_grid.py
Author: Nikolas Ovaskainen
Github: https://github.com/nialov
Description: Scripts for creating sample grids for fracture trace, branch and
node data
"""
import geopandas as gpd
import fiona
import shapely
from shapely.geometry import Polygon, LineString, Point
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from typing import Dict, Tuple, List, Optional

CC_branch = "C - C"
CI_branch = "C - I"
II_branch = "I - I"
EE_branch = "E - E"
X_node = "X"
Y_node = "Y"
I_node = "I"


def create_grid(cell_width: float, branches: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Creates an empty polygon grid for sampling fracture branch data.
    Grid is created to always contain all given branches.

    E.g.

    >>> branches = gpd.GeoSeries(
    ...     [
    ...             LineString([(1, 1), (2, 2)]),
    ...             LineString([(2, 2), (3, 3)]),
    ...             LineString([(3, 0), (2, 2)]),
    ...             LineString([(2, 2), (-2, 5)]),
    ...     ]
    ... )
    >>> create_grid(cell_width=0.1, branches=branches).head(5)
                                                geometry
    0  POLYGON ((-2.00000 5.00000, -1.90000 5.00000, ...
    1  POLYGON ((-2.00000 4.90000, -1.90000 4.90000, ...
    2  POLYGON ((-2.00000 4.80000, -1.90000 4.80000, ...
    3  POLYGON ((-2.00000 4.70000, -1.90000 4.70000, ...
    4  POLYGON ((-2.00000 4.60000, -1.90000 4.60000, ...
    """
    assert cell_width > 0
    assert len(branches) > 0
    # Get total bounds of branches
    xmin, ymin, xmax, ymax = branches.total_bounds
    cell_height = cell_width
    # Calculate cell row and column counts
    rows = int(np.ceil((ymax - ymin) / cell_height))
    cols = int(np.ceil((xmax - xmin) / cell_width))
    XleftOrigin = xmin
    XrightOrigin = xmin + cell_width
    YtopOrigin = ymax
    YbottomOrigin = ymax - cell_height
    polygons = []
    # Create grid cell polygons
    for i in range(cols):
        Ytop = YtopOrigin
        Ybottom = YbottomOrigin
        for j in range(rows):
            polygons.append(
                Polygon(
                    [
                        (XleftOrigin, Ytop),
                        (XrightOrigin, Ytop),
                        (XrightOrigin, Ybottom),
                        (XleftOrigin, Ybottom),
                    ]
                )
            )
            Ytop = Ytop - cell_height
            Ybottom = Ybottom - cell_height
        XleftOrigin = XleftOrigin + cell_width
        XrightOrigin = XrightOrigin + cell_width
    # Create GeoDataFrame with grid polygons
    grid = gpd.GeoDataFrame({"geometry": polygons}, crs=branches.crs)
    assert len(grid) != 0
    return grid


def populate_sample_cell(
    sample_cell: Polygon,
    sample_cell_area: float,
    traces_sindex: gpd.sindex.PyGEOSSTRTreeIndex,
    nodes_sindex: gpd.sindex.PyGEOSSTRTreeIndex,
    traces: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
) -> Dict[str, float]:
    """
    Takes a single grid polygon and populates it with parameters usings the
    other inputs.

    E.g.

    >>> traces = gpd.GeoDataFrame(
    ...     {
    ...             "geometry": [
    ...                         LineString([(1, 1), (2, 2), (3, 3)]),
    ...                         ]
    ...     }
    ... )
    >>> nodes = gpd.GeoDataFrame(
    ...     {
    ...             "geometry": [
    ...                         Point(2, 2),
    ...                         ],
    ...             "Class": [
    ...                         X_node,
    ...                      ],
    ...     }
    ... )
    >>> traces_sindex, nodes_sindex = traces.sindex, nodes.sindex
    >>> sample_cell = Polygon([(-2, -2), (-2, 2), (2, 2), (2, -2)])
    >>> [print(item) for item in populate_sample_cell(
    ...     sample_cell, sample_cell.area, traces_sindex, nodes_sindex, traces, nodes
    ... ).items()]
    ('Sample_Area', 112.91574565965382)
    ('Total_Length', 2.8284271247461903)
    ('Average_Trace_Length', 0)
    ('Average_Branch_Length', 1.4142135623730951)
    ('Branch_Frequency', 0.017712321592670707)
    ('Trace_Frequency', 0.0)
    ('Node_Frequency', 0.008856160796335354)
    ('P21', 0.025049005417468732)
    ('P22', 0.0)
    ('B22', 0.035424643185341415)
    ('Connections_Per_Branch', 2.0)
    ('Connections_Per_Trace', 0)
    ('Connection_Frequency', 0.008856160796335354)
    [None, None, None, None, None, None, None, None, None, None, None, None, None]

    """
    params = dict()
    sample_circle = sample_cell.centroid.buffer(
        np.sqrt(sample_cell_area) * 1.5  # type: ignore
    )
    sample_circle_area = sample_circle.area
    assert sample_circle_area > 0
    # Choose geometries that are either within the sample_circle or
    # intersect it
    # Use spatial indexing to filter to only spatially relevant traces,
    # traces and nodes
    trace_candidates_idx = list(traces_sindex.intersection(sample_circle.bounds))
    node_candidates_idx = list(nodes_sindex.intersection(sample_circle.bounds))

    trace_candidates = traces.iloc[trace_candidates_idx]
    node_candidates = nodes.iloc[node_candidates_idx]

    # Crop traces to sample circle
    # First check if any geometries intersect
    # If not: sample_features is an empty GeoDataFrame
    if any(trace_candidates.intersects(sample_circle)):  # type: ignore
        sample_traces = gpd.clip(trace_candidates, sample_circle)
    else:
        sample_traces = traces.iloc[0:0]
    if any(nodes.intersects(sample_circle)):
        sample_nodes = gpd.clip(node_candidates, sample_circle)
    else:
        sample_nodes = nodes.iloc[0:0]

    node_count_data = sample_nodes.groupby("Class").size()  # type: gpd.GeoSeries

    node_counts = dict()
    for node_type in [X_node, Y_node, I_node]:
        if node_type in node_count_data:
            node_counts[node_type] = node_count_data[node_type]
        else:
            node_counts[node_type] = 0

    x_count = node_counts[X_node]
    y_count = node_counts[Y_node]
    i_count = node_counts[I_node]
    node_count = x_count + y_count + i_count
    # Trace and branch counts also from nodes to avoid edge effects
    trace_count = (y_count + i_count) / 2.0
    branch_count = (x_count * 4 + y_count * 3 + i_count) / 2.0

    # Calculation of parameters
    total_trace_length = sum([geom.length for geom in sample_traces.geometry])
    # Total branch length is the sum of CC, CI and II branch lengths
    total_branch_length = total_trace_length
    if trace_count > 0:
        average_trace_length = total_trace_length / trace_count
    else:
        average_trace_length = 0
    if branch_count > 0:
        average_branch_length = total_branch_length / branch_count
    else:
        average_branch_length = 0

    trace_frequency = trace_count / sample_circle_area
    branch_frequency = branch_count / sample_circle_area
    node_frequency = node_count / sample_circle_area

    P21 = total_trace_length / sample_circle_area

    P22 = P21 * average_trace_length
    B22 = P21 * average_branch_length

    if branch_count > 0:
        connections_per_branch = (3 * y_count + 4 * x_count) / branch_count
        connections_per_branch = (
            0 if connections_per_branch > 2 else connections_per_branch
        )
    else:
        connections_per_branch = 0
    if trace_count > 0:
        connections_per_trace = 2 * (y_count + x_count) / trace_count
    else:
        connections_per_trace = 0

    connection_frequency = (y_count + x_count) / sample_circle_area
    # Append to results dict
    params["Sample_Area"] = sample_circle_area
    params["Total_Length"] = total_trace_length
    params["Average_Trace_Length"] = average_trace_length
    params["Average_Branch_Length"] = average_branch_length
    params["Branch_Frequency"] = branch_frequency
    params["Trace_Frequency"] = trace_frequency
    params["Node_Frequency"] = node_frequency
    params["P21"] = P21
    params["P22"] = P22
    params["B22"] = B22
    params["Connections_Per_Branch"] = connections_per_branch
    params["Connections_Per_Trace"] = connections_per_trace
    params["Connection_Frequency"] = connection_frequency
    return params


def sample_grid(
    grid: gpd.GeoDataFrame,
    traces: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Populates a sample polygon grid with geometrical and topological fracture
    network parameters.

    E.g.

    >>> branches = gpd.GeoDataFrame(
    ...     {
    ...             "geometry": [
    ...                         LineString([(1, 1), (2, 2), (3, 3)]),
    ...                         ]
    ...     }
    ... )
    >>> nodes = gpd.GeoDataFrame(
    ...     {
    ...             "geometry": [
    ...                         Point(2, 2),
    ...                         ],
    ...             "Class": [
    ...                         X_node,
    ...                      ],
    ...     }
    ... )
    >>> grid = create_grid(cell_width=0.5, branches=branches)
    >>> sample_grid(grid, branches, nodes).loc[5]
    geometry                  POLYGON ((1.5 2.5, 2 2.5, 2 2, 1.5 2, 1.5 2.5))
    Sample_Area                                                       1.76431
    Total_Length                                                      1.32287
    Average_Trace_Length                                                    0
    Average_Branch_Length                                            0.661437
    Branch_Frequency                                                  1.13359
    Trace_Frequency                                                         0
    Node_Frequency                                                   0.566794
    P21                                                              0.749798
    P22                                                                     0
    B22                                                              0.495944
    Connections_Per_Branch                                                  2
    Connections_Per_Trace                                                   0
    Connection_Frequency                                             0.566794
    Name: 5, dtype: object

    """
    sample_cell_area = grid.geometry.iloc[0].area
    assert sample_cell_area != 0
    # String identifiers as parameters
    # dict with lists of parameter values
    params = dict()  # type: Dict[str, list]
    param_keys = (
        "Sample_Area",
        "Total_Length",
        "Average_Trace_Length",
        "Average_Branch_Length",
        "Branch_Frequency",
        "Trace_Frequency",
        "Node_Frequency",
        "P21",
        "P22",
        "B22",
        "Connections_Per_Branch",
        "Connections_Per_Trace",
        "Connection_Frequency",
    )
    # Iterate over sample cells
    # Uses a buffer 1.5 times the size of the sample_cell side length
    # Make sure index doesnt cause issues TODO
    [gdf.reset_index(inplace=True, drop=True) for gdf in (traces, nodes)]
    traces_sindex = traces.sindex
    nodes_sindex = nodes.sindex

    params_for_cells = list(
        map(
            lambda sample_cell: populate_sample_cell(
                sample_cell,
                sample_cell_area,
                traces_sindex,
                nodes_sindex,
                traces,
                nodes,
            ),
            grid.geometry,
        )
    )
    for key in param_keys:
        params[key] = [cell_param[key] for cell_param in params_for_cells]

    for key in params:
        grid[key] = params[key]
    return grid


def run_grid_sampling(
    traces: gpd.GeoDataFrame,
    branches: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    cell_width=0.0,
    precursor_grid: Optional[gpd.GeoDataFrame] = None,
) -> gpd.GeoDataFrame:
    """
    Runs the contour grid sampling to passed trace, branch and node data.
    The grid extents are determined by using the passed branches unless
    precursor_grid is passed.

    If precursor_grid is passed, cell_width is not requires. Otherwise
    it must always be set to be to a non-default value.
    """
    if precursor_grid is not None:
        if not isinstance(precursor_grid, gpd.GeoDataFrame):
            raise TypeError("Expected precursor_grid to be of type: GeoDataFrame.")
        # Avoid modifying same precursor_grid multiple times
        grid = precursor_grid.copy()
    else:
        if np.isclose(cell_width, 0.0) or cell_width < 0:
            raise ValueError(
                "Expected cell_width to be non-close-to-zero positive number."
            )
        grid = create_grid(cell_width, branches)
    sampled_grid = sample_grid(grid, traces, nodes)
    return sampled_grid
