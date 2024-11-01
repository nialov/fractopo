"""
Scripts for creating sample grids for fracture trace, branch and node data.
"""

import logging
import platform
from typing import Any, Dict, Optional

import geopandas as gpd
import numpy as np
from geopandas.sindex import SpatialIndex
from joblib import Parallel, delayed
from shapely.geometry import LineString, Point, Polygon

from fractopo.analysis.parameters import (
    determine_node_type_counts,
    determine_topology_parameters,
)
from fractopo.branches_and_nodes import branches_and_nodes
from fractopo.general import (
    CLASS_COLUMN,
    GEOMETRY_COLUMN,
    JOBLIB_CACHE,
    Param,
    crop_to_target_areas,
    geom_bounds,
    safe_buffer,
    spatial_index_intersection,
)

log = logging.getLogger(__name__)


def create_grid(cell_width: float, lines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Create an empty polygon grid for sampling fracture line data.

    Grid is created to always contain all given lines.

    E.g.

    >>> lines = gpd.GeoSeries(
    ...     [
    ...         LineString([(1, 1), (2, 2)]),
    ...         LineString([(2, 2), (3, 3)]),
    ...         LineString([(3, 0), (2, 2)]),
    ...         LineString([(2, 2), (-2, 5)]),
    ...     ]
    ... )
    >>> create_grid(cell_width=0.1, lines=lines).head(5)
                                                geometry
    0   POLYGON ((-2 5, -1.9 5, -1.9 4.9, -2 4.9, -2 5))
    1  POLYGON ((-2 4.9, -1.9 4.9, -1.9 4.8, -2 4.8, ...
    2  POLYGON ((-2 4.8, -1.9 4.8, -1.9 4.7, -2 4.7, ...
    3  POLYGON ((-2 4.7, -1.9 4.7, -1.9 4.6, -2 4.6, ...
    4  POLYGON ((-2 4.6, -1.9 4.6, -1.9 4.5, -2 4.5, ...
    """
    assert cell_width > 0
    assert len(lines) > 0
    assert all(isinstance(val, LineString) for val in lines.geometry.values)

    # Get total bounds of branches
    x_min, y_min, x_max, y_max = lines.total_bounds
    cell_height = cell_width

    # Calculate cell row and column counts
    rows = int(np.ceil((y_max - y_min) / cell_height))
    cols = int(np.ceil((x_max - x_min) / cell_width))

    x_left_origin = x_min
    x_right_origin = x_min + cell_width
    y_top_origin = y_max
    y_bottom_origin = y_max - cell_height
    polygons = []
    # Create grid cell polygons
    for _ in range(cols):
        y_top = y_top_origin
        y_bottom = y_bottom_origin
        for _ in range(rows):
            polygons.append(
                Polygon(
                    [
                        (x_left_origin, y_top),
                        (x_right_origin, y_top),
                        (x_right_origin, y_bottom),
                        (x_left_origin, y_bottom),
                    ]
                )
            )
            y_top = y_top - cell_height
            y_bottom = y_bottom - cell_height
        x_left_origin = x_left_origin + cell_width
        x_right_origin = x_right_origin + cell_width

    # Create GeoDataFrame with grid polygons
    grid = gpd.GeoDataFrame({GEOMETRY_COLUMN: polygons}, crs=lines.crs)
    assert len(grid) != 0
    return grid


def populate_sample_cell(
    sample_cell: Polygon,
    sample_cell_area: float,
    traces: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    branches: gpd.GeoDataFrame,
    snap_threshold: float,
    resolve_branches_and_nodes: bool,
    traces_sindex: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Take a single grid polygon and populate it with parameters.

    Mauldon determination requires that E-nodes are defined for
    every single sample circle. If correct Mauldon values are
    wanted `resolve_branches_and_nodes` must be passed as True.
    This will result in much longer analysis time.
    """

    def resolve_samples(candidates, sample_circle):
        """
        Crop traces or branches to sample circle

        First check if any geometries intersect.
        If not: sample_features is an empty GeoDataFrame.
        """
        if any(
            candidate.intersects(sample_circle)
            for candidate in candidates.geometry.values
        ):
            samples = crop_to_target_areas(
                traces=candidates,
                areas=gpd.GeoSeries([sample_circle]),
                is_filtered=True,
                keep_column_data=False,
            )
        else:
            samples = candidates.iloc[0:0]
        return samples

    def choose_geometries(sindex, sample_circle, geometries):
        candidates_idx = spatial_index_intersection(sindex, geom_bounds(sample_circle))
        candidates = geometries.iloc[candidates_idx]
        assert isinstance(candidates, gpd.GeoDataFrame)
        return candidates

    centroid = sample_cell.centroid
    if not isinstance(centroid, Point):
        raise TypeError("Expected Point centroid.")
    sample_circle = safe_buffer(centroid, np.sqrt(sample_cell_area) * 1.5)
    sample_circle_area = sample_circle.area
    assert sample_circle_area > 0

    if traces_sindex is None:
        traces_sindex: SpatialIndex = traces.sindex

    # Choose geometries that are either within the sample_circle or
    # intersect it
    # Use spatial indexing to filter to only spatially relevant traces,
    # traces and nodes
    # trace_candidates_idx = spatial_index_intersection(
    #     traces_sindex, geom_bounds(sample_circle)
    # )
    trace_candidates = choose_geometries(
        sindex=traces_sindex, sample_circle=sample_circle, geometries=traces
    )

    if len(trace_candidates) == 0:
        return determine_topology_parameters(
            trace_length_array=np.array([]),
            node_counts=determine_node_type_counts(np.array([]), branches_defined=True),
            area=sample_circle_area,
            branch_length_array=np.array([]),
            branches_defined=True,
            correct_mauldon=resolve_branches_and_nodes,
        )
    if resolve_branches_and_nodes:
        # Solve branches and nodes for each cell if wanted
        # Only way to make sure Mauldon parameters are correct
        branches, nodes = branches_and_nodes(
            traces=trace_candidates,
            areas=gpd.GeoSeries([sample_circle], crs=traces.crs),
            snap_threshold=snap_threshold,
        )

    is_topology_defined = branches.shape[0] > 0
    if is_topology_defined:
        branch_candidates = choose_geometries(
            sindex=branches.sindex,
            sample_circle=sample_circle,
            geometries=branches,
        )

        node_candidates = choose_geometries(
            sindex=nodes.sindex,
            sample_circle=sample_circle,
            geometries=nodes,
        )
        sample_branches = resolve_samples(
            candidates=branch_candidates, sample_circle=sample_circle
        )
        if any(node.intersects(sample_circle) for node in nodes.geometry.values):
            # if any(nodes.intersects(sample_circle)):
            # TODO: Is node clipping stable?
            sample_nodes = gpd.clip(node_candidates, sample_circle)
            assert sample_nodes is not None
            assert all(isinstance(val, Point) for val in sample_nodes.geometry.values)
        else:
            sample_nodes = nodes.iloc[0:0]
        assert isinstance(sample_nodes, gpd.GeoDataFrame)
        assert isinstance(sample_branches, gpd.GeoDataFrame)
        sample_node_type_values = sample_nodes[CLASS_COLUMN].values
        assert isinstance(sample_node_type_values, np.ndarray)
        node_counts = determine_node_type_counts(
            sample_node_type_values, branches_defined=True
        )
        sample_branches_lengths = sample_branches.geometry.length.values
    else:
        sample_branches_lengths = np.array([])
        node_counts = determine_node_type_counts(np.array([]), branches_defined=True)

    sample_traces = resolve_samples(
        candidates=trace_candidates, sample_circle=sample_circle
    )

    assert isinstance(sample_traces, gpd.GeoDataFrame)

    topology_parameters = determine_topology_parameters(
        trace_length_array=sample_traces.geometry.length.values,
        branch_length_array=sample_branches_lengths,
        node_counts=node_counts,
        area=sample_circle_area,
        correct_mauldon=resolve_branches_and_nodes,
        # TODO: Sample grids are currently not implemented for Networks
        # without determined topology.
        branches_defined=is_topology_defined,
    )
    return topology_parameters


def sample_grid(
    grid: gpd.GeoDataFrame,
    traces: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    branches: gpd.GeoDataFrame,
    snap_threshold: float,
    resolve_branches_and_nodes: bool = False,
) -> gpd.GeoDataFrame:
    """
    Populate a sample polygon grid with geometrical and topological parameters.
    """
    sample_cell_area = grid.geometry.iloc[0].area
    assert sample_cell_area != 0
    # String identifiers as parameters
    # dict with lists of parameter values
    params = dict()  # type: Dict[str, list]
    # Iterate over sample cells
    # Uses a buffer 1.5 times the size of the sample_cell side length
    # Make sure index does not cause issues TODO
    traces_reset, nodes_reset, branches_reset = (
        traces.reset_index(drop=True),
        nodes.reset_index(drop=True),
        branches.reset_index(drop=True),
    )
    assert isinstance(traces_reset, gpd.GeoDataFrame)
    assert isinstance(nodes_reset, gpd.GeoDataFrame)
    assert isinstance(branches_reset, gpd.GeoDataFrame)
    traces, nodes, branches = traces_reset, nodes_reset, branches_reset

    # Use all CPUs with n_jobs=-1
    # Use only one process on Windows
    params_for_cells = Parallel(
        n_jobs=(-1 if not platform.system() == "Windows" else 1)
    )(
        delayed(populate_sample_cell)(
            sample_cell=sample_cell,
            sample_cell_area=sample_cell_area,
            traces_sindex=None,
            traces=traces,
            nodes=nodes,
            branches=branches,
            snap_threshold=snap_threshold,
            resolve_branches_and_nodes=resolve_branches_and_nodes,
        )
        for sample_cell in grid.geometry.values
    )
    assert isinstance(params_for_cells, list)
    for key in [param.value.name for param in Param]:
        assert isinstance(key, str)
        try:
            params[key] = [cell_param[key] for cell_param in params_for_cells]
        except KeyError:
            continue
    for key, value in params.items():
        grid[key] = value
    return grid


@JOBLIB_CACHE.cache
def run_grid_sampling(
    traces: gpd.GeoDataFrame,
    branches: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    cell_width: float,
    snap_threshold: float,
    precursor_grid: Optional[gpd.GeoDataFrame] = None,
    resolve_branches_and_nodes: bool = False,
) -> gpd.GeoDataFrame:
    """
    Run the contour grid sampling to passed trace, branch and node data.

    The grid extents are determined by using the passed branches unless
    precursor_grid is passed.

    If precursor_grid is passed, cell_width is not requires. Otherwise
    it must always be set to be to a non-default value.
    """
    if traces.empty:
        log.warning("Empty GeoDataFrame passed to run_grid_sampling.")
        return gpd.GeoDataFrame()
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
        is_topology_defined = branches.shape[0] > 0
        if is_topology_defined:
            lines = branches
        else:
            lines = traces
        grid = create_grid(cell_width, lines=lines)
    sampled_grid = sample_grid(
        grid,
        traces,
        nodes,
        branches=branches,
        snap_threshold=snap_threshold,
        resolve_branches_and_nodes=resolve_branches_and_nodes,
    )
    return sampled_grid
