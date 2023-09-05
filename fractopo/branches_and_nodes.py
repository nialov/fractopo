"""
Functions for extracting branches and nodes from trace maps.

branches_and_nodes is the main entrypoint.
"""
import logging
import math
from itertools import chain, compress
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
from geopandas.sindex import PyGEOSSTRTreeIndex
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseMultipartGeometry
from shapely.ops import unary_union
from shapely.wkt import dumps

from fractopo.general import (
    CLASS_COLUMN,
    CONNECTION_COLUMN,
    GEOMETRY_COLUMN,
    JOBLIB_CACHE,
    CC_branch,
    CE_branch,
    CI_branch,
    E_node,
    EE_branch,
    Error_branch,
    I_node,
    IE_branch,
    II_branch,
    X_node,
    Y_node,
    check_for_z_coordinates,
    crop_to_target_areas,
    geom_bounds,
    get_trace_coord_points,
    get_trace_endpoints,
    line_intersection_to_points,
    numpy_to_python_type,
    point_to_point_unit_vector,
    pygeos_spatial_index,
    remove_z_coordinates_from_geodata,
    spatial_index_intersection,
)

log = logging.getLogger(__name__)

UNARY_ERROR_SIZE_THRESHOLD = 100


def determine_branch_identity(
    number_of_i_nodes: int, number_of_xy_nodes: int, number_of_e_nodes: int
) -> str:
    """
    Determine the identity of a branch.

    Is based on the amount of I-, XY- and E-nodes and returns it as a string.

    E.g.

    >>> determine_branch_identity(2, 0, 0)
    'I - I'

    >>> determine_branch_identity(1, 1, 0)
    'C - I'

    >>> determine_branch_identity(1, 0, 1)
    'I - E'

    """
    branch_type = Error_branch
    if number_of_i_nodes + number_of_e_nodes + number_of_xy_nodes != 2:
        log.error("Did not find 2 EXYI-nodes that intersected branch endpoints.\n")
        branch_type = Error_branch
    elif number_of_i_nodes == 2:
        branch_type = II_branch
    elif number_of_xy_nodes == 2:
        branch_type = CC_branch
    elif number_of_e_nodes == 2:
        branch_type = EE_branch
    elif number_of_i_nodes == 1 and number_of_xy_nodes == 1:
        branch_type = CI_branch
    elif number_of_e_nodes == 1 and number_of_xy_nodes == 1:
        branch_type = CE_branch
    elif number_of_e_nodes == 1 and number_of_i_nodes == 1:
        branch_type = IE_branch
    else:
        log.error("Unknown error in determine_branch_identity")
        branch_type = EE_branch
    return branch_type


def get_branch_identities(
    branches: gpd.GeoSeries,
    nodes: gpd.GeoSeries,
    node_identities: list,
    snap_threshold: float,
) -> List[str]:
    """
    Determine the types of branches for a GeoSeries of branches.

    i.e. C-C, C-I or I-I, + (C-E, E-E, I-E)

    >>> branches = gpd.GeoSeries(
    ...     [
    ...         LineString([(1, 1), (2, 2)]),
    ...         LineString([(2, 2), (3, 3)]),
    ...         LineString([(3, 0), (2, 2)]),
    ...         LineString([(2, 2), (-2, 5)]),
    ...     ]
    ... )
    >>> nodes = gpd.GeoSeries(
    ...     [
    ...         Point(2, 2),
    ...         Point(1, 1),
    ...         Point(3, 3),
    ...         Point(3, 0),
    ...         Point(-2, 5),
    ...     ]
    ... )
    >>> node_identities = ["X", "I", "I", "I", "E"]
    >>> snap_threshold = 0.001
    >>> get_branch_identities(branches, nodes, node_identities, snap_threshold)
    ['C - I', 'C - I', 'C - I', 'C - E']

    """
    assert len(nodes) == len(node_identities)
    node_spatial_index = pygeos_spatial_index(nodes)
    branch_identities = []
    for branch in branches.geometry.values:
        assert isinstance(branch, LineString)
        node_candidate_idxs = spatial_index_intersection(
            spatial_index=node_spatial_index, coordinates=geom_bounds(branch)
        )
        # node_candidate_idxs = list(node_spatial_index.intersection(branch.bounds))
        node_candidates = nodes.iloc[node_candidate_idxs]
        node_candidate_types = [node_identities[i] for i in node_candidate_idxs]

        # Use distance instead of two polygon buffers
        inter = [
            dist < snap_threshold
            for dist in node_candidates.distance(
                MultiPoint(list(get_trace_endpoints(branch)))
            ).values
        ]
        assert len(inter) == len(node_candidates)
        # nodes_that_intersect = node_candidates.loc[inter]
        nodes_that_intersect_types = list(compress(node_candidate_types, inter))
        number_of_E_nodes = sum(
            inter_id == E_node for inter_id in nodes_that_intersect_types
        )
        number_of_I_nodes = sum(
            inter_id == I_node for inter_id in nodes_that_intersect_types
        )
        number_of_XY_nodes = sum(
            inter_id in [X_node, Y_node] for inter_id in nodes_that_intersect_types
        )
        branch_identities.append(
            determine_branch_identity(
                number_of_I_nodes, number_of_XY_nodes, number_of_E_nodes
            )
        )

    return branch_identities


def angle_to_point(
    point: Point, nearest_point: Point, comparison_point: Point
) -> float:
    """
    Calculate the angle between two vectors.

    Vectors are made from the given points: Both vectors have the same first
    point, nearest_point, and second point is either point or comparison_point.

    Returns angle in degrees.

    E.g.

    >>> point = Point(1, 1)
    >>> nearest_point = Point(0, 0)
    >>> comparison_point = Point(-1, 1)
    >>> angle_to_point(point, nearest_point, comparison_point)
    90.0

    >>> point = Point(1, 1)
    >>> nearest_point = Point(0, 0)
    >>> comparison_point = Point(-1, 2)
    >>> angle_to_point(point, nearest_point, comparison_point)
    71.56505117707799

    """
    if (
        point.intersects(nearest_point)
        or point.intersects(comparison_point)
        or nearest_point.intersects(comparison_point)
    ):
        raise ValueError("Expected points in angle_to_point not to intersect.")

    unit_vector_1 = point_to_point_unit_vector(point=nearest_point, other_point=point)
    unit_vector_2 = point_to_point_unit_vector(
        point=nearest_point, other_point=comparison_point
    )
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    if dot_product > 1 or dot_product < -1:
        rad_angle = np.nan
    else:
        rad_angle = np.arccos(dot_product)
    if np.isnan(rad_angle):
        # Cannot determine with angle.
        unit_vector_sum_len = np.linalg.norm(unit_vector_1 + unit_vector_2)
        if np.isclose(unit_vector_sum_len, 0.0, atol=1e-07):
            return 180.0
        if np.isclose(unit_vector_sum_len, 2.0, atol=1e-07):
            return 0.0
        vectors_dict = dict(
            unit_vector_1=unit_vector_1,
            unit_vector_2=unit_vector_2,
            unit_vector_sum_len=unit_vector_sum_len,
        )
        raise ValueError(
            f"Could not determine point relationships. Vectors: {vectors_dict}"
        )
    degrees = numpy_to_python_type(np.rad2deg(rad_angle))
    assert 360.0 >= degrees >= 0.0
    assert isinstance(degrees, float)
    return degrees


def insert_point_to_linestring(
    trace: LineString, point: Point, snap_threshold: float
) -> LineString:
    """
    Insert/modify point to trace LineString.

    The point location is determined to fit into the LineString without
    changing the geometrical order of LineString vertices
    (which only makes sense if LineString is sublinear.)

    TODO: Does not work for 2.5D geometries (Z-coordinates).
    Z-coordinates will be lost.

    E.g.

    >>> trace = LineString([(0, 0), (1, 0), (2, 0), (3, 0)])
    >>> point = Point(1.25, 0.1)
    >>> insert_point_to_linestring(trace, point, 0.01).wkt
    'LINESTRING (0 0, 1 0, 1.25 0.1, 2 0, 3 0)'

    >>> trace = LineString([(0, 0), (1, 0), (2, 0), (3, 0)])
    >>> point = Point(2.25, 0.1)
    >>> insert_point_to_linestring(trace, point, 0.01).wkt
    'LINESTRING (0 0, 1 0, 2 0, 2.25 0.1, 3 0)'

    """
    assert isinstance(trace, LineString)
    assert isinstance(point, Point)
    if trace.has_z:
        log.warning("Trace contains z-coordinates. These will be lost.")
    if point.has_z:
        log.warning("Point contains z-coordinates. These will be lost.")

    if any(point.intersects(Point(xy)) for xy in list(trace.coords)):
        log.error(
            "Point already matches a coordinate point in trace.\n"
            f"point: {point.wkt}, trace: {trace.wkt}\n"
            "Returning original trace without changes."
        )
        return trace
    trace_point_dists = [
        (idx, trace_point, trace_point.distance(point))
        for idx, trace_point in enumerate([Point(c) for c in list(trace.coords)])
    ]
    trace_point_dists = sorted(trace_point_dists, key=lambda vals: vals[2])
    nearest_point = trace_point_dists[0][1]
    nearest_point_idx = trace_point_dists[0][0]

    # Determine if to insert or modify trace
    idx, insert = determine_insert_approach(
        nearest_point_idx, trace_point_dists, snap_threshold, point, nearest_point
    )

    t_coords = list(trace.coords)
    if not insert:
        t_coords.pop(idx)
    t_coords.insert(idx, (point.x, point.y)) if not point.has_z else t_coords.insert(
        idx, (point.x, point.y, point.z)
    )
    # Closest points might not actually be the points which in between the
    # point is added. Have to use project and interpolate (?)
    # print(t_coords)
    new_trace = LineString(t_coords)
    # print(new_trace.wkt)
    assert new_trace.intersects(point)
    assert isinstance(new_trace, LineString)
    assert new_trace.is_valid
    # assert new_trace.is_simple
    if not new_trace.is_simple:
        log.warning(f"Non-simple geometry detected.\n{new_trace.wkt}")
    return new_trace


def determine_insert_approach(
    nearest_point_idx: int,
    trace_point_dists: List[Tuple[int, Point, float]],
    snap_threshold: float,
    point: Point,
    nearest_point: Point,
):
    """
    Determine if to insert or replace point.
    """
    insert = True
    if nearest_point_idx == 0:
        # It is the first node of linestring
        idx = nearest_point_idx + 1
    elif nearest_point_idx == max(vals[0] for vals in trace_point_dists):
        # It is the last node of linestring
        idx = nearest_point_idx
    else:
        nearest_point_distance_to_point = nearest_point.distance(point)
        if (
            nearest_point_distance_to_point < snap_threshold
            or nearest_point.wkt == point.wkt
            or np.isnan(nearest_point_distance_to_point)
        ):
            # Replace instead of insert
            insert = False
            idx = nearest_point_idx
        else:
            # It is in the middle of the linestring
            points_on_either_side = [
                (
                    vals,
                    angle_to_point(
                        point=point,
                        nearest_point=nearest_point,
                        comparison_point=vals[1],
                    ),
                )
                for vals in trace_point_dists
                if vals[0] in (nearest_point_idx - 1, nearest_point_idx + 1)
            ]

            points_on_either_side = sorted(
                points_on_either_side, key=lambda vals: vals[1]
            )

            assert sum(vals[1] == 0.0 for vals in points_on_either_side) < 2
            smallest_angle_idx = points_on_either_side[0][0][0]

            if smallest_angle_idx > nearest_point_idx:
                idx = smallest_angle_idx
            else:
                idx = nearest_point_idx
    return idx, insert


def additional_snapping_func(
    trace: LineString, idx: int, additional_snapping: List[Tuple[int, Point]]
) -> LineString:
    """
    Insert points into LineStrings to make sure trace abutting trace.

    E.g.

    >>> trace = LineString([(0, 0), (1, 0), (2, 0), (3, 0)])
    >>> idx = 0
    >>> point = Point(2.25, 0.1)
    >>> additional_snapping = [
    ...     (0, point),
    ... ]
    >>> additional_snapping_func(trace, idx, additional_snapping).wkt
    'LINESTRING (0 0, 1 0, 2 0, 2.25 0.1, 3 0)'

    When idx doesn't match -> no additional snapping

    >>> trace = LineString([(0, 0), (1, 0), (2, 0), (3, 0)])
    >>> idx = 1
    >>> point = Point(2.25, 0.1)
    >>> additional_snapping = [
    ...     (0, point),
    ... ]
    >>> additional_snapping_func(trace, idx, additional_snapping).wkt
    'LINESTRING (0 0, 1 0, 2 0, 3 0)'

    """
    indexes_to_fix, points_to_add = zip(*additional_snapping)
    if idx not in indexes_to_fix:
        return trace

    is_idx = np.array(indexes_to_fix) == idx
    filtered_points = compress(points_to_add, is_idx)
    for p in filtered_points:
        trace = insert_point_to_linestring(trace, p, snap_threshold=0.001)
    assert isinstance(trace, LineString)
    return trace


def snap_traces(
    traces: List[LineString],
    snap_threshold: float,
    areas: Optional[List[Union[Polygon, MultiPolygon]]] = None,
    final_allowed_loop=False,
) -> Tuple[List[LineString], bool]:
    """
    Snap traces to end exactly at other traces.
    """
    if len(traces) == 0:
        return ([], False)
    # Only handle LineStrings
    assert all(isinstance(trace, LineString) for trace in traces)

    # Spatial index for traces
    traces_spatial_index = pygeos_spatial_index(geodataset=gpd.GeoSeries(traces))

    # Collect simply snapped (and non-snapped) traces to list
    simply_snapped_traces, simple_changes = zip(
        *[
            snap_trace_simple(
                idx,
                trace,
                snap_threshold,
                traces,
                traces_spatial_index,
                final_allowed_loop=final_allowed_loop,
            )
            for idx, trace in enumerate(traces)
        ]
    )
    assert len(simply_snapped_traces) == len(traces)
    simply_snapped_traces_list: List[LineString] = list(simply_snapped_traces)

    # Collect snapped (and non-snapped) traces to list
    snapped_traces, changes = zip(
        *[
            snap_others_to_trace(
                idx=idx,
                trace=trace,
                snap_threshold=snap_threshold,
                traces_spatial_index=traces_spatial_index,
                areas=areas,
                traces=simply_snapped_traces_list,
                final_allowed_loop=final_allowed_loop,
            )
            for idx, trace in enumerate(simply_snapped_traces)
        ]
    )

    assert len(snapped_traces) == len(simply_snapped_traces)
    assert all(isinstance(ls, LineString) for ls in snapped_traces)

    return list(snapped_traces), any(changes + simple_changes)


def resolve_trace_candidates(
    trace: LineString,
    idx: int,
    traces_spatial_index: PyGEOSSTRTreeIndex,
    traces: List[LineString],
    snap_threshold: float,
) -> List[LineString]:
    """
    Resolve PyGEOSSTRTreeIndex intersection to actual intersection candidates.
    """
    assert isinstance(trace, LineString)

    # Get trace bounds and extend them
    # minx, miny, maxx, maxy = trace.bounds
    minx, miny, maxx, maxy = geom_bounds(trace)
    extended_bounds = (
        minx - snap_threshold * 20,
        miny - snap_threshold * 20,
        maxx + snap_threshold * 20,
        maxy + snap_threshold * 20,
    )

    # Use extended trace bounds to catch all possible intersecting traces
    trace_candidate_idxs_raw = list(traces_spatial_index.intersection(extended_bounds))
    trace_candidate_idxs = [
        i.item() for i in trace_candidate_idxs_raw if isinstance(i.item(), int)
    ]
    assert len(trace_candidate_idxs_raw) == len(trace_candidate_idxs)
    assert isinstance(trace_candidate_idxs, list)

    # Remove current trace
    trace_candidate_idxs.remove(idx)

    # Filter to only candidates based on spatial index
    # trace_candidates = traces.iloc[trace_candidate_idxs]
    # using operator.itemgetter() to
    # elements from list
    assert isinstance(trace_candidate_idxs, list)
    assert isinstance(traces, list)
    # trace_candidates = (
    #     list(itemgetter(*trace_candidate_idxs)(traces))
    #     if len(trace_candidate_idxs) > 0
    #     else []
    # )

    trace_candidates = [traces[i] for i in trace_candidate_idxs]

    return trace_candidates


def snap_trace_simple(
    idx: int,
    trace: LineString,
    snap_threshold: float,
    traces: List[LineString],
    traces_spatial_index: PyGEOSSTRTreeIndex,
    final_allowed_loop: bool = False,
) -> Tuple[LineString, bool]:
    """
    Determine whether and how to perform simple snap.
    """
    trace_candidates = resolve_trace_candidates(
        trace=trace,
        idx=idx,
        traces=traces,
        traces_spatial_index=traces_spatial_index,
        snap_threshold=snap_threshold,
    )

    # If no candidates -> no intersecting -> trace is isolated
    if len(trace_candidates) == 0:
        return trace, False

    # Handle simple case where trace can be extended to meet an absolute
    # coordinate point already in one of the candidates
    trace, was_simple_snapped = simple_snap(trace, trace_candidates, snap_threshold)

    # Debugging
    if final_allowed_loop and was_simple_snapped:
        log.error("In final_allowed_loop and still snapping:")
        log.error(f"{traces, trace}")

    # Return trace and information if it was changed
    return trace, was_simple_snapped


def snap_others_to_trace(
    idx: int,
    trace: LineString,
    snap_threshold: float,
    traces: List[LineString],
    traces_spatial_index: PyGEOSSTRTreeIndex,
    areas: Optional[List[Union[Polygon, MultiPolygon]]],
    final_allowed_loop: bool = False,
) -> Tuple[LineString, bool]:
    """
    Determine whether and how to snap `trace` to `traces`.

    E.g.

    Trace gets new coordinates to snap other traces to it:

    >>> idx = 0
    >>> trace = LineString([(0, 0), (1, 0), (2, 0), (3, 0)])
    >>> snap_threshold = 0.001
    >>> traces = [trace, LineString([(1.5, 3), (1.5, 0.00001)])]
    >>> traces_spatial_index = pygeos_spatial_index(gpd.GeoSeries(traces))
    >>> areas = None
    >>> snapped = snap_others_to_trace(
    ...     idx, trace, snap_threshold, traces, traces_spatial_index, areas
    ... )
    >>> snapped[0].wkt, snapped[1]
    ('LINESTRING (0 0, 1 0, 1.5 1e-05, 2 0, 3 0)', True)

    Trace itself is not snapped by snap_others_to_trace:

    >>> idx = 0
    >>> trace = LineString([(0, 0), (1, 0), (2, 0), (3, 0)])
    >>> snap_threshold = 0.001
    >>> traces = [trace, LineString([(3.0001, -3), (3.0001, 0), (3, 3)])]
    >>> traces_spatial_index = pygeos_spatial_index(gpd.GeoSeries(traces))
    >>> areas = None
    >>> snapped = snap_others_to_trace(
    ...     idx, trace, snap_threshold, traces, traces_spatial_index, areas
    ... )
    >>> snapped[0].wkt, snapped[1]
    ('LINESTRING (0 0, 1 0, 2 0, 3 0)', False)
    """
    trace_candidates = resolve_trace_candidates(
        trace=trace,
        idx=idx,
        traces=traces,
        traces_spatial_index=traces_spatial_index,
        snap_threshold=snap_threshold,
    )

    if trace in list(trace_candidates):
        error = "Found trace in trace_candidates."
        log.error(
            error,
            extra=dict(trace_wkt=trace.wkt, trace_candidates_len=len(trace_candidates)),
        )
        raise ValueError(error)

    # If no candidates -> no intersecting -> trace is isolated
    if len(trace_candidates) == 0:
        return trace, False

    # Get all endpoints of trace_candidates
    endpoints: List[Point] = list(
        chain(
            *[
                list(get_trace_endpoints(trace_candidate))
                for trace_candidate in trace_candidates
            ]
        )
    )

    # Filter endpoints out that are near to the area boundary
    if areas is not None:
        endpoints = [
            ep
            for ep in endpoints
            if not is_endpoint_close_to_boundary(
                ep, areas, snap_threshold=snap_threshold
            )
        ]

    # Add/replace endpoints into trace if they are within snap_threshold
    trace, was_snapped = snap_trace_to_another(
        trace_endpoints=endpoints, another=trace, snap_threshold=snap_threshold
    )

    # Debugging
    if final_allowed_loop and was_snapped:
        log.error("In final_allowed_loop and still snapping:")
        log.error(f"{traces, endpoints}")

    # Return trace and information if it was changed
    return trace, was_snapped


def simple_snap(
    trace: LineString, trace_candidates: List[LineString], snap_threshold: float
) -> Tuple[LineString, bool]:
    """
    Modify conditionally trace to snap to any of trace_candidates.

    E.g.

    >>> trace = LineString([(0, 0), (1, 0), (2, 0), (3, 0)])
    >>> trace_candidates = gpd.GeoSeries(
    ...     [LineString([(3.0001, -3), (3.0001, 0), (3, 3)])]
    ... )
    >>> snap_threshold = 0.001
    >>> snapped = simple_snap(trace, trace_candidates, snap_threshold)
    >>> snapped[0].wkt, snapped[1]
    ('LINESTRING (0 0, 1 0, 2 0, 3.0001 0)', True)

    Do not snap overlapping.

    >>> trace = LineString([(0, 0), (1, 0), (2, 0), (3.0002, 0)])
    >>> trace_candidates = gpd.GeoSeries(
    ...     [LineString([(3.0001, -3), (3.0001, 0), (3, 3)])]
    ... )
    >>> snap_threshold = 0.001
    >>> snapped = simple_snap(trace, trace_candidates, snap_threshold)
    >>> snapped[0].wkt, snapped[1]
    ('LINESTRING (0 0, 1 0, 2 0, 3.0002 0)', False)
    """
    trace_endpoints = get_trace_endpoints(trace)
    traces_to_snap_to = [
        candidate
        for candidate in trace_candidates
        if any(
            endpoint.distance(candidate) < snap_threshold
            for endpoint in trace_endpoints
        )
    ]
    replace_endpoint: Dict[str, Point] = dict()
    for trace_to_snap_to in traces_to_snap_to:
        assert isinstance(trace_to_snap_to, LineString)

        # Get coordinate points of trace_to_snap_to expect endpoints
        coord_points = get_trace_coord_points(trace_to_snap_to)[1:-1]
        if len(coord_points) == 0:
            # Cannot snap trace to any point as there are only endpoints
            continue

        # Check both trace endpoints
        for endpoint in trace_endpoints:
            # Check for already existing snap
            if any(endpoint.intersects(coord_point) for coord_point in coord_points):
                # Already snapped
                continue

            # Get distances between endpoint in coord_points
            distances: List[float] = [
                coord_point.distance(endpoint) for coord_point in coord_points
            ]
            # If not within snap_threshold -> continue
            if not min(distances) < snap_threshold:
                continue

            # Get intersection points
            intersection_points = line_intersection_to_points(
                first=trace, second=trace_to_snap_to
            )
            # Check for overlapping snap
            if any(
                intersection_point.distance(endpoint) < snap_threshold
                for intersection_point in intersection_points
            ):
                # Overlapping snap
                # These are not fixed here and fallback to the other snapping
                # method
                continue

            sorted_points = sorted(
                zip(coord_points, distances), key=lambda vals: vals[1]
            )
            if endpoint.wkt in replace_endpoint:
                error = "Found endpoint in replace_endpoint dict."
                log.error(
                    error,
                    extra=(
                        dict(
                            endpoint_wkt=endpoint.wkt,
                            replace_endpoint_len=len(replace_endpoint),
                        )
                    ),
                )
                raise ValueError(error)
            assert endpoint.wkt not in replace_endpoint
            replace_endpoint[endpoint.wkt] = sorted_points[0][0]

    # If not replacing is done -> return original
    if len(replace_endpoint) == 0:
        return trace, False

    # Replace one of the endpoints based on replace_endpoint
    trace_coords = get_trace_coord_points(trace)
    trace_coords = [
        point if point.wkt not in replace_endpoint else replace_endpoint[point.wkt]
        for point in trace_coords
    ]

    # Return modified
    modified = LineString(trace_coords)
    return modified, True


def filter_non_unique_traces(
    traces: gpd.GeoSeries, snap_threshold: float
) -> gpd.GeoSeries:
    """
    Filter out traces that are not unique.
    """
    assert isinstance(traces, gpd.GeoSeries)
    traces_set = set()
    idxs_to_keep = []
    for idx, geom in enumerate(traces.geometry.values):
        assert isinstance(geom, LineString), geom

        # Use a less strict coordinate precision when finding duplicates
        geom_wkt = dumps(geom, rounding_precision=int(-math.log10(snap_threshold)))
        if geom_wkt in traces_set:
            continue
        traces_set.add(geom_wkt)
        idxs_to_keep.append(idx)

    unique_traces = traces.iloc[idxs_to_keep]
    assert isinstance(unique_traces, gpd.GeoSeries)

    filter_count = len(traces) - len(unique_traces)
    log.info(
        f"Filtered out {filter_count} traces.",
        extra=dict(traces_len=len(traces), unique_traces_len=len(unique_traces)),
    )
    return unique_traces


def safer_unary_union(
    traces_geosrs: gpd.GeoSeries, snap_threshold: float, size_threshold: int
) -> MultiLineString:
    """
    Perform unary union to transform traces to branch segments.

    unary_union is not completely stable with large datasets but problem can be
    alleviated by dividing analysis to parts.

    TODO: Usage is deprecated as unary_union seems to give consistent results.
    """
    if traces_geosrs.empty:
        return MultiLineString()

    # Get amount of traces
    trace_count = traces_geosrs.shape[0]

    # Only one trace and self-intersects shouldn't occur -> Simply return the
    # one LineString wrapped in MultiLineString
    if trace_count == 1:
        return MultiLineString(list(traces_geosrs.geometry.values))

    # Try normal union without any funny business
    # This will be compared to the split approach result and better will
    # be returned
    normal_full_union: MultiLineString = traces_geosrs.unary_union

    if isinstance(normal_full_union, LineString):
        return MultiLineString([normal_full_union])

    if trace_count < size_threshold:
        if len(normal_full_union.geoms) > trace_count and isinstance(
            normal_full_union, MultiLineString
        ):
            return normal_full_union

    # Debugging, fail safely
    if size_threshold < UNARY_ERROR_SIZE_THRESHOLD:
        log.critical(
            "Expected size_threshold to be higher than 100. Union might be impossible."
        )

    # How many parts
    div = int(np.ceil(trace_count / size_threshold))

    # Divide with numpy
    split_traces = np.array_split(traces_geosrs, div)
    assert isinstance(split_traces, list)

    # How many in each pair
    # part_count = int(np.ceil(trace_count / div))

    assert div * sum(part.shape[0] for part in split_traces) >= trace_count
    assert all(isinstance(val, gpd.GeoSeries) for val in split_traces)
    assert isinstance(split_traces[0].iloc[0], LineString)

    # Do unary_union in parts
    part_unions = part_unary_union(
        split_traces=split_traces,
        snap_threshold=snap_threshold,
        size_threshold=size_threshold,
        div=div,
    )
    # Do full union of split unions
    full_union = unary_union(MultiLineString(list(chain(*part_unions))))

    # full_union should always be better or equivalent to normal unary_union.
    # (better when unary_union fails silently)
    if isinstance(full_union, MultiLineString):
        assert isinstance(full_union, BaseMultipartGeometry)
        if len(full_union.geoms) >= len(normal_full_union.geoms):
            return full_union

        raise ValueError(
            "Expected split union to give better results."
            " Branches and nodes should be checked for inconsistencies."
        )
    if isinstance(full_union, LineString):
        return MultiLineString([full_union])
    raise TypeError(
        f"Expected (Multi)LineString from unary_union. Got {full_union.wkt}"
    )


def part_unary_union(
    split_traces: list, snap_threshold: float, size_threshold: int, div: int
):
    """
    Conduct safer_unary_union in parts.
    """
    # Collect partly done unary_unions to part_unions list
    part_unions = []

    # Iterate over list of split trace GeoSeries
    for part in split_traces:
        # Do unary_union to part
        part_union = part.unary_union

        # Do naive check for if unary_union is successful
        if (
            not isinstance(part_union, MultiLineString)
            or len(part_union.geoms) < part.shape[0]
        ):
            # Still fails -> Try with lower threshold for part
            part_union = safer_unary_union(
                part, snap_threshold, size_threshold=size_threshold // 2
            )

        # Collect
        part_unions.append(part_union.geoms)
    assert len(part_unions) == div
    return part_unions


def report_snapping_loop(loops: int, allowed_loops: int):
    """
    Report snapping looping.
    """
    log.info(f"Loop :{ loops }")
    if loops >= 10:
        log.warning(
            f"{loops} loops have passed without resolved snapped"
            " traces_geosrs. Snapped traces_geosrs might not"
            " possibly be resolved."
        )
    if loops > allowed_loops:
        raise RecursionError(
            f"More loops have passed ({loops}) than allowed by allowed_loops "
            f"({allowed_loops})) for snapping traces_geosrs for"
            " branch determination."
        )


@JOBLIB_CACHE.cache
def branches_and_nodes(
    traces: Union[gpd.GeoSeries, gpd.GeoDataFrame],
    areas: Union[gpd.GeoSeries, gpd.GeoDataFrame],
    snap_threshold: float,
    allowed_loops=10,
    already_clipped: bool = False,
    # unary_size_threshold: int = 5000,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Determine branches and nodes of given traces.

    The traces will be cropped to the given target area(s) if not already
    clipped(already_clipped).

    TODO: unary_union will not respect identical traces or near-identical.
    Therefore cannot test if there are more branches than traces because
    there might be less due to this issue.
    """
    log.info(
        "Starting determination of branches and nodes.",
        extra=dict(
            len_traces=len(traces),
            len_areas=len(areas),
            snap_threshold=snap_threshold,
            already_clipped=already_clipped,
            allowed_loops=allowed_loops,
        ),
    )
    traces_geosrs: gpd.GeoSeries = traces.geometry
    if check_for_z_coordinates(geodata=traces_geosrs):
        log.info(
            "Removing z-coordinates from trace data before branch and node determination."
        )
        traces_without_z_coords = remove_z_coordinates_from_geodata(
            geodata=traces_geosrs
        )
        assert isinstance(traces_without_z_coords, gpd.GeoSeries)
        traces_geosrs = traces_without_z_coords

    # Filter out traces that are not unique by wkt
    # unary_union will fail to take them into account any way
    traces_geosrs = filter_non_unique_traces(
        traces_geosrs, snap_threshold=snap_threshold
    )

    areas_geosrs: gpd.GeoSeries = areas.geometry

    # Collect into lists
    areas_lists_of_polygons = [
        [poly] if isinstance(poly, Polygon) else list(poly.geoms)
        for poly in areas_geosrs.geometry.values
        if isinstance(poly, (Polygon, MultiPolygon))
    ]
    areas_list = list(chain(*areas_lists_of_polygons))
    assert all(isinstance(poly, Polygon) for poly in areas_list), areas_list

    # Collect into lists
    traces_list = [
        trace
        for trace in traces_geosrs.geometry.values
        if isinstance(trace, LineString)
    ]

    # Snapping occurs multiple times due to possible side effects of each snap
    loops = 0
    traces_list, any_changes_applied = snap_traces(
        traces_list, snap_threshold, areas=areas_list
    )
    # Snapping causes changes that might cause new errors. The snapping is looped
    # as many times as there are changed made to the data.
    # If loop count reaches allowed_loops, error is raised.
    while any_changes_applied:
        traces_list, any_changes_applied = snap_traces(
            traces_list,
            snap_threshold,
            final_allowed_loop=loops == allowed_loops,
            areas=areas_list,
        )
        loops += 1
        report_snapping_loop(loops, allowed_loops=allowed_loops)

    traces_geosrs = gpd.GeoSeries(traces_list, crs=traces.crs)

    # Clip if necessary
    if not already_clipped:
        traces_geosrs = crop_to_target_areas(
            traces_geosrs,
            areas_geosrs,
            keep_column_data=False,
        ).geometry

    # Remove too small geometries.
    traces_geosrs = traces_geosrs.loc[
        traces_geosrs.geometry.length > snap_threshold * 2.01
    ]

    # Branches are determined with shapely/geopandas unary_union
    unary_union_result = traces_geosrs.unary_union
    if isinstance(unary_union_result, MultiLineString):
        branches_all = list(unary_union_result.geoms)
    elif isinstance(unary_union_result, LineString):
        branches_all = [unary_union_result]
    else:
        raise TypeError(
            "Expected unary_union_result to be of type (Multi)LineString."
            f" Got: {type(unary_union_result), unary_union_result}"
        )

    # branches_all = list(
    #     safer_unary_union(
    #         traces_geosrs,
    #         snap_threshold=snap_threshold,
    #         size_threshold=unary_size_threshold,
    #     ).geoms
    # )

    # Filter out very short branches
    branches = gpd.GeoSeries(
        [b for b in branches_all if b.length > snap_threshold * 1.01],
        crs=traces_geosrs.crs,
    )

    # Report and error possibly unary_union failure
    if len(branches_all) < len(traces_geosrs):
        # unary_union can fail with too large datasets
        log.critical(
            "Expected more branches than traces. Possible unary_union failure."
        )

    # Determine nodes and identities
    nodes, node_identities = node_identities_from_branches(
        branches=branches, areas=areas_geosrs, snap_threshold=snap_threshold
    )

    # Collect to GeoSeries
    nodes_geosrs = gpd.GeoSeries(nodes)

    # Determine branch identities
    branch_identities = get_branch_identities(
        branches, nodes_geosrs, node_identities, snap_threshold
    )

    # Collect to GeoDataFrames
    node_gdf = gpd.GeoDataFrame(
        {GEOMETRY_COLUMN: nodes_geosrs, CLASS_COLUMN: node_identities}, crs=traces.crs
    )
    branch_gdf = gpd.GeoDataFrame(
        {GEOMETRY_COLUMN: branches, CONNECTION_COLUMN: branch_identities},
        crs=traces.crs,
    )
    return branch_gdf, node_gdf


def snap_trace_to_another(
    trace_endpoints: List[Point], another: LineString, snap_threshold: float
) -> Tuple[LineString, bool]:
    """
    Add point to another trace to snap trace to end at another trace.

    I.e. modifies and returns `another`
    """
    # Do nothing if endpoints of trace are not within snap_threshold.
    # Or if endpoint is already exact
    endpoints = [
        ep
        for ep in trace_endpoints
        if ep.distance(another) < snap_threshold and not ep.intersects(another)
    ]

    # No endpoints within snapping threshold
    if len(endpoints) == 0:
        # Do not snap
        return another, False

    for endpoint in endpoints:
        another = insert_point_to_linestring(
            another, endpoint, snap_threshold=snap_threshold
        )

    return another, True


def is_endpoint_close_to_boundary(
    endpoint: Point, areas: List[Union[Polygon, MultiPolygon]], snap_threshold: float
) -> bool:
    """
    Check if endpoint is within snap_threshold of areas boundaries.
    """
    for area in areas:
        assert isinstance(area, (Polygon, MultiPolygon))
        if endpoint.distance(area.boundary) < snap_threshold:
            return True
    return False


def node_identity(
    endpoint: Point,
    idx: int,
    areas: Union[gpd.GeoSeries, gpd.GeoDataFrame],
    endpoints_geoseries: gpd.GeoSeries,
    endpoints_spatial_index: PyGEOSSTRTreeIndex,
    snap_threshold: float,
) -> str:
    """
    Determine node identity of endpoint.
    """
    node_type = E_node
    # Check for proximity to area boundary
    if any(
        endpoint.distance(area.boundary) < snap_threshold
        for area in areas.geometry.values
    ):
        return node_type

    # Candidate points idxs
    candidate_idxs = list(endpoints_spatial_index.intersection(endpoint.coords[0]))

    # Remove current
    candidate_idxs.remove(idx)

    # Candidate points
    candidates = endpoints_geoseries.iloc[candidate_idxs]

    # Get intersect count
    intersecting_node_count = sum(
        candidate.distance(endpoint) < snap_threshold for candidate in candidates
    )

    if intersecting_node_count == 0:
        node_type = I_node
    elif intersecting_node_count == 2:
        node_type = Y_node
    elif intersecting_node_count == 3:
        node_type = X_node
    elif intersecting_node_count == 1:
        # Unresolvable
        log.error(
            f"Expected 0, 2 or 3 intersects. V-node or similar error at {endpoint.wkt}."
        )
        node_type = I_node
    else:
        # Unresolvable
        log.error(f"Expected 0, 2 or 3 intersects. Multijunction at {endpoint.wkt}.")
        node_type = X_node
    return node_type


def node_identities_from_branches(
    branches: gpd.GeoSeries,
    areas: Union[gpd.GeoSeries, gpd.GeoDataFrame],
    snap_threshold: float,
) -> Tuple[List[Point], List[str]]:
    """
    Resolve node identities from branch data.

    >>> branches_list = [
    ...     LineString([(0, 0), (1, 1)]),
    ...     LineString([(2, 2), (1, 1)]),
    ...     LineString([(2, 0), (1, 1)]),
    ... ]
    >>> area_polygon = Polygon([(-5, -5), (-5, 5), (5, 5), (5, -5)])
    >>> branches = gpd.GeoSeries(branches_list)
    >>> areas = gpd.GeoSeries([area_polygon])
    >>> snap_threshold = 0.001
    >>> nodes, identities = node_identities_from_branches(
    ...     branches, areas, snap_threshold
    ... )
    >>> [node.wkt for node in nodes]
    ['POINT (0 0)', 'POINT (1 1)', 'POINT (2 2)', 'POINT (2 0)']
    >>> identities
    ['I', 'Y', 'I', 'I']

    """
    # Get list of all branch endpoints
    all_endpoints: List[Point] = list(
        chain(
            *[list(get_trace_endpoints(branch)) for branch in branches.geometry.values]
        )
    )
    # Collect into GeoSeries
    all_endpoints_geoseries = gpd.GeoSeries(all_endpoints)

    # Get spatial index
    endpoints_spatial_index = pygeos_spatial_index(all_endpoints_geoseries)

    # Collect resolved nodes
    collected_nodes: Dict[str, Tuple[Point, str]] = dict()

    for idx, endpoint in enumerate(all_endpoints):
        # Do not resolve nodes that have already been resolved
        if endpoint.wkt in collected_nodes:
            continue

        # Determine node identity
        identity = node_identity(
            endpoint=endpoint,
            idx=idx,
            areas=areas,
            endpoints_geoseries=all_endpoints_geoseries,
            endpoints_spatial_index=endpoints_spatial_index,
            snap_threshold=snap_threshold,
        )

        # Add to resolved
        collected_nodes[endpoint.wkt] = (endpoint, identity)

    if len(collected_nodes) == 0:
        return [], []

    # Collect into two lists
    values = list(collected_nodes.values())
    nodes = [value[0] for value in values]
    identities = [value[1] for value in values]

    return nodes, identities
