"""
Functions for extracting branches and nodes from trace maps.

branches_and_nodes is the main entrypoint.
"""
import logging
from itertools import chain
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas.sindex import PyGEOSSTRTreeIndex
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.ops import unary_union

from fractopo.general import (
    CLASS_COLUMN,
    CONNECTION_COLUMN,
    GEOMETRY_COLUMN,
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
    crop_to_target_areas,
    determine_valid_intersection_points,
    geom_bounds,
    get_trace_coord_points,
    get_trace_endpoints,
    line_intersection_to_points,
    pygeos_spatial_index,
)


def remove_identical_sindex(
    geosrs: gpd.GeoSeries, snap_threshold: float
) -> gpd.GeoSeries:
    """
    Remove stacked nodes by using a search buffer the size of snap_threshold.
    """
    geosrs = geosrs.reset_index(inplace=False, drop=True)
    spatial_index = geosrs.sindex
    identical_idxs = []
    point: Point
    for idx, point in enumerate(geosrs.geometry.values):
        if idx in identical_idxs:
            continue
        # point = point.buffer(snap_threshold) if snap_threshold != 0 else point
        p_candidate_idxs = (
            list(spatial_index.intersection(point.buffer(snap_threshold).bounds))
            if snap_threshold != 0
            else list(spatial_index.intersection(point.coords[0]))
        )
        p_candidate_idxs.remove(idx)
        p_candidates = geosrs.iloc[p_candidate_idxs]
        inter = p_candidates.distance(point) < snap_threshold
        colliding = inter.loc[inter]
        if len(colliding) > 0:
            index_to_list = colliding.index.to_list()
            assert len(index_to_list) > 0
            assert all([isinstance(i, int) for i in index_to_list])
            identical_idxs.extend(index_to_list)
    return geosrs.drop(identical_idxs)


# def get_node_identities(
#     traces: gpd.GeoSeries,
#     nodes: gpd.GeoSeries,
#     areas: gpd.GeoSeries,
#     snap_threshold: float,
# ) -> List[str]:
#     """
#     Determine the type of node i.e. E, X, Y or I.

#     Uses the given traces to assess the number of intersections.

#     E.g.


#     """
#     # Only handle LineString traces
#     assert all([isinstance(trace, LineString) for trace in traces.geometry.values])

#     # Area can be Polygon or MultiPolygon
#     assert all(
#         [isinstance(area, (Polygon, MultiPolygon)) for area in areas.geometry.values]
#     )

#     # Collect node identities to list
#     identities = []

#     # Create spatial index of traces
#     trace_sindex = traces.sindex

#     # Iterate over all nodes

#     node: Point
#     for node in nodes.geometry.values:

#         # If node is within snap_threshold of area boundary -> E-node
#         if any(
#             [
#                 node.distance(area.boundary) < snap_threshold
#                 for area in areas.geometry.values
#             ]
#         ):
#             identities.append(E_node)
#             continue

#         trace_candidate_idxs = list(
#             trace_sindex.intersection(node.buffer(snap_threshold).bounds)
#         )
#         trace_candidates: gpd.GeoSeries = traces.iloc[trace_candidate_idxs]
#         # inter_with_traces = trace_candidates.intersects(node.buffer(snap_threshold))

#         # If theres 2 intersections -> X or Y
#         # 1 (must be) -> I
#         # Point + LineString -> Y
#         # LineString + Linestring -> X or Y
#         inter_with_traces_geoms: gpd.GeoSeries = trace_candidates.loc[
#             trace_candidates.distance(node) < snap_threshold
#         ]
#         assert all(
#             [isinstance(t, LineString) for t in inter_with_traces_geoms.geometry.values]
#         )
#         assert len(inter_with_traces_geoms) > 0

#         if len(inter_with_traces_geoms) == 1:
#             identities.append(I_node)
#             continue
#         # If more than 2 traces intersect -> Y-node or X-node

#         all_inter_endpoints = [
#             pt
#             for sublist in map(
#                 get_trace_endpoints,
#                 inter_with_traces_geoms,
#             )
#             for pt in sublist
#         ]
#         assert all([isinstance(pt, Point) for pt in all_inter_endpoints])
#         strict_intersect_count = sum(
#             [node.intersects(ep) for ep in all_inter_endpoints]
#         )
#         if strict_intersect_count == 1:
#             # Y-node
#             identities.append(Y_node)
#         elif strict_intersect_count == 2:
#             # Node intersects more than 1 endpoint
#             logging.error(
#                 "Expected node not to intersect more than one endpoint.\n"
#                 f"{node=} {inter_with_traces_geoms=}"
#             )
#             identities.append(Y_node)

#         elif len(inter_with_traces_geoms) == 2:
#             # X-node
#             identities.append(X_node)
#         elif len(inter_with_traces_geoms) > 2:
#             distances = inter_with_traces_geoms.distance(node).values

#             # If all traces are almost equally distanced from node the junction
#             # represents an error and is not a strict X-node
#             if all([np.isclose(distances[0], dist) for dist in distances]):
#                 raise ValueError(
#                     "Node intersects trace_candidates more than two times and\n"
#                     f"does not intersect any endpoint. Node: {node.wkt}\n"
#                     f"inter_with_traces_geoms: {inter_with_traces_geoms}\n"
#                 )

#             # If two of the traces are equal-distance from the node the third
#             # trace is not a participant in the junction -> X-node
#             found = False
#             for comb in combinations(distances, 2):
#                 if np.isclose(*comb):
#                     identities.append(X_node)
#                     found = True
#                     break

#             # If unresolvable -> Error.
#             if not found:
#                 raise ValueError(
#                     "Node intersects trace_candidates more than two times and\n"
#                     f"does not intersect any endpoint. Node: {node.wkt}\n"
#                     f"inter_with_traces_geoms: {inter_with_traces_geoms}\n"
#                 )
#         else:
#             raise ValueError(
#                 "Could not resolve node type." f"{node=} {inter_with_traces_geoms=}"
#             )

#     assert len(identities) == len(nodes)
#     return identities


def determine_branch_identity(
    number_of_I_nodes: int, number_of_XY_nodes: int, number_of_E_nodes: int
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
    if number_of_I_nodes + number_of_E_nodes + number_of_XY_nodes != 2:
        logging.error("Did not find 2 EXYI-nodes that intersected branch endpoints.\n")
        return Error_branch
    elif number_of_I_nodes == 2:
        return II_branch
    elif number_of_XY_nodes == 2:
        return CC_branch
    elif number_of_E_nodes == 2:
        return EE_branch
    elif number_of_I_nodes == 1 and number_of_XY_nodes == 1:
        return CI_branch
    elif number_of_E_nodes == 1 and number_of_XY_nodes == 1:
        return CE_branch
    elif number_of_E_nodes == 1 and number_of_I_nodes == 1:
        return IE_branch
    else:
        logging.error("Unknown error in determine_branch_identity")
        return EE_branch


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
    ...            LineString([(1, 1), (2, 2)]),
    ...            LineString([(2, 2), (3, 3)]),
    ...            LineString([(3, 0), (2, 2)]),
    ...            LineString([(2, 2), (-2, 5)]),
    ...     ]
    ... )
    >>> nodes = gpd.GeoSeries(
    ...     [Point(2, 2), Point(1, 1), Point(3, 3), Point(3, 0), Point(-2, 5),]
    ...     )
    >>> node_identities = ["X", "I", "I", "I", "E"]
    >>> snap_threshold = 0.001
    >>> get_branch_identities(branches, nodes, node_identities, snap_threshold)
    ['C - I', 'C - I', 'C - I', 'C - E']

    """
    assert len(nodes) == len(node_identities)
    nodes_buffered = gpd.GeoSeries(list(map(lambda p: p.buffer(snap_threshold), nodes)))
    node_gdf = gpd.GeoDataFrame({GEOMETRY_COLUMN: nodes, CLASS_COLUMN: node_identities})
    node_spatial_index = nodes_buffered.sindex
    branch_identities = []
    for branch in branches:
        node_candidate_idxs = list(node_spatial_index.intersection(branch.bounds))
        node_candidates = node_gdf.iloc[node_candidate_idxs]
        # Use distance instead of two polygon buffers
        inter = [
            dist < snap_threshold
            for dist in node_candidates.distance(
                MultiPoint([p for p in get_trace_endpoints(branch)])
            )
        ]
        assert len(inter) == len(node_candidates)
        nodes_that_intersect = node_candidates.loc[inter]
        number_of_E_nodes = len(
            [
                inter_id
                for inter_id in nodes_that_intersect[CLASS_COLUMN]
                if inter_id == E_node
            ]
        )
        number_of_I_nodes = len(
            [
                inter_id
                for inter_id in nodes_that_intersect[CLASS_COLUMN]
                if inter_id == I_node
            ]
        )
        number_of_XY_nodes = len(
            [
                inter_id
                for inter_id in nodes_that_intersect[CLASS_COLUMN]
                if inter_id in [X_node, Y_node]
            ]
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
        raise ValueError("Points in angle_to_point intersect.")

    x1, y1 = tuple(*nearest_point.coords)
    x2, y2 = tuple(*point.coords)
    x3, y3 = tuple(*comparison_point.coords)
    if any([np.isnan(val) for val in (x1, y1, x2, y2, x3, y3)]):
        raise ValueError(
            f"""
            np.nan in angle_to_point inputs.\n
            inputs: {point,nearest_point,comparison_point}
            """
        )
    # Vector from nearest_point to point
    vector_1 = np.array([x2 - x1, y2 - y1])
    # Vector from nearest_point to comparison_point
    vector_2 = np.array([x3 - x1, y3 - y1])
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    if 1 < dot_product or dot_product < -1:
        rad_angle = np.nan
    else:
        rad_angle = np.arccos(dot_product)
    if np.isnan(rad_angle):
        # Cannot determine with angle.
        unit_vector_sum_len = np.linalg.norm(unit_vector_1 + unit_vector_2)
        if np.isclose(unit_vector_sum_len, 0.0, atol=1e-07):
            return 180.0
        elif np.isclose(unit_vector_sum_len, 2.0, atol=1e-07):
            return 0.0
        else:
            logging.error(unit_vector_1, unit_vector_2, unit_vector_sum_len)
            raise ValueError(
                "Could not detemine point relationships. Vectors printed above."
            )
    assert 360 >= np.rad2deg(rad_angle) >= 0
    return np.rad2deg(rad_angle)


# def insert_point_to_linestring(trace: LineString, point: Point) -> LineString:
#     """
#     Insert given point to given trace LineString.

#     The point location is determined to fit into the LineString without
#     changing the geometrical order of LineString vertices
#     (which only makes sense if LineString is sublinear.)

#     TODO/Note: Does not work for 2.5D geometries (Z-coordinates).
#     Z-coordinates will be lost.

#     E.g.

#     >>> trace = LineString([(0, 0), (1, 0), (2, 0), (3, 0)])
#     >>> point = Point(1.25, 0.1)
#     >>> insert_point_to_linestring(trace, point).wkt
#     'LINESTRING (0 0, 1 0, 1.25 0.1, 2 0, 3 0)'

#     >>> trace = LineString([(0, 0), (1, 0), (2, 0), (3, 0)])
#     >>> point = Point(2.25, 0.1)
#     >>> insert_point_to_linestring(trace, point).wkt
#     'LINESTRING (0 0, 1 0, 2 0, 2.25 0.1, 3 0)'

#     """
#     assert isinstance(trace, LineString)
#     assert isinstance(point, Point)
#     if trace.has_z:
#         logging.warning("Trace contains z-coordinates. These will be lost.")
#     if point.has_z:
#         logging.warning("Point contains z-coordinates. These will be lost.")

#     if any([point.intersects(Point(xy)) for xy in trace.coords]):
#         logging.error(
#             "Point already matches a coordinate point in trace.\n"
#             f"point: {point.wkt}, trace: {trace.wkt}\n"
#             "Returning original trace without changes."
#         )
#         return trace
#     t_points_gdf = gpd.GeoDataFrame(
#         {
#             GEOMETRY_COLUMN: [Point(c) for c in trace.coords],
#             "distances": [Point(c).distance(point) for c in trace.coords],
#         }
#     )
#     t_points_gdf.sort_values(by="distances", inplace=True)
#     nearest_point = t_points_gdf.iloc[0].geometry
#     assert isinstance(nearest_point, Point)
#     nearest_point_idx = t_points_gdf.iloc[0].name
#     if nearest_point_idx == 0:
#         # It is the first node of linestring
#         add_before = nearest_point_idx + 1
#     elif nearest_point_idx == max(t_points_gdf.index):
#         # It is the last node of linestring
#         add_before = nearest_point_idx
#     else:
#         # It is in the middle of the linestring
#         points_on_either_side = t_points_gdf.loc[
#             [nearest_point_idx - 1, nearest_point_idx + 1]
#         ]

#         points_on_either_side["angle"] = list(
#             map(
#                 lambda p: angle_to_point(point, nearest_point, p),
#                 points_on_either_side.geometry,
#             )
#         )
#         assert sum(points_on_either_side["angle"] == 0.0) < 2
#         smallest_angle_idx = points_on_either_side.sort_values(by="angle").iloc[0].name
#         if smallest_angle_idx > nearest_point_idx:
#             add_before = smallest_angle_idx
#         else:
#             add_before = nearest_point_idx

#     t_coords = list(trace.coords)
#     t_coords.insert(add_before, tuple(*point.coords))
#     # Closest points might not actually be the points which inbetween the
#     # point is added. Have to use project and interpolate (?)
#     # print(t_coords)
#     new_trace = LineString(t_coords)
#     # print(new_trace.wkt)
#     assert new_trace.intersects(point)
#     assert isinstance(new_trace, LineString)
#     assert new_trace.is_valid
#     # assert new_trace.is_simple
#     if not new_trace.is_simple:
#         logging.warning(f"Non-simple geometry detected.\n{new_trace.wkt}")
#     return new_trace


def insert_point_to_linestring(
    trace: LineString, point: Point, snap_threshold: float
) -> LineString:
    """
    Insert/modify point to trace LineString.

    The point location is determined to fit into the LineString without
    changing the geometrical order of LineString vertices
    (which only makes sense if LineString is sublinear.)

    TODO/Note: Does not work for 2.5D geometries (Z-coordinates).
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
        logging.warning("Trace contains z-coordinates. These will be lost.")
    if point.has_z:
        logging.warning("Point contains z-coordinates. These will be lost.")

    if any([point.intersects(Point(xy)) for xy in trace.coords]):
        logging.error(
            "Point already matches a coordinate point in trace.\n"
            f"point: {point.wkt}, trace: {trace.wkt}\n"
            "Returning original trace without changes."
        )
        return trace
    trace_point_dists = [
        (idx, trace_point, trace_point.distance(point))
        for idx, trace_point in enumerate([Point(c) for c in trace.coords])
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
    t_coords.insert(idx, tuple(*point.coords))
    # Closest points might not actually be the points which inbetween the
    # point is added. Have to use project and interpolate (?)
    # print(t_coords)
    new_trace = LineString(t_coords)
    # print(new_trace.wkt)
    assert new_trace.intersects(point)
    assert isinstance(new_trace, LineString)
    assert new_trace.is_valid
    # assert new_trace.is_simple
    if not new_trace.is_simple:
        logging.warning(f"Non-simple geometry detected.\n{new_trace.wkt}")
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
    elif nearest_point_idx == max([vals[0] for vals in trace_point_dists]):
        # It is the last node of linestring
        idx = nearest_point_idx
    else:

        if nearest_point.distance(point) < snap_threshold:
            # Replace instead of insert
            insert = False
            idx = nearest_point_idx
        else:

            # It is in the middle of the linestring
            points_on_either_side = [
                (vals, angle_to_point(point, nearest_point, vals[1]))
                for vals in trace_point_dists
                if vals[0] in (nearest_point_idx - 1, nearest_point_idx + 1)
            ]

            points_on_either_side = sorted(
                points_on_either_side, key=lambda vals: vals[1]
            )

            assert sum([vals[1] == 0.0 for vals in points_on_either_side]) < 2
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
    ...     ]
    >>> additional_snapping_func(trace, idx, additional_snapping).wkt
    'LINESTRING (0 0, 1 0, 2 0, 2.25 0.1, 3 0)'

    When idx doesn't match -> no additional snapping

    >>> trace = LineString([(0, 0), (1, 0), (2, 0), (3, 0)])
    >>> idx = 1
    >>> point = Point(2.25, 0.1)
    >>> additional_snapping = [
    ...     (0, point),
    ...     ]
    >>> additional_snapping_func(trace, idx, additional_snapping).wkt
    'LINESTRING (0 0, 1 0, 2 0, 3 0)'

    """
    indexes_to_fix, points_to_add = zip(*additional_snapping)
    if idx in indexes_to_fix:
        df = pd.DataFrame(
            {"indexes_to_fix": indexes_to_fix, "points_to_add": points_to_add}
        )
        df = df.loc[df["indexes_to_fix"] == idx]
        for p in df["points_to_add"].values:
            trace = insert_point_to_linestring(trace, p, snap_threshold=0.001)
        assert isinstance(trace, LineString)
        return trace

    else:
        return trace


# def snap_traces_old(
#     traces: gpd.GeoSeries,
#     snap_threshold: float,
#     areas: Optional[Union[gpd.GeoDataFrame, gpd.GeoSeries]] = None,
#     final_allowed_loop: bool = False,
# ) -> Tuple[gpd.GeoSeries, bool]:
#     """
#     Snap traces to end exactly at other traces.

#     E.g. when within snap_threshold:

#     >>> traces = gpd.GeoSeries(
#     ...     [LineString([(1, 1), (2, 2), (3, 3)]), LineString([(1.9999, 2), (-2, 5)])]
#     ...     )
#     >>> snap_threshold = 0.001
#     >>> snap_traces_old(traces, snap_threshold)
#     (0    LINESTRING (1.00000 1.00000, 2.00000 2.00000, ...
#     1       LINESTRING (2.00000 2.00000, -2.00000 5.00000)
#     dtype: geometry, True)

#     TODO: Too complex and messy. But works...

#     """
#     if areas is None:
#         logging.warning("Areas not given to snap_traces. Results may vary.")
#     traces.reset_index(drop=True, inplace=True)
#     any_changed_applied = False
#     traces_spatial_index = traces.sindex
#     snapped_traces: List[LineString]
#     snapped_traces = []
#     additional_snapping: List[Point]
#     additional_snapping = []
#     trace: LineString
#     idx: int
#     for idx, trace in enumerate(traces.geometry.values):
#         minx, miny, maxx, maxy = trace.bounds
#         extended_bounds = (
#             minx - snap_threshold * 10,
#             miny - snap_threshold * 10,
#             maxx + snap_threshold * 10,
#             maxy + snap_threshold * 10,
#         )
#         trace_candidate_idxs = list(traces_spatial_index.intersection(extended_bounds))
#         trace_candidate_idxs.remove(idx)
#         trace_candidates = traces.iloc[trace_candidate_idxs]
#         if len(trace_candidates) == 0:
#             snapped_traces.append(trace)
#             continue
#         # get_trace_endpoints returns ls[0] and then ls[-1]
#         endpoints = get_trace_endpoints(trace)
#         snapped_endpoints: List[Point]
#         snapped_endpoints = []
#         n: Point
#         for n in endpoints:
#             if areas is not None and any(
#                 [
#                     n.distance(area.boundary) < snap_threshold
#                     for area in areas.geometry.values
#                 ]
#             ):
#                 # Do not try to snap traces near target area boundary
#                 snapped_endpoints.append(n)
#                 continue
#             how_many_n_intersects = sum(trace_candidates.intersects(n))
#             if how_many_n_intersects == 1:
#                 snapped_endpoints.append(n)
#                 continue
#             elif how_many_n_intersects > 1:
#                 logging.warning(
#                     "Endpoint intersects more than two traces?\n"
#                     f"Endpoint: {n.wkt}\n"
#                     f"how_many_n_intersects: {how_many_n_intersects}\n"
#                     f"trace_candidates: {[t.wkt for t in trace_candidates]}\n"
#                 )
#             distances_less_than = trace_candidates.distance(n) < snap_threshold
#             traces_to_snap_to = trace_candidates[distances_less_than]
#             if len(traces_to_snap_to) == 1:
#                 any_changed_applied = True
#                 traces_to_snap_to_vertices = gpd.GeoSeries(
#                     [Point(c) for c in traces_to_snap_to.geometry.iloc[0].coords]
#                 )
#                 # vertice_intersects = traces_to_snap_to_vertices.intersects(
#                 #     n.buffer(snap_threshold)
#                 # )
#                 vertice_intersects = (
#                     traces_to_snap_to_vertices.distance(n) < snap_threshold
#                 )
#                 if sum(vertice_intersects) > 0:
#                     # Snap endpoint to the vertice of the trace in which abutting trace
#                     # abuts.
#                     new_n = traces_to_snap_to_vertices.loc[vertice_intersects].iloc[0]
#                     assert isinstance(new_n, Point)
#                     snapped_endpoints.append(
#                         traces_to_snap_to_vertices[vertice_intersects].iloc[0]
#                     )
#                 else:
#                     # Other trace must be snapped to a vertice to assure
#                     # that they intersect. I.e. a new vertice is added into
#                     # the middle of the other trace.
#                     # This is handled later.
#                     t_idx = traces_to_snap_to.index.values[0]
#                     additional_snapping.append((t_idx, n))

#                     snapped_endpoints.append(n)
#             elif len(traces_to_snap_to) == 0:
#                 snapped_endpoints.append(n)
#             else:
#                 logging.warning(
#                     "Trace endpoint is within the snap threshold"
#                     " of two traces.\n"
#                     f"No Snapping was done.\n"
#                     f"endpoints: {list(map(lambda p: p.wkt, endpoints))}\n"
#                     f"distances: {distances_less_than}\n"
#                     f"traces_to_snap_to: {traces_to_snap_to}\n"
#                 )
#                 snapped_endpoints.append(n)
#         assert len(snapped_endpoints) == len(endpoints)
#         # Original coords
#         trace_coords = [point for point in trace.coords]
#         assert all([isinstance(tc, tuple) for tc in trace_coords])
#         trace_coords[0] = tuple(*snapped_endpoints[0].coords)
#         trace_coords[-1] = tuple(*snapped_endpoints[-1].coords)
#         snapped_traces.append(LineString(trace_coords))

#     if final_allowed_loop:
#         # Print debugging information before execution is stopped upstream.
#         logging.error("In final loop and still snapping.")
#         logging.error(f"{additional_snapping=}")
#     # Handle additional_snapping
#     if len(additional_snapping) != 0:
#         snapped_traces = list(
#             map(
#                 lambda idx, val: additional_snapping_func(
#                     val, idx, additional_snapping
#                 ),
#                 *zip(*enumerate(snapped_traces)),
#             )
#         )

#     assert len(snapped_traces) == len(traces)
#     return gpd.GeoSeries(snapped_traces), any_changed_applied


def true_candidate_endpoints(
    trace: LineString, trace_candidate: LineString, snap_threshold: float
) -> List[Point]:
    """
    Determine if trace candidate endpoints qualify for snapping.
    """
    candidate_endpoints = get_trace_endpoints(trace_candidate)
    endpoint_candidates = [
        ep for ep in candidate_endpoints if ep.distance(trace) < snap_threshold
    ]
    if len(endpoint_candidates) == 0:
        return []

    trace_endpoints = get_trace_endpoints(trace)

    return [
        ep
        for ep in endpoint_candidates
        if (
            not ep.intersects(trace)
            and (
                not any(
                    [
                        ep.distance(trace_ep) < snap_threshold
                        for trace_ep in trace_endpoints
                    ]
                )
            )
        )
    ]


# def snap_traces_alternative(
#     traces: gpd.GeoSeries, snap_threshold: float
# ) -> Tuple[gpd.GeoSeries, bool]:
#     """
#     Snap traces to end exactly at other traces.
#     """
#     # TODO: Is it necessary
#     traces.reset_index(inplace=True, drop=True)

#     # Only handle LineStrings
#     assert all([isinstance(trace, LineString) for trace in traces.geometry.values])

#     # Mark if any snapping must be done
#     any_changed_applied = False

#     # Spatial index for traces
#     traces_spatial_index = traces.sindex

#     # Collect snapped (and non-snapped) traces to list
#     snapped_traces: List[LineString] = []
#     trace: LineString
#     idx: int
#     for idx, trace in enumerate(traces.geometry.values):

#         # Get trace bounds and extend them
#         minx, miny, maxx, maxy = trace.bounds
#         extended_bounds = (
#             minx - snap_threshold * 20,
#             miny - snap_threshold * 20,
#             maxx + snap_threshold * 20,
#             maxy + snap_threshold * 20,
#         )

#         # Use extended trace bounds to catch all possible
#         trace_candidate_idxs = list(traces_spatial_index.intersection(extended_bounds))

#         # Do not check currect trace
#         trace_candidate_idxs.remove(idx)

#         # Filter to only candidates based on spatial index
#         trace_candidates = traces.iloc[trace_candidate_idxs]

#         # If no candidates -> no intersecting -> trace is isolated
#         if len(trace_candidates) == 0:
#             snapped_traces.append(trace)
#             continue

#         for trace_candidate in trace_candidates.geometry.values:
#             candidate_endpoints = true_candidate_endpoints(
#                 trace, trace_candidate, snap_threshold
#             )
#             if len(candidate_endpoints) == 0:
#                 reverse_endpoints = true_candidate_endpoints(
#                     trace_candidate, trace, snap_threshold
#                 )
#                 if len(reverse_endpoints) == 0:
#                     continue
#                 # Replace endnode of current trace to end just before other
#                 # trace and let next loop handle the exact snapping
#                 linestrings = [
#                     ls
#                     for ls in resolve_split_to_ls(trace, trace_candidate)
#                     if ls.length > snap_threshold * 0.1
#                 ]
#                 if len(linestrings) < 2:
#                     continue
#                 for ep in reverse_endpoints:
#                     for ls in linestrings:
#                         if np.isclose(ep.distance(ls), 0, atol=1e-4):
#                             if len(linestrings) == 2:
#                                 other = [line for line in linestrings if line != ls][0]
#                             elif len(linestrings) == 3:
#                                 # Determining middle is not stable
#                                 # Weird small linestrings possible
#                                 candidates = determine_middle_in_triangle(
#                                     linestrings, snap_threshold, 1.0
#                                 )
#                                 if len(candidates) == 1:
#                                     other = candidates[0]
#                                 else:
#                                     raise ValueError("Expected one candidate.")
#                             else:
#                                 raise ValueError(
#                                     "Expected no more than 3 split segments."
#                                 )

#                             other_endpoints = get_trace_endpoints(other)
#                             if not np.isclose(
#                                 other_endpoints[-1].distance(ls), 0, atol=1e-4
#                             ):
#                                 other = LineString(
#                                     reversed(get_trace_coord_points(other))
#                                 )
#                                 other_endpoints = get_trace_endpoints(other)
#                             while (
#                                 len(resolve_split_to_ls(other, trace_candidate))
#                                 == len(linestrings)
#                                 and other_endpoints[-1].distance(trace_candidate)
#                                 < snap_threshold * 0.25
#                             ):
#                                 other = substring(
#                                     other, 0, other.length - snap_threshold * 0.1
#                                 )
#                             if len(linestrings) == 3:
#                                 another_segment = [
#                                     another
#                                     for another in linestrings
#                                     if another not in (candidates[0], ls)
#                                 ]
#                                 if len(another_segment) == 1:
#                                     assert other.is_valid

#                                     other = linemerge([other, another_segment[0]])
#                                 else:
#                                     raise ValueError("Expected one another.")
#                             trace = other
#                             any_changed_applied = True

#                             # while (
#                             #     other_endpoints[-1].distance(trace_candidate)
#                             #     < snap_threshold
#                             # ):
#                             #     other = substring(
#                             #         other,
#                             #         0,
#                             #         other.length - snap_threshold,
#                             #         normalized=False,
#                             #     )

#             candidate_split_ls = resolve_split_to_ls(trace_candidate, trace)
#             for endp in candidate_endpoints:
#                 if len(candidate_split_ls) > 1 and any(
#                     [
#                         np.isclose(cand.distance(endp), 0, atol=1e-4)
#                         for cand in candidate_split_ls
#                     ]
#                 ):
#                     continue
#                 distances_to_other_traces = trace_candidates.geometry.distance(
#                     endp
#                 ).values
#                 # TODO: Gather earlier (but use same filtering) to avoid repetition
#                 distance_to_current_trace = endp.distance(trace)
#                 # Check if node is closer to some other trace instead of
#                 # current. Sum must be higher than 1 because the trace which
#                 # the endpoint belongs to is always closest (naturally).
#                 if (
#                     sum(
#                         [
#                             dist_other < distance_to_current_trace
#                             for dist_other in distances_to_other_traces
#                         ]
#                     )
#                     > 1
#                 ):
#                     # Do not insert to current trace if node is closer to some
#                     # other trace.
#                     continue
#                 trace = insert_point_to_linestring(trace, endp)
#                 any_changed_applied = True

#         snapped_traces.append(trace)
#     assert len(snapped_traces) == len(traces)
#     assert all([isinstance(ls, LineString) for ls in snapped_traces])
#     return gpd.GeoSeries(snapped_traces), any_changed_applied


def snap_traces(
    traces: List[LineString],
    snap_threshold: float,
    areas: Optional[List[Union[Polygon, MultiPolygon]]] = None,
    final_allowed_loop=False,
) -> Tuple[List[LineString], bool]:
    """
    Snap traces to end exactly at other traces.
    """
    # Only handle LineStrings
    assert all([isinstance(trace, LineString) for trace in traces])

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
    simply_snapped_traces_list = list(simply_snapped_traces)

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
    assert all([isinstance(ls, LineString) for ls in snapped_traces])

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
        logging.error("In final_allowed_loop and still snapping:")
        logging.error(f"{traces, trace}")

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

    assert trace not in list(trace_candidates)

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
        logging.error("In final_allowed_loop and still snapping:")
        logging.error(f"{traces, endpoints}")

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
            [
                endpoint.distance(candidate) < snap_threshold
                for endpoint in trace_endpoints
            ]
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
            if any([endpoint.intersects(coord_point) for coord_point in coord_points]):
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
                [
                    intersection_point.distance(endpoint) < snap_threshold
                    for intersection_point in intersection_points
                ]
            ):
                # Overlapping snap
                # These are not fixed here and fallback to the other snapping
                # method
                continue

            sorted_points = sorted(
                zip(coord_points, distances), key=lambda vals: vals[1]
            )
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


def determine_nodes(
    trace_geodataframe: gpd.GeoDataFrame,
    snap_threshold: float,
    interactions=True,
    endpoints=True,
) -> Tuple[List[Point], List[Tuple[int, ...]]]:
    """
    Determine points of interest between traces.

    The points are linked to the indexes of the traces if
    trace_geodataframe with the returned node_id_data list. node_id_data
    contains tuples of ids. The ids represent indexes of
    nodes_of_interaction. The order of the node_id_data tuples is
    equivalent to the trace_geodataframe trace indexes.

    Conditionals interactions and endpoints allow for choosing what
    to return specifically.

    TODO: Waiting for branches and nodes refactor.
    """
    # nodes_of_interaction contains all intersection points between
    # trace_geodataframe traces.
    nodes_of_interaction: List[Point] = []
    # node_id_data contains all ids that correspond to points in
    # nodes_of_interaction
    node_id_data: List[Tuple[int, ...]] = []
    trace_geodataframe.reset_index(drop=True, inplace=True)
    spatial_index = trace_geodataframe.geometry.sindex
    for idx, geom in enumerate(trace_geodataframe.geometry):
        # for idx, row in trace_geodataframe.iterrows():
        start_length = len(nodes_of_interaction)

        trace_candidates_idx = list(spatial_index.intersection(geom.bounds))
        assert idx in trace_candidates_idx
        # Remove current geometry from candidates
        trace_candidates_idx.remove(idx)
        assert idx not in trace_candidates_idx
        trace_candidates = trace_geodataframe.geometry.iloc[trace_candidates_idx]
        intersection_geoms = trace_candidates.intersection(geom)
        if interactions:
            nodes_of_interaction.extend(
                determine_valid_intersection_points(intersection_geoms)
            )
        # Add trace endpoints to nodes_of_interaction
        if endpoints:
            try:
                nodes_of_interaction.extend(
                    [
                        endpoint
                        for endpoint in get_trace_endpoints(geom)
                        if not any(
                            # intersection_geoms.intersects(
                            #     endpoint.buffer(snap_threshold)
                            # )
                            intersection_geoms.distance(endpoint)
                            < snap_threshold
                        )
                        # Checking that endpoint is also not a point of
                        # interaction is not done if interactions are not
                        # determined.
                        or not interactions
                    ]
                )
            except TypeError:
                # Error is raised when MultiLineString is passed. They do
                # not provide a coordinate sequence.
                pass
        end_length = len(nodes_of_interaction)
        node_id_data.append(tuple([i for i in range(start_length, end_length)]))

    if len(nodes_of_interaction) == 0 or len(node_id_data) == 0:
        logging.error("Both nodes_of_interaction and node_id_data are empty...")
    return nodes_of_interaction, node_id_data


def safer_unary_union(
    traces_geosrs: gpd.GeoSeries, snap_threshold: float, size_threshold: int
) -> MultiLineString:
    """
    Perform unary union to transform traces to branch segments.

    unary_union is not completely stable with large datasets but problem can be
    alleviated by dividing analysis to parts.
    """
    if size_threshold < 100:
        raise ValueError(
            "Expected size_threshold to be higher than 100. Union might be impossible."
        )
    trace_count = traces_geosrs.shape[0]
    if trace_count < 2:
        return MultiLineString([geom for geom in traces_geosrs.geometry.values])
    if trace_count < size_threshold:
        # Try normal union without any funny business
        full_union = traces_geosrs.unary_union
        if len(full_union.geoms) > trace_count and isinstance(
            full_union, MultiLineString
        ):
            return full_union
    div = int(np.ceil(trace_count / size_threshold))
    part_count = int(np.ceil(trace_count / div))
    part_unions = []
    for i in range(1, div + 1):
        if i == 1:
            part = traces_geosrs.iloc[0:part_count]
        elif i == div:
            part = traces_geosrs.iloc[part_count * i - 1 :]
        else:
            part = traces_geosrs.iloc[part_count * i - 1 : part_count * i]
        part_union = part.unary_union
        if (
            not isinstance(part_union, MultiLineString)
            or len(part_union.geoms) < part.shape[0]
        ):
            # Still fails -> Try with lower threshold for part
            part_unions.append(
                safer_unary_union(
                    part, snap_threshold, size_threshold=size_threshold // 2
                )
            )
        else:
            # assume success
            part_unions.append(part_union)
    assert len(part_unions) == div

    full_union = unary_union(MultiLineString(list(chain(*part_unions))))
    if isinstance(full_union, MultiLineString):
        return full_union
    else:
        raise TypeError(f"Expected MultiLineString from unary_union. Got {full_union}")


def report_snapping_loop(loops: int, allowed_loops: int):
    """
    Report snapping looping.
    """
    logging.info(f"Loop :{ loops }")
    if loops >= 10:
        logging.warning(
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


# def branches_and_nodes_old(
#     traces: Union[gpd.GeoSeries, gpd.GeoDataFrame],
#     areas: Union[gpd.GeoSeries, gpd.GeoDataFrame],
#     snap_threshold: float,
#     allowed_loops=10,
#     already_clipped: bool = False,
# ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
#     """
#     Determine branches and nodes of given traces.

#     The traces will be cropped to the given target area(s) if not already
#     clipped(already_clipped).
#     """
#     traces_geosrs: gpd.GeoSeries = traces.geometry
#     areas_geosrs: gpd.GeoSeries = areas.geometry

#     # Only LineStrings
#     if not all(
#         [isinstance(trace, LineString) for trace in traces_geosrs.geometry.values]
#     ):
#         raise TypeError("Expected all geometries to be of type LineString.")

#     # Snapping occurs multiple times due to possible side effects of each snap
#     loops = 0
#     traces_geosrs, any_changes_applied = snap_traces_old(
#         traces_geosrs, snap_threshold, areas=areas_geosrs
#     )
#     # Snapping causes changes that might cause new errors. The snapping is looped
#     # as many times as there are changed made to the data.
#     # If loop count reaches allowed_loops, error is raised.
#     while any_changes_applied:
#         traces_geosrs, any_changes_applied = snap_traces_old(
#             traces_geosrs,
#             snap_threshold,
#             final_allowed_loop=loops == allowed_loops,
#             areas=areas_geosrs,
#         )
#         loops += 1
#         report_snapping_loop(loops, allowed_loops=allowed_loops)

#     # Clip if necessary
#     if not already_clipped:
#         traces_geosrs = crop_to_target_areas(
#             traces_geosrs, areas_geosrs, snap_threshold=snap_threshold
#         )

#     # Remove too small geometries.
#     traces_geosrs = traces_geosrs.loc[
#         traces_geosrs.geometry.length > snap_threshold * 2.01
#     ]
#     # TODO: Works but inefficient. Waiting for refactor.
#     nodes, _ = determine_nodes(
#         gpd.GeoDataFrame({GEOMETRY_COLUMN: traces_geosrs}),
#         snap_threshold=snap_threshold,
#     )
#     nodes_geosrs: gpd.GeoSeries = gpd.GeoSeries(nodes)
#     nodes_geosrs = remove_identical_sindex(nodes_geosrs, snap_threshold)
#     node_identities = get_node_identities(
#         traces_geosrs, nodes_geosrs, areas_geosrs, snap_threshold
#     )

#     # Branches are determined with shapely/geopandas unary_union
#     branches = gpd.GeoSeries(
#         [
#             b
#             for b in safer_unary_union(
#                 traces_geosrs, snap_threshold=snap_threshold, size_threshold=5000
#             ).geoms
#             if b.length > snap_threshold * 1.01
#         ]
#     )

#     # Report and error possibly unary_union failure
#     if len(branches) < len(traces_geosrs):
#         # unary_union can fail with too large datasets
#         raise ValueError(
#             "Expected more branches than traces. Possible unary_union failure."
#         )
#     # # Determine nodes and identities
#     # nodes, node_identities = node_identities_from_branches(
#     #     branches=branches, areas=areas_geosrs, snap_threshold=snap_threshold
#     # )
#     # # Collect to GeoSeries
#     # nodes_geosrs = gpd.GeoSeries(nodes)

#     # Determine branch identities
#     branch_identities = get_branch_identities(
#         branches, nodes_geosrs, node_identities, snap_threshold
#     )

#     # Collect to GeoDataFrames
#     node_gdf = gpd.GeoDataFrame(
#         {GEOMETRY_COLUMN: nodes_geosrs, CLASS_COLUMN: node_identities}
#     )
#     branch_gdf = gpd.GeoDataFrame(
#         {GEOMETRY_COLUMN: branches, CONNECTION_COLUMN: branch_identities}
#     )
#     return branch_gdf, node_gdf


def branches_and_nodes(
    traces: Union[gpd.GeoSeries, gpd.GeoDataFrame],
    areas: Union[gpd.GeoSeries, gpd.GeoDataFrame],
    snap_threshold: float,
    allowed_loops=10,
    already_clipped: bool = False,
    unary_size_threshold: int = 5000,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Determine branches and nodes of given traces.

    The traces will be cropped to the given target area(s) if not already
    clipped(already_clipped).
    """
    traces_geosrs: gpd.GeoSeries = traces.geometry
    areas_geosrs: gpd.GeoSeries = areas.geometry

    areas_list = [
        poly
        for poly in areas_geosrs.geometry.values
        if isinstance(poly, (Polygon, MultiPolygon))
    ]
    traces_list = [
        trace for trace in traces.geometry.values if isinstance(trace, LineString)
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

    traces_geosrs = gpd.GeoSeries(traces_list)
    # Clip if necessary
    if not already_clipped:
        traces_geosrs = crop_to_target_areas(
            traces_geosrs, areas_geosrs, snap_threshold=snap_threshold
        )

    # Remove too small geometries.
    traces_geosrs = traces_geosrs.loc[
        traces_geosrs.geometry.length > snap_threshold * 2.01
    ]
    # TODO: Works but inefficient. Waiting for refactor.
    # nodes, _ = determine_nodes(
    #     gpd.GeoDataFrame({GEOMETRY_COLUMN: traces_geosrs}),
    #     snap_threshold=snap_threshold,
    # )
    # nodes_geosrs: gpd.GeoSeries = gpd.GeoSeries(nodes)
    # nodes_geosrs = remove_identical_sindex(nodes_geosrs, snap_threshold)
    # node_identities = get_node_identities(
    #     traces_geosrs, nodes_geosrs, areas_geosrs, snap_threshold
    # )

    # Branches are determined with shapely/geopandas unary_union
    branches = gpd.GeoSeries(
        [
            b
            for b in safer_unary_union(
                traces_geosrs,
                snap_threshold=snap_threshold,
                size_threshold=unary_size_threshold,
            ).geoms
            if b.length > snap_threshold * 1.01
        ]
    )

    # Report and error possibly unary_union failure
    if len(branches) < len(traces_geosrs):
        # unary_union can fail with too large datasets
        raise ValueError(
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
        {GEOMETRY_COLUMN: nodes_geosrs, CLASS_COLUMN: node_identities}
    )
    branch_gdf = gpd.GeoDataFrame(
        {GEOMETRY_COLUMN: branches, CONNECTION_COLUMN: branch_identities}
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
    if any(
        [
            endpoint.distance(area.boundary) < snap_threshold
            for area in areas.geometry.values
        ]
    ):
        return E_node

    candidate_idxs = list(endpoints_spatial_index.intersection(endpoint.coords[0]))

    candidate_idxs.remove(idx)

    candidates = endpoints_geoseries.iloc[candidate_idxs]

    intersecting_node_count = sum(
        [candidate.distance(endpoint) < snap_threshold for candidate in candidates]
    )
    if intersecting_node_count == 0:
        # I-node
        return I_node
    elif intersecting_node_count == 2:
        # Y-node
        return Y_node
    elif intersecting_node_count == 3:
        return X_node
    elif intersecting_node_count == 1:
        logging.error(
            f"Expected 0, 2 or 3 intersects. V-node or similar error at {endpoint.wkt}."
        )
        return I_node
    else:
        logging.error(
            f"Expected 0, 2 or 3 intersects. Multijunction at {endpoint.wkt}."
        )
        return X_node


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
    >>> nodes, identities = node_identities_from_branches(branches, areas, snap_threshold)
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

    # Collect into two lists
    nodes, identities = zip(*collected_nodes.values())

    return list(nodes), list(identities)
