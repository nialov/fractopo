import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import (
    MultiPoint,
    Point,
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
)
import numpy as np
from typing import List, Tuple, Union

# Import trace_validator
from fractopo.tval import trace_validator
from fractopo.tval.trace_validator import BaseValidator

import logging

logging.basicConfig(level=logging.INFO, format="%(process)d-%(levelname)s-%(message)s")

# Setup
trace_validator.BaseValidator.set_snap_threshold_and_multipliers(0.001, 1.1, 1.1)

from fractopo.general import (
    CC_branch,
    CE_branch,
    CI_branch,
    IE_branch,
    II_branch,
    EE_branch,
    Error_branch,
    X_node,
    Y_node,
    I_node,
    E_node,
    CONNECTION_COLUMN,
    CLASS_COLUMN,
    GEOMETRY_COLUMN,
    match_crs,
)


def remove_identical_sindex(
    geosrs: gpd.GeoSeries, snap_threshold: float
) -> gpd.GeoSeries:
    """
    Remove stacked nodes by using a search buffer the size of snap_threshold.
    """
    geosrs.reset_index(inplace=True, drop=True)
    spatial_index = geosrs.sindex
    marked_for_death = []
    point: Point
    for idx, point in enumerate(geosrs):
        if idx in marked_for_death:
            continue
        point = point.buffer(snap_threshold) if snap_threshold != 0 else point
        p_candidate_idxs = (
            list(spatial_index.intersection(point.bounds))
            if snap_threshold != 0
            else list(spatial_index.intersection(point.coords[0]))
        )
        p_candidate_idxs.remove(idx)
        p_candidates = geosrs.iloc[p_candidate_idxs]
        inter = p_candidates.intersects(point)
        colliding = inter.loc[inter]
        if len(colliding) > 0:
            index_to_list = colliding.index.to_list()
            assert len(index_to_list) > 0
            assert all([isinstance(i, int) for i in index_to_list])
            marked_for_death.extend(index_to_list)
    return geosrs.drop(marked_for_death)


def get_node_identities(
    traces: gpd.GeoSeries,
    nodes: gpd.GeoSeries,
    areas: gpd.GeoSeries,
    snap_threshold: float,
) -> List[str]:
    """
    Determines the type of node i.e. E, X, Y or I.
    Uses the given traces to assess the number of intersections.

    E.g.

    >>> traces = gpd.GeoSeries(
    ...     [LineString([(1, 1), (2, 2), (3, 3)]), LineString([(3, 0), (2, 2), (-2, 5)])]
    ...     )
    >>> nodes = gpd.GeoSeries(
    ...     [Point(2, 2), Point(1, 1), Point(3, 3), Point(3, 0), Point(-2, 5),]
    ...     )
    >>> areas = gpd.GeoSeries([Polygon([(-2, 5), (-2, -5), (4, -5), (4, 5)])])
    >>> snap_threshold = 0.001
    >>> get_node_identities(traces, nodes, areas, snap_threshold)
    ['X', 'I', 'I', 'I', 'E']

    """
    assert all([isinstance(trace, LineString) for trace in traces])
    identities = []
    trace_sindex = traces.sindex
    # nodes_sindex = nodes.sindex
    for i, p in enumerate(nodes):
        if any([p.buffer(snap_threshold).intersects(area.boundary) for area in areas]):
            identities.append(E_node)
            continue
        trace_candidate_idxs = list(
            trace_sindex.intersection(p.buffer(snap_threshold).bounds)
        )
        trace_candidates = traces.iloc[trace_candidate_idxs]
        inter_with_traces = trace_candidates.intersects(p.buffer(snap_threshold))
        # If theres 2 intersections -> X or Y
        # 1 (must be) -> I
        # Point + LineString -> Y
        # LineString + Linestring -> X or Y
        inter_with_traces_geoms = trace_candidates.loc[inter_with_traces]
        assert all([isinstance(t, LineString) for t in inter_with_traces_geoms])
        if not len(inter_with_traces_geoms) < 3:
            logging.error(
                "Node intersects trace_candidates more than two times.\n"
                f"Node: {p.wkt}\n"
                f"inter_with_traces_geoms: {inter_with_traces_geoms}\n"
            )
        assert len(inter_with_traces_geoms) > 0

        if len(inter_with_traces_geoms) == 1:
            identities.append(I_node)
            continue
        # If more than 2 traces intersect -> Y-node or X-node

        all_inter_endpoints = [
            pt
            for sublist in map(
                trace_validator.BaseValidator.get_trace_endpoints,
                inter_with_traces_geoms,
            )
            for pt in sublist
        ]
        assert all([isinstance(pt, Point) for pt in all_inter_endpoints])
        if any([p.intersects(ep) for ep in all_inter_endpoints]):
            # Y-node
            identities.append(Y_node)
            continue
        else:
            # X-node
            identities.append(X_node)
            continue
    assert len(identities) == len(nodes)
    return identities

    # def split_traces_with_self(traces, snap_threshold):
    #     trace_sindex = traces.sindex
    #     split_traces = []
    #     for idx, trace in enumerate(traces):
    #         minx, miny, maxx, maxy = trace.bounds
    #         extended_bounds = (
    #             minx - snap_threshold * 10,
    #             miny - snap_threshold * 10,
    #             maxx + snap_threshold * 10,
    #             maxy + snap_threshold * 10,
    #         )
    #         trace_candidate_idxs = list(trace_sindex.intersection(extended_bounds))
    #         assert len(trace_candidate_idxs) != 0
    #         trace_candidate_idxs.remove(idx)
    #         if len(trace_candidate_idxs) == 0:
    #             split_traces.append(trace)
    #         trace_candidates = traces.iloc[trace_candidate_idxs]
    #         trace_candidates_mls = MultiLineString([tc for tc in trace_candidates])
    #         intersection_points = trace.intersection(trace_candidates_mls)
    #         if isinstance(intersection_points, Point):
    #             intersection_points = MultiPoint([intersection_points])
    #         no_multipoints = []
    #         for ip in intersection_points:
    #             if isinstance(ip, Point):
    #                 no_multipoints.append(ip)
    #             elif isinstance(ip, MultiPoint):
    #                 no_multipoints.extend([p for p in ip])
    #             else:
    #                 raise ValueError("Invalid geometry from intersection.")
    #         no_multipoints = MultiPoint(no_multipoints)

    #         assert isinstance(trace_candidates_mls, MultiLineString)
    #         assert isinstance(no_multipoints, MultiPoint)
    #         assert all([isinstance(p, Point) for p in no_multipoints])
    #         split_trace = split(trace, no_multipoints)
    #         linestrings = []
    #         for geom in split_trace:
    #             if isinstance(geom, LineString):
    #                 linestrings.append(geom)
    #             elif isinstance(geom, MultiLineString):
    #                 linestrings.extend([ls for ls in geom])
    #             else:
    #                 raise ValueError(
    #                     "Uncompatible geometry from split.\n" f"geom wkt : {geom.wkt}"
    #                 )
    #         assert all([isinstance(trace, LineString) for trace in linestrings])
    #         split_traces.extend(linestrings)
    #     assert len(split_traces) > len(traces)
    #     return split_traces

    # def split_traces_to_branches_with_traces(
    #     traces, nodes, node_identities, snap_threshold
    # ):
    #     """
    #     Splits given traces to branches by a combination of Y-nodes and cutting
    #     them with themselves.
    #     """

    #     def filter_with_sindex_then_split(
    #         trace: gpd.GeoSeries,
    #         nodes: gpd.GeoSeries,
    #         node_spatial_index,
    #         traces,
    #         traces_spatial_index,
    #         idx,
    #     ) -> Union[shapely.geometry.collection.GeometryCollection, List[LineString]]:
    #         """
    #         First filters both nodes and traces with spatial indexes. Then
    #         splits given trace using both Y-nodes and other traces.
    #         """

    #         if node_spatial_index is not None:
    #             node_candidate_idxs = list(node_spatial_index.intersection(trace.bounds))
    #             node_candidates = nodes.iloc[node_candidate_idxs]
    #             mp = MultiPoint([p for p in node_candidates])
    #         else:
    #             mp = MultiPoint()
    #         if traces_spatial_index is not None:
    #             trace_candidate_idxs = list(traces_spatial_index.intersection(trace.bounds))
    #             trace_candidate_idxs.remove(idx)
    #             trace_candidates = traces.iloc[trace_candidate_idxs]
    #             mls = MultiLineString([t for t in trace_candidates])
    #         else:
    #             mls = MultiLineString()
    #         if len(mp) == 0:
    #             if len(mls) == 0:
    #                 return [trace]
    #             else:
    #                 try:
    #                     return split(trace, mls)
    #                 except GEOSException as geos_exception:
    #                     logging.error(
    #                         "GEOSException when splitting trace.\n" f"{geos_exception}"
    #                     )
    #                     return [trace]
    #         else:
    #             split_with_nodes = split(trace, mp)
    #             if len(mls) != 0:
    #                 split_with_traces = []
    #                 for branch in split_with_nodes:
    #                     try:
    #                         split_with_traces.extend([geom for geom in split(branch, mls)])
    #                     except GEOSException as geos_exception:
    #                         logging.error(
    #                             "GEOSException when splitting branch.\n" f"{geos_exception}"
    #                         )
    #                         split_with_traces.extend([branch])
    #             else:
    #                 return split_with_nodes
    #             return split_with_traces

    # assert len(nodes) == len(node_identities)
    # nodes = gpd.GeoSeries(
    #     [node for node, node_id in zip(nodes, node_identities) if node_id == Y_node]
    # )
    # # Index is made from buffer polygons but indexes will match with those
    # # of nodes
    # node_spatial_index = gpd.GeoSeries([p.buffer(snap_threshold) for p in nodes]).sindex
    # traces_spatial_index = traces.sindex
    # branches_grouped = [
    #     filter_with_sindex_then_split(
    #         trace, nodes, node_spatial_index, traces, traces_spatial_index, idx
    #     )
    #     for idx, trace in enumerate(traces)
    # ]
    # # Flatten list
    # branches = gpd.GeoSeries([g for subgroup in branches_grouped for g in subgroup])
    # return branches


def determine_branch_identity(
    number_of_I_nodes: int, number_of_XY_nodes: int, number_of_E_nodes: int
) -> str:
    """
    Determines the identity of a branch based on the amount of I-, XY- and E-nodes
    and returns it as a string.

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
    Determines the type of branch i.e. C-C, C-I or I-I for all branches of a
    GeoSeries.

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
    for idx, branch in enumerate(branches):
        node_candidate_idxs = list(node_spatial_index.intersection(branch.bounds))
        node_candidates = node_gdf.iloc[node_candidate_idxs]
        # Use distance instead of two polygon buffers
        inter = [
            dist < snap_threshold
            for dist in node_candidates.distance(
                MultiPoint(
                    [
                        p
                        for p in trace_validator.BaseValidator.get_trace_endpoints(
                            branch
                        )
                    ]
                )
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
    Calculates the angle between two vectors which are made from the given
    points: Both vectors have the same first point, nearest_point, and second
    point is either point or comparison_point.

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
            print(unit_vector_1, unit_vector_2, unit_vector_sum_len)
            raise ValueError(
                "Could not detemine point relationships." f"Vectors printed above."
            )
    assert 360 >= np.rad2deg(rad_angle) >= 0
    return np.rad2deg(rad_angle)


def insert_point_to_linestring(trace: LineString, point: Point) -> LineString:
    """
    Inserts given point to given trace LineString.
    The point location is determined to fit into the LineString without
    changing the geometrical order of LineString vertices
    (which only makes sense if LineString is sublinear.)

    TODO/Note: Does not work for 2.5D geometries (Z-coordinates).
    Z-coordinates will be lost.

    E.g.

    >>> trace = LineString([(0, 0), (1, 0), (2, 0), (3, 0)])
    >>> point = Point(1.25, 0.1)
    >>> insert_point_to_linestring(trace, point).wkt
    'LINESTRING (0 0, 1 0, 1.25 0.1, 2 0, 3 0)'

    >>> trace = LineString([(0, 0), (1, 0), (2, 0), (3, 0)])
    >>> point = Point(2.25, 0.1)
    >>> insert_point_to_linestring(trace, point).wkt
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
    t_points_gdf = gpd.GeoDataFrame(
        {
            GEOMETRY_COLUMN: [Point(c) for c in trace.coords],
            "distances": [Point(c).distance(point) for c in trace.coords],
        }
    )
    t_points_gdf.sort_values(by="distances", inplace=True)
    nearest_point = t_points_gdf.iloc[0].geometry
    assert isinstance(nearest_point, Point)
    nearest_point_idx = t_points_gdf.iloc[0].name
    if nearest_point_idx == 0:
        # It is the first node of linestring
        add_before = nearest_point_idx + 1
    elif nearest_point_idx == max(t_points_gdf.index):
        # It is the last node of linestring
        add_before = nearest_point_idx
    else:
        # It is in the middle of the linestring
        points_on_either_side = t_points_gdf.loc[
            [nearest_point_idx - 1, nearest_point_idx + 1]
        ]

        points_on_either_side["angle"] = list(
            map(
                lambda p: angle_to_point(point, nearest_point, p),
                points_on_either_side.geometry,
            )
        )
        assert sum(points_on_either_side["angle"] == 0.0) < 2
        smallest_angle_idx = points_on_either_side.sort_values(by="angle").iloc[0].name
        if smallest_angle_idx > nearest_point_idx:
            add_before = smallest_angle_idx
        else:
            add_before = nearest_point_idx

    t_coords = list(trace.coords)
    t_coords.insert(add_before, tuple(*point.coords))
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


def additional_snapping_func(
    trace: LineString, idx: int, additional_snapping: List[Tuple[int, Point]]
) -> LineString:
    """
    Inserts points into LineStrings to make sure trace intersects Y-node.

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
        for p in df["points_to_add"]:
            trace = insert_point_to_linestring(trace, p)
        assert isinstance(trace, LineString)
        return trace

    else:
        return trace


def snap_traces(
    traces: gpd.GeoSeries, snap_threshold: float
) -> Tuple[gpd.GeoSeries, bool]:
    """
    Snaps traces to end exactly at other traces when the trace endpoint
    is within the snap threshold of another trace.

    E.g. when within snap_threshold:

    >>> traces = gpd.GeoSeries(
    ...     [LineString([(1, 1), (2, 2), (3, 3)]), LineString([(1.9999, 2), (-2, 5)])]
    ...     )
    >>> snap_threshold = 0.001
    >>> snap_traces(traces, snap_threshold)
    (0    LINESTRING (1.00000 1.00000, 2.00000 2.00000, ...
    1       LINESTRING (2.00000 2.00000, -2.00000 5.00000)
    dtype: geometry, True)

    """
    any_changed_applied = False
    traces_spatial_index = traces.sindex
    snapped_traces: List[LineString]
    snapped_traces = []
    additional_snapping: List[Point]
    additional_snapping = []
    trace: LineString
    idx: int
    for idx, trace in enumerate(traces):
        minx, miny, maxx, maxy = trace.bounds
        extended_bounds = (
            minx - snap_threshold * 10,
            miny - snap_threshold * 10,
            maxx + snap_threshold * 10,
            maxy + snap_threshold * 10,
        )
        trace_candidate_idxs = list(traces_spatial_index.intersection(extended_bounds))
        trace_candidate_idxs.remove(idx)
        trace_candidates = traces.iloc[trace_candidate_idxs]
        if len(trace_candidates) == 0:
            snapped_traces.append(trace)
            continue
        # get_trace_endpoints returns ls[0] and then ls[-1]
        endpoints = BaseValidator.get_trace_endpoints(trace)
        snapped_endpoints: List[Point]
        snapped_endpoints = []
        n: Point
        for n in endpoints:
            how_many_n_intersects = sum(trace_candidates.intersects(n))
            if how_many_n_intersects == 1:
                snapped_endpoints.append(n)
                continue
            elif how_many_n_intersects > 1:
                logging.warning(
                    "Endpoint intersects more than two traces?\n"
                    f"Endpoint: {n.wkt}\n"
                    f"how_many_n_intersects: {how_many_n_intersects}\n"
                    f"trace_candidates: {[t.wkt for t in trace_candidates]}\n"
                )
            distances = np.array([t.distance(n) for t in trace_candidates])
            distances_less_than = distances < snap_threshold
            traces_to_snap_to = trace_candidates[distances_less_than]
            if len(traces_to_snap_to) == 1:
                any_changed_applied = True
                traces_to_snap_to_vertices = gpd.GeoSeries(
                    [Point(c) for c in traces_to_snap_to.geometry.iloc[0].coords]
                )
                vertice_intersects = traces_to_snap_to_vertices.intersects(
                    n.buffer(snap_threshold)
                )
                if sum(vertice_intersects) > 0:
                    # Snap endpoint to the vertice of the trace in which abutting trace
                    # abuts.
                    new_n = traces_to_snap_to_vertices[vertice_intersects].iloc[0]
                    assert isinstance(new_n, Point)
                    snapped_endpoints.append(
                        traces_to_snap_to_vertices[vertice_intersects].iloc[0]
                    )
                else:
                    # Other trace must be snapped to a vertice to assure
                    # that they intersect. I.e. a new vertice is added into
                    # the middle of the other trace.
                    # This is handled later.
                    t_idx = traces_to_snap_to.index.values[0]
                    additional_snapping.append((t_idx, n))

                    snapped_endpoints.append(n)
            elif len(traces_to_snap_to) == 0:
                snapped_endpoints.append(n)
            else:
                logging.warning(
                    "Trace endpoint is within the snap threshold"
                    " of two traces.\n"
                    f"No Snapping was done.\n"
                    f"endpoints: {list(map(lambda p: p.wkt, endpoints))}\n"
                    f"distances: {distances}\n"
                    f"traces_to_snap_to: {traces_to_snap_to}\n"
                )
                snapped_endpoints.append(n)
        assert len(snapped_endpoints) == len(endpoints)
        # Original coords
        trace_coords = [point for point in trace.coords]
        assert all([isinstance(tc, tuple) for tc in trace_coords])
        trace_coords[0] = tuple(*snapped_endpoints[0].coords)
        trace_coords[-1] = tuple(*snapped_endpoints[-1].coords)
        snapped_traces.append(LineString(trace_coords))

    # Handle additional_snapping
    if len(additional_snapping) != 0:
        snapped_traces = list(
            map(
                lambda idx, val: additional_snapping_func(
                    val, idx, additional_snapping
                ),
                *zip(*enumerate(snapped_traces)),
            )
        )

    assert len(snapped_traces) == len(traces)
    return gpd.GeoSeries(snapped_traces), any_changed_applied


def get_trace_coord_points(trace: LineString) -> List[Point]:
    """
    Get all coordinate Points of a LineString.

    >>> trace = LineString([(0, 0), (2, 0), (3, 0)])
    >>> coord_points = get_trace_coord_points(trace)
    >>> print([p.wkt for p in coord_points])
    ['POINT (0 0)', 'POINT (2 0)', 'POINT (3 0)']

    """
    assert isinstance(trace, LineString)
    return [Point(xy) for xy in trace.coords]


def snap_traces_alternative(
    traces: gpd.GeoSeries, snap_threshold: float
) -> Tuple[gpd.GeoSeries, bool]:
    """
    Snaps traces to end exactly at other traces when the trace endpoint
    is within the snap threshold of another trace.
    """
    assert all([isinstance(trace, LineString) for trace in traces])
    any_changed_applied = False
    traces_spatial_index = traces.sindex
    snapped_traces: List[LineString]
    snapped_traces = []
    trace: LineString
    idx: int
    for idx, trace in enumerate(traces):
        minx, miny, maxx, maxy = trace.bounds
        extended_bounds = (
            minx - snap_threshold * 10,
            miny - snap_threshold * 10,
            maxx + snap_threshold * 10,
            maxy + snap_threshold * 10,
        )
        trace_candidate_idxs = list(traces_spatial_index.intersection(extended_bounds))
        trace_candidate_idxs.remove(idx)
        trace_candidates = traces.iloc[trace_candidate_idxs]
        if len(trace_candidates) == 0:
            snapped_traces.append(trace)
            continue
        # get_trace_endpoints returns ls[0] and then ls[-1]
        endpoints = []
        for trace_candidate in trace_candidates:
            endpoints = BaseValidator.get_trace_endpoints(trace_candidate)
            endpoints.extend(endpoints)
        assert all([isinstance(ep, Point) for ep in endpoints])
        n: Point
        for n in endpoints:
            trace_coord_points = gpd.GeoSeries(get_trace_coord_points(trace))
            if any(trace_coord_points.intersects(n)):
                continue
            if n.buffer(snap_threshold).intersects(trace):
                # Add point as trace coordinate
                trace = insert_point_to_linestring(trace, n)
                assert any(
                    [n.intersects(point) for point in get_trace_coord_points(trace)]
                )
                any_changed_applied = True
        snapped_traces.append(trace)
    assert len(snapped_traces) == len(traces)
    assert all([isinstance(ls, LineString) for ls in snapped_traces])
    return gpd.GeoSeries(snapped_traces), any_changed_applied


def mls_to_ls(multilinestrings: List[MultiLineString]) -> List[LineString]:
    """
    Flattens a list of multilinestrings to a list of linestrings.

    >>> multilinestrings = [
    ...     MultiLineString(
    ...             [
    ...                  LineString([(1, 1), (2, 2), (3, 3)]),
    ...                  LineString([(1.9999, 2), (-2, 5)]),
    ...             ]
    ...                    ),
    ...     MultiLineString(
    ...             [
    ...                  LineString([(1, 1), (2, 2), (3, 3)]),
    ...                  LineString([(1.9999, 2), (-2, 5)]),
    ...             ]
    ...                    ),
    ... ]
    >>> result_linestrings = mls_to_ls(multilinestrings)
    >>> print([ls.wkt for ls in result_linestrings])
    ['LINESTRING (1 1, 2 2, 3 3)', 'LINESTRING (1.9999 2, -2 5)',
    'LINESTRING (1 1, 2 2, 3 3)', 'LINESTRING (1.9999 2, -2 5)']

    """
    linestrings: List[LineString] = []
    for mls in multilinestrings:
        linestrings.extend([ls for ls in mls.geoms])
    if not all([isinstance(ls, LineString) for ls in linestrings]):
        raise ValueError("MultiLineStrings within MultiLineStrings?")
    return linestrings


def crop_to_target_areas(traces: gpd.GeoSeries, areas: gpd.GeoSeries) -> gpd.GeoSeries:
    """
    Crops given traces to the gives area polygons.

    E.g.

    >>> traces = gpd.GeoSeries(
    ...     [LineString([(1, 1), (2, 2), (3, 3)]), LineString([(1.9999, 2), (-2, 5)])]
    ...     )
    >>> areas = gpd.GeoSeries(
    ...     [
    ...             Polygon([(1, 1), (-1, 1), (-1, -1), (1, -1)]),
    ...                     Polygon([(-2.5, 6), (-1.9, 6), (-1.9, 4), (-2.5, 4)]),
    ...                         ]
    ...                         )
    >>> cropped_traces = crop_to_target_areas(traces, areas)
    >>> print([trace.wkt for trace in cropped_traces])
    ['LINESTRING (-1.9 4.924998124953124, -2 5)']

    """
    if not all([isinstance(trace, LineString) for trace in traces]):
        logging.error("MultiLineString passed into crop_to_target_areas.")
    # TODO: CRS mismatch
    traces, areas = match_crs(traces, areas)
    clipped_traces = gpd.clip(traces, areas)
    clipped_traces_linestrings = [
        trace for trace in clipped_traces if isinstance(trace, LineString)
    ]
    ct_multilinestrings = [
        mls for mls in clipped_traces if isinstance(mls, MultiLineString)
    ]
    as_linestrings = mls_to_ls(ct_multilinestrings)
    return gpd.GeoSeries(clipped_traces_linestrings + as_linestrings)


def branches_and_nodes(
    traces: Union[gpd.GeoSeries, gpd.GeoDataFrame],
    areas: Union[gpd.GeoSeries, gpd.GeoDataFrame],
    snap_threshold: float,
    allowed_loops=10,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Determines branches and nodes of given traces.
    """
    if isinstance(traces, gpd.GeoDataFrame):
        traces = traces.geometry
    if isinstance(areas, gpd.GeoDataFrame):
        areas = areas.geometry
    assert all([isinstance(trace, LineString) for trace in traces])
    traces, any_changes_applied = snap_traces(traces, snap_threshold)
    loops = 0
    # Snapping causes changes that might cause new errors. The snapping is looped
    # as many times as there are changed made to the data.
    # If loop count reaches allowed_loops, error is raised.
    while any_changes_applied:
        traces, any_changes_applied = snap_traces(traces, snap_threshold)
        loops += 1
        print(f"Loop :{ loops }")
        if loops >= 10:
            logging.warning(
                f"""
                    10 or more loops have passed without resolved snapped
                    traces. Snapped traces cannot possibly be resolved.
                    """
            )
        if loops > allowed_loops:
            raise RecursionError(
                f"More loops have passed than allowed by allowed_loops ({allowed_loops}))"
            )

    traces = crop_to_target_areas(traces, areas)
    nodes, _ = trace_validator.BaseValidator.determine_nodes(
        gpd.GeoDataFrame({GEOMETRY_COLUMN: traces})
    )
    nodes = gpd.GeoSeries(nodes)
    nodes = remove_identical_sindex(nodes, snap_threshold)
    node_identities = get_node_identities(traces, nodes, areas, snap_threshold)
    branches = gpd.GeoSeries([b for b in traces.unary_union])
    branch_identities = get_branch_identities(
        branches, nodes, node_identities, snap_threshold
    )
    node_gdf = gpd.GeoDataFrame({GEOMETRY_COLUMN: nodes, CLASS_COLUMN: node_identities})
    branch_gdf = gpd.GeoDataFrame(
        {GEOMETRY_COLUMN: branches, CONNECTION_COLUMN: branch_identities}
    )
    return branch_gdf, node_gdf
