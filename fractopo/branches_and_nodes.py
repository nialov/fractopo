import logging
from typing import List, Tuple, Union

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


# Setup

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
    get_trace_endpoints,
    get_trace_coord_points,
    crop_to_target_areas,
    mls_to_ls,
    determine_valid_intersection_points,
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
            identical_idxs.extend(index_to_list)
    return geosrs.drop(identical_idxs)


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
    p: Point
    for i, p in enumerate(nodes):
        if any([p.buffer(snap_threshold).intersects(area.boundary) for area in areas]):
            identities.append(E_node)
            continue
        trace_candidate_idxs = list(
            trace_sindex.intersection(p.buffer(snap_threshold).bounds)
        )
        trace_candidates: gpd.GeoSeries = traces.iloc[trace_candidate_idxs]
        inter_with_traces = trace_candidates.intersects(p.buffer(snap_threshold))
        # If theres 2 intersections -> X or Y
        # 1 (must be) -> I
        # Point + LineString -> Y
        # LineString + Linestring -> X or Y
        inter_with_traces_geoms: gpd.GeoSeries = trace_candidates.loc[inter_with_traces]
        assert all(
            [isinstance(t, LineString) for t in inter_with_traces_geoms.geometry.values]
        )
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
                get_trace_endpoints,
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
    Determine the type of branch i.e. C-C, C-I or I-I for all branches of a
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
    Calculate the angle between two vectors which are made from the given
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
            logging.error(unit_vector_1, unit_vector_2, unit_vector_sum_len)
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
        endpoints = get_trace_endpoints(trace)
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
        endpoints_list = []
        for trace_candidate in trace_candidates:
            endpoints = get_trace_endpoints(trace_candidate)
            endpoints_list.extend(endpoints)
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


def branches_and_nodes(
    traces: Union[gpd.GeoSeries, gpd.GeoDataFrame],
    areas: Union[gpd.GeoSeries, gpd.GeoDataFrame],
    snap_threshold: float,
    allowed_loops=10,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Determine branches and nodes of given traces.

    The traces will be cropped to the given target area(s) to correctly
    """
    traces_geosrs: gpd.GeoSeries = traces.geometry
    areas_geosrs: gpd.GeoSeries = areas.geometry
    if not all(
        [isinstance(trace, LineString) for trace in traces_geosrs.geometry.values]
    ):
        raise TypeError("Expected all geometries to be of type LineString.")
    traces_geosrs, any_changes_applied = snap_traces(traces_geosrs, snap_threshold)
    loops = 0
    # Snapping causes changes that might cause new errors. The snapping is looped
    # as many times as there are changed made to the data.
    # If loop count reaches allowed_loops, error is raised.
    while any_changes_applied:
        traces_geosrs, any_changes_applied = snap_traces(traces_geosrs, snap_threshold)
        loops += 1
        logging.info(f"Loop :{ loops }")
        if loops >= 10:
            logging.warning(
                f"""
                    10 or more loops have passed without resolved snapped
                    traces_geosrs. Snapped traces_geosrs might not possibly be resolved.
                    """
            )
        if loops > allowed_loops:
            raise RecursionError(
                f"More loops have passed than allowed by allowed_loops "
                "({allowed_loops})) for snapping traces_geosrs for branch determination."
            )

    traces_geosrs = crop_to_target_areas(traces_geosrs, areas_geosrs)
    # Remove too small geometries.
    traces_geosrs = traces_geosrs.loc[
        traces_geosrs.geometry.length > snap_threshold * 2.01
    ]
    # TODO: Works but inefficient. Waiting for refactor.
    nodes, _ = determine_nodes(
        gpd.GeoDataFrame({GEOMETRY_COLUMN: traces_geosrs}),
        snap_threshold=snap_threshold,
    )
    nodes_geosrs: gpd.GeoSeries = gpd.GeoSeries(nodes)
    nodes_geosrs = remove_identical_sindex(nodes_geosrs, snap_threshold)
    node_identities = get_node_identities(
        traces_geosrs, nodes_geosrs, areas_geosrs, snap_threshold
    )
    branches = gpd.GeoSeries(
        [b for b in traces_geosrs.unary_union if b.length > snap_threshold * 2.01]
    )
    branch_identities = get_branch_identities(
        branches, nodes_geosrs, node_identities, snap_threshold
    )
    node_gdf = gpd.GeoDataFrame(
        {GEOMETRY_COLUMN: nodes_geosrs, CLASS_COLUMN: node_identities}
    )
    branch_gdf = gpd.GeoDataFrame(
        {GEOMETRY_COLUMN: branches, CONNECTION_COLUMN: branch_identities}
    )
    return branch_gdf, node_gdf


def determine_nodes(
    trace_geodataframe: gpd.GeoDataFrame,
    snap_threshold: float,
    interactions=True,
    endpoints=True,
) -> Tuple[List[Point], List[Tuple[int, ...]]]:
    """
    Determines points of interest between traces.

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
                            intersection_geoms.intersects(
                                endpoint.buffer(snap_threshold)
                            )
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
