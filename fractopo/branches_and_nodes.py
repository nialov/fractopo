import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import (
    MultiPoint,
    Point,
    LineString,
    MultiLineString,
    MultiPolygon,
)
from shapely.ops import split
import numpy as np
from typing import List, Tuple, Union

from pygeos import GEOSException

# Import trace_validator
from tval import trace_validator
from tval.trace_validator import BaseValidator

import logging

logging.basicConfig(level=logging.INFO, format="%(process)d-%(levelname)s-%(message)s")

# Setup
trace_validator.BaseValidator.set_snap_threshold_and_multipliers(0.001, 1.1, 1.1)

CC_branch = "C-C"
CE_branch = "C-E"
CI_branch = "C-I"
IE_branch = "I-E"
II_branch = "I-I"
EE_branch = "E-E"
X_node = "X"
Y_node = "Y"
I_node = "I"
E_node = "E"

type_column = "Type"
geom_column = "geometry"


def remove_identical_sindex(
    geosrs: gpd.GeoSeries, snap_threshold: float
) -> gpd.GeoSeries:
    """
    Removes stacked nodes by using a search buffer the size of snap_threshold.
    """
    geosrs.reset_index(inplace=True, drop=True)
    spatial_index = geosrs.sindex
    marked_for_death = []
    for idx, p in enumerate(geosrs):
        if idx in marked_for_death:
            continue
        p = p.buffer(snap_threshold) if snap_threshold != 0 else p
        p_candidate_idxs = (
            list(spatial_index.intersection(p.bounds))
            if snap_threshold != 0
            else list(spatial_index.intersection(p.coords[0]))
        )
        p_candidate_idxs.remove(idx)
        p_candidates = geosrs.iloc[p_candidate_idxs]
        inter = p_candidates.intersects(p)
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
    """
    identities = []
    trace_sindex = traces.sindex
    nodes_sindex = nodes.sindex
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


# def find_y_nodes_and_snap_em(
#     traces: gpd.GeoSeries,
#     nodes: gpd.GeoSeries,
#     node_ids: List[str],
#     snap_threshold: float,
# ) -> gpd.GeoSeries:
#     snapped = []
#     assert len(nodes) == len(node_ids)
#     for n, n_id in zip(nodes, node_ids):
#         if n_id == X_node or n_id == I_node:
#             snapped.append(n)
#             continue
#         # elif n.distance(t) > 0.02:
#         distances = np.array([t.distance(n) for t in traces])
#         distances_less_than = distances < snap_threshold
#         if any(distances_less_than):
#             assert len(distances[distances_less_than]) < 3
#             # print(distances[distances_less_than])
#             traces_nearby = traces.loc[distances_less_than]
#             assert any([tn.intersects(n) for tn in traces_nearby])
#             # print(traces_nearby, n, n_id)
#             # There can be two nearby <- perfectly snapped Y-node
#             assert len([tn for tn in traces_nearby if tn.intersects(n)]) < 3
#             assert len(traces_nearby) != 0
#             if all([tn.intersects(n) for tn in traces_nearby]):
#                 # TODO:
#                 # Snapping required if trace overlaps in a Y-node
#                 # for t in traces_nearby:
#                 #     endpoints = BaseValidator.get_trace_endpoints(t)
#                 #     endpoints_buffered = gpd.GeoSeries([ep.buffer(snap_threshold) for ep in endpoints])
#                 #     if n.intersects(endpoints_buffered):
#                 #         inter_projected = t.interpolate(t.project(n))
#                 #         snapped.append(inter_projected)
#                 #         continue

#                 snapped.append(n)
#                 continue
#             for tn in traces_nearby:
#                 if not tn.intersects(n):
#                     # print(f"Doesnt intersect {n}")
#                     inter_projected = tn.interpolate(tn.project(n))
#                     # print(n, n_id)
#                     snapped.append(inter_projected)
#                     continue
#         else:
#             # print(n, n_id)
#             snapped.append(n)
#     assert len(nodes) == len(snapped)
#     snapped = gpd.GeoSeries(snapped)
#     return snapped


# # def split_traces_to_branches(traces, nodes, node_identities, snap_threshold):
# #     def filter_with_sindex_then_split(
# #         trace: gpd.GeoSeries, nodes: gpd.GeoSeries, node_spatial_index
# #     ) -> Union[shapely.geometry.collection.GeometryCollection, List[LineString]]:

# #         node_candidate_idxs = list(node_spatial_index.intersection(trace.bounds))
# #         node_candidates = nodes.iloc[node_candidate_idxs]
# #         mp = MultiPoint([p for p in node_candidates])
# #         assert isinstance(mp, MultiPoint)
# #         if len(mp) == 0:
# #             return [trace]
# #         else:
# #             return split(trace, mp)

# #     assert len(nodes) == len(node_identities)
# #     nodes = gpd.GeoSeries(
# #         [
# #             node
# #             for node, node_id in zip(nodes, node_identities)
# #             if node_id == X_node or node_id == Y_node
# #         ]
# #     )
# #     # Index is made from buffer polygons but indexes will match with those
# #     # of nodes
# #     node_spatial_index = gpd.GeoSeries([p.buffer(snap_threshold) for p in nodes]).sindex
# #     branches_grouped = [
# #         filter_with_sindex_then_split(trace, nodes, node_spatial_index)
# #         for trace in traces
# #     ]
# #     # Flatten list
# #     branches = gpd.GeoSeries([g for subgroup in branches_grouped for g in subgroup])
# #     return branches


def split_traces_to_branches_with_traces(
    traces, nodes, node_identities, snap_threshold
):
    """
    Splits given traces to branches by a combination of Y-nodes and cutting
    them with themselves.
    """

    def filter_with_sindex_then_split(
        trace: gpd.GeoSeries,
        nodes: gpd.GeoSeries,
        node_spatial_index,
        traces,
        traces_spatial_index,
        idx,
    ) -> Union[shapely.geometry.collection.GeometryCollection, List[LineString]]:
        """
        First filters both nodes and traces with spatial indexes. Then
        splits given trace using both Y-nodes and other traces.
        """

        if node_spatial_index is not None:
            node_candidate_idxs = list(node_spatial_index.intersection(trace.bounds))
            node_candidates = nodes.iloc[node_candidate_idxs]
            mp = MultiPoint([p for p in node_candidates])
        else:
            mp = MultiPoint()
        if traces_spatial_index is not None:
            trace_candidate_idxs = list(traces_spatial_index.intersection(trace.bounds))
            trace_candidate_idxs.remove(idx)
            trace_candidates = traces.iloc[trace_candidate_idxs]
            mls = MultiLineString([t for t in trace_candidates])
        else:
            mls = MultiLineString()
        if len(mp) == 0:
            if len(mls) == 0:
                return [trace]
            else:
                try:
                    return split(trace, mls)
                except GEOSException as geos_exception:
                    logging.error(
                        "GEOSException when splitting trace.\n" f"{geos_exception}"
                    )
                    return [trace]
        else:
            split_with_nodes = split(trace, mp)
            if len(mls) != 0:
                split_with_traces = []
                for branch in split_with_nodes:
                    try:
                        split_with_traces.extend([geom for geom in split(branch, mls)])
                    except GEOSException as geos_exception:
                        logging.error(
                            "GEOSException when splitting branch.\n" f"{geos_exception}"
                        )
                        split_with_traces.extend([branch])
            else:
                return split_with_nodes
            return split_with_traces

    assert len(nodes) == len(node_identities)
    nodes = gpd.GeoSeries(
        [node for node, node_id in zip(nodes, node_identities) if node_id == Y_node]
    )
    # Index is made from buffer polygons but indexes will match with those
    # of nodes
    node_spatial_index = gpd.GeoSeries([p.buffer(snap_threshold) for p in nodes]).sindex
    traces_spatial_index = traces.sindex
    branches_grouped = [
        filter_with_sindex_then_split(
            trace, nodes, node_spatial_index, traces, traces_spatial_index, idx
        )
        for idx, trace in enumerate(traces)
    ]
    # Flatten list
    branches = gpd.GeoSeries([g for subgroup in branches_grouped for g in subgroup])
    return branches


def get_branch_identities(
    branches: gpd.GeoSeries,
    nodes: gpd.GeoSeries,
    node_identities: list,
    snap_threshold: float,
) -> List[str]:
    """
    Determines the type of branch i.e. C-C, C-I or I-I.
    """
    assert len(nodes) == len(node_identities)
    nodes_buffered = gpd.GeoSeries(list(map(lambda p: p.buffer(snap_threshold), nodes)))
    node_gdf = gpd.GeoDataFrame({geom_column: nodes, type_column: node_identities})
    node_spatial_index = nodes_buffered.sindex
    branch_identities = []
    for idx, branch in enumerate(branches):
        node_candidate_idxs = list(node_spatial_index.intersection(branch.bounds))
        node_candidates = node_gdf.iloc[node_candidate_idxs]
        # inter = node_candidates.intersects(
        #     MultiPolygon(
        #         [
        #             p.buffer(snap_threshold)
        #             for p in trace_validator.BaseValidator.get_trace_endpoints(branch)
        #         ]
        #     )
        # )
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
                for inter_id in nodes_that_intersect[type_column]
                if inter_id == E_node
            ]
        )
        number_of_I_nodes = len(
            [
                inter_id
                for inter_id in nodes_that_intersect[type_column]
                if inter_id == I_node
            ]
        )
        number_of_XY_nodes = len(
            [
                inter_id
                for inter_id in nodes_that_intersect[type_column]
                if inter_id in [X_node, Y_node]
            ]
        )
        # if branch.intersects(Polygon([(1, 0.9), (1.1, 1.1), (0.9, 1.1)])):
        #     if number_of_I_nodes == 2:
        #         print(nodes_that_intersect)
        #         assert False
        if number_of_I_nodes == 2:
            branch_identities.append(II_branch)
        elif number_of_XY_nodes == 2:
            branch_identities.append(CC_branch)
        elif number_of_E_nodes == 2:
            branch_identities.append(EE_branch)
        elif number_of_I_nodes == 1 and number_of_XY_nodes == 1:
            branch_identities.append(CI_branch)
        elif number_of_E_nodes == 1 and number_of_XY_nodes == 1:
            branch_identities.append(CE_branch)
        elif number_of_E_nodes == 1 and number_of_I_nodes == 1:
            branch_identities.append(IE_branch)
        elif number_of_I_nodes + number_of_E_nodes + number_of_XY_nodes != 2:
            logging.error(
                "Did not find 2 EXYI-nodes that intersected branch endpoints.\n"
                f"branch: {branch.wkt}\n"
                f"nodes_that_intersect[type_column]: {nodes_that_intersect[type_column]}\n"
            )
            branch_identities.append(EE_branch)
    return branch_identities


def angle_to_point(
    point: Point, nearest_point: Point, comparison_point: Point
) -> float:
    """
    Calculates the angle between two vectors which are made from the given
    points: Both vectors have the same first point, nearest_point, and second
    point is either point or comparison_point.
    """
    x1, y1 = tuple(*nearest_point.coords)
    x2, y2 = tuple(*point.coords)
    x3, y3 = tuple(*comparison_point.coords)
    # Vector from nearest_point to point
    vector_1 = np.array([x2 - x1, y2 - y1])
    # Vector from nearest_point to comparison_point
    vector_2 = np.array([x3 - x1, y3 - y1])
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    rad_angle = np.arccos(dot_product)
    if np.isnan(rad_angle):
        # Cannot determine with angle.
        unit_vector_sum_len = np.linalg.norm(unit_vector_1 + unit_vector_2)
        if np.isclose(unit_vector_sum_len, 0.0):
            return 180.0
        elif np.isclose(unit_vector_sum_len, 2.0):
            return 0.0
        else:
            print(unit_vector_1, unit_vector_2, unit_vector_sum_len)
            raise ValueError(
                "Could not detemine point relationships." f"Vectors printed above."
            )
    assert 360 >= np.rad2deg(rad_angle) >= 0
    return rad_angle


def insert_point_to_linestring(trace: LineString, point: Point) -> LineString:
    """
    Inserts given point to given trace LineString.
    The point location is determined to fit into the LineString without
    changing the geometrical order of LineString vertices
    (which only makes sense if LineString is sublinear.)
    """
    assert isinstance(trace, LineString)
    assert isinstance(point, Point)
    t_points_gdf = gpd.GeoDataFrame(
        {
            "geometry": [Point(c) for c in trace.coords],
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
        logging.warning("Non-simple geometry detected.\n" f"{new_trace.wkt}")
    return new_trace


def additional_snapping_func(
    trace: LineString, idx: int, additional_snapping: List[Tuple[int, Point]]
) -> LineString:
    """
    Inserts points into LineStrings to make sure trace intersects Y-node.
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


def crop_to_target_areas(traces: gpd.GeoSeries, areas: gpd.GeoSeries) -> gpd.GeoSeries:
    assert str(traces.crs) == str(areas.crs)
    clipped_traces = gpd.clip(traces, areas)
    return clipped_traces


def branches_and_nodes(
    traces: gpd.GeoSeries, areas: gpd.GeoSeries, snap_threshold: float
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Determines branches and nodes of given traces.
    """
    traces, any_changes_applied = snap_traces(traces, snap_threshold)
    loops = 0
    while any_changes_applied:
        traces, any_changes_applied = snap_traces(traces, snap_threshold)
        loops += 1
        print(f"Loop :{ loops }")
        if loops > 10:
            logging.warning(
                f"""
                    More than 10 loops have passed without resolved snapped
                    traces. Snapped traces cannot possibly be resolved.
                    """
            )
    traces = crop_to_target_areas(traces, areas)
    nodes, _ = trace_validator.BaseValidator.determine_nodes(
        gpd.GeoDataFrame({geom_column: traces})
    )
    nodes = gpd.GeoSeries(nodes)
    nodes = remove_identical_sindex(nodes, snap_threshold)
    node_identities = get_node_identities(traces, nodes, areas, snap_threshold)
    branches = split_traces_to_branches_with_traces(
        traces, nodes, node_identities, snap_threshold
    )
    branch_identities = get_branch_identities(
        branches, nodes, node_identities, snap_threshold
    )
    node_geodataframe = gpd.GeoDataFrame(
        {geom_column: nodes, type_column: node_identities}
    )
    branch_geodataframe = gpd.GeoDataFrame(
        {geom_column: branches, type_column: branch_identities}
    )
    return branch_geodataframe, node_geodataframe

