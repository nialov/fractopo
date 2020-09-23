import geopandas as gpd
import shapely
from shapely.geometry import MultiPoint, Point, LineString
from shapely.ops import split
import numpy as np
from typing import List, Tuple

# Import trace_validator
from tval import trace_validator
from tval.trace_validator import BaseValidator

import logging

logging.basicConfig(level=logging.INFO, format="%(process)d-%(levelname)s-%(message)s")

# Setup
trace_validator.BaseValidator.set_snap_threshold_and_multipliers(0.001, 1.1, 1.1)

CC_branch = "C-C"
CI_branch = "C-I"
II_branch = "I-I"
Error_branch = "E-E"
X_node = "X"
Y_node = "Y"
I_node = "I"

type_column = "Type"
geom_column = "geometry"


def remove_identical(geosrs: gpd.GeoSeries, snap_threshold: float) -> gpd.GeoSeries:
    geosrs.reset_index(inplace=True, drop=True)

    marked_for_death = []
    for idx, p in enumerate(geosrs):
        if idx in marked_for_death:
            continue
        inter = geosrs.drop(idx).intersection(p.buffer(snap_threshold))
        colliding = inter.loc[[True if not i.is_empty else False for i in inter]]
        if len(colliding) > 0:
            index_to_list = colliding.index.to_list()
            assert len(index_to_list) > 0
            assert all([isinstance(i, int) for i in index_to_list])
            marked_for_death.extend(index_to_list)
    return geosrs.drop(marked_for_death)


def remove_identical_sindex(
    geosrs: gpd.GeoSeries, snap_threshold: float
) -> gpd.GeoSeries:
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
    traces: gpd.GeoSeries, nodes: gpd.GeoSeries, snap_threshold: float
) -> List[str]:
    identities = []
    trace_sindex = traces.sindex
    for i, p in enumerate(nodes):
        trace_candidate_idxs = list(
            trace_sindex.intersection(p.buffer(snap_threshold).bounds)
        )
        trace_candidates = traces.iloc[trace_candidate_idxs]
        inter_with_traces = trace_candidates.intersection(p.buffer(snap_threshold))
        # If theres 2 intersections -> X or Y
        # 1 (must be) -> I
        # Point + LineString -> Y
        # LineString + Linestring -> X or Y
        inter_with_traces_geoms = [iwt for iwt in inter_with_traces if not iwt.is_empty]
        assert len(inter_with_traces_geoms) < 3
        assert len(inter_with_traces_geoms) > 0

        if len(inter_with_traces_geoms) == 1:
            identities.append(I_node)
            continue
        if any([isinstance(iwt, shapely.geometry.Point) for iwt in inter_with_traces]):
            identities.append(Y_node)
            continue
        # assert len(inter_with_traces_geoms) == 2

        all_inter_endpoints = [
            pt
            for sublist in map(
                trace_validator.BaseValidator.get_trace_endpoints,
                inter_with_traces_geoms,
            )
            for pt in sublist
        ]
        if any([p.intersects(ep) for ep in all_inter_endpoints]):
            # Y-node
            identities.append(Y_node)
            continue
        else:
            identities.append(X_node)
            continue
    return identities


def find_y_nodes_and_snap_em(
    traces: gpd.GeoSeries,
    nodes: gpd.GeoSeries,
    node_ids: List[str],
    snap_threshold: float,
) -> gpd.GeoSeries:
    snapped = []
    assert len(nodes) == len(node_ids)
    for n, n_id in zip(nodes, node_ids):
        if n_id == X_node or n_id == I_node:
            snapped.append(n)
            continue
        # elif n.distance(t) > 0.02:
        distances = np.array([t.distance(n) for t in traces])
        distances_less_than = distances < snap_threshold
        if any(distances_less_than):
            assert len(distances[distances_less_than]) < 3
            # print(distances[distances_less_than])
            traces_nearby = traces.loc[distances_less_than]
            assert any([tn.intersects(n) for tn in traces_nearby])
            # print(traces_nearby, n, n_id)
            # There can be two nearby <- perfectly snapped Y-node
            assert len([tn for tn in traces_nearby if tn.intersects(n)]) < 3
            assert len(traces_nearby) != 0
            if all([tn.intersects(n) for tn in traces_nearby]):
                # TODO:
                # Snapping required if trace overlaps in a Y-node
                # for t in traces_nearby:
                #     endpoints = BaseValidator.get_trace_endpoints(t)
                #     endpoints_buffered = gpd.GeoSeries([ep.buffer(snap_threshold) for ep in endpoints])
                #     if n.intersects(endpoints_buffered):
                #         inter_projected = t.interpolate(t.project(n))
                #         snapped.append(inter_projected)
                #         continue

                snapped.append(n)
                continue
            for tn in traces_nearby:
                if not tn.intersects(n):
                    # print(f"Doesnt intersect {n}")
                    inter_projected = tn.interpolate(tn.project(n))
                    # print(n, n_id)
                    snapped.append(inter_projected)
                    continue
        else:
            # print(n, n_id)
            snapped.append(n)
    assert len(nodes) == len(snapped)
    snapped = gpd.GeoSeries(snapped)
    return snapped


def split_traces_to_branches(traces, nodes, node_identities, snap_threshold):
    def filter_with_sindex_then_split(
        trace: gpd.GeoSeries, nodes: gpd.GeoSeries, node_spatial_index
    ) -> Tuple[gpd.GeoSeries, MultiPoint]:

        node_candidate_idxs = list(node_spatial_index.intersection(trace.bounds))
        node_candidates = nodes.iloc[node_candidate_idxs]
        mp = MultiPoint([p for p in node_candidates])
        assert isinstance(mp, MultiPoint)
        if len(mp) == 0:
            return [trace]
        else:
            return split(trace, mp)

    assert len(nodes) == len(node_identities)
    nodes = gpd.GeoSeries(
        [
            node
            for node, node_id in zip(nodes, node_identities)
            if node_id == X_node or node_id == Y_node
        ]
    )
    # Index is made from buffer polygons but indexes will match with those
    # of nodes
    node_spatial_index = gpd.GeoSeries([p.buffer(snap_threshold) for p in nodes]).sindex
    branches_grouped = [
        filter_with_sindex_then_split(trace, nodes, node_spatial_index)
        for trace in traces
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
    assert len(nodes) == len(node_identities)
    nodes_buffered = gpd.GeoSeries(list(map(lambda p: p.buffer(snap_threshold), nodes)))
    node_gdf = gpd.GeoDataFrame(
        {geom_column: nodes_buffered, type_column: node_identities}
    )
    node_spatial_index = nodes_buffered.sindex
    branch_identities = []
    for idx, branch in enumerate(branches):
        node_candidate_idxs = list(node_spatial_index.intersection(branch.bounds))
        node_candidates = node_gdf.iloc[node_candidate_idxs]
        inter = node_candidates.intersects(
            MultiPoint(
                [p for p in trace_validator.BaseValidator.get_trace_endpoints(branch)]
            )
        )
        assert len(inter) == len(node_candidates)
        nodes_that_intersect = node_candidates.loc[inter]
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
        if number_of_I_nodes == 2 and number_of_XY_nodes == 0:
            branch_identities.append(II_branch)
        elif number_of_I_nodes == 1 and number_of_XY_nodes == 1:
            branch_identities.append(CI_branch)
        elif number_of_XY_nodes == 2 and number_of_I_nodes == 0:
            branch_identities.append(CC_branch)
        else:
            logging.error(
                "Did not find 2 XYI-nodes that intersected branch endpoints.\n"
                f"branch: {branch.wkt}\n"
                f"nodes_that_intersect[type_column]: {nodes_that_intersect[type_column]}\n"
            )
            branch_identities.append(Error_branch)
    return branch_identities


def branches_and_nodes(
    traces: gpd.GeoSeries, snap_threshold: float
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    nodes, _ = trace_validator.BaseValidator.determine_nodes(
        gpd.GeoDataFrame({geom_column: traces})
    )
    nodes = gpd.GeoSeries(nodes)
    nodes = remove_identical_sindex(nodes, snap_threshold)
    node_identities = get_node_identities(traces, nodes, snap_threshold)
    nodes = find_y_nodes_and_snap_em(traces, nodes, node_identities, snap_threshold)
    branches = split_traces_to_branches(traces, nodes, node_identities)
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


def snap_traces(traces: gpd.GeoSeries, snap_threshold: float) -> gpd.GeoSeries:
    """
    Snaps traces to end exactly at other traces when the trace endpoint
    is within the snap threshold of another trace.
    """
    traces_spatial_index = traces.sindex
    snapped_traces = []
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
        snapped_endpoints = []
        for n in endpoints:
            distances = np.array([t.distance(n) for t in trace_candidates])
            distances_less_than = distances < snap_threshold
            traces_to_snap_to = trace_candidates[distances_less_than]
            if len(traces_to_snap_to) == 1:
                t = traces_to_snap_to.iloc[0]
                inter_projected = t.interpolate(t.project(n))
                assert inter_projected.intersects(t)
                assert isinstance(inter_projected, Point)
                snapped_endpoints.append(inter_projected)
            elif len(traces_to_snap_to) == 0:
                snapped_endpoints.append(n)
            else:
                logging.warning(
                    "Trace endpoint is within the snap threshold"
                    " of two traces.\n"
                    f"No Snapping was done.\n"
                    f"endpoints: {endpoints}\n"
                    f"distances: {distances}\n"
                    f"traces_to_snap_to: {traces_to_snap_to}\n"
                )
                snapped_endpoints.append(n)
        assert len(snapped_endpoints) == len(endpoints)
        # Original coords
        trace_coords = [point for point in trace.coords]
        trace_coords[0] = snapped_endpoints[0]
        trace_coords[-1] = snapped_endpoints[-1]
        snapped_traces.append(LineString(trace_coords))

    assert len(snapped_traces) == len(traces)
    return gpd.GeoSeries(snapped_traces)
