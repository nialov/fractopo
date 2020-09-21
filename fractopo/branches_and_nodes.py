import geopandas as gpd
import shapely
from shapely.geometry import LineString, Point, MultiPoint, MultiPolygon, Polygon
from shapely.ops import snap, split
import shapely
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Import trace_validator
from tval import trace_validator

# Setup
trace_validator.BaseValidator.set_snap_threshold_and_multipliers(0.001, 1.1, 1.1)

CC_branch = "C-C"
CI_branch = "C-I"
II_branch = "I-I"
X_node = "X"
Y_node = "Y"
I_node = "I"


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
    for i, p in enumerate(nodes):
        inter_with_traces = traces.intersection(p.buffer(snap_threshold))
        # If theres 2 intersections -> X or Y
        # 1 (must be) -> I
        # Point + LineString -> Y
        # LineString + Linestring -> X or Y
        inter_with_traces_geoms = [iwt for iwt in inter_with_traces if not iwt.is_empty]
        assert len(inter_with_traces_geoms) < 3
        assert len(inter_with_traces_geoms) > 0

        if len(inter_with_traces_geoms) == 1:
            identities.append("I")
            continue
        if any([isinstance(iwt, shapely.geometry.Point) for iwt in inter_with_traces]):
            identities.append("Y")
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
            identities.append("Y")
            continue
        else:
            identities.append("X")
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
        if n_id == "X" or n_id == "I":
            # print(n, n_id)
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
                # Both traces in a Y-node intersect -> no snapping required
                snapped.append(n)
            for tn in traces_nearby:
                if not tn.intersects(n):
                    # print(f"Doesnt intersect {n}")
                    inter_projected = tn.interpolate(tn.project(n))
                    # print(n, n_id)
                    snapped.append(inter_projected)
        else:
            # print(n, n_id)
            snapped.append(n)
    assert len(nodes) == len(snapped)
    snapped = gpd.GeoSeries(snapped)
    return snapped


def split_traces_to_branches(traces, nodes, node_identities):
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
            if node_id == "X" or node_id == "Y"
        ]
    )
    node_spatial_index = nodes.sindex
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
    node_gdf = gpd.GeoDataFrame({"geometry": nodes_buffered, "Type": node_identities})
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
                for inter_id in nodes_that_intersect["Type"]
                if inter_id == I_node
            ]
        )
        number_of_XY_nodes = len(
            [
                inter_id
                for inter_id in nodes_that_intersect["Type"]
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
            raise TypeError(
                "Node identities did not match specified strings."
                f"\nnode_identities: {nodes_that_intersect.Type}"
            )
    return branch_identities


def branches_and_nodes(
    traces: gpd.GeoSeries, snap_threshold: float
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    nodes, _ = trace_validator.BaseValidator.determine_nodes(
        gpd.GeoDataFrame({"geometry": traces})
    )
    nodes = gpd.GeoSeries(nodes)
    nodes = remove_identical_sindex(nodes, snap_threshold)
    node_identities = get_node_identities(traces, nodes, snap_threshold)
    nodes = find_y_nodes_and_snap_em(traces, nodes, node_identities, snap_threshold)
    branches = split_traces_to_branches(traces, nodes, node_identities)
    branch_identities = get_branch_identities(
        branches, nodes, node_identities, snap_threshold
    )
    node_geodataframe = gpd.GeoDataFrame({"geometry": nodes, "Type": node_identities})
    branch_geodataframe = gpd.GeoDataFrame(
        {"geometry": branches, "Type": branch_identities}
    )
    return branch_geodataframe, node_geodataframe

