"""
Functions for plotting cross-cutting and abutting relationships.
"""

from typing import Tuple, Optional, List, Dict
import math
from textwrap import wrap
from itertools import combinations, chain

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, MultiLineString
from shapely import prepared
from shapely.strtree import STRtree

from fractopo.analysis import tools
from fractopo.general import (
    X_node,
    Y_node,
    prepare_geometry_traces,
    get_trace_endpoints,
)


def determine_crosscut_abutting_relationships(
    trace_series: gpd.GeoSeries,
    node_series: gpd.GeoSeries,
    node_types: np.ndarray,
    set_array: np.ndarray,
    set_names: Tuple[str, ...],
    buffer_value: float,
):
    """
    Determines cross-cutting and abutting relationships between all
    inputted sets by using spatial intersects
    between node and trace data.

    TODO: No within set relations.....yet... Problem?
    """
    assert len(set_array) == len(trace_series)
    assert len(node_series) == len(node_types)
    # Determines xy relations and dynamically creates a dataframe as an aid for plotting the relations
    #
    relations_df = pd.DataFrame(
        columns=["name", "sets", "x", "y", "y-reverse", "error-count"]
    )

    is_xy = np.array([val in (X_node, Y_node) for val in node_types])
    node_series_xy = node_series.loc[is_xy]
    node_types_xy = node_types[is_xy]

    if len(set_names) < 2:
        raise ValueError(f"Expected more than one set to be defined.")

    set_combinations = combinations(set_names, 2)
    for first_set, second_set in set_combinations:

        trace_series_two_sets: Tuple[gpd.GeoSeries, gpd.GeoSeries] = (
            trace_series.loc[set_array == first_set],  # type: ignore
            trace_series.loc[set_array == second_set],  # type: ignore
        )
        set_names_two_sets = (first_set, second_set)
        # determine_nodes_intersecting_sets returns a boolean array
        # representing nodes that intersect both sets.
        intersects_both_sets = determine_nodes_intersecting_sets(
            trace_series_two_sets=trace_series_two_sets,  # type: ignore
            set_names_two_sets=set_names_two_sets,
            set_array=set_array,
            node_series_xy=node_series_xy,  # type: ignore
            buffer_value=buffer_value,
        )

        # TODO: Refactor get_intersect_frame to fractopo.general
        # Along with other support functions
        intersectframe = tools.get_intersect_frame(
            intersecting_nodes_frame,
            traceframe_two_sets,
            (s, c_s),
            use_length_sets=use_length_sets,
        )

        if len(intersectframe.loc[intersectframe.error == True]) > 0:
            # TODO
            pass
        intersect_series = intersectframe.groupby(["nodeclass", "sets"]).size()

        x_count = 0
        y_count = 0
        y_reverse_count = 0

        for item in [s for s in intersect_series.iteritems()]:
            value = item[1]
            if item[0][0] == "X":
                x_count = value
            elif item[0][0] == "Y":
                if item[0][1] == (s, c_s):  # it's set s abutting in set c_s
                    y_count = value
                elif item[0][1] == (
                    c_s,
                    s,
                ):  # it's set c_s abutting in set s
                    y_reverse_count = value
                else:
                    raise ValueError(
                        f"item[0][1] doesnt equal {(s, c_s)}"
                        f" nor {(c_s, s)}\nitem[0][1]: {item[0][1]}"
                    )
            else:
                raise ValueError(
                    f'item[0][0] doesnt match "X" or "Y"\nitem[0][0]: {item[0][0]}'
                )

        addition = {
            "name": name,
            "sets": (s, c_s),
            "x": x_count,
            "y": y_count,
            "y-reverse": y_reverse_count,
            "error-count": len(intersectframe.loc[intersectframe.error == True]),
        }

        relations_df = relations_df.append(addition, ignore_index=True)


def determine_nodes_intersecting_sets(
    trace_series_two_sets: Tuple[gpd.GeoSeries, gpd.GeoSeries],
    set_array: np.ndarray,
    set_names_two_sets: Tuple[str, str],
    node_series_xy: gpd.GeoSeries,
    buffer_value: float,
) -> List[bool]:
    """
    Does a spatial intersect between node GeoDataFrame with only X- and Y-nodes
    and the trace GeoDataFrame with only two sets. Returns boolean array of
    based on

    E.g.

    >>> traces = gpd.GeoSeries([LineString([(0,0), (1, 1)])]), gpd.GeoSeries([LineString([(0,1), (0, -1)])])
    >>> set_array = np.array(["1", "2"])
    >>> set_names_two_sets = ("1", "2")
    >>> node_series_xy = gpd.GeoSeries([Point(0, 0), Point(1, 1), Point(0, 1), Point(0, -1)])
    >>> buffer_value = 0.001
    >>> determine_nodes_intersecting_sets(traces, set_array, set_names_two_sets, node_series_xy, buffer_value)
    [True, False, False, False]

    """

    def _intersects_both(
        point: Point,
        prep_traces_first: prepared.PreparedGeometry,
        prep_traces_second: prepared.PreparedGeometry,
        buffer_value: float,
    ):
        return prep_traces_first.intersects(
            point.buffer(buffer_value)
        ) and prep_traces_second.intersects(point.buffer(buffer_value))

    assert len(set_names_two_sets) == 2
    if len(np.unique(set_array)) != 2:
        return [False for _ in range(len(set_array))]
    trace_series_first_set = trace_series_two_sets[0]
    trace_series_second_set = trace_series_two_sets[1]

    prep_traces_first = prepare_geometry_traces(trace_series_first_set)  # type: ignore
    prep_traces_second = prepare_geometry_traces(trace_series_second_set)  # type: ignore
    intersects_both_sets = [
        _intersects_both(
            point, prep_traces_first, prep_traces_second, buffer_value=buffer_value
        )
        for point in node_series_xy.geometry.values
    ]
    return intersects_both_sets


# intersecting_nodes_frame, traceframe, set_names_two_sets, use_length_sets=False
def determine_intersects(
    trace_series_two_sets: Tuple[gpd.GeoSeries, gpd.GeoSeries],
    set_names_two_sets: Tuple[str, str],
    buffer_value: float,
):
    """
    Does spatial intersects to determine how abutments and crosscuts occur between two sets

    E.g. where Set 2 ends in set 1 and Set 2 crosscuts set 1 (TODO: DECREPID)

    >>> nodes = gpd.GeoDataFrame({'geometry': [Point(0, 0), Point(1, 1)], 'c': ['Y', 'X']})
    >>> traces = gpd.GeoDataFrame(data={'geometry': [
    ... LineString([(-1, -1), (2, 2)]), LineString([(0, 0), (-0.5, 0.5)]), LineString([(2, 0), (0, 2)])]
    ... , 'set': [1, 2, 2]})
    >>> traces['startpoint'] = traces.geometry.apply(line_start_point)
    >>> traces['endpoint'] = traces.geometry.apply(line_end_point)
    >>> sets = (1, 2)
    >>> intersect_frame = get_intersect_frame(nodes, traces, sets)
    >>> intersect_frame
              node nodeclass    sets  error
    0  POINT (0 0)         Y  (2, 1)  False
    1  POINT (1 1)         X  (1, 2)  False

    """

    # intersectframe = pd.DataFrame(columns=["node", "nodeclass", "sets", "error"])

    first_set, second_set = (
        set_names_two_sets[0],
        set_names_two_sets[1],
    )  # sets for comparison

    trace_series_first_set = trace_series_two_sets[0]
    trace_series_second_set = trace_series_two_sets[1]

    first_set_prep = prepare_geometry_traces(trace_series_first_set)
    second_set_prep = prepare_geometry_traces(trace_series_second_set)
    # Creates a rtree from all start- and endpoints of set 1
    # Used in deducting in which set a trace abuts (Y-node)
    first_set_points = list(
        chain(
            *[
                endpoints
                for endpoints in trace_series_first_set.geometry.apply(
                    get_trace_endpoints
                ).values
            ]
        )
    )
    assert all([isinstance(p, Point) for p in first_set_points])
    rtree = STRtree()
    first_setpointtree = make_point_tree(first_setframe)

    if len(intersecting_nodes_frame) == 0:
        # No intersections between sets
        return intersectframe

    for idx, row in intersecting_nodes_frame.iterrows():
        node = row.geometry
        c = row.c

        l1 = first_set_prep.intersects(
            node.buffer(buffer_value)
        )  # Checks if node intersects set 1 traces.
        l2 = second_set_prep.intersects(
            node.buffer(buffer_value)
        )  # Checks if node intersects set 2 traces.

        if (l1 is False) and (l2 is False):  # DEBUGGING
            raise Exception(
                f"Node {node} does not intersect both sets {first_set} and {second_set}\n l1 is {l1} and l2 is {l2}"
            )

        # NO RELATIONS FOR NODE IS GIVEN AS ERROR == TRUE (ERROR).
        # addition gets overwritten if there are no errors
        addition = {
            "node": node,
            "nodeclass": c,
            "sets": set_names_two_sets,
            "error": True,
        }  # DEBUGGING

        # ALL X NODE RELATIONS
        if c == "X":
            if (l1 is True) and (l2 is True):  # It's an x-node between sets
                sets = (first_set, second_set)
                addition = {"node": node, "nodeclass": c, "sets": sets, "error": False}

            if (l1 is True) and (l2 is False):  # It's an x-node inside set 1
                raise Exception(
                    f"Node {node} does not intersect both sets {first_set} and {second_set}\n l1 is {l1} and l2 is {l2}"
                )
                # sets = (first_set, first_set)
                # addition = {'node': node, 'nodeclass': c, 'sets': sets}

            if (l1 is False) and (l2 is True):  # It's an x-node inside set 2
                raise Exception(
                    f"Node {node} does not intersect both sets {first_set} and {second_set}\n l1 is {l1} and l2 is {l2}"
                )
                # sets = (second_set, second_set)
                # addition = {'node': node, 'nodeclass': c, 'sets': sets}

        # ALL Y NODE RELATIONS
        elif c == "Y":
            if (l1 is True) and (l2 is True):  # It's an y-node between sets
                # p1 == length of list of nodes from first_set traces that intersect with X- or Y-node
                p1 = len(first_setpointtree.query(node.buffer(buffer_value)))
                if p1 != 0:  # set 1 ends in set 2
                    sets = (first_set, second_set)
                else:  # set 2 ends in set 1
                    sets = (second_set, first_set)
                addition = {"node": node, "nodeclass": c, "sets": sets, "error": False}

            if (l1 is True) and (l2 is False):  # It's a y-node inside set 1
                raise Exception(
                    f"Node {node} does not intersect both sets {first_set} and {second_set}\n l1 is {l1} and l2 is {l2}"
                )
                # sets = (first_set, first_set)
                # addition = {'node': node, 'nodeclass': c, 'sets': sets}

            if (l1 is False) and (l2 is True):  # It's a y-node inside set 2
                raise Exception(
                    f"Node {node} does not intersect both sets {first_set} and {second_set}\n l1 is {l1} and l2 is {l2}"
                )
                # sets = (second_set, second_set)
                # addition = {'node': node, 'nodeclass': c, 'sets': sets}
        else:
            raise ValueError(f"Node {node} neither X or Y")

        intersectframe = intersectframe.append(
            addition, ignore_index=True
        )  # Append frame with result

    return intersectframe


# def make_point_tree(traceframe):
#     points = []
#     for idx, row in traceframe.iterrows():
#         sp = row.startpoint
#         ep = row.endpoint
#         points.extend([sp, ep])
#     tree = strtree.STRtree(points)
#     return tree
