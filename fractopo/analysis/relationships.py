"""
Functions for plotting cross-cutting and abutting relationships.
"""

import logging
from itertools import chain, combinations
from typing import Dict, List, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from shapely import prepared
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from fractopo.general import (
    X_node,
    Y_node,
    get_trace_endpoints,
    prepare_geometry_traces,
)


def determine_crosscut_abutting_relationships(
    trace_series: gpd.GeoSeries,
    node_series: gpd.GeoSeries,
    node_types: np.ndarray,
    set_array: np.ndarray,
    set_names: Tuple[str, ...],
    buffer_value: float,
    label: str,
) -> pd.DataFrame:
    """
    Determine cross-cutting and abutting relationships between trace sets.

    Determines relationships between all inputted sets by using spatial
    intersects between node and trace data.

    E.g.

    >>> trace_series = gpd.GeoSeries(
    ...     [LineString([(0, 0), (1, 0)]), LineString([(0, 1), (0, -1)])]
    ... )
    >>> node_series = gpd.GeoSeries(
    ...     [Point(0, 0), Point(1, 0), Point(0, 1), Point(0, -1)]
    ... )
    >>> node_types = np.array(["Y", "I", "I", "I"])
    >>> set_array = np.array(["1", "2"])
    >>> set_names = ("1", "2")
    >>> buffer_value = 0.001
    >>> label = "title"
    >>> determine_crosscut_abutting_relationships(
    ...     trace_series,
    ...     node_series,
    ...     node_types,
    ...     set_array,
    ...     set_names,
    ...     buffer_value,
    ...     label,
    ... )
        name    sets  x  y y-reverse error-count
    0  title  (1, 2)  0  1         0           0

    TODO: No within set relations.....yet... Problem?

    """
    assert len(set_array) == len(trace_series)
    assert len(node_series) == len(node_types)
    assert all(isinstance(val, LineString) for val in trace_series.geometry.values)
    # Determines xy relations and dynamically creates a dataframe as an aid for
    # plotting the relations

    relations_df = pd.DataFrame(
        columns=["name", "sets", "x", "y", "y-reverse", "error-count"]
    )

    is_xy = np.array([val in (X_node, Y_node) for val in node_types])
    node_series_xy: gpd.GeoSeries = node_series.loc[is_xy]  # type: ignore
    node_types_xy = node_types[is_xy]

    assert len(node_series_xy) == len(node_types_xy)

    if len(set_names) < 2:
        raise ValueError("Expected more than one set to be defined.")

    set_combinations = combinations(set_names, 2)
    for first_set, second_set in set_combinations:

        trace_series_two_sets: Tuple[gpd.GeoSeries, gpd.GeoSeries] = (
            trace_series.loc[set_array == first_set],  # type: ignore
            trace_series.loc[set_array == second_set],  # type: ignore
        )

        if any(series.shape[0] == 0 for series in trace_series_two_sets):
            logging.warning("Expected first_set and second_set to both contain traces.")
            return relations_df
        set_names_two_sets = (first_set, second_set)
        # determine_nodes_intersecting_sets returns a boolean array
        # representing nodes that intersect both sets.
        intersects_both_sets = determine_nodes_intersecting_sets(
            trace_series_two_sets=trace_series_two_sets,  # type: ignore
            set_names_two_sets=set_names_two_sets,
            node_series_xy=node_series_xy,  # type: ignore
            buffer_value=buffer_value,
        )
        assert len(node_series_xy) == len(node_types_xy) == len(intersects_both_sets)
        node_series_xy_intersects = node_series_xy.loc[intersects_both_sets]
        node_types_xy_intersects = node_types_xy[intersects_both_sets]

        intersectframe = determine_intersects(
            trace_series_two_sets=trace_series_two_sets,
            node_series_xy_intersects=node_series_xy_intersects,  # type: ignore
            node_types_xy_intersects=node_types_xy_intersects,
            set_names_two_sets=set_names_two_sets,
            buffer_value=buffer_value,
        )
        intersect_series = intersectframe.groupby(["nodeclass", "sets"]).size()

        x_count = 0
        y_count = 0
        y_reverse_count = 0

        for item in list(intersect_series.iteritems()):
            value = item[1]
            if item[0][0] == X_node:
                x_count = value
            elif item[0][0] == Y_node:
                if item[0][1] == (
                    first_set,
                    second_set,
                ):  # it's set s abutting in set second_set
                    y_count = value
                elif item[0][1] == (
                    second_set,
                    first_set,
                ):  # it's set second_set abutting in set s
                    y_reverse_count = value
                else:
                    raise ValueError(
                        f"item[0][1] does not equal {(first_set, second_set)}"
                        f" nor {(second_set, first_set)}\nitem[0][1]: {item[0][1]}"
                    )
            else:
                raise ValueError(
                    f'item[0][0] does not match "X" or "Y"\nitem[0][0]: {item[0][0]}'
                )

        addition = {
            "name": label,
            "sets": (first_set, second_set),
            "x": x_count,
            "y": y_count,
            "y-reverse": y_reverse_count,
            "error-count": len(intersectframe.loc[intersectframe.error]),
        }

        relations_df = relations_df.append(addition, ignore_index=True)
    return relations_df


def determine_nodes_intersecting_sets(
    trace_series_two_sets: Tuple[gpd.GeoSeries, gpd.GeoSeries],
    set_names_two_sets: Tuple[str, str],
    node_series_xy: gpd.GeoSeries,
    buffer_value: float,
) -> List[bool]:
    """
    Conduct a spatial intersect between nodes and traces.

    Node GeoDataFrame contains only X- and Y-nodes and the trace GeoDataFrame
    only two sets. Returns boolean array of based on intersections.

    E.g.

    >>> traces = gpd.GeoSeries([LineString([(0, 0), (1, 1)])]), gpd.GeoSeries(
    ...     [LineString([(0, 1), (0, -1)])]
    ... )
    >>> set_names_two_sets = ("1", "2")
    >>> nodes_xy = gpd.GeoSeries([Point(0, 0), Point(1, 1), Point(0, 1), Point(0, -1)])
    >>> buffer_value = 0.001
    >>> determine_nodes_intersecting_sets(
    ...     traces, set_names_two_sets, nodes_xy, buffer_value
    ... )
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
    trace_series_first_set = trace_series_two_sets[0]
    trace_series_second_set = trace_series_two_sets[1]

    prep_traces_first = prepare_geometry_traces(trace_series_first_set)
    prep_traces_second = prepare_geometry_traces(trace_series_second_set)
    intersects_both_sets = [
        _intersects_both(
            point, prep_traces_first, prep_traces_second, buffer_value=buffer_value
        )
        for point in node_series_xy.geometry.values
    ]
    return intersects_both_sets


def determine_intersects(
    trace_series_two_sets: Tuple[gpd.GeoSeries, gpd.GeoSeries],
    set_names_two_sets: Tuple[str, str],
    node_series_xy_intersects: gpd.GeoSeries,
    node_types_xy_intersects: np.ndarray,
    buffer_value: float,
) -> pd.DataFrame:
    """
    Determine how abutments and crosscuts occur between two sets.

    E.g.

    >>> traces = gpd.GeoSeries([LineString([(0, 0), (1, 1)])]), gpd.GeoSeries(
    ...     [LineString([(0, 1), (0, -1)])]
    ... )
    >>> set_names_two_sets = ("1", "2")
    >>> node_series_xy_intersects = gpd.GeoSeries([Point(0, 0)])
    >>> node_types_xy_intersects = np.array(["Y"])
    >>> buffer_value = 0.001
    >>> determine_intersects(
    ...     traces,
    ...     set_names_two_sets,
    ...     node_series_xy_intersects,
    ...     node_types_xy_intersects,
    ...     buffer_value,
    ... )
              node nodeclass    sets  error
    0  POINT (0 0)         Y  (1, 2)  False

    """
    # TODO: No DataFrames -> Refactor to something else
    # TODO: Refactor intersect logic (but it works for now)
    intersectframe = pd.DataFrame(columns=["node", "nodeclass", "sets", "error"])
    if len(node_series_xy_intersects) == 0:
        # No intersections between sets
        logging.debug("No intersections between sets.")
        return intersectframe

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
        chain(*list(trace_series_first_set.geometry.apply(get_trace_endpoints).values))
    )
    assert all(isinstance(p, Point) for p in first_set_points)
    first_setpointtree = STRtree(first_set_points)
    node: Point
    node_class: str
    for node, node_class in zip(node_series_xy_intersects, node_types_xy_intersects):
        assert isinstance(node, Point)
        assert isinstance(node_class, str)
        # for idx, row in intersecting_nodes_frame.iterrows():
        # node = row.geometry
        # c = row.c

        # Checks if node intersects set 1 traces.
        l1 = first_set_prep.intersects(node.buffer(buffer_value))
        # Checks if node intersects set 2 traces.
        l2 = second_set_prep.intersects(node.buffer(buffer_value))

        if (l1 is False) and (l2 is False):  # DEBUGGING
            raise ValueError(
                f"Node {node} does not intersect both sets"
                f" {first_set} and {second_set}\n l1 is {l1} and l2 is {l2}"
            )

        try:
            addition = determine_intersect(
                node=node,
                node_class=node_class,
                l1=l1,
                l2=l2,
                first_set=first_set,
                second_set=second_set,
                first_setpointtree=first_setpointtree,
                buffer_value=buffer_value,
            )
        except ValueError:
            # NO RELATIONS FOR NODE IS GIVEN AS ERROR == TRUE (ERROR).
            # addition gets overwritten if there are no errors
            addition = {
                "node": node,
                "nodeclass": node_class,
                "sets": set_names_two_sets,
                "error": True,
            }  # DEBUGGING

        intersectframe = intersectframe.append(
            addition, ignore_index=True
        )  # Append frame with result

    return intersectframe


def determine_intersect(
    node: Point,
    node_class: str,
    l1: bool,
    l2: bool,
    first_set: str,
    second_set: str,
    first_setpointtree: STRtree,
    buffer_value: float,
) -> Dict[str, Union[Point, str, Tuple[str, str], bool]]:
    """
    Determine what intersection the node represents.

    TODO: R0912: Too many branches.
    """
    if node_class == "X":
        if l1 and l2:  # It's an x-node between sets
            sets = (first_set, second_set)
            addition = {
                "node": node,
                "nodeclass": node_class,
                "sets": sets,
                "error": False,
            }

        elif l1 and not l2:  # It's an x-node inside set 1
            raise ValueError(
                f"Node {node} does not intersect both sets"
                f" {first_set} and {second_set}\n l1 is {l1} and l2 is {l2}"
            )
            # sets = (first_set, first_set)
            # addition = {'node': node, 'nodeclass': c, 'sets': sets}

        elif not l1 and l2:  # It's an x-node inside set 2
            raise ValueError(
                f"Node {node} does not intersect both sets"
                f" {first_set} and {second_set}\n l1 is {l1} and l2 is {l2}"
            )
            # sets = (second_set, second_set)
            # addition = {'node': node, 'nodeclass': c, 'sets': sets}
        else:
            raise ValueError(
                f"Node {node} does not intersect both sets"
                f" {first_set} and {second_set}\n l1 is {l1} and l2 is {l2}"
            )

    # ALL Y NODE RELATIONS
    elif node_class == "Y":
        if (l1 is True) and (l2 is True):  # It's an y-node between sets
            # p1 == length of list of nodes from first_set traces
            # that intersect with X- or Y-node
            p1 = len(first_setpointtree.query(node.buffer(buffer_value)))
            if p1 != 0:  # set 1 ends in set 2
                sets = (first_set, second_set)
            else:  # set 2 ends in set 1
                sets = (second_set, first_set)
            addition = {
                "node": node,
                "nodeclass": node_class,
                "sets": sets,
                "error": False,
            }

        elif (l1 is True) and (l2 is False):  # It's a y-node inside set 1
            raise ValueError(
                f"Node {node} does not intersect both sets"
                f" {first_set} and {second_set}\n l1 is {l1} and l2 is {l2}"
            )
            # sets = (first_set, first_set)
            # addition = {'node': node, 'nodeclass': c, 'sets': sets}

        elif (l1 is False) and (l2 is True):  # It's a y-node inside set 2
            raise ValueError(
                f"Node {node} does not intersect both sets"
                f" {first_set} and {second_set}\n l1 is {l1} and l2 is {l2}"
            )
            # sets = (second_set, second_set)
            # addition = {'node': node, 'nodeclass': c, 'sets': sets}
        else:
            raise ValueError(
                f"Node {node} does not intersect both sets"
                f" {first_set} and {second_set}\n l1 is {l1} and l2 is {l2}"
            )
    else:
        raise ValueError(f"Node {node} neither X or Y")

    return addition


def plot_crosscut_abutting_relationships_plot(
    relations_df: pd.DataFrame, set_array: np.ndarray, set_names: Tuple[str, ...]
) -> Tuple[List[Figure], List[np.ndarray]]:
    """
    Plot cross-cutting and abutting relationships.
    """
    # relations_df = pd.DataFrame(
    #     columns=["name", "sets", "x", "y", "y-reverse", "error-count"]
    # )
    # SUBPLOTS, FIGURE SETUP
    # set_column = "set"
    cols = relations_df.shape[0]
    if cols == 0:
        logging.warning("Expected relations_df to have rows. Returning empty lists.")
        return [], []
    if cols == 2:
        cols = 1
    width = 12 / 3 * cols
    height = (width / cols) * 0.75
    names = set(relations_df.name.tolist())
    figs: List[Figure] = []
    fig_axes = []
    with plt.style.context("default"):
        for name in names:
            relations_df_with_name = relations_df.loc[relations_df.name == name]
            set_counts = []
            for set_name in set_names:
                set_counts.append(sum(set_array == set_name))

            fig, axes = plt.subplots(ncols=cols, nrows=1, figsize=(width, height))
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])

            prop_title = dict(
                boxstyle="square", facecolor="linen", alpha=1, linewidth=2
            )

            fig.suptitle(
                f"   {name}   ",
                x=0.19,
                y=1.0,
                fontsize=20,
                fontweight="bold",
                fontfamily="DejaVu Sans",
                va="center",
                bbox=prop_title,
            )

            for ax, idx_row in zip(axes, relations_df_with_name.iterrows()):
                row = idx_row[1]
                # TODO: More colors? change brightness or some other parameter?
                bars = ax.bar(
                    x=[0.3, 0.55, 0.65],
                    height=[row["x"], row["y"], row["y-reverse"]],
                    width=0.1,
                    color=["darkgrey", "darkolivegreen", "darkseagreen"],
                    linewidth=1,
                    edgecolor="black",
                    alpha=0.95,
                    zorder=10,
                )

                ax.legend(
                    bars,
                    (
                        f"Sets {row.sets[0]} and {row.sets[1]} cross-cut",
                        f"Set {row.sets[0]} abuts to set {row.sets[1]}",
                        f"Set {row.sets[1]} abuts to set {row.sets[0]}",
                    ),
                    framealpha=0.6,
                    loc="upper center",
                    edgecolor="black",
                    prop={"family": "DejaVu Sans"},
                )

                ax.set_ylim(0, 1.6 * max([row["x"], row["y"], row["y-reverse"]]))

                ax.grid(zorder=-10, color="black", alpha=0.5)

                xticks = [0.3, 0.6]
                xticklabels = ["X", "Y"]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)

                xticklabels_texts = ax.get_xticklabels()

                for xtick in xticklabels_texts:
                    xtick.set_fontweight("bold")
                    xtick.set_fontsize(12)

                ax.set_xlabel(
                    "Node type",
                    fontweight="bold",
                    fontsize=13,
                    fontstyle="italic",
                    fontfamily="DejaVu Sans",
                )
                ax.set_ylabel(
                    "Node count",
                    fontweight="bold",
                    fontsize=13,
                    fontstyle="italic",
                    fontfamily="DejaVu Sans",
                )

                # Set y ticks as integers
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

                plt.subplots_adjust(wspace=0.3)

                if ax == axes[-1]:
                    text = ""
                    prop = dict(boxstyle="square", facecolor="linen", alpha=1, pad=0.45)
                    for set_label, set_len in zip(set_names, set_counts):
                        text += f"Set {set_label} trace count: {set_len}"
                        if not set_label == set_names[-1]:
                            text += "\n"
                    ax.text(
                        1.1,
                        0.5,
                        text,
                        rotation=90,
                        transform=ax.transAxes,
                        va="center",
                        bbox=prop,
                        fontfamily="DejaVu Sans",
                    )
            figs.append(fig)
            fig_axes.append(axes)
    return figs, fig_axes
