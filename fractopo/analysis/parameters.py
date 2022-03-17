"""
Analysis and plotting of geometric and topological parameters.
"""
import logging
from itertools import compress
from textwrap import fill
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ternary
from matplotlib import patheffects as path_effects
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.ndimage.filters import gaussian_filter
from ternary.ternary_axes_subplot import TernaryAxesSubplot

from fractopo.general import (
    CC_branch,
    CE_branch,
    CI_branch,
    E_node,
    EE_branch,
    I_node,
    IE_branch,
    II_branch,
    Number,
    Param,
    X_node,
    Y_node,
)


def determine_node_type_counts(
    node_types: np.ndarray, branches_defined: bool
) -> Dict[str, Number]:
    """
    Determine node type counts.
    """
    return {
        str(node_class): (
            (sum(node_types == node_class) if len(node_types) > 0 else 0)
            if branches_defined
            else np.nan
        )
        for node_class in (X_node, Y_node, I_node, E_node)
    }


def determine_branch_type_counts(
    branch_types: np.ndarray, branches_defined: bool
) -> Dict[str, Number]:
    """
    Determine branch type counts.
    """
    return {
        str(branch_class): sum(branch_types == branch_class)
        if branches_defined
        else np.nan
        for branch_class in (
            CC_branch,
            CI_branch,
            II_branch,
            CE_branch,
            IE_branch,
            EE_branch,
        )
    }


def ternary_text(text: str, ax: Axes):
    """
    Add ternary text about counts.
    """
    prop = dict(boxstyle="square", facecolor="linen", alpha=1, pad=0.45)
    ax.text(
        0.85,
        1.05,
        text,
        transform=ax.transAxes,
        fontsize="medium",
        weight="roman",
        verticalalignment="top",
        bbox=prop,
        fontfamily="DejaVu Sans",
        ha="center",
    )


def decorate_xyi_ax(ax: Axes, tax: TernaryAxesSubplot, counts: Dict[str, int]):
    """
    Decorate xyi plot.
    """
    xcount, ycount, icount = _get_xyi_counts(counts)
    text = "\n".join(
        (
            f"n: {xcount+ycount+icount}",
            f"X-nodes: {xcount}",
            f"Y-nodes: {ycount}",
            f"I-nodes: {icount}",
        )
    )
    initialize_ternary_points(ax, tax)
    tern_plot_the_fing_lines(tax)
    ternary_text(ax=ax, text=text)


def decorate_count_ax(
    ax: Axes, tax: TernaryAxesSubplot, label_counts: Dict[str, int], is_nodes: bool
):
    """
    Decorate ternary count plot.
    """
    text = "\n".join([f"{label} (n={value})" for label, value in label_counts.items()])
    if is_nodes:
        initialize_ternary_points(ax=ax, tax=tax)
        tern_plot_the_fing_lines(tax)
    else:
        initialize_ternary_branches_points(ax=ax, tax=tax)
        tern_plot_branch_lines(tax=tax)
    prop = dict(boxstyle="square", facecolor="linen", alpha=1, pad=0.45)
    ax.text(
        0.85,
        1.05,
        text,
        transform=ax.transAxes,
        fontsize="medium",
        weight="roman",
        verticalalignment="top",
        bbox=prop,
        fontfamily="DejaVu Sans",
        ha="center",
    )


def branches_intersect_boundary(branch_types: np.ndarray) -> np.ndarray:
    """
    Get array of if branches have E-component (intersects target area).
    """
    return ~np.isin(branch_types, (CC_branch, CI_branch, II_branch))


def plot_ternary_plot(
    counts_list: List[Dict[str, int]],
    labels: List[str],
    is_nodes: bool,
    colors: Optional[List[Optional[str]]] = None,
) -> Tuple[Figure, Axes, TernaryAxesSubplot]:
    """
    Plot ternary plot.

    Same function is used to plot both XYI and branch type ternary plots.
    """
    # Check if plotting xyi or branches
    # Set plotter and decorator functions accordingly
    if is_nodes:
        plotter = plot_xyi_plot_ax
        decorator = decorate_xyi_ax
    else:
        plotter = plot_branch_plot_ax
        decorator = decorate_branch_ax

    # Check if only one label is given
    one_label = len(labels) == 1

    # Check if colors are given and if they are, check that there are enough
    # for all ternary points
    if colors is None:
        colors = [None for _ in counts_list]
    elif len(colors) != len(counts_list):
        raise ValueError(
            f"Expected colors (len={len(colors)}) to be of"
            f" same size as node_counts_list (len={len(counts_list)})."
        )

    # Scale in the plots is 100
    scale = 100

    # Initialize ternary plot
    fig, ax = plt.subplots(figsize=(6.5, 5.1))
    fig, tax = ternary.figure(ax=ax, scale=scale)

    # If only one set of counts is give, we are only plotting a single ternary
    # point
    if len(counts_list) == 1:

        # Call plotter with single point
        plotter(counts=counts_list[0], label=labels[0], tax=tax, color=colors[0])

        # Decorate plot
        decorator(ax, tax, counts=counts_list[0])
    else:

        # Gather points from tuples of counts
        points = [
            counts_to_point(counts=count, scale=100, is_nodes=is_nodes)
            for count in counts_list
        ]

        if one_label:

            counts_sum = sum([sum(count.values()) for count in counts_list])
            label_counts = {"n": counts_sum}
        else:
            label_counts = {
                label: sum(count.values()) for label, count in zip(labels, counts_list)
            }
        decorate_count_ax(ax=ax, tax=tax, label_counts=label_counts, is_nodes=is_nodes)

        # Create boolean list of if point is not None
        points_is_not_none = [point is not None for point in points]

        # Remove None-points
        points = list(compress(points, points_is_not_none))

        if one_label:
            # Only one label for all points is given
            tax.scatter(
                points,
                label=labels[0],
                color=colors[0],
                **ternary_point_kwargs(),
            )
        else:
            # Multiple labels are given
            labels = list(compress(labels, points_is_not_none))
            colors = list(compress(colors, points_is_not_none))
            assert len(points) == len(labels)
            assert len(points) == len(colors)
            for point, label, color in zip(points, labels, colors):
                # Plot individually while giving unique label for each
                tax.scatter([point], label=label, color=color, **ternary_point_kwargs())
    tax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        prop={"family": "DejaVu Sans", "weight": "heavy", "size": "x-large"},
        edgecolor="black",
        ncol=2,
        columnspacing=0.7,
        shadow=True,
    )

    # Return Figure, Axes and TernaryAxesSubplot
    return fig, ax, tax


def ternary_heatmapping(
    x_values: np.ndarray,
    y_values: np.ndarray,
    i_values: np.ndarray,
    number_of_bins: int,
    scale_divider: float = 1.0,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, ternary.TernaryAxesSubplot]:
    """
    Plot ternary heatmap.

    Modified from: https://github.com/marcharper/python-ternary/issues/81
    """
    scale = number_of_bins / scale_divider
    histogram, _ = np.histogramdd(
        (x_values, y_values, i_values),
        bins=(number_of_bins, number_of_bins, number_of_bins),
        range=((0, 1), (0, 1), (0, 1)),
    )
    histogram_normed = histogram / np.sum(histogram)

    # 3D smoothing and interpolation
    kde = gaussian_filter(histogram_normed, sigma=2)
    interp_dict = dict()

    binx = np.linspace(0, 1, number_of_bins)
    for i in range(len(binx)):
        for j in range(len(binx)):
            for k in range(len(binx)):
                interp_dict[(i, j, k)] = kde[i, j, k]

    fig, tax = ternary.figure(ax=ax, scale=scale)
    tax.heatmap(interp_dict)
    return fig, tax


def counts_to_point(
    counts: Dict[str, int], is_nodes: bool, scale: int = 100
) -> Optional[Tuple[float, float, float]]:
    """
    Create ternary point from node_counts.

    The order is important:

      -  For nodes: X, I, Y.
      -  For branches: CC, II, CI.
    """
    if is_nodes:
        # xcount, ycount, icount = _get_xyi_counts(node_counts)
        point_counts = _get_xyi_counts(counts)
    else:
        # cc_count, ci_count, ii_count = _get_branch_class_counts(branch_counts)
        point_counts = _get_branch_class_counts(counts)

    sumcount = sum(point_counts)
    if sumcount == 0 or np.isclose(sumcount, 0.0):
        return None
    # NOTE: Order is switched to get right order in output.
    first = scale * point_counts[0] / sumcount
    second = scale * point_counts[2] / sumcount
    third = scale * point_counts[1] / sumcount
    return (first, second, third)


def _get_xyi_counts(node_counts: Dict[str, int]) -> Tuple[int, int, int]:
    """
    Return tuple of node counts from dict of node counts.
    """
    xcount = node_counts[X_node]
    ycount = node_counts[Y_node]
    icount = node_counts[I_node]
    return xcount, ycount, icount


def _get_branch_class_counts(branch_counts: Dict[str, int]) -> Tuple[int, int, int]:
    """
    Return tuple of branch counts from dict of branch counts.
    """
    cc_count = branch_counts[CC_branch]
    ci_count = branch_counts[CI_branch]
    ii_count = branch_counts[II_branch]
    return cc_count, ci_count, ii_count


def plot_xyi_plot_ax(
    counts: Dict[str, int],
    label: str,
    tax: ternary.ternary_axes_subplot.TernaryAxesSubplot,
    color: Optional[str] = None,
):
    """
    Plot XYI pointst to given ternary axis (tax).
    """
    if color is None:
        color = "black"
    # xcount, ycount, icount = _get_xyi_counts(node_counts)
    point = counts_to_point(counts, scale=100, is_nodes=True)
    if point is not None:
        # plot_ternary_point(tax=tax, point=point, marker="o", label=label, color=color)
        tax.scatter(
            points=[point], **ternary_point_kwargs(marker="o"), label=label, color=color
        )
    tax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        prop={"family": "DejaVu Sans", "weight": "heavy", "size": "x-large"},
        edgecolor="black",
        ncol=2,
        columnspacing=0.7,
        shadow=True,
    )


def ternary_point_kwargs(
    alpha=1.0,
    zorder=4,
    s: float = 25,
    marker="X",
):
    """
    Plot point to a ternary figure.
    """
    return dict(
        alpha=alpha,
        zorder=zorder,
        s=s,
        marker=marker,
    )


def plot_branch_plot_ax(
    counts: Dict[str, int],
    label: str,
    tax: ternary.ternary_axes_subplot.TernaryAxesSubplot,
    color: Optional[str] = None,
):
    """
    Plot ternary branch plot to tax.
    """
    if color is None:
        color = "black"
    # cc_count, ci_count, ii_count = _get_branch_class_counts(branch_counts)
    point = counts_to_point(counts, is_nodes=False)
    if point is not None:
        tax.scatter(
            points=[point],
            **ternary_point_kwargs(marker="o"),
            label=label,
            color=color,
        )

        # plot_ternary_point(tax=tax, point=point, marker="o", label=label, color=color)
    tax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        prop={"family": "DejaVu Sans", "weight": "heavy", "size": "x-large"},
        edgecolor="black",
        ncol=2,
        columnspacing=0.7,
        shadow=True,
    )


def decorate_branch_ax(
    ax: Axes,
    tax: TernaryAxesSubplot,
    # label: str,
    counts: Dict[str, int],
):
    """
    Decorate ternary branch plot.
    """
    cc_count, ci_count, ii_count = _get_branch_class_counts(counts)
    text = "\n".join(
        (
            f"n: {cc_count+ci_count+ii_count}",
            f"CC-branches: {cc_count}",
            f"CI-branches: {ci_count}",
            f"II-branches: {ii_count}",
        )
    )
    # prop = dict(boxstyle="square", facecolor="linen", alpha=1, pad=0.45)
    # ax.text(
    #     0.86,
    #     1.05,
    #     text,
    #     transform=ax.transAxes,
    #     fontsize="medium",
    #     weight="roman",
    #     verticalalignment="top",
    #     bbox=prop,
    #     fontfamily="DejaVu Sans",
    #     ha="center",
    # )
    initialize_ternary_branches_points(ax, tax)
    tern_plot_branch_lines(tax)
    ternary_text(ax=ax, text=text)
    # tax.legend(
    #     loc="upper center",
    #     bbox_to_anchor=(0.1, 1.05),
    #     prop={"family": "DejaVu Sans", "weight": "heavy", "size": "x-large"},
    #     edgecolor="black",
    #     ncol=2,
    #     columnspacing=0.7,
    #     shadow=True,
    # )


def determine_topology_parameters(
    trace_length_array: np.ndarray,
    node_counts: Dict[str, int],
    area: float,
    branches_defined: bool = True,
    correct_mauldon: bool = True,
    branch_length_array: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Determine geometric (and topological) parameters.

    Number of traces (and branches) are determined by node counting.

    The passed ``trace_length_array`` should be non-weighted.
    """
    radius = np.sqrt(area / np.pi)
    if len(trace_length_array) == 0:
        min_length_traces, max_length_traces, characteristic_length_traces = (
            0.0,
            0.0,
            0.0,
        )
    else:
        min_length_traces, max_length_traces, characteristic_length_traces = (
            float(trace_length_array.min()),
            float(trace_length_array.max()),
            float(trace_length_array.mean()),
        )
    assert isinstance(min_length_traces, float), type(min_length_traces)
    assert isinstance(max_length_traces, float)
    assert isinstance(characteristic_length_traces, float)
    # characteristic_length_traces = (
    #     trace_length_array.mean() if len(trace_length_array) > 0 else 0.0
    # )
    fracture_intensity = trace_length_array.sum() / area
    dimensionless_intensity_traces = fracture_intensity * characteristic_length_traces

    # Collect parameters that do not require topology determination into a dict
    params_without_topology = {
        Param.FRACTURE_INTENSITY_B21.value.name: fracture_intensity,
        Param.FRACTURE_INTENSITY_P21.value.name: fracture_intensity,
        Param.TRACE_MIN_LENGTH.value.name: min_length_traces,
        Param.TRACE_MAX_LENGTH.value.name: max_length_traces,
        Param.TRACE_MEAN_LENGTH.value.name: characteristic_length_traces,
        Param.DIMENSIONLESS_INTENSITY_P22.value.name: dimensionless_intensity_traces,
        Param.AREA.value.name: area,
    }

    if not branches_defined:
        # Return dict with nans in place of topological parameters
        nan_dict = {
            param.value.name: np.nan
            for param in Param
            if param.value not in params_without_topology
        }
        all_params_without_topo = {**params_without_topology, **nan_dict}
        assert all(param.value.name in all_params_without_topo for param in Param)

        return all_params_without_topo

    assert isinstance(trace_length_array, np.ndarray)
    assert isinstance(node_counts, dict)
    assert isinstance(area, float)

    if branch_length_array is None:
        raise ValueError("Expected branch_length_array to not be None.")

    number_of_traces = (node_counts[Y_node] + node_counts[I_node]) / 2

    # Handle empty branch length array
    if len(branch_length_array) == 0:
        min_length_branches, max_length_branches = 0.0, 0.0
    else:
        min_length_branches, max_length_branches = (
            float(branch_length_array.min()),
            float(branch_length_array.max()),
        )
    aerial_frequency_traces = number_of_traces / area
    number_of_branches = (
        (node_counts[X_node] * 4) + (node_counts[Y_node] * 3) + node_counts[I_node]
    ) / 2
    aerial_frequency_branches = number_of_branches / area
    characteristic_length_branches = (
        (trace_length_array.sum() / number_of_branches)
        if number_of_branches > 0
        else 0.0
    )
    dimensionless_intensity_branches = (
        fracture_intensity * characteristic_length_branches
    )
    connections_per_trace = (
        (2 * (node_counts[Y_node] + node_counts[X_node]) / number_of_traces)
        if number_of_traces > 0
        else 0.0
    )
    connections_per_branch = (
        (3 * node_counts[Y_node] + 4 * node_counts[X_node]) / number_of_branches
        if number_of_branches > 0
        else 0.0
    )
    if correct_mauldon:
        trace_mean_length_mauldon = (
            (
                ((np.pi * radius) / 2)
                * (node_counts[E_node] / (node_counts[I_node] + node_counts[Y_node]))
            )
            if node_counts[I_node] + node_counts[Y_node] > 0
            else 0.0
        )
        fracture_density_mauldon = (node_counts[I_node] + node_counts[Y_node]) / (
            area * 2
        )
        fracture_intensity_mauldon = (
            trace_mean_length_mauldon * fracture_density_mauldon
        )
    else:
        # If Network target area is not circular mauldon parameters cannot be
        # determined.
        (
            trace_mean_length_mauldon,
            fracture_density_mauldon,
            fracture_intensity_mauldon,
        ) = (np.nan, np.nan, np.nan)
    connection_frequency = (node_counts[Y_node] + node_counts[X_node]) / area

    params_with_topology = {
        Param.NUMBER_OF_TRACES.value.name: number_of_traces,
        Param.BRANCH_MIN_LENGTH.value.name: min_length_branches,
        Param.BRANCH_MAX_LENGTH.value.name: max_length_branches,
        Param.BRANCH_MEAN_LENGTH.value.name: characteristic_length_branches,
        Param.AREAL_FREQUENCY_B20.value.name: aerial_frequency_branches,
        Param.AREAL_FREQUENCY_P20.value.name: aerial_frequency_traces,
        Param.DIMENSIONLESS_INTENSITY_B22.value.name: dimensionless_intensity_branches,
        Param.CONNECTIONS_PER_TRACE.value.name: connections_per_trace,
        Param.CONNECTIONS_PER_BRANCH.value.name: connections_per_branch,
        Param.FRACTURE_INTENSITY_MAULDON.value.name: fracture_intensity_mauldon,
        Param.FRACTURE_DENSITY_MAULDON.value.name: fracture_density_mauldon,
        Param.TRACE_MEAN_LENGTH_MAULDON.value.name: trace_mean_length_mauldon,
        Param.CONNECTION_FREQUENCY.value.name: connection_frequency,
        Param.NUMBER_OF_BRANCHES.value.name: number_of_branches,
    }

    all_parameters = {**params_without_topology, **params_with_topology}
    assert len(all_parameters) == sum(
        [len(params_without_topology), len(params_with_topology)]
    )
    assert all(
        param in {param.value.name for param in Param} for param in all_parameters
    )

    return all_parameters


def plot_parameters_plot(
    topology_parameters_list: List[Dict[str, float]],
    labels: List[str],
    colors: Optional[List[str]] = None,
):
    """
    Plot topological parameters.
    """
    prop = dict(boxstyle="square", facecolor="linen", alpha=1, pad=0.45)

    columns_to_plot = [
        param for param in Param if param.value.name in topology_parameters_list[0]
    ]
    figs, axes = [], []

    for column in columns_to_plot:
        # Figure size setup
        # TODO: width higher, MAYBE lower bar_width

        width = 6 + 1 * len(topology_parameters_list) / 6
        bar_width = 0.6 * len(topology_parameters_list) / 6

        fig, ax = plt.subplots(figsize=(width, 5.5))
        topology_concat = pd.DataFrame(topology_parameters_list)
        topology_concat["label"] = labels
        topology_concat.label = topology_concat.label.astype("category")

        # Trying to have sensible widths for bars:
        topology_concat.plot.bar(
            x="label",
            y=column.value.name,
            color=colors,
            zorder=5,
            alpha=0.9,
            width=bar_width,
            ax=ax,
        )
        # PLOT STYLING
        ax.set_xlabel("")
        ax.set_ylabel(
            column.value.name + " " + f"({column.value.unit})",
            fontsize="xx-large",
            fontfamily="DejaVu Sans",
            style="italic",
        )
        ax.set_title(
            x=0.5,
            y=1.09,
            label=column.value.name,
            fontsize="xx-large",
            bbox=prop,
            transform=ax.transAxes,
            zorder=-10,
        )
        legend = ax.legend()
        legend.remove()
        if column.value.plot_as_log:
            ax.set_yscale("log")
        fig.subplots_adjust(top=0.85, bottom=0.25, left=0.2)
        locs, xtick_labels = plt.xticks()
        # xtick_labels = ["\n".join(wrap(label.get_text(), 6)) for label in xtick_labels]
        xtick_labels = [fill(label.get_text(), 6) for label in xtick_labels]
        plt.yticks(fontsize="xx-large", color="black")
        plt.xticks(locs, labels, fontsize="xx-large", color="black")
        # VALUES ABOVE BARS WITH TEXTS
        rects = ax.patches
        for value, rect in zip(topology_concat[column.value.name].values, rects):
            height = rect.get_height()
            if value > 0.01:
                value = round(value, 2)
            else:
                value = f"{value:.2e}"
            if column.value.plot_as_log:
                height = height + height / 10
            else:
                max_height = max([r.get_height() for r in rects])
                height = height + max_height / 100
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height,
                value,
                ha="center",
                va="bottom",
                zorder=15,
                fontsize="x-large",
            )
        figs.append(fig)
        axes.append(ax)
    return figs, axes


def determine_set_counts(
    set_names: Tuple[str, ...], set_array: np.ndarray
) -> Dict[str, int]:
    """
    Determine counts in for each set.
    """
    return {
        set_name: sum(set_array == set_name)
        if sum(set_array == set_name) is not None
        else 0
        for set_name in set_names
    }


def plot_set_count(
    set_counts: Dict[str, int],
    label: str,
) -> Tuple[Figure, Axes]:
    """
    Plot set counts.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    logging_info = dict(set_counts=set_counts, label=label)

    # Handle case with 0 values for all keys in set_counts dict.
    if all(value == 0 for value in set_counts.values()):
        logging.warning(
            "set_counts with only 0 values passed into plot_set_count.",
            extra=logging_info,
        )
        return fig, ax

    # Report creation.
    logging.info(
        "Creating pie plot for set counts.",
        extra=logging_info,
    )

    # Create the plot
    _, label_texts, _ = ax.pie(
        x=[set_counts[key] for key in set_counts],
        labels=list(set_counts),
        autopct="%.1f%%",
        explode=[0.025 for _ in set_counts],
        pctdistance=0.5,
        textprops=dict(weight="bold"),
        wedgeprops=dict(linewidth=2, edgecolor="black"),
    )

    # Set title
    ax.set_title(label)
    for label_text in label_texts:
        label_text.set_fontsize("large")
    return fig, ax


def initialize_ternary_points(ax: Axes, tax: TernaryAxesSubplot):
    """
    Initialize ternary points figure ax and tax.
    """
    fdict = initialize_ternary_ax(ax=ax, tax=tax)
    ax.text(-0.1, -0.03, "Y", transform=ax.transAxes, fontdict=fdict)
    ax.text(1.03, -0.03, "X", transform=ax.transAxes, fontdict=fdict)
    ax.text(0.5, 1.07, "I", transform=ax.transAxes, fontdict=fdict, ha="center")


def tern_plot_the_fing_lines(tax: TernaryAxesSubplot, cs_locs=(1.3, 1.5, 1.7, 1.9)):
    """
    Plot *connections per branch* parameter to XYI-plot.

    If not using the pre-determined locations the texts will not be correctly
    placed as they use absolute positions and labels.

    :param tax: Ternary axis to plot to.
    :param cs_locs: Pre-determined locations for the lines.
    """

    def tern_find_last_x(c, x_start=0):
        x, _, y = tern_yi_func(c, x_start)
        while y > 0:
            x_start += 0.01
            x, _, y = tern_yi_func(c, x_start)
        return x

    def tern_yi_func_perc(c, x):
        temp = 6 * (1 - 0.5 * c)
        temp2 = 3 - (3 / 2) * c
        temp3 = 1 + c / temp
        y = (c + 3 * c * x) / (temp * temp3) - (4 * x) / (temp2 * temp3)
        i = 1 - x - y
        return 100 * x, 100 * i, 100 * y

    for c in cs_locs:
        last_x = tern_find_last_x(c)
        x1 = 0
        x2 = last_x
        point1 = tern_yi_func_perc(c, x1)
        point2 = tern_yi_func_perc(c, x2)
        tax.line(
            point1,
            point2,
            alpha=0.6,
            color="k",
            zorder=-5,
            linestyle="dashed",
            linewidth=0.75,
        )
        ax = plt.gca()
        rot = 6.5
        rot2 = 4.5
        ax.text(x=55, y=59, s=r"$C_B = 1.3$", fontsize=10, rotation=rot, ha="center")
        ax.text(x=61, y=50, s=r"$C_B = 1.5$", fontsize=10, rotation=rot, ha="center")
        ax.text(
            x=68.5,
            y=36.6,
            s=r"$C_B = 1.7$",
            fontsize=10,
            rotation=rot2 + 1,
            ha="center",
        )
        ax.text(x=76, y=17, s=r"$C_B = 1.9$", fontsize=10, rotation=rot2, ha="center")


def initialize_ternary_ax(ax: Axes, tax: TernaryAxesSubplot) -> dict:
    """
    Decorate ternary ax for both XYI and branch types.

    Returns a font dict for use in plotting text.
    """
    # Remove normal matplotlib axes
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Remove frame
    ax.set_frame_on(False)

    # Set ternary boundary
    tax.boundary(linewidth=1.5)

    # Add gridlines in the tax
    tax.gridlines(linewidth=0.15, multiple=20, color="black", alpha=0.6, zorder=-100)

    # Add ticks
    tax.ticks(
        axis="lbr",
        linewidth=0.9,
        multiple=20,
        offset=0.03,
        tick_formats="%d%%",
        fontsize="small",
        alpha=0.6,
    )

    # Clear the default matplotlib ticks
    tax.clear_matplotlib_ticks()

    # Another check to remove matplotlib axes
    tax.get_axes().axis("off")

    # Create font dict for use in plotting text
    fontsize = 20
    fdict = {
        "path_effects": [path_effects.withStroke(linewidth=3, foreground="k")],
        "color": "w",
        "family": "DejaVu Sans",
        "size": fontsize,
        "weight": "bold",
    }
    return fdict


def initialize_ternary_branches_points(ax, tax):
    """
    Initialize ternary branches plot ax and tax.
    """
    fdict = initialize_ternary_ax(ax=ax, tax=tax)
    ax.text(-0.1, -0.06, "I - C", transform=ax.transAxes, fontdict=fdict)
    ax.text(1.0, -0.06, "C - C", transform=ax.transAxes, fontdict=fdict)
    ax.text(0.5, 1.07, "I - I", transform=ax.transAxes, fontdict=fdict, ha="center")


def tern_plot_branch_lines(tax):
    """
    Plot line of random assignment of nodes to branches to a branch ternary tax.

    Line positions taken from NetworkGT open source code.
    Credit to:
    https://github.com/BjornNyberg/NetworkGT

    :param tax: Ternary axis to plot to
    :type tax: ternary.TernaryAxesSubplot
    """
    ax = tax.get_axes()
    tax.boundary()
    points = [
        (0, 1, 0),
        (0.01, 0.81, 0.18),
        (0.04, 0.64, 0.32),
        (0.09, 0.49, 0.42),
        (0.16, 0.36, 0.48),
        (0.25, 0.25, 0.5),
        (0.36, 0.16, 0.48),
        (0.49, 0.09, 0.42),
        (0.64, 0.04, 0.32),
        (0.81, 0.01, 0.18),
        (1, 0, 0),
    ]
    for idx, _ in enumerate(points):
        points[idx] = points[idx][0] * 100, points[idx][1] * 100, points[idx][2] * 100

    text_loc = [(0.37, 0.2), (0.44, 0.15), (0.52, 0.088), (0.64, 0.055), (0.79, 0.027)]
    for idx, t in enumerate(text_loc):
        text_loc[idx] = t[0] * 100, t[1] * 100
    text = [r"$C_B = 1.0$", r"$1.2$", r"$1.4$", r"$1.6$", r"$1.8$"]
    rots = [-61, -44, -28, -14, -3]
    # rot = -65
    for t, l, rot in zip(text, text_loc, rots):
        ax.annotate(t, xy=l, fontsize=9, rotation=rot)
        # rot += 17
    tax.plot(
        points,
        linewidth=1.5,
        marker="o",
        color="k",
        linestyle="dashed",
        markersize=3,
        zorder=-5,
        alpha=0.6,
    )


def tern_yi_func(c, x):
    """
    Plot *Connections per branch* threshold line to branch ternary plot.

    Uses absolute values.
    """
    temp = 6 * (1 - 0.5 * c)
    temp2 = 3 - (3 / 2) * c
    temp3 = 1 + c / temp
    y = (c + 3 * c * x) / (temp * temp3) - (4 * x) / (temp2 * temp3)
    i = 1 - x - y
    return x, i, y


def convert_counts(counts: Dict[str, Number]) -> Dict[str, int]:
    """
    Convert float and int value in counts to ints only.

    Used by branches and nodes count calculators.
    """
    if any(np.isnan(list(counts.values()))):
        raise ValueError(f"Expected no nan in counts: {counts}")
    counts_ints = {
        key: int(value) for key, value in counts.items() if not np.isnan(value)
    }
    if len(counts_ints) != len(counts):
        raise ValueError(f"Expected no nan in node_counts: {counts}")

    assert all(isinstance(value, int) for value in counts_ints.values())
    return counts_ints
