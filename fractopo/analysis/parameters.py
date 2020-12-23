"""
Analysis and plotting of geometric and topological parameters.
"""
from typing import Dict, Tuple, Optional, List, Union
from functools import lru_cache

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import ternary
from fractopo.general import (
    X_node,
    Y_node,
    I_node,
    E_node,
    CC_branch,
    CI_branch,
    II_branch,
    Param,
    styled_prop,
)
from fractopo.analysis import tools
from textwrap import wrap


def determine_node_classes(node_types: np.ndarray) -> Dict[str, int]:
    return {
        str(node_class): amount
        if (amount := sum(node_types == node_class)) is not None
        else 0
        for node_class in (X_node, Y_node, I_node, E_node)
    }


def determine_branch_classes(branch_types: np.ndarray) -> Dict[str, int]:
    return {
        str(branch_class): amount
        if (amount := sum(branch_types == branch_class)) is not None
        else 0
        for branch_class in (CC_branch, CI_branch, II_branch)
    }


def decorate_xyi_ax(
    ax: matplotlib.axes.Axes, tax: ternary.ternary_axes_subplot.TernaryAxesSubplot, label: str, node_counts: Dict[str, int]  # type: ignore
):
    xcount, ycount, icount = _get_xyi_counts(node_counts)
    text = f"n: {xcount+ycount+icount}\n"
    f"X-nodes: {xcount}\n"
    f"Y-nodes: {ycount}\n"
    f"I-nodes: {icount}\n"
    tools.initialize_ternary_points(ax, tax)
    tools.tern_plot_the_fing_lines(tax)
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
        fontfamily="Calibri",
        ha="center",
    )


def plot_xyi_plot(
    node_counts_list: List[Dict[str, int]],
    labels: List[str],
    colors: Optional[List[Optional[str]]] = None,
) -> Tuple[
    matplotlib.figure.Figure,  # type: ignore
    matplotlib.axes.Axes,  # type: ignore
    ternary.ternary_axes_subplot.TernaryAxesSubplot,
]:
    """
    Plot ternary XYI-plot.

    By default accepts a list of node_types -arrays but a list with
    a single node_types -array is accepted i.e. a single XYI-value is easily
    plotted.
    """
    if colors is None:
        colors = [None for _ in node_counts_list]
    scale = 100
    fig, ax = plt.subplots(figsize=(6.5, 5.1))
    fig, tax = ternary.figure(ax=ax, scale=scale)
    plot_xyi_plot_ax(
        node_counts=node_counts_list[0], label=labels[0], tax=tax, color=colors[0]
    )
    decorate_xyi_ax(ax, tax, label=labels[0], node_counts=node_counts_list[0])
    if len(node_counts_list) > 1:
        for node_counts, label, color in zip(
            node_counts_list[1:], labels[1:], colors[1:]
        ):
            point = node_counts_to_point(node_counts)
            if point is None:
                continue
            plot_ternary_point(point=point, marker="X", label=label, tax=tax)

    return fig, ax, tax


def node_counts_to_point(node_counts: Dict[str, int]):
    xcount, ycount, icount = _get_xyi_counts(node_counts)
    sumcount = xcount + ycount + icount
    if sumcount == 0:
        return None
    else:
        xp = 100 * xcount / sumcount
        yp = 100 * ycount / sumcount
        ip = 100 * icount / sumcount
        point = [(xp, ip, yp)]
        return point


def branch_counts_to_point(branch_counts: Dict[str, int]):
    cc_count, ci_count, ii_count = _get_branch_class_counts(branch_counts)
    sumcount = cc_count + ci_count + ii_count
    if sumcount == 0:
        return None
    ccp = 100 * cc_count / sumcount
    cip = 100 * ci_count / sumcount
    iip = 100 * ii_count / sumcount

    point = [(ccp, iip, cip)]
    return point


def _get_xyi_counts(node_counts: Dict[str, int]) -> Tuple[int, int, int]:
    xcount = node_counts[X_node]
    ycount = node_counts[Y_node]
    icount = node_counts[I_node]
    return xcount, ycount, icount


def _get_branch_class_counts(branch_counts: Dict[str, int]) -> Tuple[int, int, int]:
    cc_count = branch_counts[CC_branch]
    ci_count = branch_counts[CI_branch]
    ii_count = branch_counts[II_branch]
    return cc_count, ci_count, ii_count


def plot_xyi_plot_ax(
    node_counts: Dict[str, int],
    label: str,
    tax: ternary.ternary_axes_subplot.TernaryAxesSubplot,
    color: Optional[str] = None,
):
    if color is None:
        color = "black"
    xcount, ycount, icount = _get_xyi_counts(node_counts)
    point = node_counts_to_point(node_counts)
    if point is not None:
        plot_ternary_point(tax=tax, point=point, marker="o", label=label, color=color)
    tax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        prop={"family": "Calibri", "weight": "heavy", "size": "x-large"},
        edgecolor="black",
        ncol=2,
        columnspacing=0.7,
        shadow=True,
    )


def plot_branch_plot(
    branch_counts_list: List[Dict[str, int]],
    labels: List[str],
    colors: Optional[List[Optional[str]]] = None,
) -> Tuple[
    matplotlib.figure.Figure,  # type: ignore
    matplotlib.axes.Axes,  # type: ignore
    ternary.ternary_axes_subplot.TernaryAxesSubplot,
]:
    """
    Plot a branch classification ternary plot to a new ternary figure. Single point in each figure.
    """
    if colors is None:
        colors = [None for _ in branch_counts_list]
    scale = 100
    fig, ax = plt.subplots(figsize=(6.5, 5.1))
    fig, tax = ternary.figure(ax=ax, scale=scale)
    plot_branch_plot_ax(
        branch_counts=branch_counts_list[0], label=labels[0], tax=tax, color=colors[0]
    )
    decorate_branch_ax(
        ax=ax, tax=tax, label=labels[0], branch_counts=branch_counts_list[0]
    )
    if len(branch_counts_list) > 1:
        for branch_counts, label, color in zip(
            branch_counts_list[1:], labels[1:], colors[1:]
        ):
            point = branch_counts_to_point(branch_counts)
            if point is None:
                continue
            plot_ternary_point(point=point, marker="o", label=label, tax=tax)

    return fig, ax, tax


def plot_ternary_point(
    point: List[Tuple[float, float, float]],
    marker: str,
    label: str,
    tax: ternary.ternary_axes_subplot.TernaryAxesSubplot,
    color: Optional[str] = "black",
    s: float = 25,
):
    tax.scatter(
        point,
        marker=marker,
        label=label,
        alpha=1,
        zorder=4,
        s=s,
        color=color,
    )


def plot_branch_plot_ax(
    branch_counts: Dict[str, int],
    label: str,
    tax: ternary.ternary_axes_subplot.TernaryAxesSubplot,
    color: Optional[str] = None,
):
    if color is None:
        color = "black"
    cc_count, ci_count, ii_count = _get_branch_class_counts(branch_counts)
    point = branch_counts_to_point(branch_counts)
    if point is not None:
        plot_ternary_point(tax=tax, point=point, marker="o", label=label, color=color)
    tax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        prop={"family": "Calibri", "weight": "heavy", "size": "x-large"},
        edgecolor="black",
        ncol=2,
        columnspacing=0.7,
        shadow=True,
    )


def decorate_branch_ax(
    ax: matplotlib.axes.Axes,  # type: ignore
    tax: ternary.ternary_axes_subplot.TernaryAxesSubplot,
    label: str,
    branch_counts: Dict[str, int],
):
    cc_count, ci_count, ii_count = _get_branch_class_counts(branch_counts)
    text = f"n: {cc_count+ci_count+ii_count}\n"
    f"CC-branches: {cc_count}\n"
    f"CI-branches: {ci_count}\n"
    f"II-branches: {ii_count}\n"
    prop = dict(boxstyle="square", facecolor="linen", alpha=1, pad=0.45)
    ax.text(
        0.86,
        1.05,
        text,
        transform=ax.transAxes,
        fontsize="medium",
        weight="roman",
        verticalalignment="top",
        bbox=prop,
        fontfamily="Calibri",
        ha="center",
    )
    tools.initialize_ternary_branches_points(ax, tax)
    tools.tern_plot_branch_lines(tax)
    tax.legend(
        loc="upper center",
        bbox_to_anchor=(0.1, 1.05),
        prop={"family": "Calibri", "weight": "heavy", "size": "x-large"},
        edgecolor="black",
        ncol=2,
        columnspacing=0.7,
        shadow=True,
    )


def determine_topology_parameters(
    trace_length_array: np.ndarray,
    branch_length_array: np.ndarray,
    node_counts: Dict[str, int],
    area: float,
):
    """
    Determine topology parameters.

    Number of traces and branches are determined by node counting.
    """
    assert isinstance(trace_length_array, np.ndarray)
    assert isinstance(branch_length_array, np.ndarray)
    assert isinstance(node_counts, dict)
    assert isinstance(area, float)
    number_of_traces = (node_counts[Y_node] + node_counts[I_node]) / 2
    number_of_branches = (
        (node_counts[X_node] * 4) + (node_counts[Y_node] * 3) + node_counts[I_node]
    ) / 2
    fracture_intensity = branch_length_array.sum() / area
    aerial_frequency_traces = number_of_traces / area
    aerial_frequency_branches = number_of_branches / area
    characteristic_length_traces = trace_length_array.mean()
    characteristic_length_branches = branch_length_array.mean()
    dimensionless_intensity_traces = fracture_intensity * characteristic_length_traces
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
    topology_parameters = {
        Param.NUMBER_OF_TRACES.value: number_of_traces,
        Param.NUMBER_OF_BRANCHES.value: number_of_branches,
        Param.FRACTURE_INTENSITY_B21.value: fracture_intensity,
        Param.FRACTURE_INTENSITY_P21.value: fracture_intensity,
        Param.AREAL_FREQUENCY_P20.value: aerial_frequency_traces,
        Param.AREAL_FREQUENCY_B20.value: aerial_frequency_branches,
        Param.TRACE_MEAN_LENGTH.value: characteristic_length_traces,
        Param.BRANCH_MEAN_LENGTH.value: characteristic_length_branches,
        Param.DIMENSIONLESS_INTENSITY_P22.value: dimensionless_intensity_traces,
        Param.DIMENSIONLESS_INTENSITY_B22.value: dimensionless_intensity_branches,
        Param.CONNECTIONS_PER_TRACE.value: connections_per_trace,
        Param.CONNECTIONS_PER_BRANCH.value: connections_per_branch,
    }
    return topology_parameters


def plot_parameters_plot(
    topology_parameters_list: List[Dict[str, float]],
    labels: List[str],
    colors=Optional[List[str]],
):
    """
    Plot topological parameters.

    """
    log_scale_columns = Param.log_scale_columns()
    prop = dict(boxstyle="square", facecolor="linen", alpha=1, pad=0.45)

    columns_to_plot = [param.value for param in Param]
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
            y=column,
            color=colors,
            zorder=5,
            alpha=0.9,
            width=bar_width,
            ax=ax,
        )
        # PLOT STYLING
        ax.set_xlabel("")
        ax.set_ylabel(
            column + " " + f"({Param.get_unit_for_column(column)})",
            fontsize="xx-large",
            fontfamily="Calibri",
            style="italic",
        )
        ax.set_title(
            x=0.5,
            y=1.09,
            label=column,
            fontsize="xx-large",
            bbox=prop,
            transform=ax.transAxes,
            zorder=-10,
        )
        legend = ax.legend()
        legend.remove()
        if column in log_scale_columns:
            ax.set_yscale("log")
        fig.subplots_adjust(top=0.85, bottom=0.25, left=0.2)
        locs, labels = plt.xticks()
        labels = ["\n".join(wrap(l.get_text(), 6)) for l in labels]
        plt.yticks(fontsize="xx-large", color="black")
        plt.xticks(locs, labels, fontsize="xx-large", color="black")
        # VALUES ABOVE BARS WITH TEXTS
        rects = ax.patches
        for value, rect in zip(topology_concat[column].values, rects):
            height = rect.get_height()
            if value > 0.01:
                value = round(value, 2)
            else:
                value = "{0:.2e}".format(value)
            if column in log_scale_columns:
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
    set_names: np.ndarray, set_array: np.ndarray
) -> Dict[str, int]:
    return {
        set_name: amount if (amount := sum(set_array == set_name)) is not None else 0
        for set_name in set_names
    }


def plot_set_count(
    set_counts: Dict[str, int],
    label: str,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, label_texts, count_texts = ax.pie(
        x=[set_counts[key] for key in set_counts],
        labels=[key for key in set_counts],
        autopct="%.1f%%",
        explode=[0.025 for _ in set_counts],
        pctdistance=0.5,
        textprops=dict(weight="bold"),
        wedgeprops=dict(linewidth=2, edgecolor="black"),
    )
    ax.set_title(label)
    for label_text in label_texts:
        label_text.set_fontsize("large")
    return fig, ax
