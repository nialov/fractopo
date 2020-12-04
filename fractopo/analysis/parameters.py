"""
Analysis and plotting of geometric and topological parameters.
"""
from typing import Dict, Tuple, Optional, List
from functools import singledispatch

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
)
from fractopo.analysis import tools


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
    ax: matplotlib.axes.Axes, tax: ternary.ternary_axes_subplot.TernaryAxesSubplot, label: str, node_dict: Dict[str, int]  # type: ignore
):
    xcount, ycount, icount = _get_xyi_counts(node_dict)
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


@singledispatch
def plot_xyi_plot(
    node_types: np.ndarray,
    label: str,
    color: Optional[str] = None,
    node_dict: Optional[Dict[str, int]] = None,
) -> Tuple[
    Dict[str, int],
    matplotlib.figure.Figure,  # type: ignore
    matplotlib.axes.Axes,  # type: ignore
    ternary.ternary_axes_subplot.TernaryAxesSubplot,
]:
    if node_dict is None:
        node_dict = determine_node_classes(node_types)
    # Scatter Plot
    scale = 100
    fig, ax = plt.subplots(figsize=(6.5, 5.1))
    fig, tax = ternary.figure(ax=ax, scale=scale)
    plot_xyi_plot_ax(node_dict=node_dict, label=label, tax=tax, color=color)
    decorate_xyi_ax(ax, tax, label=label, node_dict=node_dict)
    return node_dict, fig, ax, tax


@plot_xyi_plot.register
def _(
    node_types: List[np.ndarray],
    label: str,
    color: Optional[str] = None,
    node_dict: Optional[Dict[str, int]] = None,
) -> Tuple[
    List[Dict[str, int]],
    matplotlib.figure.Figure,  # type: ignore
    matplotlib.axes.Axes,  # type: ignore
    ternary.ternary_axes_subplot.TernaryAxesSubplot,
]:
    # Scatter Plot
    scale = 100
    fig, ax = plt.subplots(figsize=(6.5, 5.1))
    fig, tax = ternary.figure(ax=ax, scale=scale)
    for node_types_array in node_types:
        node_dict = determine_node_classes(node_types)
        decorate_xyi_ax(ax, tax, label=label, node_dict=node_dict)
        plot_xyi_plot_ax(node_dict=node_dict, label=label, tax=tax, color=color)
    return node_dict, fig, ax, tax


def _get_xyi_counts(node_dict: Dict[str, int]) -> Tuple[int, int, int]:
    xcount = node_dict[X_node]
    ycount = node_dict[Y_node]
    icount = node_dict[I_node]
    return xcount, ycount, icount


def _get_branch_class_counts(branch_dict: Dict[str, int]) -> Tuple[int, int, int]:
    cc_count = branch_dict[CC_branch]
    ci_count = branch_dict[CI_branch]
    ii_count = branch_dict[II_branch]
    return cc_count, ci_count, ii_count


def plot_xyi_plot_ax(
    node_dict: Dict[str, int],
    label: str,
    tax: ternary.ternary_axes_subplot.TernaryAxesSubplot,
    color: Optional[str] = None,
):
    if color is None:
        color = "black"
    xcount, ycount, icount = _get_xyi_counts(node_dict)

    sumcount = xcount + ycount + icount
    if sumcount == 0:
        return
    else:
        xp = 100 * xcount / sumcount
        yp = 100 * ycount / sumcount
        ip = 100 * icount / sumcount
        point = [(xp, ip, yp)]

    plot_ternary_point(tax=tax, point=point, marker="o", label=label)
    # tax.scatter(point, s=50, marker="o", label=label, alpha=1, zorder=4, color=color)
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
    branch_types: np.ndarray,
    label: str,
    color: Optional[str] = None,
    branch_dict: Optional[Dict[str, int]] = None,
):
    """
    Plot a branch classification ternary plot to a new ternary figure. Single point in each figure.
    """
    if branch_dict is None:
        branch_dict = determine_branch_classes(branch_types)
    fig, ax = plt.subplots(figsize=(6.5, 5.1))
    scale = 100
    fig, tax = ternary.figure(ax=ax, scale=scale)
    plot_branch_plot_ax(branch_dict=branch_dict, label=label, tax=tax, color=color)
    decorate_branch_ax(ax=ax, tax=tax, label=label, branch_dict=branch_dict)
    return branch_dict, fig, ax, tax


def plot_ternary_point(
    point: List[Tuple[float, float, float]],
    marker: str,
    label: str,
    tax: ternary.ternary_axes_subplot.TernaryAxesSubplot,
    color: Optional[str] = "black",
):
    tax.scatter(
        point,
        marker=marker,
        label=label,
        alpha=1,
        zorder=4,
        s=125,
        color=color,
    )


def plot_branch_plot_ax(
    branch_dict: Dict[str, int],
    label: str,
    tax: ternary.ternary_axes_subplot.TernaryAxesSubplot,
    color: Optional[str] = None,
):
    cc_count, ci_count, ii_count = _get_branch_class_counts(branch_dict)
    sumcount = cc_count + ci_count + ii_count
    if sumcount == 0:
        return
    ccp = 100 * cc_count / sumcount
    cip = 100 * ci_count / sumcount
    iip = 100 * ii_count / sumcount

    point = [(ccp, iip, cip)]
    plot_ternary_point(tax=tax, point=point, marker="X", label=label, color=color)


def decorate_branch_ax(
    ax: matplotlib.axes.Axes,  # type: ignore
    tax: ternary.ternary_axes_subplot.TernaryAxesSubplot,
    label: str,
    branch_dict: Dict[str, int],
):
    cc_count, ci_count, ii_count = _get_branch_class_counts(branch_dict)
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
