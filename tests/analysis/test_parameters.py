"""
Test parameters.py.
"""
import matplotlib
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from ternary.ternary_axes_subplot import TernaryAxesSubplot

import tests
from fractopo.analysis import parameters

# import os


# @pytest.mark.skipif(
#     os.environ.get("RUNNER_OS") == "Windows",
#     reason="""
# Odd bug with Windows and TKinter for only this single test.


# See:
#     """,
# )
@pytest.mark.parametrize("node_counts_list,labels", tests.test_plot_xyi_plot_params)
@tests.plotting_test
def test_plot_xyi_plot(node_counts_list, labels):
    """
    Test plotting xyi.
    """
    fig, ax, tax = parameters.plot_ternary_plot(
        counts_list=node_counts_list, labels=labels, is_nodes=True
    )
    assert isinstance(fig, matplotlib.figure.Figure)  # type: ignore
    assert isinstance(ax, matplotlib.axes.Axes)  # type: ignore
    assert isinstance(tax, TernaryAxesSubplot)
    plt.close("all")


@pytest.mark.parametrize(
    "branch_counts_list,labels,colors", tests.test_plot_branch_plot_params
)
@tests.plotting_test
def test_plot_branch_plot(branch_counts_list, labels, colors):
    """
    Test plotting branches.
    """
    fig, ax, tax = parameters.plot_ternary_plot(
        counts_list=branch_counts_list, labels=labels, is_nodes=False, colors=colors
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert isinstance(tax, TernaryAxesSubplot)
    plt.close("all")


@pytest.mark.parametrize(
    "trace_length_array,node_counts,area,"
    "branches_defined,correct_mauldon,branch_length_array",
    tests.test_determine_topology_parameters_params,
)
def test_determine_topology_parameters(
    trace_length_array,
    node_counts,
    area,
    branches_defined,
    correct_mauldon,
    branch_length_array,
):
    """
    Test determining parameters.
    """
    assert isinstance(area, float)
    topology_parameters = parameters.determine_topology_parameters(
        trace_length_array=trace_length_array,
        node_counts=node_counts,
        area=area,
        branches_defined=branches_defined,
        correct_mauldon=correct_mauldon,
        branch_length_array=branch_length_array,
    )
    assert all(param >= 0 for param in topology_parameters.values())
    assert all(
        isinstance(param, (float, int)) for param in topology_parameters.values()
    )


@pytest.mark.parametrize(
    "topology_parameters_list, labels, colors", tests.test_plot_topology_params
)
@tests.plotting_test
def test_plot_parameters_plot(topology_parameters_list, labels, colors):
    """
    Test plotting parameters.
    """
    figs, axes = parameters.plot_parameters_plot(
        topology_parameters_list, labels, colors
    )
    assert all(isinstance(fig, Figure) for fig in figs)
    assert all(isinstance(ax, Axes) for ax in axes)
    plt.close("all")


@pytest.mark.parametrize(
    "x_values,y_values,i_values,number_of_bins",
    tests.test_ternary_heatmapping_params(),
)
@tests.plotting_test
def test_ternary_heatmapping(x_values, y_values, i_values, number_of_bins):
    """
    Test ternary_heatmapping.
    """
    fig, tax = parameters.ternary_heatmapping(
        x_values, y_values, i_values, number_of_bins
    )
    assert isinstance(fig, Figure)
    assert isinstance(tax, TernaryAxesSubplot)
    plt.close("all")
