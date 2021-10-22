"""
Test parameters.py.
"""
import matplotlib
import pytest
import ternary
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import tests
from fractopo.analysis import parameters
from fractopo.general import Param
from tests import Helpers


@pytest.mark.parametrize("node_counts_list,labels", Helpers.test_plot_xyi_plot_params)
def test_plot_xyi_plot(node_counts_list, labels):
    """
    Test plotting xyi.
    """
    fig, ax, tax = parameters.plot_xyi_plot(
        node_counts_list=node_counts_list, labels=labels
    )
    assert isinstance(fig, matplotlib.figure.Figure)  # type: ignore
    assert isinstance(ax, matplotlib.axes.Axes)  # type: ignore
    assert isinstance(tax, ternary.ternary_axes_subplot.TernaryAxesSubplot)
    plt.close()


@pytest.mark.parametrize(
    "branch_counts_list,labels", Helpers.test_plot_branch_plot_params
)
def test_plot_branch_plot(branch_counts_list, labels):
    """
    Test plotting branches.
    """
    fig, ax, tax = parameters.plot_branch_plot(
        branch_counts_list=branch_counts_list, labels=labels
    )
    assert isinstance(fig, matplotlib.figure.Figure)  # type: ignore
    assert isinstance(ax, matplotlib.axes.Axes)  # type: ignore
    assert isinstance(tax, ternary.ternary_axes_subplot.TernaryAxesSubplot)
    plt.close()


@pytest.mark.parametrize(
    "trace_length_array, node_counts, area",
    Helpers.test_determine_topology_parameters_params,
)
def test_determine_topology_parameters(
    trace_length_array,
    node_counts,
    area,
):
    """
    Test determining parameters.
    """
    topology_parameters = parameters.determine_topology_parameters(
        trace_length_array,
        node_counts,
        area,
    )
    assert all(param >= 0 for param in topology_parameters.values())
    assert all(
        isinstance(param, (float, int)) for param in topology_parameters.values()
    )


@pytest.mark.parametrize(
    "topology_parameters_list, labels, colors", Helpers.test_plot_topology_params
)
def test_plot_parameters_plot(topology_parameters_list, labels, colors):
    """
    Test plotting parameters.
    """
    figs, axes = parameters.plot_parameters_plot(
        topology_parameters_list, labels, colors
    )
    assert all(isinstance(fig, Figure) for fig in figs)
    assert all(isinstance(ax, Axes) for ax in axes)
    plt.close()


@pytest.mark.parametrize(
    "x_values,y_values,i_values,number_of_bins",
    tests.test_ternary_heatmapping_params(),
)
def test_ternary_heatmapping(x_values, y_values, i_values, number_of_bins):
    """
    Test ternary_heatmapping.
    """
    fig, tax = parameters.ternary_heatmapping(
        x_values, y_values, i_values, number_of_bins
    )
    assert isinstance(fig, Figure)
    assert isinstance(tax, ternary.TernaryAxesSubplot)
