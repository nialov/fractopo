import numpy as np
import powerlaw
import matplotlib
import matplotlib.pyplot as plt
import ternary
import pytest
from hypothesis import given, settings
from hypothesis.extra import numpy
from hypothesis.strategies import floats

import fractopo.analysis.parameters as parameters
from fractopo.general import Param


from tests import Helpers


@pytest.mark.parametrize("node_counts_list,labels", Helpers.test_plot_xyi_plot_params)
def test_plot_xyi_plot(node_counts_list, labels):
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
    fig, ax, tax = parameters.plot_branch_plot(
        branch_counts_list=branch_counts_list, labels=labels
    )
    assert isinstance(fig, matplotlib.figure.Figure)  # type: ignore
    assert isinstance(ax, matplotlib.axes.Axes)  # type: ignore
    assert isinstance(tax, ternary.ternary_axes_subplot.TernaryAxesSubplot)
    plt.close()


@pytest.mark.parametrize(
    "trace_length_array, branch_length_array, node_counts, area",
    Helpers.test_determine_topology_parameters_params,
)
def test_determine_topology_parameters(
    trace_length_array,
    branch_length_array,
    node_counts,
    area,
):
    topology_parameters = parameters.determine_topology_parameters(
        trace_length_array,
        branch_length_array,
        node_counts,
        area,
    )
    assert all([key in topology_parameters for key in [param.value for param in Param]])
    assert all([param >= 0 for param in topology_parameters.values()])
    assert all(
        [isinstance(param, (float, int)) for param in topology_parameters.values()]
    )


@pytest.mark.parametrize(
    "topology_parameters_list, labels, colors", Helpers.test_plot_topology_params
)
def test_plot_parameters_plot(topology_parameters_list, labels, colors):
    figs, axes = parameters.plot_parameters_plot(
        topology_parameters_list, labels, colors
    )
    assert all([isinstance(fig, matplotlib.figure.Figure) for fig in figs])  # type: ignore
    assert all([isinstance(ax, matplotlib.axes.Axes) for ax in axes])  # type: ignore
    plt.close()
