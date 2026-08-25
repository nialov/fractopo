"""
Test azimuth.py azimuth rose plotting functions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.projections import PolarAxes

import tests
from fractopo.analysis import azimuth
from fractopo.analysis.azimuth import AzimuthBins


@pytest.mark.parametrize(
    ",".join(
        [
            "azimuth_array",
            "label",
            "append_azimuth_set_text",
            "add_abundance_order",
            "axial",
            "plain",
            "color",
        ]
    ),
    tests.test_plot_azimuth_plot_params(),
)
@tests.plotting_test
def test_plot_azimuth_plot(
    azimuth_array,
    label,
    append_azimuth_set_text,
    add_abundance_order,
    axial,
    plain,
    color,
):
    """
    Test plot_azimuth_plot.
    """
    length_array = np.ones(len(azimuth_array))
    azimuth_set_array = np.ones(len(azimuth_array))
    azimuth_set_names = tuple(np.ones(len(azimuth_array)).astype(str))
    assert len(azimuth_array) == len(length_array)
    assert len(length_array) == len(azimuth_set_array)
    assert len(length_array) == len(azimuth_set_names)
    azimuth_set_ranges = ((0, 60), (60, 120), (120, 180))
    bins, fig, ax = azimuth.plot_azimuth_plot(
        azimuth_array=azimuth_array,
        length_array=length_array,
        azimuth_set_array=azimuth_set_array,
        azimuth_set_names=azimuth_set_names,
        label=label,
        append_azimuth_set_text=append_azimuth_set_text,
        add_abundance_order=add_abundance_order,
        axial=axial,
        azimuth_set_ranges=azimuth_set_ranges,
        plain=plain,
        bar_color=color,
    )

    assert isinstance(bins, AzimuthBins)
    assert isinstance(fig, Figure)
    assert isinstance(ax, PolarAxes)
    plt.close("all")


def test_plot_azimuth_plot_centers_follow_visualize_sets():
    """Center and range guides are opt-in and use distinct line styles."""
    kwargs = dict(
        azimuth_array=np.array([10.0, 90.0, 170.0]),
        length_array=np.ones(3),
        azimuth_set_array=np.array(["1", "2", "1"]),
        azimuth_set_names=("1", "2"),
        azimuth_set_ranges=((0, 30), (60, 120)),
        label="test",
        plain=False,
        axial=True,
        azimuth_set_centers=np.array([15.0, 90.0]),
    )
    _, fig, ax = azimuth.plot_azimuth_plot(**kwargs)
    assert len(ax.lines) == 0
    plt.close(fig)

    _, fig, ax = azimuth.plot_azimuth_plot(**kwargs, visualize_sets=True)
    assert len(ax.lines) == 8
    assert sum(line.get_color() == "darkgray" for line in ax.lines) == 4
    assert sum(line.get_color() == "black" for line in ax.lines) == 4
    plt.close(fig)
