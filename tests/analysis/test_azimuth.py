"""
Test azimuth.py azimuth rose plotting functions.
"""

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
            "length_array",
            "azimuth_set_array",
            "azimuth_set_names",
            "label",
            "append_azimuth_set_text",
            "axial",
            "plain",
            "color",
        ]
    ),
    tests.test_plot_azimuth_plot_params(),
)
def test_plot_azimuth_plot(
    azimuth_array,
    length_array,
    azimuth_set_array,
    azimuth_set_names,
    label,
    append_azimuth_set_text,
    axial,
    plain,
    color,
):
    """
    Test plot_azimuth_plot.
    """
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
        axial=axial,
        azimuth_set_ranges=azimuth_set_ranges,
        plain=plain,
        bar_color=color,
    )

    assert isinstance(bins, AzimuthBins)
    assert isinstance(fig, Figure)
    assert isinstance(ax, PolarAxes)
