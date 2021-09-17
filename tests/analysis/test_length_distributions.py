"""
Tests for length distributions utilities.
"""
import numpy as np
import powerlaw
import pytest
from hypothesis import given, settings
from hypothesis.extra import numpy
from hypothesis.strategies import floats
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from fractopo.analysis import length_distributions
from tests import Helpers


@given(
    numpy.arrays(dtype=float, shape=1),
    floats(min_value=0, allow_infinity=False, allow_nan=False),
)
@settings(max_examples=25)
def test_determine_fit(length_array: np.ndarray, cut_off: float):
    """
    Test determine_fit.
    """
    fit = length_distributions.determine_fit(length_array, cut_off)
    assert isinstance(fit, powerlaw.Fit)


@pytest.mark.parametrize("lengths,label", Helpers.test_describe_powerlaw_fit_params)
def test_describe_powerlaw_fit(lengths, label):
    """
    Test describe_powerlaw_fit.
    """
    fit = length_distributions.determine_fit(lengths)
    result = length_distributions.describe_powerlaw_fit(fit, label)
    assert isinstance(result, dict)


@pytest.mark.parametrize("lengths,_", Helpers.test_describe_powerlaw_fit_params)
def test_all_fit_attributes_dict(lengths, _):
    """
    Test describe_powerlaw_fit.
    """
    fit = length_distributions.determine_fit(lengths)
    result = length_distributions.all_fit_attributes_dict(fit)
    assert isinstance(result, dict)


@pytest.mark.parametrize(
    "list_of_length_arrays,list_of_area_values,names",
    [
        (
            [Helpers.kb11_traces.geometry.length, Helpers.kb11_traces.geometry.length],
            [
                Helpers.kb11_area.geometry.area.sum(),
                Helpers.kb11_area.geometry.area.sum() * 10,
            ],
            ["kb11_1", "kb11_2"],
        ),
        (
            [
                Helpers.kb11_traces.geometry.length,
                Helpers.hastholmen_traces.geometry.length,
            ],
            [
                Helpers.kb11_area.geometry.area.sum(),
                Helpers.hastholmen_area.geometry.area.sum(),
            ],
            ["kb11", "hastholmen"],
        ),
    ],
)
@pytest.mark.parametrize("auto_cut_off", [True, False])
@pytest.mark.parametrize("using_branches", [True, False])
def test_multi_scale_length_distribution_fit(
    list_of_length_arrays,
    list_of_area_values,
    names,
    auto_cut_off,
    using_branches,
):
    """
    Test multi_scale_length_distribution_fit.
    """
    assert isinstance(list_of_length_arrays, list)
    assert isinstance(list_of_area_values, list)
    assert isinstance(names, list)
    assert len(list_of_length_arrays) == len(list_of_area_values) == len(names)
    fig, ax = length_distributions.multi_scale_length_distribution_fit(
        distributions=[
            length_distributions.LengthDistribution(
                name=name,
                lengths=lengths,
                area_value=area_value,
            )
            for name, lengths, area_value in zip(
                names, list_of_length_arrays, list_of_area_values
            )
        ],
        auto_cut_off=auto_cut_off,
        using_branches=using_branches,
    )

    fig.suptitle(f"auto,using: {auto_cut_off, using_branches}")

    assert isinstance(fig, Figure) and isinstance(ax, Axes)
