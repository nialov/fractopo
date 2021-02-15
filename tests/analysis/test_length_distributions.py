"""
Tests for length distributions utilities.
"""
import numpy as np
import powerlaw
from hypothesis import given, settings
from hypothesis.extra import numpy
from tests import Helpers
from hypothesis.strategies import floats

import pytest
from fractopo.analysis import length_distributions


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
