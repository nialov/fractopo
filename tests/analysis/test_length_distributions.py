import numpy as np
import powerlaw
from hypothesis import given, settings
from hypothesis.extra import numpy
from hypothesis.strategies import floats

from fractopo.analysis import length_distributions


@given(
    numpy.arrays(dtype=float, shape=1),
    floats(min_value=0, allow_infinity=False, allow_nan=False),
)
@settings(max_examples=25)
def test_determine_fit(length_array: np.ndarray, cut_off: float):
    fit = length_distributions.determine_fit(length_array, cut_off)
    assert isinstance(fit, powerlaw.Fit)
