import numpy as np
import powerlaw
from hypothesis import given
from hypothesis.extra import numpy
from hypothesis.strategies import floats

import fractopo.analysis.length_distributions as length_distributions


@given(
    numpy.arrays(dtype=float, shape=1),
    floats(min_value=0, allow_infinity=False, allow_nan=False),
)
def test_determine_fit(length_array: np.ndarray, cut_off: float):
    fit = length_distributions.determine_fit(length_array, cut_off)
    assert isinstance(fit, powerlaw.Fit)
