import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import assume, example, given, settings
from hypothesis.extra import numpy as hypothesis_numpy
from hypothesis.strategies import integers

from fractopo.analysis.automatic_azimuth_sets import (
    _smallest_covering_axial_range,
    automatic_azimuth_sets,
)

RNG = np.random.default_rng(0)


def _range_contains_azimuth(range_tuple, azimuth):
    start, end = range_tuple
    azimuth = azimuth % 180
    if start <= end:
        return start <= azimuth <= end
    return azimuth >= start or azimuth <= end


def test_smallest_covering_axial_range_wraparound():
    """Test that wraparound axial ranges are represented with start > end."""
    result = _smallest_covering_axial_range(np.array([170.0, 175.0, 5.0, 10.0]))
    assert np.allclose(result, (170.0, 10.0))


def test_automatic_azimuth_sets_perfect_clusters():
    """Test automatic_azimuth_sets with well-separated deterministic clusters."""
    azimuths = np.concatenate(
        [
            RNG.normal(10, 2, 10),
            RNG.normal(80, 2, 10),
            RNG.normal(150, 2, 10),
        ]
    )
    azimuths = azimuths % 180
    centers, ranges = automatic_azimuth_sets(azimuths, n_sets=3, random_state=0)
    assert len(centers) == 3
    assert len(ranges) == 3
    assert np.allclose(np.sort(centers), [10, 80, 150], atol=10)
    for detected_range in ranges:
        assert isinstance(detected_range, tuple)
        assert len(detected_range) == 2


def test_automatic_azimuth_sets_axial_wraparound_cluster():
    """Test that wraparound clusters produce a wraparound range."""
    azimuths = np.array([178.0, 179.0, 1.0, 2.0, 88.0, 92.0])
    centers, ranges = automatic_azimuth_sets(azimuths, n_sets=2, random_state=0)
    assert any(np.isclose(center, 90.0, atol=5.0) for center in centers)
    wraparound_ranges = [
        range_tuple for range_tuple in ranges if range_tuple[0] > range_tuple[1]
    ]
    assert len(wraparound_ranges) == 1
    assert all(
        _range_contains_azimuth(wraparound_ranges[0], azimuth)
        for azimuth in np.array([178.0, 179.0, 1.0, 2.0])
    )


def test_automatic_azimuth_sets_returns_centers_and_ranges():
    """Test output shapes for detected centers and ranges."""
    azimuths = np.array([0, 10, 20, 90, 100, 110, 170, 175])
    centers, ranges = automatic_azimuth_sets(azimuths, n_sets=3, random_state=0)
    assert centers.shape == (3,)
    assert len(ranges) == 3


@example(np.array([0.0, 90.0]))
@example(np.array([178.0, 2.0, 88.0, 92.0]))
@given(
    hypothesis_numpy.arrays(
        dtype=np.float64,
        shape=integers(min_value=2, max_value=20),
        elements=integers(min_value=0, max_value=179),
    )
)
@settings(max_examples=25)
def test_automatic_azimuth_sets_is_invariant_under_adding_180_degrees(azimuths):
    """Test that adding 180 degrees does not change axial set detection."""
    assume(np.unique(azimuths % 180).size >= 2)
    centers, ranges = automatic_azimuth_sets(azimuths, n_sets=2, random_state=0)
    shifted_centers, shifted_ranges = automatic_azimuth_sets(
        azimuths + 180.0, n_sets=2, random_state=0
    )
    assert np.allclose(np.sort(centers), np.sort(shifted_centers))
    assert sorted(tuple(np.round(range_tuple, 6)) for range_tuple in ranges) == sorted(
        tuple(np.round(range_tuple, 6)) for range_tuple in shifted_ranges
    )


@pytest.mark.parametrize(
    ("azimuths", "n_sets", "expected_exception"),
    [
        (np.array([0.0, 90.0, 180.0]), 0, BeartypeCallHintParamViolation),
        (np.array([], dtype=float), 1, BeartypeCallHintParamViolation),
        (np.array([[0.0, 90.0]]), 1, BeartypeCallHintParamViolation),
        (np.array([0.0, np.nan]), 1, ValueError),
        (np.array([0.0, 1.0]), 3, ValueError),
    ],
)
def test_automatic_azimuth_sets_invalid_inputs(azimuths, n_sets, expected_exception):
    """Test that invalid inputs are rejected by contract checks or runtime guards."""
    with pytest.raises(expected_exception):
        automatic_azimuth_sets(azimuths, n_sets=n_sets)
