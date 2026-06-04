import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import assume, example, given, settings
from hypothesis.extra import numpy as hypothesis_numpy
from hypothesis.strategies import composite, integers

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
    lengths = np.ones_like(azimuths)
    centers, ranges = automatic_azimuth_sets(
        azimuths,
        lengths,
        n_sets=3,
        random_state=0,
    )
    assert len(centers) == 3
    assert len(ranges) == 3
    assert np.allclose(np.sort(centers), [10, 80, 150], atol=10)
    for detected_range in ranges:
        assert isinstance(detected_range, tuple)
        assert len(detected_range) == 2


def test_automatic_azimuth_sets_length_weighting_changes_center():
    """Test that longer fractures influence the detected center more strongly."""
    centers, _ = automatic_azimuth_sets(
        np.array([0.0, 20.0]),
        np.array([1.0, 9.0]),
        n_sets=1,
        random_state=0,
    )
    assert np.isclose(centers[0], 18.0, atol=1.0)


def test_automatic_azimuth_sets_axial_wraparound_cluster():
    """Test that wraparound clusters produce a wraparound range."""
    azimuths = np.array([178.0, 179.0, 1.0, 2.0, 88.0, 92.0])
    lengths = np.ones_like(azimuths)
    centers, ranges = automatic_azimuth_sets(
        azimuths, lengths, n_sets=2, random_state=0
    )
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
    lengths = np.ones_like(azimuths, dtype=float)
    centers, ranges = automatic_azimuth_sets(
        azimuths, lengths, n_sets=3, random_state=0
    )
    assert centers.shape == (3,)
    assert len(ranges) == 3


@composite
def azimuths_and_lengths(draw):
    size = draw(integers(min_value=2, max_value=20))
    azimuths = draw(
        hypothesis_numpy.arrays(
            dtype=np.float64,
            shape=size,
            elements=integers(min_value=0, max_value=179),
        )
    )
    lengths = draw(
        hypothesis_numpy.arrays(
            dtype=np.float64,
            shape=size,
            elements=integers(min_value=1, max_value=20),
        )
    )
    return azimuths, lengths


@example((np.array([0.0, 90.0]), np.array([1.0, 1.0])))
@example((np.array([178.0, 2.0, 88.0, 92.0]), np.array([1.0, 1.0, 1.0, 1.0])))
@given(azimuths_and_lengths())
@settings(max_examples=25)
def test_automatic_azimuth_sets_is_invariant_under_adding_180_degrees(data):
    azimuths, lengths = data
    """Test that adding 180 degrees does not change axial set detection."""
    assume(azimuths.shape == lengths.shape)
    assume(np.unique(azimuths % 180).size >= 2)
    centers, ranges = automatic_azimuth_sets(
        azimuths,
        lengths,
        n_sets=2,
        random_state=0,
    )
    shifted_centers, shifted_ranges = automatic_azimuth_sets(
        (azimuths + 180.0) % 180.0,
        lengths,
        n_sets=2,
        random_state=0,
    )
    assert np.allclose(np.sort(centers), np.sort(shifted_centers))
    assert sorted(tuple(np.round(range_tuple, 6)) for range_tuple in ranges) == sorted(
        tuple(np.round(range_tuple, 6)) for range_tuple in shifted_ranges
    )


@pytest.mark.parametrize(
    ("azimuths", "lengths", "n_sets", "expected_exception"),
    [
        (
            np.array([0.0, 90.0, 180.0]),
            np.ones(3),
            0,
            BeartypeCallHintParamViolation,
        ),
        (np.array([], dtype=float), np.array([], dtype=float), 1, ValueError),
        (
            np.array([[0.0, 90.0]]),
            np.array([[1.0, 1.0]]),
            1,
            BeartypeCallHintParamViolation,
        ),
        (np.array([0.0, np.nan]), np.ones(2), 1, BeartypeCallHintParamViolation),
        (np.array([0.0, 1.0]), np.ones(2), 3, ValueError),
        (np.array([0.0, 1.0]), np.ones(3), 1, ValueError),
        (np.array([0.0, 1.0]), np.array([1.0, np.inf]), 1, ValueError),
        (
            np.array([0.0, 1.0]),
            np.array([1.0, 0.0]),
            1,
            BeartypeCallHintParamViolation,
        ),
    ],
)
def test_automatic_azimuth_sets_invalid_inputs(
    azimuths, lengths, n_sets, expected_exception
):
    """Test that invalid inputs are rejected by contract checks or runtime guards."""
    with pytest.raises(expected_exception):
        automatic_azimuth_sets(azimuths, lengths, n_sets=n_sets)
