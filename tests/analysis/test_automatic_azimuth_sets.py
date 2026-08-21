import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import assume, example, given, settings
from hypothesis.extra import numpy as hypothesis_numpy
from hypothesis.strategies import composite, integers

from fractopo.analysis.automatic_azimuth_sets import (
    _smallest_covering_axial_range,
    automatic_azimuth_sets,
    trim_azimuth_set_ranges,
)
from fractopo.general import is_set

RNG = np.random.default_rng(0)


@pytest.mark.parametrize(
    ("azimuths", "expected_range"),
    [
        pytest.param(
            np.array([170.0, 175.0, 5.0, 10.0]),
            (170.0, 10.0),
            id="wraparound-range",
        ),
        pytest.param(
            np.array([70.0]),
            (70.0, 70.0),
            id="singleton-range",
        ),
    ],
)
def test_smallest_covering_axial_range_known_cases(azimuths, expected_range):
    """Test known smallest-covering axial range cases."""
    result = _smallest_covering_axial_range(azimuths)
    assert np.allclose(result, expected_range)


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
        is_set(azimuth, wraparound_ranges[0], loop_around=True)
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


def test_trim_azimuth_set_ranges_creates_background_points():
    """Test trimming a set range can reclassify edge fractures as background."""
    azimuths = np.array([10.0, 12.0, 14.0, 30.0])
    lengths = np.array([1.0, 3.0, 3.0, 3.0])
    trimmed_ranges, labels = trim_azimuth_set_ranges(
        azimuths,
        lengths,
        ((10.0, 30.0),),
        retained_length_fraction=0.6,
    )
    assert np.allclose(trimmed_ranges[0], (12.0, 14.0))
    assert tuple(labels) == ("background", "0", "0", "background")


def test_trim_azimuth_set_ranges_wraparound():
    """Test trimming preserves wraparound semantics for axial sets."""
    azimuths = np.array([178.0, 179.0, 1.0, 2.0, 20.0])
    lengths = np.array([4.0, 4.0, 4.0, 4.0, 1.0])
    trimmed_ranges, labels = trim_azimuth_set_ranges(
        azimuths,
        lengths,
        ((178.0, 20.0),),
        retained_length_fraction=0.75,
    )
    assert trimmed_ranges[0][0] > trimmed_ranges[0][1]
    assert all(
        is_set(azimuth, trimmed_ranges[0], loop_around=True)
        for azimuth in np.array([178.0, 179.0, 1.0])
    )
    assert labels[-1] == "background"


@pytest.mark.parametrize(
    (
        "azimuths",
        "lengths",
        "set_ranges",
        "retained_length_fraction",
        "expected_ranges",
        "expected_labels",
    ),
    [
        pytest.param(
            np.array([10.0, 12.0, 14.0, 30.0]),
            np.array([1.0, 3.0, 3.0, 3.0]),
            ((10.0, 30.0),),
            1.0,
            ((10.0, 30.0),),
            ("0", "0", "0", "0"),
            id="full-retained-fraction-keeps-single-range",
        ),
        pytest.param(
            np.array([178.0, 179.0, 1.0, 2.0]),
            np.ones(4),
            ((178.0, 20.0),),
            1.0,
            ((178.0, 2.0),),
            ("0", "0", "0", "0"),
            id="full-retained-fraction-shrinks-wraparound-to-data-support",
        ),
        pytest.param(
            np.array([10.0, 12.0, 14.0]),
            np.ones(3),
            ((10.0, 20.0), (80.0, 100.0)),
            1.0,
            ((10.0, 14.0), (80.0, 100.0)),
            ("0", "0", "0"),
            id="empty-set-range-is-left-unchanged",
        ),
    ],
)
def test_trim_azimuth_set_ranges_known_cases(
    azimuths,
    lengths,
    set_ranges,
    retained_length_fraction,
    expected_ranges,
    expected_labels,
):
    """Test known trimming cases."""
    trimmed_ranges, labels = trim_azimuth_set_ranges(
        azimuths,
        lengths,
        set_ranges,
        retained_length_fraction=retained_length_fraction,
    )
    assert trimmed_ranges == expected_ranges
    assert tuple(labels) == expected_labels


@composite
def azimuths_and_lengths(draw):
    size = draw(integers(min_value=2, max_value=12))
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


@example(np.array([170.0, 175.0, 5.0, 10.0]))
@given(
    hypothesis_numpy.arrays(
        dtype=np.int64,
        shape=integers(min_value=1, max_value=20),
        elements=integers(min_value=0, max_value=179),
    )
)
@settings(max_examples=25)
def test_smallest_covering_axial_range_contains_all_input_azimuths(azimuths):
    """Test returned axial range contains every input azimuth."""
    result = _smallest_covering_axial_range(azimuths)
    assert all(is_set(azimuth, result, loop_around=True) for azimuth in azimuths)


@example((np.array([0.0, 90.0]), np.array([1.0, 1.0])))
@example((np.array([178.0, 2.0, 88.0, 92.0]), np.array([1.0, 1.0, 1.0, 1.0])))
@given(azimuths_and_lengths())
@settings(max_examples=10, deadline=None)
def test_automatic_azimuth_sets_is_invariant_under_adding_180_degrees(data):
    """Test that adding 180 degrees does not change axial set detection."""
    azimuths, lengths = data
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
        pytest.param(
            np.array([0.0, 90.0, 180.0]),
            np.ones(3),
            0,
            BeartypeCallHintParamViolation,
            id="n_sets-must-be-positive",
        ),
        pytest.param(
            np.array([], dtype=float),
            np.array([], dtype=float),
            1,
            ValueError,
            id="empty-input",
        ),
        pytest.param(
            np.array([[0.0, 90.0]]),
            np.array([[1.0, 1.0]]),
            1,
            BeartypeCallHintParamViolation,
            id="azimuths-must-be-one-dimensional",
        ),
        pytest.param(
            np.array([0.0, np.nan]),
            np.ones(2),
            1,
            BeartypeCallHintParamViolation,
            id="azimuths-must-be-finite",
        ),
        pytest.param(
            np.array([0.0, 1.0]),
            np.ones(2),
            3,
            ValueError,
            id="too-many-sets",
        ),
        pytest.param(
            np.array([0.0, 1.0]),
            np.ones(3),
            1,
            ValueError,
            id="azimuths-and-lengths-shape-mismatch",
        ),
        pytest.param(
            np.array([0.0, 1.0]),
            np.array([1.0, np.inf]),
            1,
            ValueError,
            id="lengths-contain-infinity",
        ),
        pytest.param(
            np.array([0.0, 1.0]),
            np.array([1.0, 0.0]),
            1,
            BeartypeCallHintParamViolation,
            id="lengths-must-be-positive",
        ),
    ],
)
def test_automatic_azimuth_sets_invalid_inputs(
    azimuths, lengths, n_sets, expected_exception
):
    """Test that invalid inputs are rejected by contract checks or runtime guards."""
    with pytest.raises(expected_exception):
        automatic_azimuth_sets(azimuths, lengths, n_sets=n_sets)


@pytest.mark.parametrize(
    (
        "azimuths",
        "lengths",
        "set_ranges",
        "retained_length_fraction",
        "expected_exception",
    ),
    [
        pytest.param(
            np.array([0.0, 1.0]),
            np.ones(3),
            ((0.0, 1.0),),
            0.6,
            ValueError,
            id="azimuths-and-lengths-shape-mismatch",
        ),
        pytest.param(
            np.array([0.0, 1.0]),
            np.ones(2),
            (),
            0.6,
            ValueError,
            id="set-ranges-must-not-be-empty",
        ),
        pytest.param(
            np.array([0.0, 1.0]),
            np.ones(2),
            ((0.0, 1.0),),
            0.0,
            BeartypeCallHintParamViolation,
            id="retained-fraction-must-be-positive",
        ),
        pytest.param(
            np.array([[0.0, 1.0]]),
            np.ones((1, 2)),
            ((0.0, 1.0),),
            0.6,
            BeartypeCallHintParamViolation,
            id="azimuths-must-be-one-dimensional",
        ),
        pytest.param(
            np.array([0.0, np.nan]),
            np.ones(2),
            ((0.0, 1.0),),
            0.6,
            BeartypeCallHintParamViolation,
            id="azimuths-must-be-finite",
        ),
        pytest.param(
            np.array([0.0, 1.0]),
            np.array([1.0, 0.0]),
            ((0.0, 1.0),),
            0.6,
            BeartypeCallHintParamViolation,
            id="lengths-must-be-positive",
        ),
        pytest.param(
            np.array([0.0, 1.0]),
            np.ones(2),
            ((-1.0, 1.0),),
            0.6,
            ValueError,
            id="set-ranges-must-be-within-axial-domain",
        ),
        pytest.param(
            np.array([0.0, 1.0]),
            np.ones(2),
            ((0.0, np.inf),),
            0.6,
            ValueError,
            id="set-ranges-must-be-finite",
        ),
        pytest.param(
            np.array([0.0, 1.0]),
            np.ones(2),
            ((0.0,),),
            0.6,
            BeartypeCallHintParamViolation,
            id="set-ranges-must-contain-two-values",
        ),
    ],
)
def test_trim_azimuth_set_ranges_invalid_inputs(
    azimuths,
    lengths,
    set_ranges,
    retained_length_fraction,
    expected_exception,
):
    """Test invalid inputs to trimming are rejected."""
    with pytest.raises(expected_exception):
        trim_azimuth_set_ranges(
            azimuths,
            lengths,
            set_ranges,
            retained_length_fraction=retained_length_fraction,
        )
