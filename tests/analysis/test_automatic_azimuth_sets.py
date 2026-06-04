import numpy as np
import pytest
from hypothesis import assume, example, given, settings
from hypothesis.extra import numpy as hypothesis_numpy
from hypothesis.strategies import integers

from fractopo.analysis.automatic_azimuth_sets import automatic_azimuth_sets

RNG = np.random.default_rng(0)


def test_automatic_azimuth_sets_perfect_clusters():
    azimuths = np.concatenate(
        [
            RNG.normal(10, 2, 10),
            RNG.normal(80, 2, 10),
            RNG.normal(150, 2, 10),
        ]
    )
    azimuths = azimuths % 180
    labels, centers = automatic_azimuth_sets(azimuths, n_sets=3, random_state=0)
    assert len(labels) == len(azimuths)
    assert len(centers) == 3
    counts = [np.sum(labels == i) for i in range(3)]
    assert all(c >= 7 for c in counts)
    sorted_centers = np.sort(centers)
    assert np.allclose(sorted_centers, [10, 80, 150], atol=10)


def test_automatic_azimuth_sets_axial_wraparound_cluster():
    azimuths = np.array([178.0, 179.0, 1.0, 2.0, 88.0, 92.0])
    labels, centers = automatic_azimuth_sets(azimuths, n_sets=2, random_state=0)
    assert len(set(labels[:4])) == 1
    assert labels[4] == labels[5]
    assert labels[0] != labels[4]
    assert any(np.isclose(center, 90.0, atol=5.0) for center in centers)
    assert any(
        np.isclose(center, 0.0, atol=5.0) or np.isclose(center, 180.0, atol=5.0)
        for center in centers
    )


def test_automatic_azimuth_sets_returns_labels_and_centers():
    azimuths = np.array([0, 10, 20, 90, 100, 110, 170, 175])
    labels, centers = automatic_azimuth_sets(azimuths, n_sets=3)
    assert labels.shape == (len(azimuths),)
    assert centers.shape == (3,)


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
    assume(np.unique(azimuths % 180).size >= 2)
    labels, centers = automatic_azimuth_sets(azimuths, n_sets=2, random_state=0)
    shifted_labels, shifted_centers = automatic_azimuth_sets(
        azimuths + 180.0, n_sets=2, random_state=0
    )
    assert labels.shape == shifted_labels.shape
    assert np.array_equal(labels, shifted_labels)
    assert np.allclose(np.sort(centers), np.sort(shifted_centers))


@pytest.mark.parametrize(
    ("azimuths", "n_sets"),
    [
        (np.array([0.0, 90.0, 180.0]), 0),
        (np.array([], dtype=float), 1),
        (np.array([[0.0, 90.0]]), 1),
        (np.array([0.0, np.nan]), 1),
        (np.array([0.0, 1.0]), 3),
    ],
)
def test_automatic_azimuth_sets_invalid_inputs(azimuths, n_sets):
    with pytest.raises(ValueError):
        automatic_azimuth_sets(azimuths, n_sets=n_sets)
