import numpy as np
import pytest

from fractopo.analysis.automatic_azimuth_sets import automatic_azimuth_sets


def test_automatic_azimuth_sets_perfect_clusters():
    # Cluster 1: near 10 deg, Cluster 2: near 80 deg, Cluster 3: near 150 deg
    azimuths = np.concatenate(
        [
            np.random.normal(10, 2, 10),
            np.random.normal(80, 2, 10),
            np.random.normal(150, 2, 10),
        ]
    )
    # Force into [0,180)
    azimuths = azimuths % 180
    labels, centers = automatic_azimuth_sets(azimuths, n_sets=3, random_state=0)
    assert len(labels) == len(azimuths)
    assert len(centers) == 3
    # Each cluster should have at least 7
    counts = [np.sum(labels == i) for i in range(3)]
    assert all(c >= 7 for c in counts)
    # Sorted centers should be close to true centers
    sorted_centers = np.sort(centers)
    assert np.allclose(sorted_centers, [10, 80, 150], atol=10)


def test_automatic_azimuth_sets_returns_labels_and_centers():
    azimuths = np.array([0, 10, 20, 90, 100, 110, 170, 175])
    labels, centers = automatic_azimuth_sets(azimuths, n_sets=3)
    assert labels.shape == (len(azimuths),)
    assert centers.shape == (3,)


def test_automatic_azimuth_sets_invalid_nsets():
    azimuths = np.array([0, 90, 180])
    with pytest.raises(ValueError):
        automatic_azimuth_sets(azimuths, n_sets=0)
