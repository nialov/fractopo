"""
Automatic detection of azimuth clusters (fracture sets, orientation clusters)
in orientation data, using circular embeddings and clustering algorithms.
"""

import logging

import numpy as np
from beartype import beartype
from beartype.typing import Annotated, Optional, Tuple
from beartype.vale import Is
from sklearn.cluster import KMeans

from fractopo.typing import NDArray1DNotEmpty

log = logging.getLogger(__name__)


@beartype
def _azimuths_to_axial_unit_vectors(azimuths_deg: np.ndarray) -> np.ndarray:
    """
    Represent axial azimuths on the unit circle.

    Axial data wraps over 180 degrees, so the angle is doubled before the
    circular embedding. This makes 0° and 180° equivalent.

    Examples
    --------
    >>> vectors = _azimuths_to_axial_unit_vectors(np.array([0.0, 180.0, 90.0]))
    >>> np.allclose(vectors[0], vectors[1])
    True
    >>> np.allclose(vectors[2], np.array([-1.0, 0.0]))
    True
    """
    doubled_azimuths_rad = np.deg2rad((azimuths_deg % 180) * 2)
    return np.column_stack((np.cos(doubled_azimuths_rad), np.sin(doubled_azimuths_rad)))


@beartype
def _cluster_centers_to_axial_azimuths(cluster_centers: np.ndarray) -> np.ndarray:
    """Convert doubled-angle cluster centers back to axial azimuths in [0, 180)."""
    doubled_center_azimuths_rad = np.arctan2(
        cluster_centers[:, 1], cluster_centers[:, 0]
    )
    return (np.rad2deg(doubled_center_azimuths_rad) / 2) % 180


@beartype
def automatic_azimuth_sets(
    azimuths_deg: NDArray1DNotEmpty,
    n_sets: Optional[Annotated[int, Is[lambda value: value > 0]]] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Automatically detect fracture sets in axial azimuth data.

    Axial azimuths are circular data where 0° and 180° are equivalent. To
    respect this topology, clustering is performed on doubled-angle unit-circle
    coordinates before converting cluster centers back to axial azimuths.

    :param azimuths_deg: 1D array of axial azimuths in degrees.
    :param n_sets: Number of sets to find. If None, raises ValueError because
        automatic set-count detection is not yet implemented.
    :param random_state: Optional random state passed to
        sklearn.cluster.KMeans. If None, no explicit random state is set.
    :return: Cluster labels for each azimuth and center azimuths for each set.

    Examples:
        Azimuths close to 0° and 180° are treated as belonging to the same
        axial direction.

        >>> azimuths = np.array([178.0, 179.0, 1.0, 2.0, 88.0, 92.0])
        >>> labels, centers = automatic_azimuth_sets(azimuths, n_sets=2, random_state=0)
        >>> len(set(labels[:4])) == 1
        True
        >>> bool(labels[4] == labels[5])
        True
        >>> any(np.isclose(center, 90.0, atol=5.0) for center in centers)
        True

        A simple three-cluster example returns one label per input and one
        center per requested set.

        >>> azimuths = np.array([
        ...     5.0,
        ...     10.0,
        ...     15.0,
        ...     75.0,
        ...     80.0,
        ...     85.0,
        ...     145.0,
        ...     150.0,
        ...     155.0,
        ... ])
        >>> labels, centers = automatic_azimuth_sets(azimuths, n_sets=3, random_state=0)
        >>> labels.shape
        (9,)
        >>> np.allclose(np.sort(centers), np.array([10.0, 80.0, 150.0]), atol=10.0)
        True
    """
    azimuths = np.asarray(azimuths_deg, dtype=float)
    if n_sets is None:
        raise ValueError(
            "n_sets must be specified and >0 (auto-detection not implemented yet)."
        )
    if n_sets > azimuths.size:
        raise ValueError("n_sets cannot be larger than the number of azimuths.")
    log.debug(
        "Clustering %s azimuths into %s sets with random_state=%s.",
        azimuths.size,
        n_sets,
        random_state,
    )
    unit_vectors = _azimuths_to_axial_unit_vectors(azimuths)
    kmeans = KMeans(n_clusters=n_sets, random_state=random_state, n_init=10)
    set_labels = kmeans.fit_predict(unit_vectors)
    set_centers_deg = _cluster_centers_to_axial_azimuths(kmeans.cluster_centers_)
    log.debug("Automatic azimuth set centers determined: %s", set_centers_deg)
    return set_labels, set_centers_deg
