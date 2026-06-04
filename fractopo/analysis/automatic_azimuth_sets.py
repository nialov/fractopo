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

from fractopo.general import SetRangeTuple
from fractopo.typing import NDArrayWithAxialAzimuths, NDArrayWithPositives

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
def _smallest_covering_axial_range(
    cluster_azimuths_deg: np.ndarray,
) -> Tuple[float, float]:
    """
    Return the smallest axial range covering the cluster azimuths.

    The returned range is expressed in [0, 180) and may wrap around, e.g.
    ``(170.0, 20.0)``.

    Examples
    --------
    >>> _smallest_covering_axial_range(np.array([170.0, 175.0, 5.0, 10.0]))
    (170.0, 10.0)
    >>> _smallest_covering_axial_range(np.array([70.0, 90.0, 110.0]))
    (70.0, 110.0)
    """
    azimuths = np.sort(cluster_azimuths_deg % 180)
    if azimuths.size == 1:
        angle = float(azimuths[0])
        return angle, angle
    wrapped_azimuths = np.concatenate((azimuths, azimuths[:1] + 180.0))
    gaps = np.diff(wrapped_azimuths)
    largest_gap_index = int(np.argmax(gaps))
    start = float(wrapped_azimuths[largest_gap_index + 1] % 180)
    end = float(wrapped_azimuths[largest_gap_index] % 180)
    return start, end


@beartype
def _cluster_ranges(
    azimuths_deg: np.ndarray, set_labels: np.ndarray, n_sets: int
) -> SetRangeTuple:
    """Determine smallest covering axial ranges for each detected cluster."""
    return tuple(
        _smallest_covering_axial_range(azimuths_deg[set_labels == label])
        for label in range(n_sets)
    )


@beartype
def automatic_azimuth_sets(
    azimuths_deg: NDArrayWithAxialAzimuths,
    length_array: NDArrayWithPositives,
    n_sets: Optional[Annotated[int, Is[lambda value: value > 0]]] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, SetRangeTuple]:
    """
    Automatically detect fracture sets in axial azimuth data.

    Axial azimuths are circular data where 0° and 180° are equivalent. To
    respect this topology, clustering is performed on doubled-angle unit-circle
    coordinates before converting cluster centers back to axial azimuths.
    Fracture lengths are used as sample weights in clustering so longer
    fractures influence the detected set centers more strongly.

    :param azimuths_deg: 1D array of axial azimuths in degrees.
    :param length_array: 1D array of strictly positive fracture lengths with the
        same shape as ``azimuths_deg``.
    :param n_sets: Number of sets to find. Must be specified because
        automatic set-count detection is not yet implemented.
    :param random_state: Optional random state passed to
        sklearn.cluster.KMeans. If None, no explicit random state is set.
    :return: Detected center azimuths and smallest covering axial ranges for
        each set.

    Examples:
        A longer fracture can pull the detected center towards its azimuth.

        >>> azimuths = np.array([0.0, 20.0])
        >>> centers, _ = automatic_azimuth_sets(
        ...     azimuths,
        ...     np.array([1.0, 9.0]),
        ...     n_sets=1,
        ...     random_state=0,
        ... )
        >>> float(np.round(centers[0], 1))
        18.1

        Azimuths close to 0° and 180° are treated as belonging to the same
        axial direction and produce a wraparound range.

        >>> azimuths = np.array([178.0, 179.0, 1.0, 2.0, 88.0, 92.0])
        >>> lengths = np.ones_like(azimuths)
        >>> centers, ranges = automatic_azimuth_sets(
        ...     azimuths,
        ...     lengths,
        ...     n_sets=2,
        ...     random_state=0,
        ... )
        >>> len(centers) == 2
        True
        >>> any(start > end for start, end in ranges)
        True
    """
    azimuths = np.asarray(azimuths_deg, dtype=float)
    lengths = np.asarray(length_array, dtype=float)
    if azimuths.shape != lengths.shape:
        raise ValueError("length_array must have the same shape as azimuths_deg.")
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
    set_labels = kmeans.fit_predict(unit_vectors, sample_weight=lengths)
    log.debug("KMeans determined set labels: %s", set_labels)
    set_centers_deg = _cluster_centers_to_axial_azimuths(kmeans.cluster_centers_)
    set_ranges = _cluster_ranges(azimuths, set_labels, n_sets)
    log.debug("Automatic azimuth set centers determined: %s", set_centers_deg)
    log.debug("Automatic azimuth set ranges determined: %s", set_ranges)
    return set_centers_deg, set_ranges
