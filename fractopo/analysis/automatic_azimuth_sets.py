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

from fractopo.general import SetRangeTuple, determine_set
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
def _unwrap_axial_range(
    azimuths_deg: np.ndarray, range_tuple: Tuple[float, float]
) -> np.ndarray:
    """Unwrap axial azimuths into a monotonic interval defined by ``range_tuple``."""
    start, end = range_tuple
    azimuths = azimuths_deg % 180
    if start <= end:
        return azimuths
    return np.where(azimuths < start, azimuths + 180.0, azimuths)


@beartype
def _trim_unwrapped_cluster_range_by_length_fraction(
    unwrapped_azimuths_deg: np.ndarray,
    length_array: np.ndarray,
    retained_length_fraction: float,
) -> Tuple[float, float]:
    """
    Return the narrowest unwrapped range containing the target length fraction.

    Examples
    --------
    Keep only the narrowest interval that still contains the requested share
    of the total fracture length.

    >>> _trim_unwrapped_cluster_range_by_length_fraction(
    ...     np.array([10.0, 12.0, 14.0, 30.0]),
    ...     np.array([1.0, 3.0, 3.0, 3.0]),
    ...     0.6,
    ... )
    (12.0, 14.0)
    """
    sort_idx = np.argsort(unwrapped_azimuths_deg)
    sorted_azimuths = unwrapped_azimuths_deg[sort_idx]
    sorted_lengths = length_array[sort_idx]
    target_length = retained_length_fraction * sorted_lengths.sum()
    cumulative_lengths = np.concatenate(([0.0], np.cumsum(sorted_lengths)))

    best_range = (float(sorted_azimuths[0]), float(sorted_azimuths[-1]))
    best_width = best_range[1] - best_range[0]
    right = 0

    for left in range(sorted_azimuths.size):
        right = max(right, left)
        while right < sorted_azimuths.size:
            retained_length = cumulative_lengths[right + 1] - cumulative_lengths[left]
            if retained_length >= target_length:
                break
            right += 1
        if right == sorted_azimuths.size:
            break
        candidate_range = (float(sorted_azimuths[left]), float(sorted_azimuths[right]))
        candidate_width = candidate_range[1] - candidate_range[0]
        if candidate_width < best_width:
            best_range = candidate_range
            best_width = candidate_width

    return best_range


@beartype
def trim_azimuth_set_ranges(
    azimuths_deg: NDArrayWithAxialAzimuths,
    length_array: NDArrayWithPositives,
    set_ranges: SetRangeTuple,
    retained_length_fraction: Annotated[float, Is[lambda value: 0 < value <= 1]],
    background_set_name: str = "background",
) -> Tuple[SetRangeTuple, np.ndarray]:
    """
    Trim detected axial set ranges to retain only a target fracture-length fraction.

    The current implementation trims each detected set independently by finding
    the narrowest axial sub-range that contains at least the requested fraction
    of the fracture length already assigned to that set.

    :param azimuths_deg: 1D array of axial azimuths in degrees.
    :param length_array: 1D array of strictly positive fracture lengths with the
        same shape as ``azimuths_deg``.
    :param set_ranges: Initial axial set ranges, typically from
        :func:`automatic_azimuth_sets`.
    :param retained_length_fraction: Fraction of assigned fracture length to
        retain inside each trimmed set range.
    :param background_set_name: Label assigned to fractures outside all trimmed
        set ranges.
    :return: Trimmed set ranges and fracture labels using integer-like string
        set names plus the background label.

    Examples
    --------
    Tight clusters do not always justify keeping the full original set range.
    Here the edge fractures at 10° and 30° are reclassified as background after
    trimming the set to retain 60% of its assigned total length in the tight
    core of the cluster.

    >>> azimuths = np.array([10.0, 12.0, 14.0, 30.0])
    >>> lengths = np.array([1.0, 3.0, 3.0, 3.0])
    >>> trimmed_ranges, labels = trim_azimuth_set_ranges(
    ...     azimuths,
    ...     lengths,
    ...     ((10.0, 30.0),),
    ...     retained_length_fraction=0.6,
    ... )
    >>> trimmed_ranges
    ((12.0, 14.0),)
    >>> tuple(str(label) for label in labels)
    ('background', '0', '0', 'background')

    Axial wraparound is preserved, so a set around 0°/180° can still be
    trimmed without losing its circular meaning.

    >>> azimuths = np.array([178.0, 179.0, 1.0, 2.0, 20.0])
    >>> lengths = np.array([4.0, 4.0, 4.0, 4.0, 1.0])
    >>> trimmed_ranges, labels = trim_azimuth_set_ranges(
    ...     azimuths,
    ...     lengths,
    ...     ((178.0, 20.0),),
    ...     retained_length_fraction=0.75,
    ... )
    >>> trimmed_ranges[0][0] > trimmed_ranges[0][1]
    True
    >>> tuple(str(label) for label in labels)
    ('0', '0', '0', '0', 'background')
    """
    azimuths = np.asarray(azimuths_deg, dtype=float)
    lengths = np.asarray(length_array, dtype=float)

    if azimuths.shape != lengths.shape:
        raise ValueError("length_array must have the same shape as azimuths_deg.")
    if len(set_ranges) == 0:
        raise ValueError("set_ranges must not be empty.")

    set_names = tuple(str(idx) for idx in range(len(set_ranges)))
    initial_labels = np.array(
        [
            determine_set(
                value=azimuth,
                value_ranges=set_ranges,
                set_names=set_names,
                loop_around=True,
                null_set=background_set_name,
            )
            for azimuth in azimuths
        ]
    )

    trimmed_ranges = []
    for set_name, range_tuple in zip(set_names, set_ranges):
        set_mask = initial_labels == set_name
        if not np.any(set_mask):
            trimmed_ranges.append(range_tuple)
            continue
        unwrapped_azimuths = _unwrap_axial_range(azimuths[set_mask], range_tuple)
        trimmed_start, trimmed_end = _trim_unwrapped_cluster_range_by_length_fraction(
            unwrapped_azimuths,
            lengths[set_mask],
            retained_length_fraction,
        )
        trimmed_ranges.append((trimmed_start % 180.0, trimmed_end % 180.0))

    trimmed_ranges_tuple = tuple(trimmed_ranges)
    trimmed_labels = np.array(
        [
            determine_set(
                value=azimuth,
                value_ranges=trimmed_ranges_tuple,
                set_names=set_names,
                loop_around=True,
                null_set=background_set_name,
            )
            for azimuth in azimuths
        ]
    )
    return trimmed_ranges_tuple, trimmed_labels


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
        >>> lengths = np.array([1.0, 9.0])
        >>> centers, ranges = automatic_azimuth_sets(
        ...     azimuths,
        ...     lengths,
        ...     n_sets=1,
        ...     random_state=0,
        ... )
        >>> np.round(centers, 1)
        array([18.1])
        >>> ranges
        ((0.0, 20.0),)

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
        >>> np.round(np.sort(centers), 1)
        array([ 90., 180.])
        >>> tuple(sorted((round(start, 1), round(end, 1)) for start, end in ranges))
        ((88.0, 92.0), (178.0, 2.0))
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
