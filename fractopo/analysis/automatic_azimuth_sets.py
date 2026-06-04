"""
Automatic detection of azimuth clusters (fracture sets, orientation clusters)
in orientation data, using circular embeddings and clustering algorithms.
"""

import numpy as np
from beartype import beartype
from beartype.typing import Optional, Tuple
from sklearn.cluster import KMeans


@beartype
def _validate_automatic_azimuth_set_inputs(
    azimuths_deg: np.ndarray, n_sets: Optional[int]
) -> np.ndarray:
    """
    Validate inputs for automatic azimuth set detection.
    """
    azimuths = np.asarray(azimuths_deg, dtype=float)
    if azimuths.ndim != 1:
        raise ValueError("azimuths_deg must be a one-dimensional array.")
    if azimuths.size == 0:
        raise ValueError("azimuths_deg must not be empty.")
    if not np.all(np.isfinite(azimuths)):
        raise ValueError("azimuths_deg must contain only finite values.")
    if n_sets is None or n_sets < 1:
        raise ValueError(
            "n_sets must be specified and >0 (auto-detection not implemented yet)."
        )
    if n_sets > azimuths.size:
        raise ValueError("n_sets cannot be larger than the number of azimuths.")
    return azimuths


@beartype
def _azimuths_to_axial_unit_vectors(azimuths_deg: np.ndarray) -> np.ndarray:
    """
    Represent axial azimuths on the unit circle.

    Axial data wraps over 180 degrees, so the angle is doubled before the
    circular embedding. This makes 0° and 180° equivalent.
    """
    doubled_azimuths_rad = np.deg2rad((azimuths_deg % 180) * 2)
    return np.column_stack((np.cos(doubled_azimuths_rad), np.sin(doubled_azimuths_rad)))


@beartype
def _cluster_centers_to_axial_azimuths(cluster_centers: np.ndarray) -> np.ndarray:
    """
    Convert doubled-angle cluster centers back to axial azimuths in [0, 180).
    """
    doubled_center_azimuths_rad = np.arctan2(
        cluster_centers[:, 1], cluster_centers[:, 0]
    )
    return (np.rad2deg(doubled_center_azimuths_rad) / 2) % 180


@beartype
def automatic_azimuth_sets(
    azimuths_deg: np.ndarray,
    n_sets: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Automatically detect fracture sets in axial azimuth data.

    Parameters
    ----------
    azimuths_deg : np.ndarray
        1D array of axial azimuths in degrees.
    n_sets : int or None
        Number of sets to find. If None, raises ValueError because automatic
        set-count detection is not yet implemented.
    random_state : int
        Random state passed to sklearn.cluster.KMeans.

    Returns
    -------
    set_labels : np.ndarray
        Cluster label for each azimuth.
    set_centers : np.ndarray
        Center (mean axial azimuth, degrees in [0, 180)) of each set.

    Notes
    -----
    Axial azimuths are circular data where 0° and 180° are equivalent. To
    respect this topology, clustering is performed on doubled-angle unit-circle
    coordinates before converting cluster centers back to axial azimuths.
    """
    azimuths = _validate_automatic_azimuth_set_inputs(
        azimuths_deg=azimuths_deg, n_sets=n_sets
    )
    unit_vectors = _azimuths_to_axial_unit_vectors(azimuths)
    kmeans = KMeans(n_clusters=n_sets, random_state=random_state, n_init=10)
    set_labels = kmeans.fit_predict(unit_vectors)
    set_centers_deg = _cluster_centers_to_axial_azimuths(kmeans.cluster_centers_)
    return set_labels, set_centers_deg
