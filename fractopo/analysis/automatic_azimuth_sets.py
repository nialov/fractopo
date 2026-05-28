"""
Automatic detection of azimuth clusters (fracture sets, orientation clusters) in orientation data, using Fisher mixture models and clustering algorithms.
"""

from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans


def automatic_azimuth_sets(
    azimuths_deg: np.ndarray,
    n_sets: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Automatically detects fracture sets (clusters of similar orientation) in azimuth data.

    Parameters
    ----------
    azimuths_deg : np.ndarray
        1D array of azimuths (degrees, [0, 180) for axial or [0, 360) for vector data).
    n_sets : int or None
        Number of sets to find. If None, uses the elbow/Knee method (not implemented here, set explicitly).
    random_state : int
        For reproducibility.

    Returns
    -------
    set_labels : np.ndarray
        Cluster label for each azimuth.
    set_centers : np.ndarray
        Center (mean azimuth, deg) of each set (cluster).

    Notes
    -----
    - For circular data, azimuths are better clustered as unit vectors on a circle or sphere.
    - Here, clustering is performed using KMeans on unit circle (cos,sin of azimuth).
    - For best scientific reproducibility, see OCFMD (Fisher Mixture Models), Spherical/KMeans or hierarchical clustering as in cited works.
    """
    # Convert azimuths to radians (for trigonometry)
    azimuths_rad = np.deg2rad(azimuths_deg)
    # Represent each azimuth as a point on the unit circle
    X = np.c_[np.cos(azimuths_rad), np.sin(azimuths_rad)]
    if n_sets is None or n_sets < 1:
        raise ValueError(
            "n_sets must be specified and >0 (auto-detection not implemented yet)."
        )
    kmeans = KMeans(n_clusters=n_sets, random_state=random_state, n_init=10)
    set_labels = kmeans.fit_predict(X)
    # Calculate mean azimuth for each set
    set_centers_cart = kmeans.cluster_centers_
    set_centers_rad = np.arctan2(set_centers_cart[:, 1], set_centers_cart[:, 0])
    set_centers_deg = np.rad2deg(set_centers_rad) % 180  # Keep in [0,180) for axial
    return set_labels, set_centers_deg
