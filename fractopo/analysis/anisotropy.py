"""
Anisotropy of connectivity determination utilities.
"""
from typing import Tuple

import matplotlib
import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline


def determine_anisotropy_classification(branch_classification: str) -> int:
    """
    Return value based on branch classification.

    Only C-C branches have a value, but this can be changed here.
    Classification can differ from 'C - C', 'C - I', 'I - I' (e.g. 'C - E') in
    which case a value (0) is still returned.

    :param branch_classification: Branch classification string.
    :return: Classification encoded as integer.

    E.g.

    >>> determine_anisotropy_classification("C - C")
    1

    >>> determine_anisotropy_classification("C - E")
    0
    """
    if branch_classification not in ("C - C", "C - I", "I - I"):
        return 0
    if branch_classification == "C - C":
        return 1
    if branch_classification == "C - I":
        return 0
    if branch_classification == "I - I":
        return 0
    return 0


def determine_anisotropy_sum(
    azimuth_array: np.ndarray,
    branch_types: np.ndarray,
    length_array: np.ndarray,
    sample_intervals: np.ndarray = np.arange(0, 179, 30),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Determine the sums of branch anisotropies.

    :param azimuth_array: Array of branch azimuth values.
    :param branch_types: Array of branch type classication strings.
    :param length_array: Array of branch lengths.
    :param sample_intervals: Array of the sampling intervals.
    :return: Sums of branch anisotropies.

    E.g.

    >>> from pprint import pprint
    >>> azimuths = np.array([20, 50, 60, 70])
    >>> lengths = np.array([2, 5, 6, 7])
    >>> branch_types = np.array(["C - C", "C - C", "C - C", "C - I"])
    >>> pprint(determine_anisotropy_sum(azimuths, branch_types, lengths))
    (array([ 8.09332329, 11.86423103, 12.45612765,  9.71041492,  5.05739707,
            2.15381611]),
     array([  0,  30,  60,  90, 120, 150]))

    """
    # Array of anisotropy_array-s
    anisotropy_arrays = np.array(
        [
            determine_anisotropy_value(azimuth, branch_type, length, sample_intervals)
            for azimuth, branch_type, length in zip(
                azimuth_array, branch_types, length_array
            )
        ]
    )
    return anisotropy_arrays.sum(axis=0), sample_intervals


def determine_anisotropy_value(
    azimuth: float,
    branch_type: str,
    length: float,
    sample_intervals: np.ndarray = np.arange(0, 179, 30),
) -> np.ndarray:
    """
    Calculate anisotropy of connectivity for a branch.

    Based on azimuth, branch_type and length.  Value is calculated for preset
    angles (sample_intervals = np.arange(0, 179, 30))

    E.g.

    Anisotropy for a C-C classified branch:

    >>> determine_anisotropy_value(50, "C - C", 1)
    array([0.64278761, 0.93969262, 0.98480775, 0.76604444, 0.34202014,
           0.17364818])

    Other classification for branch:

    >>> determine_anisotropy_value(50, "C - I", 1)
    array([0, 0, 0, 0, 0, 0])

    """
    classification = determine_anisotropy_classification(branch_type)
    # CALCULATION
    results = []
    for angle in sample_intervals:
        if classification == 0:
            results.append(0)
            continue
        diff = np.abs(angle - azimuth)
        if diff > 90:
            diff = 180 - max([angle, azimuth]) + min([angle, azimuth])
        cos_diff = np.cos(np.deg2rad(diff))  # type: ignore
        result = length * classification * cos_diff
        results.append(result)
    # print(results)
    return np.array(results)


def plot_anisotropy_plot(
    anisotropy_sum: np.ndarray,
    sample_intervals: np.ndarray,
    # label: str,
    # color: Optional[str] = None,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:  # type: ignore
    """
    Plot anisotropy values to new figure.
    """
    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    plot_anisotropy_ax(
        anisotropy_sum=anisotropy_sum, ax=ax, sample_intervals=sample_intervals
    )
    return fig, ax


def plot_anisotropy_ax(
    anisotropy_sum: np.ndarray,
    ax: plt.PolarAxes,
    sample_intervals: np.ndarray = np.arange(0, 179, 30),
):
    """
    Plot a styled anisotropy of connectivity to a given PolarAxes.

    Spline done with:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
    """
    double_anisotropy = np.concatenate([anisotropy_sum, anisotropy_sum])
    opp_angles = [i + 180 for i in sample_intervals]
    angles = list(sample_intervals) + opp_angles
    # PLOT SETUP
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    max_aniso = max(anisotropy_sum)

    for theta, r in zip(angles, double_anisotropy):
        theta = np.deg2rad(theta)  # type: ignore
        arrowstyle = patches.ArrowStyle.CurveB(head_length=1, head_width=0.5)
        ax.annotate(
            "",
            xy=(theta, r),
            xytext=(theta, 0),
            arrowprops=dict(
                edgecolor="black", facecolor="seashell", arrowstyle=arrowstyle
            ),
        )
    ax.axis("off")
    # CREATE CURVED STRUCTURE AROUND SCATTER AND ARROWS
    angles.append(359.999)
    double_anisotropy = np.concatenate([double_anisotropy, double_anisotropy[0:1]])
    angles_arr = np.array(angles)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
    theta = np.deg2rad(angles_arr)  # type: ignore
    cs = CubicSpline(theta, double_anisotropy, bc_type="periodic")
    xnew = np.linspace(theta.min(), theta.max(), 300)
    power_smooth = cs(xnew)
    ax.plot(xnew, power_smooth, linewidth=1.5, color="black")

    circle = patches.Circle(
        (0, 0),
        0.0025 * max_aniso,
        transform=ax.transData._b,
        edgecolor="black",
        facecolor="gray",
        zorder=10,
    )
    ax.add_artist(circle)
