import numpy as np
import fractopo.analysis.config as config
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from typing import Dict, Optional, Tuple, Union


def aniso_get_class_as_value(c) -> int:
    """
    Return value based on branch classification. Only C-C branches have a
    value, but this can be changed here.
    Classification can differ from ('C - C', 'C - I', 'I - I') (e.g. 'C - E')
    in which case a value is still returned.

    E.g.

    >>> aniso_get_class_as_value('C - C')
    1

    >>> aniso_get_class_as_value('C - E')
    0

    """
    if c not in ("C - C", "C - I", "I - I"):
        return 0
    if c == "C - C":
        return 1
    elif c == "C - I":
        return 0
    elif c == "I - I":
        return 0
    else:
        return 0


def aniso_calc_anisotropy(
    azimuth: float,
    branch_type: str,
    length: float,
    sample_intervals: np.ndarray = np.arange(0, 179, 30),
) -> np.ndarray:
    """
    Calculates anisotropy of connectivity for a branch based on azimuth,
    classification and length.
    Value is calculated for preset angles (sample_intervals = np.arange(0, 179,
    30))

    E.g.

    Anisotropy for a C-C classified branch:

    >>> aniso_calc_anisotropy(90, 'C - C', 10)
    array([6.12323400e-16, 5.00000000e+00, 8.66025404e+00, 1.00000000e+01,
           8.66025404e+00, 5.00000000e+00])

    Other classification for branch:

    >>> aniso_calc_anisotropy(90, 'C - I', 10)
    array([0, 0, 0, 0, 0, 0])

    """
    c_value = aniso_get_class_as_value(branch_type)
    # CALCULATION
    results = []
    for angle in sample_intervals:
        if c_value == 0:
            results.append(0)
            continue
        diff = np.abs(angle - azimuth)
        if diff > 90:
            diff = 180 - max([angle, azimuth]) + min([angle, azimuth])
        cos_diff = np.cos(np.deg2rad(diff))
        result = length * c_value * cos_diff
        results.append(result)
    # print(results)
    return np.array(results)


def plot_anisotropy_styled_ax(
    anisotropy_array: np.ndarray,
    ax: plt.PolarAxes,
    sample_intervals: np.ndarray = np.arange(0, 179, 30),
) -> plt.Axes:
    """
    Plots a styled anisotropy of connectivity to a given PolarAxes.

    Spline done with:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
    """
    double_anisotropy = np.concatenate([anisotropy_array, anisotropy_array])
    opp_angles = [i + 180 for i in sample_intervals]
    angles = list(sample_intervals) + opp_angles
    # PLOT SETUP
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    max_aniso = max(anisotropy_array)

    for theta, r in zip(angles, double_anisotropy):
        theta = np.deg2rad(theta)
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
    angles = np.array(angles)

    # TODO: testing CubicSpline
    # And it works!?
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
    theta = np.deg2rad(angles)
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

    return ax
