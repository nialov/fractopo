"""
Configuration file.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import seaborn as sns
from typing import List

# Switch to 'False' to disable the analysis from running
choose_your_analyses = {
    "Branches": True,
    "LengthDistributions": True,
    "Azimuths": True,
    "XYI": True,
    "BranchClassification": True,
    "Topology": True,
    "Cross-cuttingAbutting": True,
    "Anisotropy": True,
    "Hexbin": True,
}

# Half the bin width in azimuth plots
# Applicable when sample size low and lines have preferred orientations i.e.
# discrete fracture sets.
half_the_bin_width = False

# ---------------------------------------------------------
# Plot colors
# ---------------------------------------------------------
# TODO: Plot color setup

# Values used in spatial estimates of intersections
buffer_value = 0.001
snap_value = 0.001

# Lists for columns and units for plotting abundance, size and topological
# parameters
columns_to_plot_branches = [
    "Mean Length",
    "Connections per Branch",
    "Areal Frequency B20",
    "Fracture Intensity B21",
    "Dimensionless Intensity B22",
]
columns_to_plot_traces = [
    "Mean Length",
    "Connections per Trace",
    "Areal Frequency P20",
    "Fracture Intensity P21",
    "Dimensionless Intensity P22",
]
units_for_columns = {
    "Mean Length": "m",
    "Connections per Branch": r"$\frac{1}{n}$",
    "Areal Frequency B20": r"$\frac{1}{m^2}$",
    "Fracture Intensity B21": r"$\frac{m}{m^2}$",
    "Dimensionless Intensity P22": "-",
    "Connections per Trace": r"$\frac{1}{n}$",
    "Areal Frequency P20": r"$\frac{1}{m^2}$",
    "Fracture Intensity P21": r"$\frac{m}{m^2}$",
    "Dimensionless Intensity B22": "-",
}

# Angles used for anisotropy calculations
angles_for_examination = np.arange(0, 179, 30)

# Dictionary for styled text
styled_text_dict = {
    "path_effects": [patheffects.withStroke(linewidth=3, foreground="k")],
    "color": "white",
}

# Dictionary for a styled prop
styled_prop = dict(
    boxstyle="round",
    pad=0.6,
    facecolor="wheat",
    path_effects=[patheffects.SimplePatchShadow(), patheffects.Normal()],
)

# Bounding box with wheat color
props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
prop = dict(boxstyle="square", facecolor="linen", alpha=1, pad=0.45)

# Number of target areas. Should be changed before analysis.
n_ta = -1
# Number of groups. Should be changed before analysis.
n_g = -1
# Target area name list
ta_list: List[str]
ta_list = []
# Group name list
g_list: List[str]
g_list = []


def get_color_dict(unified: bool) -> dict:
    """
    Returns the default color dict, which was setup for the correct number of
    target areas and groups
    (sns.color_palette('dark', n_colors)). Assertations will fail if setup
    hasn't been done.

    :param unified: Whether the cycle for target areas or grouped data is wanted.
    :type unified: bool
    :return: Default dictionary with either target area names or group names as keys
        and colors as values.
    :rtype: dict
    :raise AssertationError: Assertations will fail if setup of target area and
    group counts hasn't been done.

    """
    assert n_ta != -1
    assert n_g != -1
    assert len(ta_list) != 0
    assert len(g_list) != 0
    n_colors = n_g if unified else n_ta
    color_list = sns.color_palette("dark", n_colors)
    name_list = g_list if unified else ta_list
    color_dict = {}
    for color, name in zip(color_list, name_list):
        color_dict[name] = color
    return color_dict


# Used for styling plots
def styling_plots(style):
    """
    Styles matplotlib plots by changing default matplotlib parameters (plt.rc).

    :param style: String to determine how to stylize. Options: 'traces', 'branches', 'gray'
    :type style: str

    """
    plt.rc("font", family="Calibri")
    if style == "traces":
        plt.rc("axes", facecolor="oldlace", linewidth=0.75, grid=True)
    if style == "branches":
        plt.rc("axes", facecolor="whitesmoke", linewidth=0.75, grid=True)
    if style == "gray":
        plt.rc("axes", facecolor="lightgrey", linewidth=0.75, grid=True)
    plt.rc("grid", linewidth=0.75, c="k", alpha=0.5)
    plt.rc("legend", facecolor="white", shadow=False, framealpha=1, edgecolor="k")
    plt.rc("scatter", marker="+")
    plt.rc("figure", edgecolor="k", frameon=True, figsize=(8, 6))
    plt.rc("xtick", bottom=True)
    plt.rc("ytick", left=True)
    # plt.rc('lines', markersize=16)
    # plt.rc('path', effects=[path_effects.withStroke(linewidth=5, foreground='w')])
    # plt.rc('xlabel', {'alpha':1})


POWERLAW = "powerlaw"
LOGNORMAL = "lognormal"
EXPONENTIAL = "exponential"
