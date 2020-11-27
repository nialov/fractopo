"""
Contains general calculation and plotting tools.
"""
import powerlaw
import geopandas as gpd
import pandas as pd
import math
from typing import Tuple, Dict, List, Union
import math
from textwrap import wrap
import os
from pathlib import Path
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import shapely
import ternary
from shapely import strtree
from shapely.geometry import LineString, Point
from shapely.ops import linemerge
from shapely import prepared
import logging

# Own code imports
from fractopo.analysis import target_area as ta, config

style = config.styled_text_dict
prop = config.styled_prop

# Columns for report_df

CC_branch = "C - C"
CE_branch = "C - E"
CI_branch = "C - I"
IE_branch = "I - E"
II_branch = "I - I"
EE_branch = "E - E"
Error_branch = "Error"
X_node = "X"
Y_node = "Y"
I_node = "I"
E_node = "E"

CONNECTION_COLUMN = "Connection"
CLASS_COLUMN = "Class"
GEOMETRY_COLUMN = "geometry"


def determine_set(
    value: Union[float, int], value_range: Tuple[float, float], loop_around: bool
):
    """
    Determines if value fits within the given value_range.

    If the value range has the possibility of looping around loop_around can be
    set to true.

    >>> determine_set(5, (0, 10), False)
    True

    >>> determine_set(5, (175, 15), True)
    True
    """
    if loop_around:
        if value_range[0] > value_range[1]:
            # Loops around case
            if (value >= value_range[0]) | (value <= value_range[1]):
                return True
    if value_range[0] <= value <= value_range[1]:
        return True


def determine_azimuth(line: LineString, halved: bool) -> float:
    """
    Calculates azimuth of given line.

    If halved -> return is in range [0, 180]
    Else -> [0, 360]

    e.g.:
    Accepts LineString

    >>> determine_azimuth(LineString([(0, 0), (1, 1)]), True)
    45.0

    >>> determine_azimuth(LineString([(0, 0), (0, 1)]), True)
    0.0

    >>> determine_azimuth(LineString([(0, 0), (-1, -1)]), False)
    225.0

    >>> determine_azimuth(LineString([(0, 0), (-1, -1)]), True)
    45.0
    """
    coord_list = list(line.coords)
    start_x = coord_list[0][0]
    start_y = coord_list[0][1]
    end_x = coord_list[-1][0]
    end_y = coord_list[-1][1]
    azimuth = 90 - math.degrees(math.atan2((end_y - start_y), (end_x - start_x)))
    if azimuth < 0:
        azimuth = azimuth + 360
    if azimuth > 360:
        azimuth -= 360
    if halved:
        azimuth = azimuth if not azimuth >= 180 else azimuth - 180
    return azimuth


def calc_strike(dip_direction):
    """
    Calculates strike from dip direction. Right-handed rule.
    Examples:

    >>> calc_strike(50)
    320

    >>> calc_strike(180)
    90

    :param dip_direction: Dip direction of plane
    :type dip_direction: float
    :return: Strike of plane
    :rtype: float
    """
    strike = dip_direction - 90
    if strike < 0:
        strike = 360 + strike
    return strike


def azimu_half(degrees):
    """
    Transforms azimuths from 180-360 range to range 0-180

    :param degrees: Degrees in range 0 - 360
    :type degrees: float
    :return: Degrees in range 0 - 180
    :rtype: float
    """
    if degrees >= 180:
        degrees = degrees - 180
    return degrees


def sd_calc(data):
    """
    TODO: Wrong results atm. Needs to take into account real length, not just orientation of unit vector.
    Calculates standard deviation for radial data (degrees)

    E.g.

    >>> sd_calc(np.array([2, 5, 8]))
    (3.0, 5.00)

    :param data: Array of degrees
    :type data: np.ndarray
    :return: Standard deviation
    :rtype: float
    """

    n = len(data)
    if n <= 1:
        return 0.0
    mean, sd = avg_calc(data), 0.0
    # calculate stan. dev.
    for el in data:
        diff = abs(mean - float(el))
        if diff > 180:
            diff = 360 - diff

        sd += diff ** 2
    sd = math.sqrt(sd / float(n - 1))

    return sd, mean


def avg_calc(data):
    # TODO: Should take length into calculations.......................... not real average atm
    n, mean = len(data), 0.0

    if n <= 1:
        return data[0]
    vectors = []
    # calculate average
    for el in data:
        rad = math.radians(el)
        v = np.array([math.cos(rad), math.sin(rad)])
        vectors.append(v)

    meanv = np.mean(np.array(vectors), axis=0)

    mean = math.degrees(math.atan2(meanv[1], meanv[0]))
    # print(mean)
    if mean < 0:
        mean = 360 + mean
    return mean


def tern_yi_func(c, x):
    """
    Function for plotting *Connections per branch* line to branch ternary plot. Absolute values.
    """
    temp = 6 * (1 - 0.5 * c)
    temp2 = 3 - (3 / 2) * c
    temp3 = 1 + c / temp
    y = (c + 3 * c * x) / (temp * temp3) - (4 * x) / (temp2 * temp3)
    i = 1 - x - y
    return x, i, y


def tern_plot_the_fing_lines(tax, cs_locs=(1.3, 1.5, 1.7, 1.9)):
    """
    Plots *connections per branch* parameter to XYI-plot.

    :param tax: Ternary axis to plot to
    :type tax: ternary.TernaryAxesSubplot
    :param cs_locs: Pre-determined locations for lines
    :type cs_locs: tuple
    """

    def tern_find_last_x(c, x_start=0):
        x, i, y = tern_yi_func(c, x_start)
        while y > 0:
            x_start += 0.01
            x, i, y = tern_yi_func(c, x_start)
        return x

    def tern_yi_func_perc(c, x):
        temp = 6 * (1 - 0.5 * c)
        temp2 = 3 - (3 / 2) * c
        temp3 = 1 + c / temp
        y = (c + 3 * c * x) / (temp * temp3) - (4 * x) / (temp2 * temp3)
        i = 1 - x - y
        return 100 * x, 100 * i, 100 * y

    for c in cs_locs:
        last_x = tern_find_last_x(c)
        x1 = 0
        x2 = last_x
        point1 = tern_yi_func_perc(c, x1)
        point2 = tern_yi_func_perc(c, x2)
        tax.line(
            point1,
            point2,
            alpha=0.4,
            color="k",
            zorder=-5,
            linestyle="dashed",
            linewidth=0.5,
        )
        ax = plt.gca()
        rot = 6.5
        rot2 = 4.5
        ax.text(x=55, y=59, s=r"$C_B = 1.3$", fontsize=10, rotation=rot, ha="center")
        ax.text(x=61, y=50, s=r"$C_B = 1.5$", fontsize=10, rotation=rot, ha="center")
        ax.text(
            x=68.5,
            y=36.6,
            s=r"$C_B = 1.7$",
            fontsize=10,
            rotation=rot2 + 1,
            ha="center",
        )
        ax.text(x=76, y=17, s=r"$C_B = 1.9$", fontsize=10, rotation=rot2, ha="center")


def tern_plot_branch_lines(tax):
    """
    Plot line of random assignment of nodes to branches to a given branch ternary tax.
    Line positions taken from NetworkGT open source code.
    Credit to:
    https://github.com/BjornNyberg/NetworkGT

    :param tax: Ternary axis to plot to
    :type tax: ternary.TernaryAxesSubplot
    """
    ax = tax.get_axes()
    tax.boundary()
    points = [
        (0, 1, 0),
        (0.01, 0.81, 0.18),
        (0.04, 0.64, 0.32),
        (0.09, 0.49, 0.42),
        (0.16, 0.36, 0.48),
        (0.25, 0.25, 0.5),
        (0.36, 0.16, 0.48),
        (0.49, 0.09, 0.42),
        (0.64, 0.04, 0.32),
        (0.81, 0.01, 0.18),
        (1, 0, 0),
    ]
    for idx, p in enumerate(points):
        points[idx] = points[idx][0] * 100, points[idx][1] * 100, points[idx][2] * 100

    text_loc = [(0.37, 0.2), (0.44, 0.15), (0.52, 0.088), (0.64, 0.055), (0.79, 0.027)]
    for idx, t in enumerate(text_loc):
        text_loc[idx] = t[0] * 100, t[1] * 100
    text = [r"$C_B = 1.0$", r"$1.2$", r"$1.4$", r"$1.6$", r"$1.8$"]
    rots = [-61, -44, -28, -14, -3]
    # rot = -65
    for t, l, rot in zip(text, text_loc, rots):
        ax.annotate(t, xy=l, fontsize=9, rotation=rot)
        # rot += 17
    tax.plot(
        points,
        linewidth=1.5,
        marker="o",
        color="k",
        linestyle="dashed",
        markersize=3,
        zorder=-5,
        alpha=0.6,
    )


def calc_xlims(lineframe) -> Tuple[float, float]:
    left = lineframe.length.min() / 50
    right = lineframe.length.max() * 50
    return left, right


def calc_ylims(lineframe) -> Tuple[float, float]:
    # TODO: Take y series instead of while dataframe...
    top = lineframe.y.max() * 50
    bottom = lineframe.y.min() / 50
    return top, bottom


def define_length_set(length: float, set_df: pd.DataFrame) -> str:
    """
    Defines sets based on the length of the traces or branches.
    """
    if length < 0:
        raise ValueError("length value wasnt positive.\n Value: {length}")

    # Set_num is -1 when length is in no length set range
    set_label = -1
    for _, row in set_df.iterrows():
        set_range: Tuple[float, float] = row.LengthSetLimits
        # Checks if length degree value is within the length set range
        if set_range[0] <= length <= set_range[1]:
            set_label = row.LengthSet

    return str(set_label)


def construct_length_distribution_base(
    lineframe: gpd.GeoDataFrame,
    areaframe: gpd.GeoDataFrame,
    name: str,
    group: str,
    cut_off_length=1,
    using_branches=False,
):
    """
    Helper function to construct TargetAreaLines to a pandas DataFrame using apply().
    """
    ld = ta.TargetAreaLines(
        lineframe, areaframe, name, group, using_branches, cut_off_length
    )
    return ld


def curviness(linestring):
    try:
        coords = list(linestring.coords)
    except NotImplementedError:
        return np.NaN
    df = pd.DataFrame(columns=["azimu", "length"])
    for i in range(len(coords) - 1):
        start = Point(coords[i])
        end = Point(coords[i + 1])
        l = LineString([start, end])
        azimu = calc_azimu(l)
        # halved = tools.azimu_half(azimu)
        length = l.length
        addition = {
            "azimu": azimu,
            "length": length,
        }  # Addition to DataFrame with fit x and fit y values
        df = df.append(addition, ignore_index=True)

    std = sd_calc(df.azimu.values.tolist())
    azimu_std = std

    return azimu_std


def curviness_initialize_sub_plotting(filecount, ncols=4):
    nrows = filecount
    width = 20
    height = (width / 4) * nrows
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(width, height),
        gridspec_kw=dict(wspace=0.45, hspace=0.3),
    )
    fig.patch.set_facecolor("#CDFFE6")
    return fig, axes, nrows, ncols


def plot_curv_plot(lineframe, ax=plt.gca(), name=""):
    lineframe["curviness"] = lineframe.geometry.apply(curviness)

    # labels = ["{0} - {1}".format(i, i + 9) for i in range(0, 180, 9)]
    lineframe["group"] = pd.cut(lineframe.halved, range(0, 181, 30), right=False)

    sns.boxplot(data=lineframe, x="curviness", y="group", notch=True, ax=ax)
    ax.set_title(name, fontsize=14, fontweight="heavy", fontfamily="Times New Roman")
    ax.set_ylabel("Set (°)", fontfamily="Times New Roman")
    ax.set_xlabel("Curvature (°)", fontfamily="Times New Roman", style="italic")
    ax.grid(True, color="k", linewidth=0.3)


def prepare_geometry_traces(traceframe):
    """
    Prepare tracefrace geometries for a spatial analysis (faster).
    The try-except clause is required due to QGIS differences in the shapely package

    :param traceframe:
    :type traceframe:
    :return:
    :rtype:
    """
    traces = traceframe.geometry.values
    traces = np.asarray(traces).tolist()  # type: ignore
    trace_col = shapely.geometry.MultiLineString(traces)
    prep_col = prepared.prep(trace_col)
    return prep_col, trace_col


def make_point_tree(traceframe):
    points = []
    for idx, row in traceframe.iterrows():
        sp = row.startpoint
        ep = row.endpoint
        points.extend([sp, ep])
    tree = strtree.STRtree(points)
    return tree


def initialize_ternary_points(ax, tax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    tax.boundary(linewidth=1.5)
    tax.gridlines(linewidth=0.1, multiple=20, color="grey", alpha=0.6)
    tax.ticks(
        axis="lbr",
        linewidth=1,
        multiple=20,
        offset=0.035,
        tick_formats="%d%%",
        fontsize="small",
    )
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")
    fontsize = 25
    fdict = {
        "path_effects": [path_effects.withStroke(linewidth=3, foreground="k")],
        "color": "white",
        "family": "Calibri",
        "size": fontsize,
        "weight": "bold",
    }
    ax.text(-0.1, -0.03, "Y", transform=ax.transAxes, fontdict=fdict)
    ax.text(1.03, -0.03, "X", transform=ax.transAxes, fontdict=fdict)
    ax.text(0.5, 1.07, "I", transform=ax.transAxes, fontdict=fdict, ha="center")
    # tax.set_title(name, x=0.1, y=1, fontsize=14, fontweight='heavy', fontfamily='Times New Roman')


def initialize_ternary_branches_points(ax, tax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    tax.boundary(linewidth=1.5)
    tax.gridlines(linewidth=0.9, multiple=20, color="black", alpha=0.6)
    tax.ticks(
        axis="lbr",
        linewidth=0.9,
        multiple=20,
        offset=0.03,
        tick_formats="%d%%",
        fontsize="small",
        alpha=0.6,
    )
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")
    fontsize = 20
    fdict = {
        "path_effects": [path_effects.withStroke(linewidth=3, foreground="k")],
        "color": "w",
        "family": "Calibri",
        "size": fontsize,
        "weight": "bold",
    }
    ax.text(-0.1, -0.06, "I - C", transform=ax.transAxes, fontdict=fdict)
    ax.text(1.0, -0.06, "C - C", transform=ax.transAxes, fontdict=fdict)
    ax.text(0.5, 1.07, "I - I", transform=ax.transAxes, fontdict=fdict, ha="center")


def setup_ax_for_ld(ax_for_setup, using_branches, indiv_fit=False):
    """
    Function to setup ax for length distribution plots.

    :param ax_for_setup: Ax to setup.
    :type ax_for_setup: matplotlib.axes.Axes
    :param using_branches: Are the lines branches or traces.
    :type using_branches: bool
    """
    #
    ax = ax_for_setup
    # LABELS
    label = "Branch length $(m)$" if using_branches else "Trace Length $(m)$"
    ax.set_xlabel(
        label,
        fontsize="xx-large",
        fontfamily="Calibri",
        style="italic",
        labelpad=16,
    )
    # Individual powerlaw fits are not normalized to area because they aren't
    # multiscale
    ccm_unit = r"$(\frac{1}{m^2})$" if not indiv_fit else ""
    ax.set_ylabel(
        "Complementary Cumulative Number " + ccm_unit,
        fontsize="xx-large",
        fontfamily="Calibri",
        style="italic",
    )
    # TICKS
    plt.xticks(color="black", fontsize="x-large")
    plt.yticks(color="black", fontsize="x-large")
    plt.tick_params(axis="both", width=1.2)
    # LEGEND
    handles, labels = ax.get_legend_handles_labels()
    labels = ["\n".join(wrap(l, 13)) for l in labels]
    lgnd = plt.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(1.37, 1.02),
        ncol=2,
        columnspacing=0.3,
        shadow=True,
        prop={"family": "Calibri", "weight": "heavy", "size": "large"},
    )
    for lh in lgnd.legendHandles:
        # lh._sizes = [750]
        lh.set_linewidth(3)
    ax.grid(zorder=-10, color="black", alpha=0.5)


def report_powerlaw_fit_statistics_df(
    name: str,
    fit: powerlaw.Fit,
    report_df: pd.DataFrame,
    using_branches: bool,
    lengths: Union[np.ndarray, pd.Series],
):
    """
    Writes powerlaw module Fit object statistics to a given report_df. Included
    powerlaw, lognormal and exponential fit statistics and parameters.
    """
    powerlaw_exponent = -(fit.power_law.alpha - 1)
    # traces_or_branches = "branches" if using_branches else "traces"

    powerlaw_sigma = fit.power_law.sigma
    powerlaw_cutoff = fit.power_law.xmin
    powerlaw_manual = str(fit.fixed_xmin)
    lognormal_mu = fit.lognormal.mu
    lognormal_sigma = fit.lognormal.sigma
    exponential_lambda = fit.exponential.Lambda
    R, p = fit.distribution_compare("power_law", "lognormal")
    powerlaw_vs_lognormal_r = R
    powerlaw_vs_lognormal_p = p
    R, p = fit.distribution_compare("power_law", "exponential")
    powerlaw_vs_exponential_r = R
    powerlaw_vs_exponential_p = p
    R, p = fit.distribution_compare("lognormal", "exponential")
    lognormal_vs_exponential_r = R
    lognormal_vs_exponential_p = p

    remaining_percent_of_data = (sum(lengths > fit.power_law.xmin) / len(lengths)) * 100

    report_df = report_df.append(
        {
            ReportDfCols.name: name,
            ReportDfCols.powerlaw_exponent: powerlaw_exponent,
            ReportDfCols.powerlaw_sigma: powerlaw_sigma,
            ReportDfCols.powerlaw_cutoff: powerlaw_cutoff,
            ReportDfCols.powerlaw_manual: powerlaw_manual,
            ReportDfCols.lognormal_mu: lognormal_mu,
            ReportDfCols.lognormal_sigma: lognormal_sigma,
            ReportDfCols.exponential_lambda: exponential_lambda,
            ReportDfCols.powerlaw_vs_lognormal_r: powerlaw_vs_lognormal_r,
            ReportDfCols.powerlaw_vs_lognormal_p: powerlaw_vs_lognormal_p,
            ReportDfCols.powerlaw_vs_exponential_r: powerlaw_vs_exponential_r,
            ReportDfCols.powerlaw_vs_exponential_p: powerlaw_vs_exponential_p,
            ReportDfCols.lognormal_vs_exponential_r: lognormal_vs_exponential_r,
            ReportDfCols.lognormal_vs_exponential_p: lognormal_vs_exponential_p,
            ReportDfCols.lognormal_vs_exponential_p: lognormal_vs_exponential_p,
            ReportDfCols.remaining_percent_of_data: remaining_percent_of_data,
        },
        ignore_index=True,
    )
    return report_df


def plotting_directories(results_folder, name):
    """
    Creates plotting directories and handles FileExistsErrors when raised.

    :param results_folder: Base folder to create plots_{name} folder to.
    :type results_folder: str
    :param name: Analysis name.
    :type name: str
    :return: Newly made path to plotting directory where all plots will be saved to.
    :rtype: str
    """
    plotting_directory = f"{results_folder}/plots_{name}"
    try:
        try:
            os.mkdir(Path(plotting_directory))
        except FileExistsError:
            print("Earlier plots exist. Overwriting old ones.")
            return plotting_directory
        os.mkdir(Path(f"{plotting_directory}/age_relations"))
        os.mkdir(Path(f"{plotting_directory}/age_relations/indiv"))
        os.mkdir(Path(f"{plotting_directory}/length_age_relations"))
        os.mkdir(Path(f"{plotting_directory}/length_age_relations/indiv"))
        os.mkdir(Path(f"{plotting_directory}/anisotropy"))
        os.mkdir(Path(f"{plotting_directory}/anisotropy/indiv"))
        os.mkdir(Path(f"{plotting_directory}/azimuths"))
        os.mkdir(Path(f"{plotting_directory}/azimuths/indiv"))
        os.mkdir(Path(f"{plotting_directory}/azimuths/equal_radius"))
        os.mkdir(Path(f"{plotting_directory}/azimuths/equal_radius/traces"))
        os.mkdir(Path(f"{plotting_directory}/azimuths/equal_radius/branches"))

        os.mkdir(Path(f"{plotting_directory}/azimuths/equal_area"))
        os.mkdir(Path(f"{plotting_directory}/azimuths/equal_area/traces"))
        os.mkdir(Path(f"{plotting_directory}/azimuths/equal_area/branches"))

        os.mkdir(Path(f"{plotting_directory}/azimuths/indiv/equal_radius"))
        os.mkdir(Path(f"{plotting_directory}/azimuths/indiv/equal_radius/traces"))
        os.mkdir(Path(f"{plotting_directory}/azimuths/indiv/equal_radius/branches"))

        os.mkdir(Path(f"{plotting_directory}/azimuths/indiv/equal_area"))
        os.mkdir(Path(f"{plotting_directory}/azimuths/indiv/equal_area/traces"))
        os.mkdir(Path(f"{plotting_directory}/azimuths/indiv/equal_area/branches"))

        os.mkdir(Path(f"{plotting_directory}/branch_class"))
        os.mkdir(Path(f"{plotting_directory}/branch_class/indiv"))
        os.mkdir(Path(f"{plotting_directory}/length_distributions"))
        os.mkdir(Path(f"{plotting_directory}/length_distributions/branches"))
        os.mkdir(
            Path(f"{plotting_directory}/length_distributions/branches/predictions")
        )
        os.mkdir(Path(f"{plotting_directory}/length_distributions/traces"))
        os.mkdir(Path(f"{plotting_directory}/length_distributions/traces/predictions"))
        os.mkdir(Path(f"{plotting_directory}/length_distributions/indiv"))
        os.mkdir(Path(f"{plotting_directory}/length_distributions/indiv/branches"))
        os.mkdir(Path(f"{plotting_directory}/length_distributions/indiv/traces"))
        os.mkdir(Path(f"{plotting_directory}/topology"))
        os.mkdir(Path(f"{plotting_directory}/topology/branches"))
        os.mkdir(Path(f"{plotting_directory}/topology/traces"))
        os.mkdir(Path(f"{plotting_directory}/xyi"))
        os.mkdir(Path(f"{plotting_directory}/xyi/indiv"))
        os.mkdir(Path(f"{plotting_directory}/hexbinplots"))

    # Should not be needed (shutil.rmtree(Path(f"{plotting_directory}"))).
    # Would run if only SOME of the above folders are present.
    # i.e. Folder creation has failed and same folder is used again or if some folders have been removed and same
    # plotting directory is used again. Edge cases.
    except FileExistsError:
        raise
        # print("Earlier decrepit directories found. Deleting decrepit result-plots folder in plots and remaking.")
        # shutil.rmtree(Path(f"{plotting_directory}"))

    return plotting_directory


def initialize_report_df() -> pd.DataFrame:
    report_df = pd.DataFrame(columns=ReportDfCols.cols)
    return report_df


def setup_powerlaw_axlims(ax: matplotlib.axes.Axes, lineframe_main, powerlaw_cut_off):
    # :type ax: matplotlib.axes.Axes
    # TODO: Very inefficient
    lineframe_main = lineframe_main.copy()
    lineframe_main = lineframe_main.loc[lineframe_main["length"] > powerlaw_cut_off]
    lineframe_main["y"] = lineframe_main["y"] / lineframe_main["y"].max()
    left, right = calc_xlims(lineframe_main)
    top, bottom = calc_ylims(lineframe_main)
    left = left * 5
    right = right / 5
    bottom = bottom * 5
    top = top / 5
    try:
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
    except ValueError:
        # Don't try setting if it errors
        pass


def match_crs(
    first: Union[gpd.GeoSeries, gpd.GeoDataFrame],
    second: Union[gpd.GeoSeries, gpd.GeoDataFrame],
) -> Tuple[
    Union[gpd.GeoSeries, gpd.GeoDataFrame], Union[gpd.GeoSeries, gpd.GeoDataFrame]
]:
    all_crs = [crs for crs in (first.crs, second.crs) if crs is not None]
    if len(all_crs) == 0:
        # No crs in either input
        return first, second
    elif len(all_crs) == 2 and all_crs[0] == all_crs[1]:
        # Two valid and they're the same
        return first, second
    elif len(all_crs) == 1:
        # One valid crs in inputs
        crs = all_crs[0]
        first.crs = crs
        second.crs = crs
        return first, second
    else:
        # Two crs that are not the same
        return first, second
