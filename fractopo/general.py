"""
Contains general calculation and plotting tools.
"""
import logging
import math
from bisect import bisect
from enum import Enum, unique
from itertools import accumulate, chain, zip_longest
from typing import Any, List, Set, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
from fractopo import SetRangeTuple
from matplotlib import patheffects as path_effects, pyplot as plt
from shapely import prepared
from shapely.affinity import scale
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
    box,
)
from sklearn.linear_model import LinearRegression


styled_text_dict = {
    "path_effects": [path_effects.withStroke(linewidth=3, foreground="k")],
    "color": "white",
}
styled_prop = dict(
    boxstyle="round",
    pad=0.6,
    facecolor="wheat",
    path_effects=[path_effects.SimplePatchShadow(), path_effects.Normal()],
)

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
POWERLAW = "powerlaw"
LOGNORMAL = "lognormal"
EXPONENTIAL = "exponential"

NULL_SET = "-1"


@unique
class Col(Enum):

    LENGTH = "length"
    AZIMUTH = "azimuth"
    AZIMUTH_SET = "azimuth_set"
    LENGTH_SET = "length_set"


@unique
class Param(Enum):

    NUMBER_OF_TRACES = "Number of Traces"
    NUMBER_OF_BRANCHES = "Number of Branches"
    TRACE_MEAN_LENGTH = "Trace Mean Length"
    BRANCH_MEAN_LENGTH = "Branch Mean Length"
    CONNECTIONS_PER_BRANCH = "Connections per Branch"
    AREAL_FREQUENCY_B20 = "Areal Frequency B20"
    FRACTURE_INTENSITY_B21 = "Fracture Intensity B21"
    DIMENSIONLESS_INTENSITY_B22 = "Dimensionless Intensity B22"
    CONNECTIONS_PER_TRACE = "Connections per Trace"
    AREAL_FREQUENCY_P20 = "Areal Frequency P20"
    FRACTURE_INTENSITY_P21 = "Fracture Intensity P21"
    DIMENSIONLESS_INTENSITY_P22 = "Dimensionless Intensity P22"

    @classmethod
    def log_scale_columns(cls) -> List[str]:
        return [
            param.value
            for param in (
                cls.TRACE_MEAN_LENGTH,
                cls.BRANCH_MEAN_LENGTH,
                cls.AREAL_FREQUENCY_B20,
                cls.FRACTURE_INTENSITY_B21,
                cls.FRACTURE_INTENSITY_P21,
                cls.AREAL_FREQUENCY_P20,
            )
        ]

    @classmethod
    def get_unit_for_column(cls, column: str) -> str:
        units_for_columns = {
            cls.NUMBER_OF_TRACES.value: "-",
            cls.NUMBER_OF_BRANCHES.value: "-",
            cls.TRACE_MEAN_LENGTH.value: "m",
            cls.BRANCH_MEAN_LENGTH.value: "m",
            cls.CONNECTIONS_PER_BRANCH.value: r"$\frac{1}{n}$",
            cls.CONNECTIONS_PER_TRACE.value: r"$\frac{1}{n}$",
            cls.AREAL_FREQUENCY_B20.value: r"$\frac{1}{m^2}$",
            cls.DIMENSIONLESS_INTENSITY_P22.value: "-",
            cls.AREAL_FREQUENCY_P20.value: r"$\frac{1}{m^2}$",
            cls.FRACTURE_INTENSITY_B21.value: r"$\frac{m}{m^2}$",
            cls.FRACTURE_INTENSITY_P21.value: r"$\frac{m}{m^2}$",
            cls.DIMENSIONLESS_INTENSITY_B22.value: "-",
        }
        assert len(units_for_columns) == len([param for param in cls])
        return units_for_columns[column]


def determine_set(
    value: float,
    value_ranges: SetRangeTuple,
    set_names: Tuple[str, ...],
    loop_around: bool,
) -> np.ndarray:
    """
    Determine which named value range, if any, value is within.

    loop_around defines behavior expected for radial data i.e. when value range
    can loop back around e.g. [160, 50]

    E.g.

    >>> determine_set(10.0, [(0, 20), (30, 160)], ["0-20", "30-160"], False)
    '0-20'

    Example with `loop_around = True`

    >>> determine_set(50.0, [(0, 20), (160, 60)], ["0-20", "160-60"], True)
    '160-60'

    """
    assert len(value_ranges) == len(set_names)
    possible_set_name = [
        set_name
        for set_name, value_range in zip(set_names, value_ranges)
        if is_set(value, value_range, loop_around)
    ]
    if len(possible_set_name) == 0:
        return NULL_SET
    elif len(possible_set_name) == 1:
        return possible_set_name[0]
    else:
        raise ValueError("Expected set value ranges to not overlap.")


def is_set(
    value: Union[float, int], value_range: Tuple[float, float], loop_around: bool
) -> bool:
    """
    Determines if value fits within the given value_range.

    If the value range has the possibility of looping around loop_around can be
    set to true.

    >>> is_set(5, (0, 10), False)
    True

    >>> is_set(5, (175, 15), True)
    True
    """
    if loop_around:
        if value_range[0] > value_range[1]:
            # Loops around case
            if (value >= value_range[0]) | (value <= value_range[1]):
                return True
    if value_range[0] <= value <= value_range[1]:
        return True
    return False


def is_azimuth_close(first: float, second: float, tolerance: float, halved=True):
    """
    Determine are azimuths first and second within tolerance.

    Takes into account the radial nature of azimuths.

    >>> is_azimuth_close(0, 179, 15)
    True

    >>> is_azimuth_close(166, 179, 15)
    True

    >>> is_azimuth_close(20, 179, 15)
    False

    """
    diff = abs(first - second)
    if halved:
        diff = diff if diff <= 90 else 180 - diff
        assert 0 <= diff <= 90
    else:
        diff = diff if diff <= 180 else 360 - diff
        assert 0 <= diff <= 180
    return diff < tolerance


def determine_regression_azimuth(line: LineString) -> float:
    """
    Determine azimuth of line LineString with linear regression.

    A scikit-learn LinearRegression is fitted to the x, y coordinates of the
    given `line` and the azimuth of the fitted linear line is returned.

    The azimuth is returned in range [0, 180].

    E.g.

    >>> line = LineString([(0, 0), (1, 1), (2, 2), (3, 3)])
    >>> determine_regression_azimuth(line)
    45.0

    >>> line = LineString([(-1, -5), (3, 3)])
    >>> determine_regression_azimuth(line)
    26.565051177077994

    >>> line = LineString([(0, 0), (0, 3)])
    >>> determine_regression_azimuth(line)
    0.0

    """
    x, y = zip(*line.coords)
    x = np.array(x).reshape((-1, 1))
    y = np.array(y)
    model = LinearRegression().fit(x, y)
    coefs = model.coef_
    if len(coefs) == 1:
        coef = coefs[0]
    else:
        raise ValueError("Expected model.coef_ to be an array of length 1.")
    if np.isclose(coef, 0.0):
        return 0.0
    azimuth = np.rad2deg(np.arctan(1 / coef))  # type: ignore
    if azimuth < 0:
        azimuth = 90 - np.rad2deg(np.arctan(coef))  # type: ignore
    assert isinstance(azimuth, float) and 0 <= azimuth <= 180
    return azimuth


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
    TODO: Wrong results atm. Needs to take into account real length, not just
    orientation of unit vector.  Calculates standard deviation for radial data
    (degrees)

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
        azimu = determine_azimuth(l, halved=True)
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


def prepare_geometry_traces(trace_series: gpd.GeoSeries) -> prepared.PreparedGeometry:
    """
    Prepare trace_series geometries for a faster spatial analysis.

    Assumes geometries are LineStrings which are consequently collected into
    a single MultiLineString which is prepared with shapely.prepared.prep.

    >>> traces = gpd.GeoSeries([LineString([(0,0), (1, 1)]), LineString([(0,1), (0, -1)])])
    >>> prepare_geometry_traces(traces).context.wkt
    'MULTILINESTRING ((0 0, 1 1), (0 1, 0 -1))'

    """
    traces = trace_series.geometry.values
    traces = np.asarray(traces).tolist()  # type: ignore
    trace_col = MultiLineString(traces)
    prepared_traces = prepared.prep(trace_col)
    return prepared_traces


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


def get_trace_endpoints(trace: LineString,) -> Tuple[Point, Point]:
    """
    Return endpoints (shapely.geometry.Point) of a given LineString
    """
    if not isinstance(trace, LineString):
        raise TypeError(
            "Non LineString geometry passed into get_trace_endpoints.\n"
            f"trace: {trace}"
        )
    return tuple(
        (endpoint for endpoint in (Point(trace.coords[0]), Point(trace.coords[-1]),))
    )


def get_trace_coord_points(trace: LineString) -> List[Point]:
    """
    Get all coordinate Points of a LineString.

    >>> trace = LineString([(0, 0), (2, 0), (3, 0)])
    >>> coord_points = get_trace_coord_points(trace)
    >>> print([p.wkt for p in coord_points])
    ['POINT (0 0)', 'POINT (2 0)', 'POINT (3 0)']

    """
    assert isinstance(trace, LineString)
    return [Point(xy) for xy in trace.coords]


def point_to_xy(point: Point) -> Tuple[float, float]:
    x, y = point.xy
    x, y = [val[0] for val in (x, y)]
    return (x, y)


def determine_general_nodes(
    traces: Union[gpd.GeoSeries, gpd.GeoDataFrame], snap_threshold: float
) -> Tuple[List[Tuple[Point, ...]], List[Tuple[Point, ...]]]:
    """
    Determine points of intersection and endpoints of given trace LineStrings.

    The points are linked to the indexes of the traces.
    """

    if traces.shape[0] == 0 or traces.empty:
        raise ValueError("Expected non-empty dataset in determine_nodes.")
    if not (
        traces.index.values[0] == 0 and traces.index.values[-1] == traces.shape[0] - 1
    ):
        raise ValueError(
            "Expected traces to have a continous index from 0...len(traces) - 1"
        )

    intersect_nodes: List[Tuple[Point, ...]] = []
    endpoint_nodes: List[Tuple[Point, ...]] = []
    spatial_index = traces.geometry.sindex
    for idx, geom in enumerate(traces.geometry.values):
        if not isinstance(geom, LineString):
            # Intersections and endpoints cannot be defined for
            # MultiLineStrings
            # TODO: Or can they? Probably shouldn't
            intersect_nodes.append(())
            endpoint_nodes.append(())
            continue
        # Get trace candidates for intersection
        trace_candidates_idx: List[int] = sorted(  # type: ignore
            list(spatial_index.intersection(geom.bounds))
        )
        # Remove current geometry from candidates
        trace_candidates_idx.remove(idx)
        trace_candidates: gpd.GeoSeries = traces.geometry.iloc[trace_candidates_idx]  # type: ignore
        # trace_candidates.index = trace_candidates_idx
        # TODO: Is intersection enough? Most Y-intersections might be underlapping.
        intersection_geoms = trace_candidates.intersection(geom)
        intersect_nodes.append(
            tuple(determine_valid_intersection_points(intersection_geoms))
        )
        endpoints = tuple(
            (
                endpoint
                for endpoint in get_trace_endpoints(geom)
                if not any(
                    [
                        endpoint.distance(intersection_geom) < snap_threshold
                        for intersection_geom in intersection_geoms
                        if not intersection_geom.is_empty
                    ]
                )
            )
        )
        endpoint_nodes.append(endpoints)
    return intersect_nodes, endpoint_nodes


def determine_valid_intersection_points(
    intersection_geoms: gpd.GeoSeries,
) -> List[Point]:
    assert isinstance(intersection_geoms, gpd.GeoSeries)
    valid_interaction_points = []
    for geom in intersection_geoms.geometry.values:
        if isinstance(geom, Point):
            valid_interaction_points.append(geom)
        elif isinstance(geom, MultiPoint):
            valid_interaction_points.extend([p for p in geom.geoms])
        else:
            pass
    assert all([isinstance(p, Point) for p in valid_interaction_points])
    return valid_interaction_points


def flatten_tuples(
    list_of_tuples: List[Tuple[Any, ...]]
) -> Tuple[List[int], List[Any]]:
    """
    Flatten collection of tuples and return index references.

    Indexes are from original tuple groupings.

    E.g.

    >>> tuples = [(1, 1, 1), (2, 2, 2, 2), (3,)]
    >>> flatten_tuples(tuples)
    ([0, 0, 0, 1, 1, 1, 1, 2], [1, 1, 1, 2, 2, 2, 2, 3])

    """
    accumulated_idxs = list(
        accumulate([len(val_tuple) for idx, val_tuple in enumerate(list_of_tuples)])
    )
    flattened_tuples = list(chain(*list_of_tuples))
    flattened_idx_reference = [
        bisect(accumulated_idxs, idx) for idx in range(len(flattened_tuples))
    ]
    return flattened_idx_reference, flattened_tuples


def determine_node_junctions(
    nodes: List[Tuple[Point, ...]],
    snap_threshold: float,
    snap_threshold_error_multiplier: float,
    error_threshold: int,
) -> Set[int]:
    """

    E.g.

    >>> nodes = [
    ...     (Point(0, 0), Point(1, 1)),
    ...     (Point(1, 1),),
    ...     (Point(5, 5),),
    ...     (Point(0, 0), Point(1, 1)),
    ... ]
    >>> snap_threshold = 0.01
    >>> snap_threshold_error_multiplier = 1.1
    >>> error_threshold = 2
    >>> determine_node_junctions(
    ...     nodes, snap_threshold, snap_threshold_error_multiplier, error_threshold
    ... )
    {0, 1, 3}

    """

    if len(nodes) == 0:
        return set()

    flattened_idx_reference, flattened_node_tuples = flatten_tuples(nodes)

    if len(flattened_node_tuples) == 0:
        return set()

    # Collect nodes into GeoSeries
    flattened_nodes_geoseries = gpd.GeoSeries(flattened_node_tuples)

    # Create spatial index of nodes
    nodes_geoseries_sindex = flattened_nodes_geoseries.sindex

    # Set collection for indexes with junctions
    indexes_with_junctions: Set[int] = set()

    # Iterate over node tuples i.e. points is a tuple with Points
    # The node tuple indexes represent the trace indexes
    for idx, points in enumerate(nodes):
        associated_point_count = len(points)

        # Because node indexes represent traces, we can remove all nodes of the
        # current trace by using the idx.
        other_nodes_geoseries: gpd.GeoSeries = flattened_nodes_geoseries.loc[  # type: ignore
            [idx_reference != idx for idx_reference in flattened_idx_reference]
        ]
        if len(points) == 0:
            continue

        # Iterate over the actual Points of the current trace
        for i, point in enumerate(points):

            # Get node candidates from spatial index
            node_candidates_idx: List[int] = list(  # type: ignore
                nodes_geoseries_sindex.intersection(
                    point.buffer(
                        snap_threshold * snap_threshold_error_multiplier * 10
                    ).bounds
                )
            )

            # Shift returned indexes by associated_point_count to match to
            # correct points
            remaining_idxs = set(other_nodes_geoseries.index.values)

            # Only choose node candidates that are not part of current traces
            # nodes
            node_candidates_idx = [
                val if val <= idx else val - associated_point_count
                for val in node_candidates_idx
                if val in remaining_idxs
            ]
            node_candidates: gpd.GeoSeries = other_nodes_geoseries.iloc[  # type: ignore
                node_candidates_idx
            ]
            # snap_threshold * snap_threshold_error_multiplier represents the
            # threshold for intersection. Actual intersection is not necessary.
            intersection_data = [
                intersecting_point.distance(point)
                < snap_threshold * snap_threshold_error_multiplier
                for intersecting_point in node_candidates.geometry.values
            ]
            assert all([isinstance(val, bool) for val in intersection_data])
            # If no intersects for current node -> continue to next
            if sum(intersection_data) == 0:
                continue

            # Different error thresholds for endpoints and intersections
            if sum(intersection_data) >= error_threshold:
                # Add idx of current trace
                indexes_with_junctions.add(idx)
                # Add other point idxs that correspond to the error
                for other_index in node_candidates.loc[
                    intersection_data
                ].index.to_list():  # type: ignore
                    indexes_with_junctions.add(flattened_idx_reference[other_index])

    return indexes_with_junctions


def zip_equal(*iterables):
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


def create_unit_vector(start_point: Point, end_point: Point) -> np.ndarray:
    """
    Create numpy unit vector from two shapely Points.
    """
    # Convert Point coordinates to (x, y)
    segment_start = point_to_xy(start_point)
    segment_end = point_to_xy(end_point)
    segment_vector = np.array(
        [segment_end[0] - segment_start[0], segment_end[1] - segment_start[1]]
    )
    segment_unit_vector = segment_vector / np.linalg.norm(segment_vector)
    return segment_unit_vector


def compare_unit_vector_orientation(
    vec_1: np.ndarray, vec_2: np.ndarray, threshold_angle: float
):
    """
    If vec_1 and vec_2 are too different in orientation, will return False.
    """
    if np.linalg.norm(vec_1 + vec_2) < np.sqrt(2):
        # If they face opposite side -> False
        return False
    dot_product = np.dot(vec_1, vec_2)
    if np.isclose(dot_product, 1):
        return True
    if 1 < dot_product or dot_product < -1 or np.isnan(dot_product):
        return False
    rad_angle = np.arccos(dot_product)
    deg_angle = np.rad2deg(rad_angle)
    if deg_angle > threshold_angle:
        # If angle between more than threshold_angle -> False
        return False
    return True


def bounding_polygon(geoseries: Union[gpd.GeoSeries, gpd.GeoDataFrame]) -> Polygon:
    """
    Create bounding polygon around GeoSeries.

    The geoseries geometries will always be completely enveloped by the
    polygon. The geometries will not intersect the polygon boundary.


    >>> geom = LineString([(1, 0), (1, 1), (-1, -1)])
    >>> geoseries = gpd.GeoSeries([geom])
    >>> poly = bounding_polygon(geoseries)
    >>> poly.wkt
    'POLYGON ((2 -2, 2 2, -2 2, -2 -2, 2 -2))'
    >>> geoseries.intersects(poly.boundary)
    0    False
    dtype: bool

    """
    total_bounds = geoseries.total_bounds
    bounding_polygon: Polygon = scale(box(*total_bounds), xfact=2, yfact=2)
    if any(geoseries.intersects(bounding_polygon.boundary)):
        bounding_polygon: Polygon = bounding_polygon.buffer(1)
        assert not any(geoseries.intersects(bounding_polygon.boundary))
    assert all(geoseries.within(bounding_polygon))
    return bounding_polygon


def mls_to_ls(multilinestrings: List[MultiLineString]) -> List[LineString]:
    """
    Flattens a list of multilinestrings to a list of linestrings.

    >>> multilinestrings = [
    ...     MultiLineString(
    ...             [
    ...                  LineString([(1, 1), (2, 2), (3, 3)]),
    ...                  LineString([(1.9999, 2), (-2, 5)]),
    ...             ]
    ...                    ),
    ...     MultiLineString(
    ...             [
    ...                  LineString([(1, 1), (2, 2), (3, 3)]),
    ...                  LineString([(1.9999, 2), (-2, 5)]),
    ...             ]
    ...                    ),
    ... ]
    >>> result_linestrings = mls_to_ls(multilinestrings)
    >>> print([ls.wkt for ls in result_linestrings])
    ['LINESTRING (1 1, 2 2, 3 3)', 'LINESTRING (1.9999 2, -2 5)',
    'LINESTRING (1 1, 2 2, 3 3)', 'LINESTRING (1.9999 2, -2 5)']

    """
    linestrings: List[LineString] = []
    for mls in multilinestrings:
        linestrings.extend([ls for ls in mls.geoms])
    if not all([isinstance(ls, LineString) for ls in linestrings]):
        raise ValueError("MultiLineStrings within MultiLineStrings?")
    return linestrings


def crop_to_target_areas(traces: gpd.GeoSeries, areas: gpd.GeoSeries) -> gpd.GeoSeries:
    """
    Crop traces to the area polygons.

    E.g.

    >>> traces = gpd.GeoSeries(
    ...     [LineString([(1, 1), (2, 2), (3, 3)]), LineString([(1.9999, 2), (-2, 5)])]
    ...     )
    >>> areas = gpd.GeoSeries(
    ...     [
    ...            Polygon([(1, 1), (-1, 1), (-1, -1), (1, -1)]),
    ...            Polygon([(-2.5, 6), (-1.9, 6), (-1.9, 4), (-2.5, 4)]),
    ...     ]
    ... )
    >>> cropped_traces = crop_to_target_areas(traces, areas)
    >>> print([trace.wkt for trace in cropped_traces])
    ['LINESTRING (-1.9 4.924998124953124, -2 5)']

    """
    if not all([isinstance(trace, LineString) for trace in traces.geometry.values]):
        logging.error("Expected no MultiLineStrings in crop_to_target_areas.")
    traces, areas = match_crs(traces, areas)
    clipped_traces = gpd.clip(traces, areas)
    clipped_traces_linestrings = [
        trace for trace in clipped_traces if isinstance(trace, LineString)
    ]
    ct_multilinestrings = [
        mls for mls in clipped_traces if isinstance(mls, MultiLineString)
    ]
    as_linestrings = mls_to_ls(ct_multilinestrings)
    return gpd.GeoSeries(clipped_traces_linestrings + as_linestrings)


def is_empty_area(area: gpd.GeoDataFrame, traces: gpd.GeoDataFrame):
    """
    Check if any traces intersect the area(s) in area GeoDataFrame.
    """
    for area in area.geometry.values:
        for trace in traces.geometry.values:
            # Only one trace intersect required. No need to loop
            # through all traces if one found.
            if trace.intersects(area):
                return False
    return True

