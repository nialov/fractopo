"""
Contains general calculation and plotting tools.
"""
import math
import random
from bisect import bisect
from enum import Enum, unique
from itertools import accumulate, chain, zip_longest
from pathlib import Path
from typing import Any, List, Set, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas.sindex import PyGEOSSTRTreeIndex
from matplotlib import patheffects as path_effects
from matplotlib import pyplot as plt
from shapely import prepared
from shapely.affinity import scale
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.ops import split
from sklearn.linear_model import LinearRegression

from fractopo import BoundsTuple, PointTuple, SetRangeTuple

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

    """
    GeoDataFrame column names for attributes.
    """

    LENGTH = "length"
    AZIMUTH = "azimuth"
    AZIMUTH_SET = "azimuth_set"
    LENGTH_SET = "length_set"
    LENGTH_WEIGHTS = "boundary_weight"
    LENGTH_NON_WEIGHTED = "length non-weighted"


@unique
class Param(Enum):

    """
    Column names for geometric and topological parameters.
    """

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
    TRACE_MEAN_LENGTH_MAULDON = "Trace Mean Length (Mauldon)"
    FRACTURE_INTENSITY_MAULDON = "Fracture Intensity (Mauldon)"
    FRACTURE_DENSITY_MAULDON = "Fracture Density (Mauldon)"
    CONNECTION_FREQUENCY = "Connection Frequency"

    @classmethod
    def log_scale_columns(cls) -> List[str]:
        """
        Return collection of column names that can/should be plotted in log.
        """
        return [
            param.value
            for param in (
                cls.TRACE_MEAN_LENGTH,
                cls.BRANCH_MEAN_LENGTH,
                cls.AREAL_FREQUENCY_B20,
                cls.FRACTURE_INTENSITY_B21,
                cls.FRACTURE_INTENSITY_P21,
                cls.AREAL_FREQUENCY_P20,
                cls.TRACE_MEAN_LENGTH_MAULDON,
                cls.FRACTURE_INTENSITY_MAULDON,
                cls.FRACTURE_DENSITY_MAULDON,
            )
        ]

    @classmethod
    def get_unit_for_column(cls, column: str) -> str:
        """
        Return unit for parameter name.
        """
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
            cls.TRACE_MEAN_LENGTH_MAULDON.value: "m",
            cls.FRACTURE_INTENSITY_MAULDON.value: r"$\frac{m}{m^2}$",
            cls.FRACTURE_DENSITY_MAULDON.value: r"$\frac{1}{m^2}$",
            cls.CONNECTION_FREQUENCY.value: r"$\frac{1}{m^2}$",
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
    Determine if value fits within the given value_range.

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
    Calculate azimuth of given line.

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
    Calculate strike from dip direction. Right-handed rule.

    E.g.:

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
    Transform azimuth from 180-360 range to range 0-180.

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
    Calculate standard deviation for radial data.

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
    """
    Calculate average for radial data.

    TODO: Should take length into calculations.......................... not
    real average atm
    """
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
    Plot Connections Per Branch values to branch ternary plot.
    """
    temp = 6 * (1 - 0.5 * c)
    temp2 = 3 - (3 / 2) * c
    temp3 = 1 + c / temp
    y = (c + 3 * c * x) / (temp * temp3) - (4 * x) / (temp2 * temp3)
    i = 1 - x - y
    return x, i, y


def tern_plot_the_fing_lines(tax, cs_locs=(1.3, 1.5, 1.7, 1.9)):
    """
    Plot *connections per branch* parameter to XYI-plot.

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
    Plot line of random assignment of nodes to a given branch ternary tax.

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
    """
    Calculate x limits for length distribution plot.
    """
    left = lineframe.length.min() / 50
    right = lineframe.length.max() * 50
    return left, right


def calc_ylims(lineframe) -> Tuple[float, float]:
    """
    Calculate y limits for length distribution plot.
    """
    # TODO: Take y series instead of while dataframe...
    top = lineframe.y.max() * 50
    bottom = lineframe.y.min() / 50
    return top, bottom


def define_length_set(length: float, set_df: pd.DataFrame) -> str:
    """
    Define sets based on the length of the traces or branches.
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
    """
    Determine curviness of LineString.

    TODO: Invalid.
    """
    try:
        coords = list(linestring.coords)
    except NotImplementedError:
        return np.NaN
    df = pd.DataFrame(columns=["azimu", "length"])
    for i in range(len(coords) - 1):
        start = Point(coords[i])
        end = Point(coords[i + 1])
        line = LineString([start, end])
        azimu = determine_azimuth(line, halved=True)
        # halved = tools.azimu_half(azimu)
        length = line.length
        addition = {
            "azimu": azimu,
            "length": length,
        }  # Addition to DataFrame with fit x and fit y values
        df = df.append(addition, ignore_index=True)

    std = sd_calc(df.azimu.values.tolist())
    azimu_std = std

    return azimu_std


def prepare_geometry_traces(trace_series: gpd.GeoSeries) -> prepared.PreparedGeometry:
    """
    Prepare trace_series geometries for a faster spatial analysis.

    Assumes geometries are LineStrings which are consequently collected into
    a single MultiLineString which is prepared with shapely.prepared.prep.

    >>> traces = gpd.GeoSeries([LineString([(0,0),(1,1)]),LineString([(0,1),(0,-1)])])
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
    """
    Match crs between two geopandas data structures.
    """
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


def replace_coord_in_trace(
    trace: LineString, index: int, replacement: Point
) -> LineString:
    """
    Replace coordinate Point of LineString at index with replacement Point.
    """
    coord_points = get_trace_coord_points(trace)
    coord_points.pop(index)
    coord_points.insert(index, replacement)
    new_linestring = LineString(coord_points)
    return new_linestring


def get_next_point_in_trace(trace: LineString, point: Point) -> Point:
    """
    Determine next coordinate point towards middle of LineString from point.
    """
    coord_points = get_trace_coord_points(trace)
    assert point in coord_points
    if point == coord_points[-1]:
        return coord_points[-2]
    elif point == coord_points[0]:
        return coord_points[1]
    else:
        raise ValueError("Expected point to be a coord in trace.")


def get_trace_endpoints(
    trace: LineString,
) -> Tuple[Point, Point]:
    """
    Return endpoints (shapely.geometry.Point) of a given LineString.
    """
    if not isinstance(trace, LineString):
        raise TypeError(
            "Non LineString geometry passed into get_trace_endpoints.\n"
            f"trace: {trace}"
        )
    return tuple(
        (
            endpoint
            for endpoint in (
                Point(trace.coords[0]),
                Point(trace.coords[-1]),
            )
        )
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
    """
    Get x and y coordinates of Point.
    """
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
    # spatial_index = traces.geometry.sindex
    try:
        spatial_index = pygeos_spatial_index(traces.geometry)
    except TypeError:
        spatial_index = None
    for idx, geom in enumerate(traces.geometry.values):
        if not isinstance(geom, LineString):
            # Intersections and endpoints cannot be defined for
            # MultiLineStrings
            # TODO: Or can they? Probably shouldn't
            intersect_nodes.append(())
            endpoint_nodes.append(())
            continue
        # Get trace candidates for intersection
        trace_candidates_idx: List[int] = sorted(
            spatial_index_intersection(spatial_index, geom_bounds(geom))
        )
        # Remove current geometry from candidates
        trace_candidates_idx.remove(idx)
        trace_candidates: gpd.GeoSeries = traces.geometry.iloc[trace_candidates_idx]
        # trace_candidates.index = trace_candidates_idx
        # TODO: Is intersection enough? Most Y-intersections might be underlapping.
        intersection_geoms = determine_valid_intersection_points_no_vnode(
            trace_candidates, geom
        )
        intersect_nodes.append(tuple(intersection_geoms))
        endpoints = tuple(
            (
                endpoint
                for endpoint in get_trace_endpoints(geom)
                if not any(
                    [
                        # TODO: Intersection results in inaccurate geoms ->
                        # tolerance is actually close to a snap_threshold
                        # This represents a hard limit to snapping threshold.
                        np.isclose(endpoint.distance(intersection_geom), 0, atol=1e-3)
                        for intersection_geom in intersection_geoms
                        if not intersection_geom.is_empty
                    ]
                )
            )
        )
        endpoint_nodes.append(endpoints)
    return intersect_nodes, endpoint_nodes


def determine_valid_intersection_points_no_vnode(
    trace_candidates: gpd.GeoSeries, geom: LineString
) -> List[Point]:
    """
    Filter intersection points between trace candidates and geom with no vnodes.

    V-node intersections are validated by looking at the endpoints. If V-nodes
    were kept as intersection points the VNodeValidator could not find V-node
    errors.
    """
    inter = determine_valid_intersection_points(trace_candidates.intersection(geom))
    geom_endpoints = get_trace_endpoints(geom)
    for trace_candidate in trace_candidates.geometry.values:
        candidate_endpoints = get_trace_endpoints(trace_candidate)
        for ce in candidate_endpoints:
            for ge in geom_endpoints:
                for p in inter:
                    if np.isclose(ce.distance(ge), 0, atol=1e-4) and np.isclose(
                        ge.distance(p), 0, atol=1e-4
                    ):
                        inter.remove(p)
    return inter


def determine_valid_intersection_points(
    intersection_geoms: gpd.GeoSeries,
) -> List[Point]:
    """
    Filter intersection points between trace candidates and geom.

    Only allows Point geometries as intersections. LineString intersections
    would be possible if geometries are stacked.
    """
    assert isinstance(intersection_geoms, gpd.GeoSeries)
    valid_interaction_points = []
    for geom in intersection_geoms.geometry.values:
        if isinstance(geom, Point):
            valid_interaction_points.append(geom)
        elif isinstance(geom, MultiPoint):
            valid_interaction_points.extend([p for p in geom.geoms])
        elif geom.is_empty:
            pass
        else:
            # TODO: LineStrings occur here due to stacked traces. These can
            # occur because of validation.
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
    Determine faulty trace junctions using nodes.

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
    # nodes_geoseries_sindex = flattened_nodes_geoseries.sindex
    nodes_geoseries_sindex = pygeos_spatial_index(flattened_nodes_geoseries)

    # Set collection for indexes with junctions
    indexes_with_junctions: Set[int] = set()

    # Iterate over node tuples i.e. points is a tuple with Points
    # The node tuple indexes represent the trace indexes
    for idx, points in enumerate(nodes):
        associated_point_count = len(points)
        if associated_point_count == 0:
            continue

        # Because node indexes represent traces, we can remove all nodes of the
        # current trace by using the idx.
        other_nodes_geoseries: gpd.GeoSeries = flattened_nodes_geoseries.loc[
            [idx_reference != idx for idx_reference in flattened_idx_reference]
        ]

        # Iterate over the actual Points of the current trace
        for i, point in enumerate(points):

            # Get node candidates from spatial index
            node_candidates_idx = spatial_index_intersection(
                nodes_geoseries_sindex,
                geom_bounds(
                    safe_buffer(
                        point, snap_threshold * snap_threshold_error_multiplier * 10
                    )
                ),
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
    """
    Zip iterables of only equal lengths.
    """
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


def crop_to_target_areas(
    traces: Union[gpd.GeoSeries, gpd.GeoDataFrame],
    areas: Union[gpd.GeoSeries, gpd.GeoDataFrame],
    snap_threshold: float,
) -> Union[gpd.GeoSeries, gpd.GeoDataFrame]:
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
    >>> cropped_traces = crop_to_target_areas(traces, areas, 0.01)
    >>> print([trace.wkt for trace in cropped_traces])
    ['LINESTRING (-1.9 4.924998124953124, -2 5)']

    """
    # Only handle LineStrings
    if not all([isinstance(trace, LineString) for trace in traces.geometry.values]):
        raise TypeError(
            "Expected no MultiLineString geometries in crop_to_target_areas."
        )
    # Match the crs
    traces, areas = match_crs(traces, areas)

    # Clip traces to target areas
    clipped_traces = gpd.clip(traces, areas)

    # Clipping might result in Point geometries
    # Filter to only LineStrings and MultiLineStrings
    clipped_traces = clipped_traces.loc[
        [
            isinstance(geom, (LineString, MultiLineString))
            for geom in clipped_traces.geometry.values
        ]
    ]

    # Some traces might be converted to MultiLineStrings if they become
    # disjointed with the clip.
    # This presents a data management problem -> Data is either lost or
    # duplicated to different parts of a MultiLineString converted trace
    clipped_and_dissolved_traces = dissolve_multi_part_traces(clipped_traces)
    assert (
        clipped_and_dissolved_traces.shape[0] >= clipped_and_dissolved_traces.shape[0]
    )

    # Remove small traces that might cause topology errors
    clipped_and_dissolved_traces = clipped_and_dissolved_traces.loc[
        clipped_and_dissolved_traces.geometry.length > snap_threshold * 2.01
    ]

    # Reset index
    clipped_and_dissolved_traces.reset_index(inplace=True, drop=True)

    return clipped_and_dissolved_traces


def dissolve_multi_part_traces(
    traces: Union[gpd.GeoDataFrame, gpd.GeoSeries]
) -> Union[gpd.GeoDataFrame, gpd.GeoSeries]:
    """
    Dissolve MultiLineStrings in GeoDataFrame or GeoSeries.

    Copies all attribute data of rows with MultiLineStrings to new LineStrings.
    """
    # Get MultiLineString rows
    mls_bools = [isinstance(trace, MultiLineString) for trace in traces.geometry.values]
    mls_traces = traces.loc[
        [isinstance(trace, MultiLineString) for trace in traces.geometry.values]
    ]
    # If no MultiLineString geoms -> return original
    if mls_traces.shape[0] == 0:
        return traces

    # Gather LineString geoms
    ls_traces = traces.loc[[not val for val in mls_bools]]

    # Dissolve MultiLineString geoms but keep same row data
    dissolved_rows = []

    for _, row in mls_traces.iterrows():
        as_linestrings = mls_to_ls([row.geometry])
        if len(as_linestrings) == 0:
            raise ValueError("Expected atleast one geom from mls_to_ls.")

        for new_geom in as_linestrings:
            # Do not modify mls_traces inplace
            new_row = row.copy()
            new_row[GEOMETRY_COLUMN] = new_geom
            dissolved_rows.append(new_row)

    # Merge with ls_traces
    dissolved_traces = pd.concat([ls_traces, gpd.GeoDataFrame(dissolved_rows)])

    if not all(
        [isinstance(val, LineString) for val in dissolved_traces.geometry.values]
    ):
        raise TypeError("Expected all LineStrings in dissolved_traces.")
    return dissolved_traces


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


def resolve_split_to_ls(geom: LineString, splitter: LineString) -> List[LineString]:
    """
    Resolve split between two LineStrings to only LineString results.
    """
    split_current = list(split(geom, splitter))
    linestrings = [
        geom
        for geom in split_current
        if isinstance(geom, LineString) and not geom.is_empty
    ]
    linestrings.extend(
        mls_to_ls([geom for geom in split_current if isinstance(geom, MultiLineString)])
    )
    return linestrings


def safe_buffer(geom: Union[Point, LineString], radius: float, **kwargs) -> Polygon:
    """
    Get type checked Polygon buffer.
    """
    buffer = geom.buffer(radius, **kwargs)
    if not isinstance(buffer, Polygon):
        raise TypeError("Expected Polygon buffer.")
    return buffer


def random_points_within(poly: Polygon, num_points: int) -> List[Point]:
    """
    Get random points within Polygon.
    """
    min_x, min_y, max_x, max_y = poly.bounds
    points = []

    while len(points) < num_points:
        random_point = Point(
            [random.uniform(min_x, max_x), random.uniform(min_y, max_y)]
        )
        if random_point.within(poly):
            points.append(random_point)

    return points


def spatial_index_intersection(
    spatial_index: PyGEOSSTRTreeIndex, coordinates: Union[BoundsTuple, PointTuple]
) -> List[int]:
    """
    Type-checked spatial index intersection.
    """
    if spatial_index is None:
        return []
    result = spatial_index.intersection(coordinates)
    indexes = []
    for idx in result:
        if isinstance(idx, int):
            indexes.append(idx)
        elif idx == int(idx):
            indexes.append(idx)
        else:
            raise TypeError("Expected integer results from intersection.")
    return indexes


def geom_bounds(geom: Union[LineString, Polygon]) -> Tuple[float, float, float, float]:
    """
    Get LineString or Polygon bounds.
    """
    types = (LineString, Polygon)
    if not isinstance(geom, types):
        raise TypeError(f"Expected {types} as type.")
    bounds = []
    for val in geom.bounds:
        if isinstance(val, (int, float)):
            bounds.append(val)
        else:
            raise TypeError("Expected numerical bounds.")
    if not len(bounds) == 4:
        raise ValueError("Expected bounds of length 4.")
    return tuple(bounds)


def pygeos_spatial_index(
    geodataset: Union[gpd.GeoDataFrame, gpd.GeoSeries]
) -> PyGEOSSTRTreeIndex:
    """
    Get PyGEOSSTRTreeIndex from geopandas dataset.
    """
    sindex = geodataset.sindex
    if not isinstance(sindex, PyGEOSSTRTreeIndex):
        raise TypeError("Expected PyGEOSSTRTreeIndex as spatial index.")
    return sindex


def read_geofile(path: Path) -> gpd.GeoDataFrame:
    """
    Read a filepath for a GeoDataFrame representable geo-object.
    """
    data = gpd.read_file(path)
    if not isinstance(data, gpd.GeoDataFrame):
        raise TypeError("Expected GeoDataFrame as file read result.")
    return data


def determine_boundary_intersecting_lines(
    line_gdf: gpd.GeoDataFrame, area_gdf: gpd.GeoDataFrame, snap_threshold: float
):
    """
    Determine lines that intersect any target area boundary.
    """
    assert isinstance(line_gdf, (gpd.GeoSeries, gpd.GeoDataFrame))
    # line_gdf = line_gdf.reset_index(inplace=False, drop=True)
    spatial_index = line_gdf.sindex
    intersecting_idxs = []
    cuts_through_idxs = []
    for target_area in area_gdf.geometry.values:
        assert isinstance(target_area, (MultiPolygon, Polygon))
        target_area_bounds = target_area.boundary.bounds
        assert len(target_area_bounds) == 4 and isinstance(target_area_bounds, tuple)
        min_x, min_y, max_x, max_y = target_area_bounds
        intersection = spatial_index.intersection(
            extend_bounds(
                min_x=min_x,
                min_y=min_y,
                max_x=max_x,
                max_y=max_y,
                extend_amount=snap_threshold * 100,
            )
        )
        candidate_idxs = list(intersection if intersection is not None else [])
        if len(candidate_idxs) == 0:
            continue

        for candidate_idx in candidate_idxs:
            line = line_gdf.iloc[candidate_idx].geometry
            assert isinstance(line, LineString)
            if line.distance(target_area.boundary) < snap_threshold:
                intersecting_idxs.append(candidate_idx)
                endpoints = get_trace_endpoints(line)
                if all(
                    [
                        endpoint.distance(target_area.boundary) < snap_threshold
                        for endpoint in endpoints
                    ]
                ):
                    cuts_through_idxs.append(candidate_idx)

                elif not any(
                    [endpoint.within(target_area) for endpoint in endpoints]
                ) and np.isclose(line.distance(target_area), 0):
                    cuts_through_idxs.append(candidate_idx)

    intersecting_lines = np.array(
        [idx in intersecting_idxs for idx in line_gdf.index.values]
    )
    cuts_through_lines = np.array(
        [idx in cuts_through_idxs for idx in line_gdf.index.values]
    )
    assert intersecting_lines.dtype == "bool"
    assert cuts_through_lines.dtype == "bool"
    return intersecting_lines, cuts_through_lines


def extend_bounds(
    min_x: float, min_y: float, max_x: float, max_y: float, extend_amount: float
) -> Tuple[float, float, float, float]:
    """
    Extend bounds by addition and reduction.
    """
    return (
        min_x - extend_amount,
        min_y - extend_amount,
        max_x + extend_amount,
        max_y + extend_amount,
    )


def bool_arrays_sum(arr_1: np.ndarray, arr_2: np.ndarray) -> np.ndarray:
    """
    Calculate integer sum of two arrays.
    """
    assert arr_1.dtype == "bool"
    assert arr_2.dtype == "bool"
    return np.array([int(val_1) + int(val_2) for val_1, val_2 in zip(arr_1, arr_2)])


def intersection_count_to_boundary_weight(intersection_count: int) -> int:
    """
    Get actual weight factor for boundary intersection count.
    """
    if not isinstance(intersection_count, int):
        intersection_count = intersection_count.item()
    assert isinstance(intersection_count, int)
    if intersection_count == 0:
        return 1
    elif intersection_count == 1:
        return 2
    elif intersection_count == 2:
        return 0
    else:
        raise ValueError(
            f"Expected 0,1,2 as intersection_count. Got: {intersection_count}"
        )
