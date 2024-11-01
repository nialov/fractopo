"""
Contains general calculation and plotting tools.
"""

import json
import logging
import math
import os
import random
from bisect import bisect
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from enum import Enum, unique
from functools import wraps
from io import StringIO
from itertools import accumulate, chain, compress, zip_longest
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple, Union, overload

import geopandas as gpd
import numpy as np
import pandas as pd
import sklearn.metrics as sklm
from geopandas.sindex import SpatialIndex
from joblib import Memory
from matplotlib import patheffects as path_effects
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from shapely import prepared, wkb
from shapely.affinity import scale
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.ops import split
from sklearn.linear_model import LinearRegression

log = logging.getLogger(__name__)

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
BOUNDARY_INTERSECT_KEYS = ("0", "1", "2")

CONNECTION_COLUMN = "Connection"
CLASS_COLUMN = "Class"
GEOMETRY_COLUMN = "geometry"
POWERLAW = "powerlaw"
LOGNORMAL = "lognormal"
EXPONENTIAL = "exponential"

NULL_SET = "-1"

REPRESENTATIVE_POINT = "Representative Point"
RADIUS = "Radius"
NAME = "Name"
CENSORING = "Censoring"
RELATIVE_CENSORING = "Relative Censoring"

GEOJSON_DRIVER = "GeoJSON"

SetRangeTuple = Tuple[Tuple[float, float], ...]
BoundsTuple = Tuple[float, float, float, float]
PointTuple = Tuple[float, float]
Number = Union[float, int]
ParameterValuesType = Dict[str, Union[float, int, str]]
ParameterListType = List[ParameterValuesType]

MINIMUM_LINE_LENGTH = 1e-18

DEFAULT_FRACTOPO_CACHE_PATH = Path(".cache/fractopo")

JOBLIB_CACHE = Memory(
    (
        os.environ.get("FRACTOPO_CACHE_PATH", DEFAULT_FRACTOPO_CACHE_PATH)
        # Disk caching is disabled in pytest execution by setting an environment
        # variable FRACTOPO_DISABLE_CACHE to a non-zero string (i.e. "1") However,
        # it can be enabled even in tests by setting FRACTOPO_DISABLE_CACHE to 0
        if os.environ.get("FRACTOPO_DISABLE_CACHE") in (None, "0")
        else None
    ),
    verbose=int(os.environ.get("FRACTOPO_JOBLIB_CACHE_VERBOSITY", 0)),
)


@dataclass
class ProcessResult:
    """
    Dataclass for multiprocessing result parsing.
    """

    identifier: str
    result: Any
    error: bool


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


def sum_aggregation(values, **_) -> Number:
    """
    Aggregate by calculating sum.
    """
    value_sum = np.array(values).sum()
    assert isinstance(value_sum, (float, int))
    return value_sum


def mean_aggregation(values, weights) -> Number:
    """
    Aggregate by calculating mean.
    """
    average_value = np.average(values, weights=weights)
    assert isinstance(average_value, (float, int))
    return average_value


def fallback_aggregation(values) -> str:
    """
    Fallback aggregation where values are simply joined into a string.
    """
    return str(values)


@unique
class Aggregator(Enum):
    """
    Define how to aggregate during subsample aggragation.
    """

    SUM = sum_aggregation
    MEAN = mean_aggregation


@dataclass
class ParamInfo:
    """
    Parameter with name and metadata.
    """

    name: str
    plot_as_log: bool
    unit: str
    # TODO: Currently not used.
    needs_topology: bool
    aggregator: Aggregator


@unique
class Param(Enum):
    """
    Column names for geometric and topological parameters.

    The ``ParamInfo`` instances contain additional metadata.
    """

    AREA = ParamInfo("Area", False, r"$m^2$", False, Aggregator.SUM)
    AREAL_FREQUENCY_B20 = ParamInfo(
        "Areal Frequency B20", True, r"$\frac{1}{m^2}$", True, Aggregator.MEAN
    )
    AREAL_FREQUENCY_P20 = ParamInfo(
        "Areal Frequency P20", True, r"$\frac{1}{m^2}$", True, Aggregator.MEAN
    )
    BRANCH_MEAN_LENGTH = ParamInfo(
        "Branch Mean Length", True, "$m$", True, Aggregator.MEAN
    )
    BRANCH_MIN_LENGTH = ParamInfo(
        "Branch Min Length", True, "$m$", True, Aggregator.MEAN
    )
    BRANCH_MAX_LENGTH = ParamInfo(
        "Branch Max Length", True, "$m$", True, Aggregator.MEAN
    )
    CONNECTIONS_PER_BRANCH = ParamInfo(
        "Connections per Branch", False, r"$\frac{1}{n}$", True, Aggregator.MEAN
    )
    CONNECTIONS_PER_TRACE = ParamInfo(
        "Connections per Trace", False, r"$\frac{1}{n}$", True, Aggregator.MEAN
    )
    CONNECTION_FREQUENCY = ParamInfo(
        "Connection Frequency", False, r"$\frac{1}{m^2}$", True, Aggregator.MEAN
    )
    DIMENSIONLESS_INTENSITY_B22 = ParamInfo(
        "Dimensionless Intensity B22", False, "-", True, Aggregator.MEAN
    )
    DIMENSIONLESS_INTENSITY_P22 = ParamInfo(
        "Dimensionless Intensity P22", False, "-", False, Aggregator.MEAN
    )
    FRACTURE_DENSITY_MAULDON = ParamInfo(
        "Fracture Density (Mauldon)", True, r"$\frac{1}{m^2}$", True, Aggregator.MEAN
    )
    FRACTURE_INTENSITY_B21 = ParamInfo(
        "Fracture Intensity B21", True, r"$\frac{m}{m^2}$", False, Aggregator.MEAN
    )
    FRACTURE_INTENSITY_MAULDON = ParamInfo(
        "Fracture Intensity (Mauldon)", True, r"$\frac{m}{m^2}$", True, Aggregator.MEAN
    )
    FRACTURE_INTENSITY_P21 = ParamInfo(
        "Fracture Intensity P21", True, r"$\frac{m}{m^2}$", False, Aggregator.MEAN
    )
    NUMBER_OF_BRANCHES = ParamInfo(
        "Number of Branches", False, "-", True, Aggregator.SUM
    )
    NUMBER_OF_BRANCHES_TRUE = ParamInfo(
        "Number of Branches (Real)", False, "-", True, Aggregator.SUM
    )
    NUMBER_OF_TRACES = ParamInfo("Number of Traces", False, "-", True, Aggregator.SUM)
    NUMBER_OF_TRACES_TRUE = ParamInfo(
        "Number of Traces (Real)", False, "-", True, Aggregator.SUM
    )
    TRACE_MEAN_LENGTH = ParamInfo(
        "Trace Mean Length", True, "$m$", False, Aggregator.MEAN
    )
    TRACE_MIN_LENGTH = ParamInfo(
        "Trace Min Length", True, "$m$", False, Aggregator.MEAN
    )
    TRACE_MAX_LENGTH = ParamInfo(
        "Trace Max Length", True, "$m$", False, Aggregator.MEAN
    )
    TRACE_MEAN_LENGTH_MAULDON = ParamInfo(
        "Trace Mean Length (Mauldon)", True, "$m$", True, Aggregator.MEAN
    )
    CIRCLE_COUNT = ParamInfo(
        "Circle Count",
        plot_as_log=False,
        unit="-",
        needs_topology=False,
        aggregator=Aggregator.SUM,
    )


def determine_set(
    value: float,
    value_ranges: SetRangeTuple,
    set_names: Tuple[str, ...],
    loop_around: bool,
) -> str:
    """
    Determine which named value range, if any, value is within.

    loop_around defines behavior expected for radial data i.e. when value range
    can loop back around e.g. [160, 50]

    E.g.

    >>> determine_set(10.0, [(0, 20), (30, 160)], ["0-20", "30-160"], False)
    '0-20'

    Example with

    >>> determine_set(50.0, [(0, 20), (160, 60)], ["0-20", "160-60"], True)
    '160-60'

    :param value: Value to determine set of.
    :param value_ranges: Ranges of each set.
    :param set_names: Names of each set.
    :param loop_around: Whether the sets loop around. This is the case for
        radial data such as azimuths but not the case for length data.
    :return: Set string in which value belongs.
    :raises ValueError: When set value ranges overlap.
    """
    assert len(value_ranges) == len(set_names)
    possible_set_name = [
        set_name
        for set_name, value_range in zip(set_names, value_ranges)
        if is_set(value, value_range, loop_around)
    ]
    if len(possible_set_name) == 0:
        return NULL_SET
    if len(possible_set_name) == 1:
        return possible_set_name[0]
    raise ValueError("Expected set value ranges to not overlap.")


def is_set(value: Number, value_range: Tuple[float, float], loop_around: bool) -> bool:
    """
    Determine if value fits within the given value_range.

    If the value range has the possibility of looping around loop_around can be
    set to true.

    >>> is_set(5, (0, 10), False)
    True

    >>> is_set(5, (175, 15), True)
    True

    :param value: Value to determine.
    :param value_range Tuple[float,: The range of values.
    :param loop_around: Whether the range loops around. This is the case for
        radial data such as azimuths but not the case for length data.
    :return: Is it within range.
    """
    if loop_around:
        if value_range[0] > value_range[1]:
            # Loops around case
            if (value >= value_range[0]) | (value <= value_range[1]):
                return True
    if value_range[0] <= value <= value_range[1]:
        return True
    return False


def is_azimuth_close(
    first: float, second: float, tolerance: float, halved: bool = True
):
    """
    Determine are azimuths first and second within tolerance.

    Takes into account the radial nature of azimuths.

    >>> is_azimuth_close(0, 179, 15)
    True

    >>> is_azimuth_close(166, 179, 15)
    True

    >>> is_azimuth_close(20, 179, 15)
    False

    :param first: First azimuth value to compare.
    :param second: Second azimuth value to compare.
    :param tolerance: Tolerance for closeness.
    :param halved: Are the azimuths azial (i.e. ``halved=True``) or vectors.
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
    given  and the azimuth of the fitted linear line is returned.

    The azimuth is returned in range [0, 180].

    E.g.

    >>> line = LineString([(0, 0), (1, 1), (2, 2), (3, 3)])
    >>> determine_regression_azimuth(line)
    45.0

    >>> line = LineString([(-1, -5), (3, 3)])
    >>> round(determine_regression_azimuth(line), 3)
    26.565

    >>> line = LineString([(0, 0), (0, 3)])
    >>> determine_regression_azimuth(line)
    0.0


    :param line: The line of which azimuth is determined.
    :return: The determined azimuth in range [0, 180].
    :raises ValueError: When ``LinearRegression`` returns
        unexpected coefficients.
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
    return float(azimuth)


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

    :param line: The line of which azimuth is determined.
    :param halved: Whether to return result in range [0, 180]
        (halved=True) or [0, 360] (halved=False).
    :return: The determined azimuth.
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


def calc_strike(dip_direction: float) -> float:
    """
    Calculate strike from dip direction. Right-handed rule.

    E.g.:

    >>> calc_strike(50.0)
    320.0

    >>> calc_strike(180.0)
    90.0

    :param dip_direction: The direction of dip.
    :return: Converted strike.
    """
    strike = dip_direction - 90
    if strike < 0:
        strike = 360 + strike
    return strike


def azimu_half(degrees: float) -> float:
    """
    Transform azimuth from 180-360 range to range 0-180.

    :param degrees: Degrees in range 0 - 360
    :return: Degrees in range 0 - 180
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

        sd += diff**2
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


def define_length_set(length: float, set_df: pd.DataFrame) -> str:
    """
    Define sets based on the length of the traces or branches.
    """
    if length < 0:
        raise ValueError("Length value was not positive.\n Value: {length}")

    # Set_num is -1 when length is in no length set range
    set_label = -1
    for _, row in set_df.iterrows():
        set_range: Tuple[float, float] = row.LengthSetLimits
        # Checks if length degree value is within the length set range
        if set_range[0] <= length <= set_range[1]:
            set_label = row.LengthSet

    return str(set_label)


# def curviness(linestring):
#     """
#     Determine curviness of LineString.
#     """
#     try:
#         coords = list(linestring.coords)
#     except NotImplementedError:
#         return np.NaN
#     df = pd.DataFrame(columns=["azimu", "length"])
#     for i in range(len(coords) - 1):
#         start = Point(coords[i])
#         end = Point(coords[i + 1])
#         line = LineString([start, end])
#         azimu = determine_azimuth(line, halved=True)
#         # halved = tools.azimu_half(azimu)
#         length = line.length
#         addition = {
#             "azimu": azimu,
#             "length": length,
#         }  # Addition to DataFrame with fit x and fit y values
#         df = df.append(addition, ignore_index=True)

#     std = sd_calc(df.azimu.values.tolist())
#     azimu_std = std

#     return azimu_std


def prepare_geometry_traces(trace_series: gpd.GeoSeries) -> prepared.PreparedGeometry:
    """
    Prepare trace_series geometries for a faster spatial analysis.

    Assumes geometries are LineStrings which are consequently collected into a
    single MultiLineString which is prepared with shapely.prepared.prep.

    >>> traces = gpd.GeoSeries(
    ...     [LineString([(0, 0), (1, 1)]), LineString([(0, 1), (0, -1)])]
    ... )
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

    >>> first = gpd.GeoSeries([Point(1, 1)], crs="EPSG:3067")
    >>> second = gpd.GeoSeries([Point(1, 1)])
    >>> m_first, m_second = match_crs(first, second)
    >>> m_first.crs == m_second.crs
    True
    """
    all_crs = [crs for crs in (first.crs, second.crs) if crs is not None]
    if len(all_crs) == 0:
        # No crs in either input
        return first, second
    if len(all_crs) == 2 and all_crs[0] == all_crs[1]:
        # Two valid and they're the same
        return first, second
    if len(all_crs) == 1:
        # One valid crs in inputs
        crs = all_crs[0]
        first = first.set_crs(crs)
        second = second.set_crs(crs)
        return first, second
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
    if point == coord_points[0]:
        return coord_points[1]
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
    points = Point(trace.coords[0]), Point(trace.coords[-1])
    return points


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


@JOBLIB_CACHE.cache
def determine_general_nodes(
    traces: Union[gpd.GeoSeries, gpd.GeoDataFrame],
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
            "Expected traces to have a continuous index from 0...len(traces) - 1"
        )

    intersect_nodes: List[Tuple[Point, ...]] = []
    endpoint_nodes: List[Tuple[Point, ...]] = []
    # spatial_index = traces.geometry.sindex
    try:
        spatial_index = traces.sindex
    except TypeError:
        spatial_index = None
    for idx, geom in enumerate(traces.geometry.values):
        if not isinstance(geom, LineString) or geom.is_empty:
            # Intersections and endpoints are not defined for
            # MultiLineStrings or empty LineStrings
            intersect_nodes.append(())
            endpoint_nodes.append(())
            continue

        # Get trace candidates for intersection
        trace_candidates_idx: List[int] = (
            sorted(spatial_index_intersection(spatial_index, geom_bounds(geom)))
            if spatial_index is not None
            else [idx]
        )

        # Remove current geometry from candidates
        trace_candidates_idx.remove(idx)
        trace_candidates: gpd.GeoSeries = traces.geometry.iloc[trace_candidates_idx]

        # Only accept LineString candidates
        trace_candidates = trace_candidates.loc[
            [isinstance(geom, LineString) for geom in trace_candidates.geometry.values]
        ]

        intersection_geoms = determine_valid_intersection_points_no_vnode(
            trace_candidates, geom
        )
        intersect_nodes.append(tuple(intersection_geoms))
        endpoints = tuple(
            (
                endpoint
                for endpoint in get_trace_endpoints(geom)
                if not any(
                    # Intersection results in inaccurate geoms ->
                    # tolerance is actually close to a snap_threshold
                    # This represents a hard limit to snapping threshold.
                    np.isclose(endpoint.distance(intersection_geom), 0, atol=1e-3)
                    for intersection_geom in intersection_geoms
                    if not intersection_geom.is_empty
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
    # Determine intersections between candidate LineStrings and the geom LineString
    inter = determine_valid_intersection_points(trace_candidates.intersection(geom))

    # If empty, exit
    if len(inter) == 0:
        return inter

    # Get endpoints of the geom LineString
    geom_endpoints = get_trace_endpoints(geom)

    # Upkeep a boolean list of which indexes (i.e., valid intersection points)
    # to keep
    # This will be used to compress `inter` before returning
    p_to_keep = [True] * len(inter)

    # Iterate over the trace candidates
    for trace_candidate in trace_candidates.geometry.values:
        candidate_endpoints = get_trace_endpoints(trace_candidate)
        for ce in candidate_endpoints:
            for ge in geom_endpoints:
                # First check: is the candidate endpoint close to the geom endpoint
                candidate_endpoint_is_close_to_geom_endpoint = np.isclose(
                    ce.distance(ge), 0, atol=1e-4
                )
                # If not, keep it in `inter`
                if not candidate_endpoint_is_close_to_geom_endpoint:
                    continue

                # Enumerate over the valid intersection points
                for idx, p in enumerate(inter):
                    # If the current valid intersection point is already set
                    # for removal, continue the loop to the next
                    if not p_to_keep[idx]:
                        continue
                    # Second check: is the `geom` LineString endpoint close to
                    # the intersection point `p`. If yes, mark the intersection
                    # point for removal from `interl`
                    if np.isclose(ge.distance(p), 0, atol=1e-4):
                        p_to_keep[idx] = False
                        # inter.remove(p)
                        # p_to_remove.add(idx)

    # Remove the marked intersection points from `inter` based on the boolean
    # list `p_to_keep`
    inter_filtered = list(compress(inter, selectors=p_to_keep))
    return inter_filtered


def determine_valid_intersection_points(
    intersection_geoms: gpd.GeoSeries,
) -> List[Point]:
    """
    Filter intersection points between trace candidates and geom.

    Only allows Point geometries as intersections. LineString intersections
    would be possible if geometries are stacked.

    :param intersection_geoms: GeoSeries of intersection (Point) geometries.
    :return: The valid intersections (Points).
    """
    assert isinstance(intersection_geoms, gpd.GeoSeries)
    valid_interaction_points = []
    for geom in intersection_geoms.geometry.values:
        assert isinstance(geom, (BaseGeometry, BaseMultipartGeometry))
        if isinstance(geom, Point):
            valid_interaction_points.append(geom)
        elif isinstance(geom, MultiPoint):
            valid_interaction_points.extend(list(geom.geoms))
        elif geom.is_empty:
            log.info(
                f"Empty geometry in determine_valid_intersection_points: {geom.wkt}"
            )

        # TODO: Should these clauses error or report the invalid geometries upstream?
        elif isinstance(geom, LineString):
            log.error(f"Expected geom ({geom.wkt}) not to be of type LineString.")
        elif isinstance(geom, MultiLineString):
            log.error(f"Expected geom ({geom.wkt}) not to be of type MultiLineString.")
        elif isinstance(geom, GeometryCollection):
            for collection_geom in geom.geoms:
                if isinstance(collection_geom, Point):
                    valid_interaction_points.append(collection_geom)
                else:
                    log.warning(
                        f"Expected collection_geom ({collection_geom.wkt}) to be of type Point"
                    )

        else:
            raise TypeError(
                "Expected (Multi)Point or (Multi)LineString geometries"
                " in determine_valid_intersection_points."
            )
    assert all(isinstance(p, Point) for p in valid_interaction_points)
    return valid_interaction_points


def line_intersection_to_points(first: LineString, second: LineString) -> List[Point]:
    """
    Perform shapely intersection between two LineStrings.

    Enforces only Point returns.
    """
    intersection = first.intersection(second)
    collect_points: List[Point] = []
    if isinstance(intersection, LineString) and intersection.is_empty:
        pass
    elif isinstance(intersection, Point):
        collect_points = [intersection]
    elif isinstance(intersection, MultiPoint):
        collect_points = list(intersection.geoms)
    else:
        log.error(f"Expected Point or empty intersection, got: {intersection}")
    return collect_points


def flatten_tuples(
    list_of_tuples: List[Tuple[Any, ...]],
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
        accumulate([len(val_tuple) for _, val_tuple in enumerate(list_of_tuples)])
    )
    flattened_tuples = list(chain(*list_of_tuples))
    flattened_idx_reference = [
        bisect(accumulated_idxs, idx) for idx in range(len(flattened_tuples))
    ]
    return flattened_idx_reference, flattened_tuples


@JOBLIB_CACHE.cache
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
    nodes_geoseries_sindex: SpatialIndex = flattened_nodes_geoseries.sindex

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
        for _, point in enumerate(points):
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
            assert all(isinstance(val, bool) for val in intersection_data)
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

    :param start_point: The start point.
    :param end_point: The end point.
    :return: The unit vector that points from ``start_point`` to
        ``end_point``.
    """
    # Convert Point coordinates to (x, y)
    segment_start = point_to_xy(start_point)
    segment_end = point_to_xy(end_point)
    segment_vector = np.array(
        [segment_end[0] - segment_start[0], segment_end[1] - segment_start[1]]
    )
    if any(np.isnan(segment_vector)):
        return np.array([np.nan, np.nan])
    segment_unit_vector = segment_vector / np.linalg.norm(segment_vector)
    assert isinstance(segment_unit_vector, np.ndarray)
    return segment_unit_vector


def compare_unit_vector_orientation(
    vec_1: np.ndarray, vec_2: np.ndarray, threshold_angle: float
) -> bool:
    """
    If vec_1 and vec_2 are too different in orientation, will return False.
    """
    if np.linalg.norm(vec_1 + vec_2) < np.sqrt(2):
        # If they face opposite side -> False
        return False
    dot_product = np.dot(vec_1, vec_2)
    if np.isclose(dot_product, 1):
        return True
    if dot_product > 1 or dot_product < -1 or np.isnan(dot_product):
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
    total_bounds_geoseries = total_bounds(geoseries)
    bounding_poly: Polygon = scale(box(*total_bounds_geoseries), xfact=2, yfact=2)
    assert isinstance(bounding_poly, Polygon)
    if any(
        geom.intersects(bounding_poly.boundary) for geom in geoseries.geometry.values
    ):
        bounding_poly = safe_buffer(bounding_poly, radius=1.0)
        if any(
            geom.intersects(bounding_poly.boundary)
            for geom in geoseries.geometry.values
        ):
            # if any(geoseries.intersects(bounding_polygon.boundary)):
            raise ValueError("Expected no intersects after buffer.")
    assert all(geoseries.within(bounding_poly))
    return bounding_poly


def mls_to_ls(multilinestrings: List[MultiLineString]) -> List[LineString]:
    """
    Flattens a list of multilinestrings to a list of linestrings.

    >>> multilinestrings = [
    ...     MultiLineString(
    ...         [
    ...             LineString([(1, 1), (2, 2), (3, 3)]),
    ...             LineString([(1.9999, 2), (-2, 5)]),
    ...         ]
    ...     ),
    ...     MultiLineString(
    ...         [
    ...             LineString([(1, 1), (2, 2), (3, 3)]),
    ...             LineString([(1.9999, 2), (-2, 5)]),
    ...         ]
    ...     ),
    ... ]
    >>> result_linestrings = mls_to_ls(multilinestrings)
    >>> print([ls.wkt for ls in result_linestrings])
    ['LINESTRING (1 1, 2 2, 3 3)', 'LINESTRING (1.9999 2, -2 5)',
    'LINESTRING (1 1, 2 2, 3 3)', 'LINESTRING (1.9999 2, -2 5)']

    """
    linestrings: List[LineString] = []
    for mls in multilinestrings:
        linestrings.extend(list(mls.geoms))
    if not all(isinstance(ls, LineString) for ls in linestrings):
        raise ValueError("MultiLineStrings within MultiLineStrings?")
    return linestrings


@JOBLIB_CACHE.cache
def crop_to_target_areas(
    traces: Union[gpd.GeoSeries, gpd.GeoDataFrame],
    areas: Union[gpd.GeoSeries, gpd.GeoDataFrame],
    is_filtered: bool = False,
    keep_column_data: bool = False,
    allow_multilinestring_input: bool = False,
) -> Union[gpd.GeoSeries, gpd.GeoDataFrame]:
    """
    Crop traces to the area polygons.

    E.g.

    >>> traces = gpd.GeoSeries(
    ...     [LineString([(1, 1), (2, 2), (3, 3)]), LineString([(1.9999, 2), (-2, 5)])]
    ... )
    >>> areas = gpd.GeoSeries(
    ...     [
    ...         Polygon([(1, 1), (-1, 1), (-1, -1), (1, -1)]),
    ...         Polygon([(-2.5, 6), (-1.9, 6), (-1.9, 4), (-2.5, 4)]),
    ...     ]
    ... )
    >>> cropped_traces = crop_to_target_areas(traces, areas, 0.01)
    >>> print([trace.wkt for trace in cropped_traces.geometry.values])
    ['LINESTRING (-1.9 4.924998124953124, -2 5)']

    :param traces: Trace data.
    :param areas: Area data.
    :param snap_threshold: Distance threshold for snapping.
    :param is_filtered: Has preliminary spatial filtering already been
        done. If not the traces will be filtered first with a spatial index.
    :param keep_column_data: Is column data of traces required to persist.
    :return: Cropped traces.
    :raises TypeError: If all geometries in the end result are not
        ``LineString's``.
    """
    # Only handle LineStrings
    if (
        not all(isinstance(trace, LineString) for trace in traces.geometry.values)
        and not allow_multilinestring_input
    ):
        raise TypeError(
            "Expected no MultiLineString geometries in crop_to_target_areas."
        )
    # Match the crs
    traces, areas = match_crs(traces, areas)

    if not is_filtered:
        traces.reset_index(drop=True, inplace=True)
        spatial_index = traces.sindex
        assert isinstance(spatial_index, SpatialIndex), type(spatial_index)

        areas_bounds = total_bounds(areas)
        assert len(areas_bounds) == 4

        candidate_idxs = spatial_index_intersection(
            spatial_index=spatial_index, coordinates=areas_bounds
        )
        candidate_traces: Union[gpd.GeoSeries, gpd.GeoDataFrame] = traces.iloc[
            candidate_idxs
        ]
    else:
        candidate_traces = traces

    # TODO: Remove environment check
    clipped_traces = gpd.clip(candidate_traces, areas)

    assert hasattr(clipped_traces, "geometry")
    assert isinstance(clipped_traces, (gpd.GeoDataFrame, gpd.GeoSeries))

    # Debug log for traces smaller than MINIMUM_LINE_LENGTH
    sum_smaller_than_minimum = sum(clipped_traces.geometry.length < MINIMUM_LINE_LENGTH)
    if sum_smaller_than_minimum > 0:
        # Log if found
        log.info(
            "Traces smaller than MINIMUM_LINE_LENGTH found after crop.",
            extra=dict(
                MINIMUM_LINE_LENGTH=MINIMUM_LINE_LENGTH,
                found_amount=sum_smaller_than_minimum,
            ),
        )

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
    # Filtering out traces smaller in length than
    # fractopo.general.MINIMUM_LINE_LENGTH
    clipped_and_dissolved_traces = clipped_and_dissolved_traces.loc[
        clipped_and_dissolved_traces.geometry.length > MINIMUM_LINE_LENGTH
    ]
    log.info(
        "Filtered out small traces with MINIMUM_LINE_LENGTH.",
        extra=dict(
            MINIMUM_LINE_LENGTH=MINIMUM_LINE_LENGTH,
            new_minimum=clipped_and_dissolved_traces.geometry.length.min(),
            maximum=clipped_and_dissolved_traces.geometry.length.max(),
        ),
    )

    # clipped_and_dissolved_traces = clipped_and_dissolved_traces.loc[
    #     clipped_and_dissolved_traces.geometry.length > snap_threshold * 2.01
    # ]

    # Reset index
    clipped_and_dissolved_traces.reset_index(inplace=True, drop=True)

    return clipped_and_dissolved_traces


@overload
def dissolve_multi_part_traces(traces: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Overload for gpd.GeoDataFrame.
    """
    ...


@overload
def dissolve_multi_part_traces(traces: gpd.GeoSeries) -> gpd.GeoSeries:
    """
    Overload for gpd.GeoSeries.
    """
    ...


def dissolve_multi_part_traces(
    traces: Union[gpd.GeoDataFrame, gpd.GeoSeries],
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

    as_linestrings_list = [mls_to_ls([geom]) for geom in mls_traces.geometry.values]
    if isinstance(traces, gpd.GeoSeries):
        return gpd.GeoSeries(list(chain(*as_linestrings_list)), crs=traces.crs)

    for (_, row), as_linestrings in zip(mls_traces.iterrows(), as_linestrings_list):
        # as_linestrings = mls_to_ls([row.geometry])
        if len(as_linestrings) == 0:
            raise ValueError("Expected at least one geom from mls_to_ls.")

        for new_geom in as_linestrings:
            # Do not modify mls_traces inplace
            new_row = row.copy()
            new_row[GEOMETRY_COLUMN] = new_geom
            dissolved_rows.append(new_row)

    # Merge with ls_traces
    dissolved_rows_gdf = gpd.GeoDataFrame(dissolved_rows, crs=ls_traces.crs)
    assert ls_traces.crs == dissolved_rows_gdf.crs
    dissolved_traces = pd.concat([ls_traces, dissolved_rows_gdf])
    assert isinstance(dissolved_traces, gpd.GeoDataFrame)

    if not all(isinstance(val, LineString) for val in dissolved_traces.geometry.values):
        raise TypeError("Expected all LineStrings in dissolved_traces.")
    return dissolved_traces


def is_empty_area(area: gpd.GeoDataFrame, traces: gpd.GeoDataFrame):
    """
    Check if any traces intersect the area(s) in area GeoDataFrame.
    """
    for area_polygon in area.geometry.values:
        sindex: SpatialIndex = traces.sindex
        intersection = spatial_index_intersection(sindex, geom_bounds(area_polygon))
        potential_traces = traces.iloc[intersection]

        for trace in potential_traces.geometry.values:
            # Only one trace intersect required. No need to loop
            # through all traces if one found.
            if trace.intersects(area_polygon):
                return False
    return True


def resolve_split_to_ls(geom: LineString, splitter: LineString) -> List[LineString]:
    """
    Resolve split between two LineStrings to only LineString results.
    """
    split_current = list(split(geom, splitter).geoms)
    linestrings = [
        geom
        for geom in split_current
        if isinstance(geom, LineString) and not geom.is_empty
    ]
    linestrings.extend(
        mls_to_ls([geom for geom in split_current if isinstance(geom, MultiLineString)])
    )
    return linestrings


def safe_buffer(
    geom: Union[Point, LineString, Polygon], radius: float, **kwargs
) -> Polygon:
    """
    Get type checked Polygon buffer.

    >>> result = safe_buffer(Point(0, 0), 1)
    >>> isinstance(result, Polygon), round(result.area, 3)
    (True, 3.137)
    """
    buffer = geom.buffer(radius, **kwargs)
    if not isinstance(buffer, Polygon):
        raise TypeError("Expected Polygon buffer.")
    return buffer


def random_points_within(poly: Polygon, num_points: int) -> List[Point]:
    """
    Get random points within Polygon.

    >>> from pprint import pprint
    >>> random.seed(10)
    >>> poly = box(0, 0, 1, 1)
    >>> result = random_points_within(poly, 2)
    >>> pprint([point.within(poly) for point in result])
    [True, True]
    """
    min_x, min_y, max_x, max_y = geom_bounds(poly)
    points: List[Point] = []

    while len(points) < num_points:
        random_point = Point(
            [random.uniform(min_x, max_x), random.uniform(min_y, max_y)]
        )
        if random_point.within(poly):
            points.append(random_point)

    return points


def spatial_index_intersection(
    spatial_index: Any, coordinates: Union[BoundsTuple, PointTuple]
) -> List[int]:
    """
    Type-checked spatial index intersection.

    TODO: Maybe use spatial_index.quary_bulk for faster execution?
    TODO: Maybe use a predicate (e.g., "contains") to specify operation?
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


def within_bounds(
    x: float, y: float, min_x: float, min_y: float, max_x: float, max_y: float
):
    """
    Are x and y within the bounds.

    >>> within_bounds(1, 1, 0, 0, 2, 2)
    True
    """
    return (min_x <= x <= max_x) and (min_y <= y <= max_y)


def geom_bounds(
    geom: Union[LineString, Polygon, MultiPolygon],
) -> Tuple[float, float, float, float]:
    """
    Get LineString or Polygon bounds.

    >>> geom_bounds(LineString([(-10, -10), (10, 10)]))
    (-10.0, -10.0, 10.0, 10.0)
    """
    types = (LineString, Polygon, MultiPolygon)
    if not isinstance(geom, types):
        raise TypeError(f"Expected {types} as geom type.")
    bounds = []
    for val in geom.bounds:
        if isinstance(val, (int, float)):
            bounds.append(val)
        else:
            raise TypeError("Expected numerical bounds.")
    if len(bounds) != 4:
        raise ValueError(f"Expected bounds of length 4. Got: {bounds}.")
    bounds_tuple: Tuple[float, float, float, float] = (
        bounds[0],
        bounds[1],
        bounds[2],
        bounds[3],
    )
    return bounds_tuple


def total_bounds(
    geodata: Union[gpd.GeoSeries, gpd.GeoDataFrame],
) -> Tuple[float, float, float, float]:
    """
    Get total bounds of geodataset.

            >>> geodata = gpd.GeoSeries([Point(-10, 10), Point(10, 10)])
    >>> total_bounds(geodata)
    (-10.0, 10.0, 10.0, 10.0)
    """
    if geodata.empty or all(
        geom is None or geom.is_empty for geom in geodata.geometry.values
    ):
        return (np.nan, np.nan, np.nan, np.nan)
    bounds = geodata.total_bounds
    if not len(bounds) == 4:
        raise ValueError(
            f"Expected total_bounds to return an array of length 4: {bounds}."
        )
    return float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])


def read_geofile(path: Path) -> gpd.GeoDataFrame:
    """
    Read a filepath for a ``GeoDataFrame`` representable geo-object.

    :param path: ``pathlib.Path`` to a ``GeoDataFrame``
        representable spatial file.
    :return: ``geopandas.GeoDataFrame`` read from the file.
    :raises TypeError: If the file could not be parsed as a ``GeoDataFrame``
        by ``geopandas``.
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
    # spatial_index = line_gdf.sindex
    spatial_index = line_gdf.sindex
    assert isinstance(spatial_index, SpatialIndex), type(spatial_index)
    intersecting_idxs = []
    cuts_through_idxs = []
    for target_area in area_gdf.geometry.values:
        assert isinstance(target_area, (MultiPolygon, Polygon))
        # target_area_bounds = target_area.boundary.bounds
        target_area_bounds = geom_bounds(target_area)
        assert len(target_area_bounds) == 4 and isinstance(target_area_bounds, tuple)
        min_x, min_y, max_x, max_y = target_area_bounds
        # TODO: Use spatial_index.quary_bulk for faster execution?
        # TODO: Use a predicate (e.g., "contains") to specify operation?
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
                    endpoint.distance(target_area.boundary) < snap_threshold
                    for endpoint in endpoints
                ):
                    cuts_through_idxs.append(candidate_idx)

                elif not any(
                    endpoint.within(target_area) for endpoint in endpoints
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

    >>> extend_bounds(0, 0, 10, 10, 10)
    (-10, -10, 20, 20)
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

    Resulting array consists only of integers 0, 1 and 2.

    >>> arr_1 = np.array([True, False, False])
    >>> arr_2 = np.array([True, True, False])
    >>> bool_arrays_sum(arr_1, arr_2)
    array([2, 1, 0])

    >>> arr_1 = np.array([True, True])
    >>> arr_2 = np.array([True, True])
    >>> bool_arrays_sum(arr_1, arr_2)
    array([2, 2])
    """
    assert arr_1.dtype == "bool"
    assert arr_2.dtype == "bool"
    return np.array([int(val_1) + int(val_2) for val_1, val_2 in zip(arr_1, arr_2)])


def intersection_count_to_boundary_weight(intersection_count: int) -> int:
    """
    Get actual weight factor for boundary intersection count.

    >>> intersection_count_to_boundary_weight(2)
    0
    >>> intersection_count_to_boundary_weight(0)
    1
    >>> intersection_count_to_boundary_weight(1)
    2
    """
    if not isinstance(intersection_count, int):
        intersection_count = intersection_count.item()
    assert isinstance(intersection_count, int)
    if intersection_count == 0:
        return 1
    if intersection_count == 1:
        return 2
    if intersection_count == 2:
        return 0
    raise ValueError(f"Expected 0,1,2 as intersection_count. Got: {intersection_count}")


def numpy_to_python_type(value: Any):
    """
    Convert numpy dtype variable to Python type, if possible.
    """
    try:
        return value.item()
    except AttributeError:
        return value


def calc_circle_area(radius: float) -> float:
    """
    Calculate area of circle.

    >>> calc_circle_area(1.78)
    9.953822163633902
    """
    return np.pi * radius**2


def calc_circle_radius(area: float) -> float:
    """
    Calculate radius from area.

    >>> calc_circle_radius(10.0)
    1.7841241161527712

    """
    assert not area < 0
    radius = numpy_to_python_type(np.sqrt(area / np.pi))
    assert isinstance(radius, float)
    return radius


def point_to_point_unit_vector(point: Point, other_point: Point) -> np.ndarray:
    """
    Create unit vector from point to other point.

    >>> point = Point(0, 0)
    >>> other_point = Point(1, 1)
    >>> point_to_point_unit_vector(point, other_point)
    array([0.70710678, 0.70710678])
    """
    x1, y1 = point.x, point.y
    x2, y2 = other_point.x, other_point.y
    if any(np.isnan([x1, y1, x2, y2])):
        raise ValueError(f"Expected no nan values: {x1, y1, x2, y2}")
    vector = np.array([x2 - x1, y2 - y1])
    if np.isclose(point.distance(other_point), 0.0, rtol=1e-15, atol=1e-15):
        logging.info(
            "Cannot compute unit vector between points due to close approximity."
        )
        return np.array([np.nan, np.nan])
    normed_vector = vector / np.linalg.norm(vector)
    assert isinstance(normed_vector, np.ndarray)
    return normed_vector


def raise_determination_error(
    attribute: str,
    verb: str = "determining",
    determine_target: str = "topology (=branches and nodes)",
):
    """
    Raise AttributeError if attribute cannot be determined.

    >>> try:
    ...     raise_determination_error("parameters")
    ...     assert False
    ... except AttributeError as exc:
    ...     print(f"{str(exc)[0:20]}...")
    ...
    Cannot determine par...

    """
    raise AttributeError(
        f"Cannot determine {attribute} without {verb} {determine_target}."
    )


def multiprocess(
    function_to_call: Callable,
    keyword_arguments: Sequence,
    arguments_identifier=lambda _: "",
    repeats: int = 0,
) -> List[ProcessResult]:
    """
    Process function calls in parallel.

    Returns result as a list where the error is appended when execution fails.
    """
    # Collect results into a list
    collect_results: List[ProcessResult] = []

    # multiprocessing!
    with ProcessPoolExecutor() as executor:
        keyword_arguments = list(keyword_arguments) * (repeats + 1)

        # Iterate over invalids. submit as tasks
        futures = {
            executor.submit(function_to_call, keyword_arg): keyword_arg
            for keyword_arg in keyword_arguments
        }

        # Collect all tasks as they complete
        # Will not be in same order as submitted
        for future in as_completed(futures):
            identifier = arguments_identifier(futures[future])
            # If execution critically fails it will be caught and logged
            try:
                # Get result from Future
                # This will throw an error (if it happened in process)
                result = future.result()

                # Collect result
                collect_results.append(
                    ProcessResult(identifier=identifier, result=result, error=False)
                )
            except Exception as exc:
                # Catch and log critical failures
                log.error(f"Process exception with {futures[future]}:\n\n" f"{exc}")
                collect_results.append(
                    ProcessResult(identifier=identifier, error=True, result=exc)
                )
    return collect_results


def save_fig(fig: Figure, results_dir: Path, name: str) -> List[Path]:
    """
    Save figure as svg image to results dir.
    """
    assert results_dir.exists() and results_dir.is_dir()
    assert len(name) > 0
    output_paths = []
    for suffix in ("png", "svg"):
        output_path = results_dir / f"{name}.{suffix}"
        fig.savefig(output_path, bbox_inches="tight")
        output_paths.append(output_path)
    return output_paths


@contextmanager
def silent_output(name: str):
    """
    General method to silence output from general func.
    """
    # Create memory files for stdout and stderr
    tmp_io_stdout = StringIO()
    tmp_io_stderr = StringIO()

    # Use double context managers and yield within both
    try:
        with redirect_stdout(tmp_io_stdout):
            with redirect_stderr(tmp_io_stderr):
                yield

    # Report stdout and stderr to log.info
    finally:
        tmp_io_stdout_value = tmp_io_stdout.getvalue()
        tmp_io_stderr_value = tmp_io_stderr.getvalue()
        # Do not log without output
        if len(tmp_io_stdout_value) + len(tmp_io_stderr_value) > 5:
            log.info(
                "silent_output output stdout and stderr.",
                extra=dict(
                    func_name=name,
                    stdout=tmp_io_stdout_value,
                    stderr=tmp_io_stderr_value,
                ),
            )


def wrap_silence(func):
    """
    Wrap function to capture and silence its output.

    Used primarily to silence output from ``powerlaw`` functions and methods.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        with silent_output(func.__name__):
            results = func(*args, **kwargs)
        return results

    return wrapper


def convert_list_columns(gdf: gpd.GeoDataFrame, allow: bool = True) -> gpd.GeoDataFrame:
    """
    Convert list type columns to string.
    """
    gdf = gdf.copy()
    if gdf.empty:
        return gdf
    for column in gdf.columns.values:
        column_data = gdf[column]
        assert isinstance(column_data, pd.Series)
        if isinstance(column_data.values[0], list):
            if not allow:
                raise ValueError(
                    "Trying to write geodata to disk but it has columns with"
                    "`list`-type data and `allow` is set to `False`.\n"
                    "Either set allow to True or fix the data."
                )
            log.info(f"Converting {column} from list to str.")
            gdf[column] = [str(tuple(item)) for item in column_data.values]
    return gdf


def write_geodata(
    gdf: gpd.GeoDataFrame,
    path: Path,
    driver: str = GEOJSON_DRIVER,
    allow_list_column_transform: bool = False,
):
    """
    Write geodata with driver.

    Default is GeoJSON.
    """
    if gdf.empty:
        # Handle empty GeoDataFrames
        path.write_text(gdf.to_json(sort_keys=True))
    else:
        # Convert list type columns to string.
        gdf = convert_list_columns(gdf, allow=allow_list_column_transform)

        # Write to disk
        gdf.to_file(path, driver=driver)

    # Only geojson output is formatted afterwards.
    if driver != GEOJSON_DRIVER:
        return

    # Format geojson with indent of 1
    read_json = path.read_text()
    loaded_json = json.loads(read_json)
    dumped_json = json.dumps(loaded_json, indent=1, sort_keys=True)
    path.write_text(dumped_json)


def write_topodata(gdf: gpd.GeoDataFrame, output_path: Path):
    """
    Write branch or nodes GeoDataFrame to `output_path`.
    """
    write_geodata(gdf=gdf, path=output_path, allow_list_column_transform=False)


def azimuth_to_unit_vector(azimuth: float) -> np.ndarray:
    """
    Convert azimuth to unit vector.
    """
    azimuth_rad = np.deg2rad(azimuth)
    return np.array([np.sin(azimuth_rad), np.cos(azimuth_rad)])


def r2_scorer(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Score fit with r2 metric.

    Changes the scoring to be best at a value of 0.
    """
    score = abs(1 - sklm.r2_score(y_true=y_true, y_pred=y_predicted))
    assert isinstance(score, float)
    return score


def focus_plot_to_bounds(ax: Axes, total_bounds: np.ndarray) -> Axes:
    """
    Focus plot to given bounds.
    """
    xmin, ymin, xmax, ymax = total_bounds
    extend_x = (xmax - xmin) * 0.05
    extend_y = (ymax - ymin) * 0.05
    ax.set_xlim(xmin - extend_x, xmax + extend_x)
    ax.set_ylim(ymin - extend_y, ymax + extend_y)
    return ax


def assign_branch_and_node_colors(feature_type: str) -> str:
    """
    Determine color for each branch and node type.

    Defaults to red.

    >>> assign_branch_and_node_colors("C-C")
    'red'
    """
    if feature_type in (CC_branch, X_node):
        return "green"
    if feature_type in (CI_branch, Y_node):
        return "blue"
    if feature_type in (II_branch, I_node):
        return "black"
    return "red"


def remove_duplicate_caseinsensitive_columns(columns) -> set:
    """
    Remove duplicate columns case-insensitively.
    """
    lower_case_columns = set(column.lower() for column in columns)
    new_cols = set(columns)
    for column in columns:
        if column not in lower_case_columns and column.lower() in columns:
            new_cols.remove(column)
    return new_cols


def write_geodataframe(geodataframe: gpd.GeoDataFrame, name: str, results_dir: Path):
    """
    Save geodataframe as GeoPackage, GeoJSON and shapefile.
    """
    non_dupl_columns = remove_duplicate_caseinsensitive_columns(geodataframe.columns)
    fid_col = "fid"
    for column in geodataframe.columns:
        if column not in non_dupl_columns:
            # print(f"Dropping column: {column}")
            geodataframe.drop(columns=[column], inplace=True)
        if column.lower() == fid_col:
            # Remove fid columns
            # print(f"Dropping column: {column} due to case-insensitive match to fid.")
            geodataframe.drop(columns=[column], inplace=True)
    error_base = "Failed to write {} as {}."
    try:
        geodataframe.to_file(results_dir / f"{name}.gpkg", driver="GPKG")
    except Exception:
        log.error(error_base.format(name, "GeoPackage"), exc_info=True)
    try:
        geodataframe.to_file(results_dir / f"{name}.geojson", driver="GeoJSON")
    except Exception:
        log.error(error_base.format(name, "GeoJSON"), exc_info=True)
    try:
        shp_dir = results_dir / f"{name}_as_shp"
        shp_dir.mkdir(exist_ok=False)
        geodataframe.to_file(shp_dir / f"{name}.shp")
    except Exception:
        log.error(error_base.format(name, "ESRI Shapefile"), exc_info=True)


def check_for_z_coordinates(geodata: Union[gpd.GeoDataFrame, gpd.GeoSeries]) -> bool:
    """
    Check if geopandas data contains Z-coordinates.
    """
    return any(geom.has_z for geom in geodata.geometry.values if hasattr(geom, "has_z"))


def remove_z_coordinates_from_geodata(
    geodata: Union[gpd.GeoDataFrame, gpd.GeoSeries],
) -> Union[gpd.GeoDataFrame, gpd.GeoSeries]:
    """
    Remove Z-coordinates from geometries in geopandasgeodata.
    """
    geodata_without_z = geodata.copy()
    geodata_without_z_geometries = geodata_without_z.geometry.apply(
        remove_z_coordinates
    )
    if isinstance(geodata_without_z, gpd.GeoDataFrame):
        geodata_without_z["geometry"] = geodata_without_z_geometries
    else:
        geodata_without_z = geodata_without_z_geometries

    assert isinstance(geodata_without_z, (gpd.GeoDataFrame, gpd.GeoSeries))
    return geodata_without_z


def remove_z_coordinates(geometry: Union[BaseGeometry, BaseMultipartGeometry]) -> Any:
    """
    Remove z-coordinates from a geometry.

    A ``shapely`` geometry should be provided. Output will always be the same
    type of geometry with the z-coordinates removed.

    :param geometry: Shapely geometry
    :return: Shapely geometry with the same geometry type
    """
    return wkb.loads(wkb.dumps(geometry, output_dimension=2))


def sanitize_name(name: str) -> str:
    """
    Return only alphanumeric parts of name string.
    """
    return "".join(filter(str.isalnum, name))


def check_for_wrong_geometries(traces: gpd.GeoDataFrame, area: gpd.GeoDataFrame):
    """
    Check that traces are line geometries and area contains area geometries.
    """
    accepted_line_types = (LineString, MultiLineString)
    accepted_area_types = (Polygon, MultiPolygon)
    for name, geoms, accepted_types in zip(
        ("traces", "area"), (traces, area), (accepted_line_types, accepted_area_types)
    ):
        if not all(isinstance(geom, accepted_types) for geom in geoms.geometry.values):
            raise TypeError(f"Expected {name} to contain only any of {accepted_types}.")
