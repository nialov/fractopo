"""
Determine traces that could be integrated together.

determine_proximal_traces takes an input of GeoSeries or GeoDataFrame
of LineString geometries and returns a GeoDataFrame with a new
column `Merge` which has values of True or False depending on if
nearby proximal traces were found.
"""
from typing import List, Union

import geopandas as gpd
from fractopo.general import (
    determine_azimuth,
    determine_regression_azimuth,
    is_azimuth_close,
)
from shapely.geometry import LineString, Point


MERGE_COLUMN = "Merge"


def is_within_buffer_distance(
    trace: LineString, other: LineString, buffer_value: float
):
    """
    Determine if `trace` and `other` are within buffer distance.

    Threshold distance is buffer_value (both are buffered with half of
    buffer_value) of each other.

    E.g.

    >>> trace = LineString([(0, 0), (0, 3)])
    >>> other = LineString([(0, 0), (0, 3)])
    >>> is_within_buffer_distance(trace, other, 0.1)
    True

    >>> line = LineString([(0, 0), (0, 3)])
    >>> other = LineString([(3, 0), (3, 3)])
    >>> is_within_buffer_distance(trace, other, 2)
    False
    >>> is_within_buffer_distance(trace, other, 4)
    True

    """
    return trace.buffer(buffer_value / 2).intersects(other.buffer(buffer_value / 2))


def is_similar_azimuth(trace: LineString, other: LineString, tolerance: float):
    """
    Determine if azimuths of `trace` and `other` are close.

    Checks both the start -- end -azimuth and regression-based azimuth.

    E.g.

    >>> trace = LineString([(0, 0), (0, 3)])
    >>> other = LineString([(0, 0), (0, 4)])
    >>> is_similar_azimuth(trace, other, 1)
    True

    >>> trace = LineString([(0, 0), (1, 1)])
    >>> other = LineString([(0, 0), (0, 4)])
    >>> is_similar_azimuth(trace, other, 40)
    False
    >>> is_similar_azimuth(trace, other, 50)
    True
    >>> is_similar_azimuth(trace, other, 45)
    False

    """
    return is_azimuth_close(
        determine_azimuth(trace, halved=True),
        determine_azimuth(other, halved=True),
        tolerance=tolerance,
        halved=True,
    ) or is_azimuth_close(
        determine_regression_azimuth(trace),
        determine_regression_azimuth(other),
        tolerance=tolerance,
        halved=True,
    )


def determine_proximal_traces(
    traces: Union[gpd.GeoSeries, gpd.GeoDataFrame],
    buffer_value: float,
    azimuth_tolerance: float,
) -> gpd.GeoDataFrame:
    """
    Determine proximal traces.

    Takes an input of GeoSeries or GeoDataFrame
    of LineString geometries and returns a GeoDataFrame with a new
    column `Merge` which has values of True or False depending on if
    nearby proximal traces were found.

    E.g.

    >>> lines = [
    ...     LineString([(0, 0), (0, 3)]),
    ...     LineString([(1, 0), (1, 3)]),
    ...     LineString([(5, 0), (5, 3)]),
    ...     LineString([(0, 0), (-3, -3)]),
    ... ]
    >>> traces = gpd.GeoDataFrame({"geometry": lines})
    >>> buffer_value = 1.1
    >>> azimuth_tolerance = 10
    >>> determine_proximal_traces(traces, buffer_value, azimuth_tolerance)
                                              geometry  Merge
    0    LINESTRING (0.00000 0.00000, 0.00000 3.00000)   True
    1    LINESTRING (1.00000 0.00000, 1.00000 3.00000)   True
    2    LINESTRING (5.00000 0.00000, 5.00000 3.00000)  False
    3  LINESTRING (0.00000 0.00000, -3.00000 -3.00000)  False

    """
    assert isinstance(traces, (gpd.GeoSeries, gpd.GeoDataFrame))
    if isinstance(traces, gpd.GeoSeries):
        traces_as_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(geometry=traces)
    else:
        traces_as_gdf = traces
    traces_as_gdf = traces_as_gdf.reset_index(inplace=False, drop=True)  # type: ignore
    spatial_index = traces_as_gdf.sindex
    trace: LineString
    proximal_traces: List[int] = []
    for idx, trace in enumerate(traces_as_gdf.geometry):  # type: ignore
        candidate_idxs = list(
            spatial_index.intersection(trace.buffer(buffer_value * 5).bounds)
        )
        candidate_idxs.remove(idx)  # type: ignore
        candidate_traces: Union[gpd.GeoSeries, gpd.GeoDataFrame] = traces_as_gdf.iloc[
            candidate_idxs
        ]
        candidate_traces = candidate_traces.loc[  # type: ignore
            [
                is_within_buffer_distance(trace, other, buffer_value)
                and is_similar_azimuth(trace, other, tolerance=azimuth_tolerance)
                for other in candidate_traces.geometry  # type: ignore
            ]
        ]
        if len(candidate_traces) > 0:
            proximal_traces.extend(
                [
                    i
                    for i in list(candidate_traces.index) + [idx]  # type: ignore
                    if i not in proximal_traces
                ]
            )
    traces_as_gdf[MERGE_COLUMN] = [i in proximal_traces for i in traces_as_gdf.index]
    return traces_as_gdf
