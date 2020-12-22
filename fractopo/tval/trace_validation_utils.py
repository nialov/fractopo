from typing import Union, Tuple, List, Optional, Any, Set, Type
import logging

import numpy as np
from shapely.ops import split
from shapely.geometry import (
    MultiPoint,
    Point,
    LineString,
    MultiLineString,
    Polygon,
)
import geopandas as gpd
from geopandas.sindex import PyGEOSSTRTreeIndex

from fractopo.general import point_to_xy, get_trace_endpoints

# Order is important


def segment_within_buffer(
    linestring: LineString,
    multilinestring: MultiLineString,
    snap_threshold: float,
    snap_threshold_error_multiplier: float,
    overlap_detection_multiplier: float,
):
    """
    First checks if given linestring completely overlaps any part of
    multilinestring and if it does, returns True.

    Next it starts to segmentize the multilinestring to smaller
    linestrings and consequently checks if these segments are completely
    within a buffer made of the given linestring.
    It also checks that the segment size is reasonable.

    TODO: segmentize_linestring is very inefficient.
    """
    # Test for a single segment overlap
    if linestring.overlaps(multilinestring):
        return True
    buffered_linestring = linestring.buffer(
        snap_threshold * snap_threshold_error_multiplier
    )
    assert isinstance(linestring, LineString)
    assert isinstance(buffered_linestring, Polygon)
    assert isinstance(multilinestring, MultiLineString)
    assert buffered_linestring.area > 0
    # Test for overlap with a buffered linestring
    all_segments: List[LineString]
    all_segments = []
    ls: LineString
    for ls in multilinestring:
        all_segments.extend(
            segmentize_linestring(ls, snap_threshold * overlap_detection_multiplier)
        )
    seg: LineString
    for seg in all_segments:
        if (
            seg.within(buffered_linestring)
            and seg.length > snap_threshold * overlap_detection_multiplier
        ):
            return True
    return False


def segmentize_linestring(linestring: LineString, threshold_length: float):
    start_point, end_point = get_trace_endpoints(linestring)
    segments = []
    for dist in np.arange(0.0, linestring.length, threshold_length):
        start = linestring.interpolate(dist)
        end = linestring.interpolate(dist + threshold_length)
        segment = LineString([start, end])
        if segment.length < threshold_length:
            continue
        segments.append(segment)
    return segments


def split_to_determine_triangle_errors(
    trace: LineString,
    splitter_trace: LineString,
    snap_threshold: float,
    triangle_error_snap_multiplier: float,
):
    assert isinstance(trace, LineString)
    assert isinstance(splitter_trace, LineString)
    try:
        segments = split(trace, splitter_trace)
    except ValueError:
        # split not possible, the traces overlap
        return True
    if len(segments) > 2:
        middle = determine_middle_in_triangle(
            [ls for ls in segments.geoms],
            snap_threshold=snap_threshold,
            snap_threshold_error_multiplier=triangle_error_snap_multiplier,
        )
        if len(middle) > 0:
            seg_lengths: List[float] = [seg.length for seg in middle]
        else:
            seg_lengths: List[float] = [seg.length for seg in segments]
        for seg_length in seg_lengths:
            if (
                snap_threshold / triangle_error_snap_multiplier
                < seg_length
                < snap_threshold * triangle_error_snap_multiplier
            ):
                return True
    return False


def determine_middle_in_triangle(
    segments: List[LineString],
    snap_threshold: float,
    snap_threshold_error_multiplier: float,
) -> List[LineString]:
    """
    Determine the middle segment within a triangle error.

    The middle segment always intersects the other two.
    """
    buffered = [
        ls.buffer(snap_threshold * snap_threshold_error_multiplier) for ls in segments
    ]
    candidates = []
    for idx, (buffer, linestring) in enumerate(zip(buffered, segments)):
        others = buffered.copy()
        others.pop(idx)
        if sum([buffer.intersects(other) for other in others]) >= 2:
            candidates.append(segments[idx])
    return candidates


def determine_trace_candidates(
    geom: LineString,
    idx: int,
    traces: gpd.GeoDataFrame,
    spatial_index: Optional[PyGEOSSTRTreeIndex],
):
    if spatial_index is None:
        logging.error("Expected spatial_index not be None.")
        return gpd.GeoSeries()
    assert isinstance(traces, (gpd.GeoSeries, gpd.GeoDataFrame))
    assert isinstance(spatial_index, PyGEOSSTRTreeIndex)
    candidate_idxs = list(spatial_index.intersection(geom.bounds))
    candidate_idxs.remove(idx)  # type: ignore
    candidate_traces: gpd.GeoSeries = traces.geometry.iloc[candidate_idxs]
    candidate_traces = candidate_traces.loc[  # type: ignore
        [isinstance(geom, LineString) for geom in candidate_traces.geometry.values]
    ]
    return candidate_traces
