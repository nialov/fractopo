"""
Direct utilities of trace validation.
"""

import logging
from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
from geopandas.sindex import SpatialIndex
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point, Polygon
from shapely.ops import split

from fractopo.general import (
    geom_bounds,
    safe_buffer,
    spatial_index_intersection,
    within_bounds,
)

log = logging.getLogger(__name__)


def segment_within_buffer(
    linestring: LineString,
    multilinestring: MultiLineString,
    snap_threshold: float,
    snap_threshold_error_multiplier: float,
    overlap_detection_multiplier: float,
    stacked_detector_buffer_multiplier: float,
) -> bool:
    """
    Check if segment is within buffer of multilinestring.

    First check if given linestring completely overlaps any part of
    multilinestring and if it does, returns True.

    Next it starts to segmentize the multilinestring to smaller
    linestrings and consequently checks if these segments are completely
    within a buffer made of the given linestring.
    It also checks that the segment size is reasonable.

    TODO: segmentize_linestring is very inefficient.
    """
    if multilinestring.is_empty:
        return False
    # Test for a single segment overlap
    if linestring.overlaps(multilinestring):
        if not isinstance(
            linestring.intersection(multilinestring), (Point, MultiPoint)
        ):
            return True
        log.warning(
            "Expected the intersection of overlapping geometries to not be a Point"
        )

    buffered_linestring = safe_buffer(
        linestring,
        (snap_threshold * snap_threshold_error_multiplier)
        * stacked_detector_buffer_multiplier,
    )
    assert isinstance(linestring, LineString)
    assert isinstance(buffered_linestring, Polygon)
    assert isinstance(multilinestring, MultiLineString)
    assert buffered_linestring.area > 0
    min_x, min_y, max_x, max_y = geom_bounds(buffered_linestring)

    # Crop MultiLineString near to the buffered_linestring
    if not buffered_linestring.intersects(multilinestring):
        return False
    cropped_mls = buffered_linestring.intersection(multilinestring)

    # Check for cases with no chance of stacking
    if cropped_mls.is_empty or (
        isinstance(cropped_mls, LineString)
        and cropped_mls.length < snap_threshold * overlap_detection_multiplier
    ):
        return False

    assert isinstance(cropped_mls, (MultiLineString, LineString))

    all_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    ls: LineString

    # Create list of LineStrings from within the crop
    mls_geoms: List[LineString] = (
        list(cropped_mls.geoms)
        if isinstance(cropped_mls, MultiLineString)
        else [cropped_mls]
    )

    # Iterate over list of LineStrings
    for ls in mls_geoms:
        all_segments.extend(
            segmentize_linestring(ls, snap_threshold * overlap_detection_multiplier)
        )
    for start, end in all_segments:
        if within_bounds(
            x=start[0], y=start[1], min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y
        ) and within_bounds(
            x=end[0],
            y=end[1],
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
            # *end, min_x, min_y, max_x, max_y
        ):
            ls = LineString([start, end])
            if ls.length > snap_threshold * overlap_detection_multiplier and ls.within(
                buffered_linestring
            ):
                return True
    return False


def segmentize_linestring(
    linestring: LineString, threshold_length: float
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Segmentize LineString to smaller parts.

    Resulting parts are not guaranteed to be mergeable back to the original.
    """
    assert isinstance(linestring, LineString)
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for dist in np.arange(0.0, linestring.length, threshold_length):
        segments.append(linestring_segment(linestring, dist, threshold_length))

    return segments


def linestring_segment(linestring: LineString, dist: float, threshold_length: float):
    """
    Get LineString segment from dist to dist + threshold_length.
    """
    coord_1 = linestring.interpolate(dist).coords[0]
    coord_2 = linestring.interpolate(dist + threshold_length).coords[0]
    return coord_1, coord_2


def split_to_determine_triangle_errors(
    trace: LineString,
    splitter_trace: LineString,
    snap_threshold: float,
    triangle_error_snap_multiplier: float,
) -> bool:
    """
    Split trace with splitter_trace to determine triangle intersections.
    """
    assert isinstance(trace, LineString)
    assert isinstance(splitter_trace, LineString)
    try:
        segments = split(trace, splitter_trace)
    except (ValueError, TypeError):
        trace_intersection = trace.intersection(splitter_trace)
        log.info(
            "Failed to split trace with splitter_trace.",
            extra=dict(
                trace=trace.wkt,
                splitter_trace=splitter_trace.wkt,
                trace_intersection_wkt=(
                    trace_intersection.wkt
                    if hasattr(trace_intersection, "wkt")
                    else "No wkt attribute."
                ),
            ),
            exc_info=True,
        )

        if isinstance(trace_intersection, Point):
            # Splitting can fail in a few different ways e.g. with
            # TypeError: object of type 'LineString' has no len()
            # shapely/**/collection.py\", line 64, in geos_geometrycollection_from_py
            # In this case a simple solution if there's just a single point
            # intersection between traces == No triangle error
            # NOTE: not an exhaustive test

            # Check that case follows expectation... (no overlap)
            # assert not trace.overlaps(splitter_trace)
            log.info(
                "Failed to split but intersection was a single point.",
                extra=dict(
                    trace_intersection_wkt=trace_intersection.wkt,
                    trace=trace.wkt,
                    splitter_trace=splitter_trace.wkt,
                    traces_overlap=trace.overlaps(splitter_trace),
                ),
            )
            return False

        # split not possible, the traces overlap
        return True
    if len(segments.geoms) > 2:
        if len(segments.geoms) > 3:
            return True
        middle = determine_middle_in_triangle(
            list(segments.geoms),
            snap_threshold=snap_threshold,
            snap_threshold_error_multiplier=triangle_error_snap_multiplier,
        )
        if len(middle) > 0:
            seg_lengths: List[float] = [seg.length for seg in middle]
        else:
            seg_lengths = [seg.length for seg in segments.geoms]
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
    candidates = []
    for idx, linestring in enumerate(segments):
        others = segments.copy()
        others.pop(idx)
        if (
            sum(
                linestring.distance(other)
                < snap_threshold * snap_threshold_error_multiplier
                for other in others
            )
            >= 2
        ):
            candidates.append(segments[idx])
    return candidates


def determine_trace_candidates(
    geom: LineString,
    idx: int,
    traces: gpd.GeoDataFrame,
    spatial_index: Optional[SpatialIndex],
) -> gpd.GeoSeries:
    """
    Determine potentially intersecting traces with spatial index.
    """
    if spatial_index is None:
        log.error("Expected spatial_index not be None.")
        return gpd.GeoSeries()
    assert isinstance(traces, (gpd.GeoSeries, gpd.GeoDataFrame))
    assert isinstance(spatial_index, SpatialIndex)
    candidate_idxs = spatial_index_intersection(spatial_index, geom_bounds(geom))
    candidate_idxs.remove(idx)
    candidate_traces: gpd.GeoSeries = traces.geometry.iloc[candidate_idxs]
    candidate_traces = candidate_traces.loc[  # type: ignore
        [isinstance(geom, LineString) for geom in candidate_traces.geometry.values]
    ]
    return candidate_traces


def is_underlapping(
    geom: LineString,
    trace: LineString,
    endpoint: Point,
    snap_threshold: float,
    snap_threshold_error_multiplier: float,
) -> Optional[bool]:
    """
    Determine if a geom is underlapping.
    """
    try:
        split_results = list(split(geom, trace).geoms)
    except ValueError:
        log.warning(
            "Expected split to work between geom and trace. Probably overlapping geometries.",
            exc_info=True,
        )
        return None
    if len(split_results) == 1:
        # Do not intersect
        return True
    if len(split_results) > 1:
        for segment in split_results:
            if (
                segment.distance(endpoint)
                < snap_threshold * snap_threshold_error_multiplier
            ):
                # Dangling end, overlapping
                return False
    log_prints = {
        "geom": geom,
        "trace": trace,
        "endpoint": endpoint,
        "snap_threshold": snap_threshold,
        "snap_threshold_error_multiplier": snap_threshold_error_multiplier,
    }
    log.error(f"Expected is_underlapping to be resolvable.\nvalues:{log_prints}")
    return None
