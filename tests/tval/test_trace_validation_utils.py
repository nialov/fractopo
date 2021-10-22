"""
Test trace_validation_utils.
"""
import pytest
from shapely.geometry import LineString

import fractopo.tval.trace_validation_utils as trace_validation_utils
from tests import Helpers


@pytest.mark.parametrize(
    "linestring,multilinestring,snap_threshold,snap_threshold_error_multiplier,"
    "overlap_detection_multiplier,assume_result",
    Helpers.test_segment_within_buffer_params,
)
def test_segment_within_buffer(
    linestring,
    multilinestring,
    snap_threshold,
    snap_threshold_error_multiplier,
    overlap_detection_multiplier,
    assume_result,
):
    """
    Test segment_within_buffer.
    """
    result = trace_validation_utils.segment_within_buffer(
        linestring,
        multilinestring,
        snap_threshold,
        snap_threshold_error_multiplier,
        overlap_detection_multiplier,
    )
    assert isinstance(result, bool)
    assert assume_result == result


@pytest.mark.parametrize(
    "linestring,threshold_length,assume_result_size",
    Helpers.test_segmentize_linestring_params,
)
def test_segmentize_linestring(linestring, threshold_length, assume_result_size):
    """
    Test segmentize_linestring.
    """
    result = trace_validation_utils.segmentize_linestring(linestring, threshold_length)
    assert isinstance(result, list)
    assert all(isinstance(val, tuple) for val in result)
    assert len(result) == assume_result_size


@pytest.mark.parametrize(
    "trace,splitter_trace,snap_threshold,"
    "triangle_error_snap_multiplier,assumed_result",
    Helpers.test_split_to_determine_triangle_errors_params,
)
def test_split_to_determine_triangle_errors(
    trace,
    splitter_trace,
    snap_threshold,
    triangle_error_snap_multiplier,
    assumed_result,
):
    """
    Test split_to_determine_triangle_errors.
    """
    result = trace_validation_utils.split_to_determine_triangle_errors(
        trace,
        splitter_trace,
        snap_threshold,
        triangle_error_snap_multiplier,
    )
    assert isinstance(result, bool)
    assert assumed_result == result


@pytest.mark.parametrize(
    "segments,snap_threshold," "snap_threshold_error_multiplier,assumed_result",
    Helpers.test_determine_middle_in_triangle_params,
)
def test_determine_middle_in_triangle(
    segments, snap_threshold, snap_threshold_error_multiplier, assumed_result
):
    """
    Test determine_middle_in_triangle.
    """
    result = trace_validation_utils.determine_middle_in_triangle(
        segments, snap_threshold, snap_threshold_error_multiplier
    )

    assert isinstance(result, list)
    assert all(isinstance(val, LineString) for val in result)
    assert assumed_result == result
