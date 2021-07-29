"""
Tests for proximal trace detection utility.
"""
from typing import Union

import geopandas as gpd
import pytest

from fractopo.tval import proximal_traces
from tests import Helpers


@pytest.mark.parametrize(
    "traces,buffer_value,azimuth_tolerance",
    Helpers.test_is_within_buffer_distance_params,
)
def test_is_within_buffer_distance(
    traces: Union[gpd.GeoDataFrame, gpd.GeoSeries], buffer_value, azimuth_tolerance
):
    """
    Test is_within_buffer_distance.
    """
    result = proximal_traces.determine_proximal_traces(
        traces, buffer_value, azimuth_tolerance
    )
    assert proximal_traces.MERGE_COLUMN in result.columns
    assert isinstance(result, gpd.GeoDataFrame)


def test_is_within_buffer_distance_regression(file_regression):
    """
    Test is_within_buffer_distance with regression.
    """
    traces = gpd.read_file(Helpers.sample_trace_data).iloc[0:100]
    assert isinstance(traces, gpd.GeoDataFrame) and len(traces) > 0
    buffer_value = 1
    azimuth_tolerance = 30
    result = proximal_traces.determine_proximal_traces(
        traces, buffer_value, azimuth_tolerance  # type: ignore
    )
    assert proximal_traces.MERGE_COLUMN in result.columns
    assert isinstance(result, gpd.GeoDataFrame)
    file_regression.check(result.sort_index().to_json(indent=1))
