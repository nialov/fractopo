"""
Tests for proximal trace detection utility.
"""
from typing import Union

import geopandas as gpd
import pytest

import tests
from fractopo.general import read_geofile
from fractopo.tval import proximal_traces


@pytest.mark.parametrize(
    "traces,buffer_value,azimuth_tolerance",
    tests.test_determine_proximal_traces_params,
)
def test_determine_proximal_traces(
    traces: Union[gpd.GeoDataFrame, gpd.GeoSeries], buffer_value, azimuth_tolerance
):
    """
    Test determine_proximal_traces.
    """
    result = proximal_traces.determine_proximal_traces(
        traces, buffer_value, azimuth_tolerance
    )
    assert proximal_traces.MERGE_COLUMN in result.columns
    assert isinstance(result, gpd.GeoDataFrame)


def test_determine_proximal_traces_regression(file_regression):
    """
    Test determine_proximal_traces with regression.
    """
    traces = read_geofile(tests.sample_trace_100_data)
    traces.reset_index(drop=True, inplace=True)
    assert isinstance(traces, gpd.GeoDataFrame) and len(traces) > 0
    buffer_value = 1
    azimuth_tolerance = 30
    result = proximal_traces.determine_proximal_traces(
        traces, buffer_value, azimuth_tolerance  # type: ignore
    )
    assert proximal_traces.MERGE_COLUMN in result.columns
    assert isinstance(result, gpd.GeoDataFrame)
    # Sort index and then sort by columns so file refression can work
    result.sort_index(inplace=True)
    result.sort_index(axis="columns", inplace=True)
    file_regression.check(result.to_json(indent=1, sort_keys=True))
