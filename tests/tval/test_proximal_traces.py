"""
Tests for proximal trace detection utility.
"""
from typing import Union

import geopandas as gpd
import pytest

from fractopo.general import read_geofile
from fractopo.tval import proximal_traces
from tests import Helpers


@pytest.mark.parametrize(
    "traces,buffer_value,azimuth_tolerance",
    Helpers.test_determine_proximal_traces_params,
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
    traces = read_geofile(Helpers.sample_trace_100_data)
    traces.reset_index(drop=True, inplace=True)
    assert isinstance(traces, gpd.GeoDataFrame) and len(traces) > 0
    buffer_value = 1
    azimuth_tolerance = 30
    result = proximal_traces.determine_proximal_traces(
        traces, buffer_value, azimuth_tolerance  # type: ignore
    )
    assert proximal_traces.MERGE_COLUMN in result.columns
    assert isinstance(result, gpd.GeoDataFrame)
    file_regression.check(result.sort_index().to_json(indent=1))
