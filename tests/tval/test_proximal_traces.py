from typing import Union

import geopandas as gpd
import hypothesis
import numpy as np
import pandas as pd
import pytest
import shapely
from hypothesis import given
from hypothesis.strategies import (
    booleans,
    floats,
    integers,
    lists,
    one_of,
    sets,
    text,
    tuples,
)
from hypothesis_geometry import planar
from shapely import wkt
from shapely.geometry import LineString, MultiLineString, Point
from tests import (
    AREA_EDGE_SNAP_MULTIPLIER,
    ERROR_COLUMN,
    GEOMETRY_COLUMN,
    SNAP_THRESHOLD,
    SNAP_THRESHOLD_ERROR_MULTIPLIER,
    BaseValidator,
    GeomNullValidator,
    GeomTypeValidator,
    Helpers,
    MultiJunctionValidator,
    MultipleCrosscutValidator,
    SharpCornerValidator,
    SimpleGeometryValidator,
    StackedTracesValidator,
    TargetAreaSnapValidator,
    UnderlappingSnapValidator,
    VNodeValidator,
)
from tests.sample_data import stacked_test
from tests.sample_data.py_samples.stacked_traces_sample import non_stacked_traces_ls

from fractopo.tval import proximal_traces


@pytest.mark.parametrize(
    "traces,buffer_value,azimuth_tolerance",
    Helpers.test_is_within_buffer_distance_params,
)
def test_is_within_buffer_distance(
    traces: Union[gpd.GeoDataFrame, gpd.GeoSeries], buffer_value, azimuth_tolerance
):
    result = proximal_traces.determine_proximal_traces(
        traces, buffer_value, azimuth_tolerance
    )
    assert proximal_traces.MERGE_COLUMN in result.columns
    assert isinstance(result, gpd.GeoDataFrame)


def test_is_within_buffer_distance_regression(file_regression, tmp_path):
    traces = gpd.read_file(Helpers.sample_trace_data).iloc[0:100]
    assert isinstance(traces, gpd.GeoDataFrame) and len(traces) > 0
    buffer_value = 1
    azimuth_tolerance = 30
    result = proximal_traces.determine_proximal_traces(
        traces, buffer_value, azimuth_tolerance  # type: ignore
    )
    assert proximal_traces.MERGE_COLUMN in result.columns
    assert isinstance(result, gpd.GeoDataFrame)
    as_geojson = result.to_json()
    file_regression.check(as_geojson)
