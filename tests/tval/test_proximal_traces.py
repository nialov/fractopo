import pandas as pd
import geopandas as gpd
import shapely
from typing import Union
from shapely.geometry import Point, LineString, MultiLineString
import shapely.wkt as wkt
import numpy as np
import hypothesis
import pytest
from hypothesis.strategies import (
    booleans,
    floats,
    sets,
    lists,
    tuples,
    one_of,
    text,
    integers,
)
from hypothesis import given
from hypothesis_geometry import planar

from tests import (
    trace_validator,
    Helpers,
    SNAP_THRESHOLD,
    SNAP_THRESHOLD_ERROR_MULTIPLIER,
    AREA_EDGE_SNAP_MULTIPLIER,
    GEOMETRY_COLUMN,
    ERROR_COLUMN,
    BaseValidator,
    GeomTypeValidator,
    MultiJunctionValidator,
    VNodeValidator,
    MultipleCrosscutValidator,
    TargetAreaSnapValidator,
    UnderlappingSnapValidator,
    GeomNullValidator,
    StackedTracesValidator,
    EmptyGeometryValidator,
    SimpleGeometryValidator,
    SharpCornerValidator,
)
from tests.sample_data import stacked_test
from tests.sample_data.py_samples.stacked_traces_sample import non_stacked_traces_ls

import fractopo.tval.proximal_traces as proximal_traces


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
