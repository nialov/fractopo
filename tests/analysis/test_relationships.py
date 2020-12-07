from typing import Tuple, Optional, List, Dict

import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import MultiLineString
from shapely.prepared import PreparedGeometry

from tests import Helpers
from fractopo.general import prepare_geometry_traces
from fractopo.analysis.relationships import (
    determine_crosscut_abutting_relationships,
    determine_nodes_intersecting_sets,
)


@pytest.mark.parametrize(
    "trace_series_two_sets, set_array, set_names_two_sets, node_series_xy, buffer_value, assumed_intersections",
    Helpers.test_determine_nodes_intersecting_sets_params,
)
def test_determine_nodes_intersecting_sets(
    trace_series_two_sets: Tuple[gpd.GeoSeries, gpd.GeoSeries],
    set_array: np.ndarray,
    set_names_two_sets: Tuple[str, str],
    node_series_xy: gpd.GeoSeries,
    buffer_value: float,
    assumed_intersections: Optional[List[bool]],
):
    intersects_both_sets = determine_nodes_intersecting_sets(
        trace_series_two_sets,
        set_array,
        set_names_two_sets,
        node_series_xy,
        buffer_value,
    )
    assert isinstance(intersects_both_sets, list)
    if assumed_intersections is not None:
        assert len(assumed_intersections) == len(intersects_both_sets)
        assert sum(assumed_intersections) == sum(intersects_both_sets)
        assert assumed_intersections == intersects_both_sets


@pytest.mark.parametrize(
    "trace_series",
    Helpers.test_prepare_geometry_traces_params,
)
def test_prepare_geometry_traces(trace_series: gpd.GeoSeries):
    prepared_traces = prepare_geometry_traces(trace_series)
    assert isinstance(prepared_traces, PreparedGeometry)
    assert isinstance(prepared_traces.context, MultiLineString)
    assert all(
        [prepared_traces.intersects(trace) for trace in trace_series.geometry.values]
    )
