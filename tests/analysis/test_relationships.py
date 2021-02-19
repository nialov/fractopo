"""
Tests for relationship detection.
"""
from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from shapely.geometry import MultiLineString
from shapely.prepared import PreparedGeometry

from fractopo.analysis.relationships import (
    determine_crosscut_abutting_relationships,
    determine_intersects,
    determine_nodes_intersecting_sets,
    plot_crosscut_abutting_relationships_plot,
)
from fractopo.general import prepare_geometry_traces
from tests import Helpers


@pytest.mark.parametrize(
    "trace_series_two_sets, set_array, set_names_two_sets,"
    "node_series_xy, buffer_value, assumed_intersections",
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
    """
    Test determine_nodes_intersecting_sets.
    """
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


@pytest.mark.parametrize(
    "trace_series_two_sets, set_names_two_sets, node_series_xy_intersects, node_types_xy_intersects, buffer_value",
    Helpers.test_determine_intersects_params,
)
def test_determine_intersects(
    trace_series_two_sets: Tuple[gpd.GeoSeries, gpd.GeoSeries],
    set_names_two_sets: Tuple[str, str],
    node_series_xy_intersects: gpd.GeoSeries,
    node_types_xy_intersects: np.ndarray,
    buffer_value: float,
):
    """
    Test determine_intersects.
    """
    assert isinstance(trace_series_two_sets, tuple)
    assert isinstance(set_names_two_sets, tuple)
    assert isinstance(node_series_xy_intersects, gpd.GeoSeries)
    assert isinstance(node_types_xy_intersects, np.ndarray)
    assert isinstance(buffer_value, float)
    intersectframe = determine_intersects(
        trace_series_two_sets=trace_series_two_sets,
        set_names_two_sets=set_names_two_sets,
        node_series_xy_intersects=node_series_xy_intersects,
        node_types_xy_intersects=node_types_xy_intersects,
        buffer_value=buffer_value,
    )
    assert isinstance(intersectframe, pd.DataFrame)
    expected_cols = ["node", "nodeclass", "sets", "error"]
    assert all([col in intersectframe.columns for col in expected_cols])


@pytest.mark.parametrize(
    "trace_series, node_series, node_types, set_array, set_names, buffer_value, label",
    Helpers.test_determine_crosscut_abutting_relationships_params,
)
def test_determine_crosscut_abutting_relationships(
    trace_series: gpd.GeoSeries,
    node_series: gpd.GeoSeries,
    node_types: np.ndarray,
    set_array: np.ndarray,
    set_names: Tuple[str, ...],
    buffer_value: float,
    label: str,
):
    """
    Test determine_crosscut_abutting_relationships.
    """
    assert isinstance(trace_series, gpd.GeoSeries)
    assert isinstance(node_series, gpd.GeoSeries)
    assert isinstance(node_types, np.ndarray)
    assert isinstance(set_array, np.ndarray)
    assert isinstance(set_names, tuple)
    assert isinstance(buffer_value, float)
    assert isinstance(label, str)
    relations_df = determine_crosscut_abutting_relationships(
        trace_series=trace_series,
        node_series=node_series,
        node_types=node_types,
        set_array=set_array,
        set_names=set_names,
        buffer_value=buffer_value,
        label=label,
    )
    assert isinstance(relations_df, pd.DataFrame)
    expected_cols = ["name", "sets", "x", "y", "y-reverse", "error-count"]
    assert all([col in relations_df.columns for col in expected_cols])


def test_plot_crosscut_abutting_relationships_plot():
    """
    Test plot_crosscut_abutting_relationships_plot.
    """
    params = Helpers.test_determine_crosscut_abutting_relationships_params[0]
    relations_df = determine_crosscut_abutting_relationships(*params)
    set_array = params[3]
    set_names = params[4]
    assert isinstance(set_array, np.ndarray)
    assert isinstance(set_names, tuple)
    figs, fig_axes = plot_crosscut_abutting_relationships_plot(
        relations_df=relations_df, set_array=set_array, set_names=set_names
    )
    assert all([isinstance(fig, Figure) for fig in figs])  # type: ignore
    plt.close()
