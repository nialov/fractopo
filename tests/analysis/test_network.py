"""
Tests for Network.
"""
from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Polygon

from fractopo import SetRangeTuple
from fractopo.analysis.network import Network
from tests import Helpers


def test_azimuth_set_relationships_regression(file_regression):
    """
    Test for azimuth set relationship regression.
    """
    azimuth_set_ranges: SetRangeTuple = (
        (0, 60),
        (60, 120),
        (120, 180),
    )
    azimuth_set_names: Tuple[str, ...] = ("1", "2", "3")
    relations_df: pd.DataFrame = Network(
        Helpers.kb7_traces,  # type: ignore
        Helpers.kb7_area,  # type: ignore
        name="kb7",
        determine_branches_nodes=True,
        azimuth_set_ranges=azimuth_set_ranges,
        azimuth_set_names=azimuth_set_names,
        snap_threshold=0.001,
    ).azimuth_set_relationships
    file_regression.check(relations_df.to_string())


def test_length_set_relationships_regression(file_regression):
    """
    Test for length set relationship regression.
    """
    trace_length_set_ranges: SetRangeTuple = (
        (0, 2),
        (2, 4),
        (4, 6),
    )
    trace_length_set_names: Tuple[str, ...] = ("a", "b", "c")
    relations_df: pd.DataFrame = Network(
        Helpers.kb7_traces,  # type: ignore
        Helpers.kb7_area,  # type: ignore
        name="kb7",
        determine_branches_nodes=True,
        trace_length_set_names=trace_length_set_names,
        trace_length_set_ranges=trace_length_set_ranges,
        snap_threshold=0.001,
    ).azimuth_set_relationships
    file_regression.check(relations_df.to_string())


@pytest.mark.parametrize(
    "traces,area,name,determine_branches_nodes,truncate_traces,snap_threshold",
    Helpers.test_network_params,
)
def test_network(
    traces,
    area,
    name,
    determine_branches_nodes,
    truncate_traces,
    snap_threshold,
    file_regression,
    data_regression,
    num_regression,
):
    """
    Test Network object creation and attributes with general datasets.

    Tests for regression and general assertions.
    """
    network = Network(
        trace_gdf=traces,
        area_gdf=area,
        name=name,
        determine_branches_nodes=determine_branches_nodes,
        truncate_traces=truncate_traces,
        snap_threshold=snap_threshold,
    )

    assert area.shape[0] == len(network.representative_points())
    assert isinstance(network.numerical_network_description(), dict)
    for key, item in network.numerical_network_description().items():
        assert isinstance(key, str)
        if not isinstance(item, (int, float)):
            assert isinstance(item.item(), (int, float))

    assert isinstance(network.target_areas, list)
    assert all(
        [isinstance(val, (Polygon, MultiPolygon)) for val in network.target_areas]
    )

    if network.branch_gdf.shape[0] < 2500:
        # Do not check massive branch counts
        file_regression.check(network.branch_gdf.sort_index().to_json())

    network_attributes = dict()
    for attribute in ("node_counts", "branch_counts"):
        # network_attributes[attribute] = getattr(network, attribute)
        for key, value in getattr(network, attribute).items():
            network_attributes[key] = int(value)

        for key, value in network.numerical_network_description().items():
            try:
                network_attributes[key] = value.item()
            except AttributeError:
                network_attributes[key] = value

    data_regression.check(network_attributes)

    assert isinstance(network.trace_intersects_target_area_boundary, np.ndarray)
    assert network.trace_intersects_target_area_boundary.dtype == "int"
    assert isinstance(network.branch_intersects_target_area_boundary, np.ndarray)
    assert network.branch_intersects_target_area_boundary.dtype == "int64"

    num_regression.check(
        {
            "branch_boundary_intersects": list(
                network.branch_intersects_target_area_boundary
            ),
        }
    )


def test_network_kb11_manual():
    """
    Test Network analysis with KB11 data.

    Returns the Network.
    """
    trace_gdf = Helpers.kb11_traces
    area_gdf = Helpers.kb11_area
    network = Network(
        trace_gdf=trace_gdf,
        area_gdf=area_gdf,
        name="KB11 test",
        determine_branches_nodes=True,
        truncate_traces=True,
        snap_threshold=0.001,
    )
    return network


@pytest.mark.parametrize(
    "trace_gdf,area_gdf,name",
    Helpers.test_network_circular_target_area_params,
)
def test_network_circular_target_area(trace_gdf, area_gdf, name):
    """
    Test circular_target_area var.
    """
    network_circular = Network(
        trace_gdf=trace_gdf,
        area_gdf=area_gdf,
        name=name,
        circular_target_area=True,
        determine_branches_nodes=False,
    )
    network_non_circular = Network(
        trace_gdf=trace_gdf,
        area_gdf=area_gdf,
        name=name,
        circular_target_area=False,
        determine_branches_nodes=False,
    )

    lengths_circular = network_circular.trace_length_array
    lengths_non_circular = network_non_circular.trace_length_array

    assert np.isclose(lengths_circular[0], lengths_non_circular[0] * 2)
    assert np.isclose(lengths_circular[1], lengths_non_circular[1])
    assert np.isclose(lengths_circular[2], 0)
    assert not np.isclose(lengths_non_circular[2], 0)

    assert all(
        np.isclose(
            network_circular.trace_length_array_non_weighted, lengths_non_circular
        )
    )
    assert all(
        np.isclose(
            network_non_circular.trace_length_array_non_weighted, lengths_non_circular
        )
    )


def test_getaberget_fault_network(data_regression):
    """
    Debug test with material causing unary_union fault.
    """
    traces = Helpers.unary_err_traces
    areas = Helpers.unary_err_areas
    assert isinstance(traces, gpd.GeoDataFrame)
    assert isinstance(areas, gpd.GeoDataFrame)

    network = Network(
        trace_gdf=traces,
        area_gdf=areas,
        determine_branches_nodes=True,
        truncate_traces=True,
        snap_threshold=0.001,
        unary_size_threshold=13000,
    )

    network_attributes = dict()
    for attribute in ("node_counts", "branch_counts"):
        # network_attributes[attribute] = getattr(network, attribute)
        for key, value in getattr(network, attribute).items():
            network_attributes[key] = int(value)

        for key, value in network.numerical_network_description().items():
            try:
                network_attributes[key] = value.item()
            except AttributeError:
                network_attributes[key] = value

    data_regression.check(network_attributes)

    assert network.branch_gdf.shape[0] > network.trace_gdf.shape[0]
