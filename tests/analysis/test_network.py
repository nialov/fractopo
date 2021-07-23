"""
Tests for Network.
"""
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import powerlaw
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.projections import PolarAxes
from pandas.testing import assert_frame_equal
from shapely.geometry import MultiPolygon, Polygon
from ternary.ternary_axes_subplot import TernaryAxesSubplot

from fractopo.analysis.azimuth import AzimuthBins
from fractopo.analysis.network import Network
from fractopo.general import SetRangeTuple
from tests import Helpers


def relations_df_to_dict(df: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Turn relations_df to a dict.
    """
    relations_df_dict = dict()

    for set_name, values in df.groupby("sets"):
        assert isinstance(set_name, tuple)
        relations_df_dict[
            str(set_name)
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")
        ] = list(values[["x", "y", "y-reverse"]].values[0])
    return relations_df_dict


def test_azimuth_set_relationships_regression(num_regression):
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
        circular_target_area=False,
    ).azimuth_set_relationships

    relations_df_dict = relations_df_to_dict(relations_df)

    num_regression.check(relations_df_dict)


def test_length_set_relationships_regression(num_regression):
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
        circular_target_area=False,
    ).azimuth_set_relationships

    relations_df_dict = relations_df_to_dict(relations_df)

    num_regression.check(relations_df_dict)


@pytest.mark.parametrize(
    """
    traces,
    area,
    name,
    determine_branches_nodes,
    truncate_traces,
    snap_threshold,
    circular_target_area
    """,
    Helpers.test_network_params,
)
def test_network(
    traces,
    area,
    name,
    determine_branches_nodes,
    truncate_traces,
    snap_threshold,
    circular_target_area,
    file_regression,
    data_regression,
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
        trace_length_set_names=("a", "b"),
        branch_length_set_names=("A", "B"),
        trace_length_set_ranges=((0.1, 1), (1, 2)),
        branch_length_set_ranges=((0.1, 1), (1, 2)),
        circular_target_area=circular_target_area,
    )

    assert area.shape[0] == len(network.representative_points())
    assert isinstance(network.numerical_network_description(), dict)
    for key, item in network.numerical_network_description().items():
        assert isinstance(key, str)
        if not isinstance(item, (int, float, str)):
            assert isinstance(item.item(), (int, float))

    assert isinstance(network.target_areas, list)
    assert all(
        [isinstance(val, (Polygon, MultiPolygon)) for val in network.target_areas]
    )

    network_attributes = dict()
    for attribute in ("node_counts", "branch_counts"):
        # network_attributes[attribute] = getattr(network, attribute)
        for key, value in getattr(network, attribute).items():
            if not np.isnan(value):
                network_attributes[key] = int(value)

        for key, value in network.numerical_network_description().items():
            if isinstance(value, (float, int)):
                network_attributes[key] = round(
                    value.item() if hasattr(value, "item") else value, 2
                )

    data_regression.check(network_attributes)

    if determine_branches_nodes and network.get_branch_gdf().shape[0] < 500:

        sorted_branch_gdf = network.get_branch_gdf().sort_index()
        assert isinstance(sorted_branch_gdf, gpd.GeoDataFrame)
        # Do not check massive branch counts
        file_regression.check(sorted_branch_gdf.to_json(indent=1))
        network_extensive_testing(
            network=network, traces=traces, area=area, snap_threshold=snap_threshold
        )

    assert isinstance(network.trace_intersects_target_area_boundary, np.ndarray)
    assert network.trace_intersects_target_area_boundary.dtype == "int"
    assert isinstance(network.branch_intersects_target_area_boundary, np.ndarray)
    assert network.branch_intersects_target_area_boundary.dtype == "int64"


def network_extensive_testing(
    network: Network,
    traces: gpd.GeoDataFrame,
    area: gpd.GeoDataFrame,
    snap_threshold: float,
):
    """
    Test Network attributes extensively.
    """
    # Test resetting
    copy_trace_gdf = network.trace_data.line_gdf.copy()
    copy_branch_gdf = network.branch_data.line_gdf.copy()

    # Test resetting
    network.reset_length_data()
    assert_frame_equal(copy_trace_gdf, network.trace_data.line_gdf)
    assert_frame_equal(copy_branch_gdf, network.branch_data.line_gdf)

    assert isinstance(network.anisotropy, tuple)
    assert isinstance(network.anisotropy[0], np.ndarray)
    assert isinstance(network.trace_length_set_array, np.ndarray)
    assert isinstance(network.branch_length_set_array, np.ndarray)

    # Test passing old branch and node data
    branch_copy = network.branch_gdf.copy()
    network_test = Network(
        trace_gdf=traces,
        area_gdf=area,
        name="teeest_with_old",
        branch_gdf=branch_copy,
        node_gdf=network.node_gdf.copy(),
        determine_branches_nodes=False,
        truncate_traces=True,
        snap_threshold=snap_threshold,
    )

    assert_frame_equal(network_test.branch_gdf, branch_copy)

    # Test plotting
    fig_returns = network.plot_branch()
    assert fig_returns is not None
    fig, ax, tax = fig_returns
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert isinstance(tax, TernaryAxesSubplot)
    plt.close()

    # Test plotting
    fig_returns = network.plot_xyi()
    assert fig_returns is not None
    fig, ax, tax = fig_returns
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert isinstance(tax, TernaryAxesSubplot)
    plt.close()

    for plot in (
        "plot_parameters",
        "plot_anisotropy",
        "plot_trace_azimuth_set_count",
        "plot_branch_azimuth_set_count",
        "plot_trace_length_set_count",
        "plot_branch_length_set_count",
        "plot_trace_azimuth",
        "plot_branch_azimuth",
        "plot_trace_lengths",
        "plot_branch_lengths",
    ):
        # Test plotting
        fig_returns = getattr(network, plot)()
        assert fig_returns is not None
        if len(fig_returns) == 2:
            fig, ax = fig_returns
        elif len(fig_returns) == 3 and "length" in plot:
            other, fig, ax = fig_returns
            assert isinstance(other, powerlaw.Fit)
        elif len(fig_returns) == 3 and "azimuth" in plot:
            other, fig, ax = fig_returns
            assert isinstance(other, AzimuthBins)
        else:
            raise ValueError("Expected 3 max returns.")
        assert isinstance(fig, Figure)
        assert isinstance(ax, (Axes, PolarAxes))
        plt.close()


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
        circular_target_area=False,
    )
    return network


@pytest.mark.parametrize(
    "trace_gdf,area_gdf,name",
    Helpers.test_network_circular_target_area_params,
)
def test_network_circular_target_area(trace_gdf, area_gdf, name, data_regression):
    """
    Test network circular_target_area.
    """
    network_circular = Network(
        trace_gdf=trace_gdf,
        area_gdf=area_gdf,
        name=name,
        circular_target_area=True,
        determine_branches_nodes=True,
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

    lengths_circular_sum = np.sum(lengths_circular)
    lengths_non_circular_sum = np.sum(lengths_non_circular)

    data_regression.check(
        {
            "circular_sum": lengths_circular_sum.item(),
            "non_circular_sum": lengths_non_circular_sum.item(),
        }
    )

    # test both traces and branches for right boundary_intersect_counts
    for boundary_intersect_count, gdf, name in zip(
        (
            network_circular.trace_boundary_intersect_count,
            network_circular.branch_boundary_intersect_count,
        ),
        (network_circular.trace_gdf, network_circular.branch_gdf),
        ("Trace", "Branch"),
    ):

        assert all([isinstance(val, str) for val in boundary_intersect_count])
        assert all([isinstance(val, int) for val in boundary_intersect_count.values()])
        assert sum(boundary_intersect_count.values()) == gdf.shape[0]
        assert sum(gdf.geometry.intersects(area_gdf.geometry.iloc[0].boundary)) == sum(
            (
                boundary_intersect_count[f"{name} Boundary 1 Intersect Count"],
                boundary_intersect_count[f"{name} Boundary 2 Intersect Count"],
            )
        )
