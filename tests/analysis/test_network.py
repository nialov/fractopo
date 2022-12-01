"""
Tests for Network.
"""
from pathlib import Path
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import powerlaw
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.projections import PolarAxes
from pandas.testing import assert_frame_equal, assert_series_equal
from shapely.geometry import MultiPolygon, Polygon
from ternary.ternary_axes_subplot import TernaryAxesSubplot

import tests
from fractopo.analysis import length_distributions
from fractopo.analysis.azimuth import AzimuthBins
from fractopo.analysis.network import CachedNetwork, Network
from fractopo.general import SetRangeTuple, read_geofile


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


@pytest.mark.parametrize(
    "azimuth_set_ranges",
    [
        (
            (0, 60),
            (60, 120),
            (120, 180),
        ),
        (
            (0, 30),
            (30, 60),
            (70, 80),
        ),
        (
            (30, 50),
            (50, 60),
        ),
    ],
)
def test_azimuth_set_relationships_regression(
    azimuth_set_ranges: SetRangeTuple, num_regression
):
    """
    Test for azimuth set relationship regression.
    """
    azimuth_set_names: Tuple[str, ...] = ("1", "2", "3")[0 : len(azimuth_set_ranges)]
    relations_df: pd.DataFrame = Network(
        tests.kb7_traces,  # type: ignore
        tests.kb7_area,  # type: ignore
        name="kb7",
        determine_branches_nodes=True,
        azimuth_set_ranges=azimuth_set_ranges,
        azimuth_set_names=azimuth_set_names,
        snap_threshold=0.001,
        circular_target_area=False,
        truncate_traces=True,
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
        tests.kb7_traces,  # type: ignore
        tests.kb7_area,  # type: ignore
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
    circular_target_area,
    try_export_of_data,
    """,
    tests.test_network_params,
)
def test_network(
    traces,
    area,
    name,
    determine_branches_nodes,
    truncate_traces,
    snap_threshold,
    circular_target_area,
    try_export_of_data,
    file_regression,
    data_regression,
    tmp_path,
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
    numerical_description = network.numerical_network_description()
    assert isinstance(numerical_description, dict)
    for key, item in numerical_description.items():
        assert isinstance(key, str)
        if not isinstance(item, (int, float, str)):
            assert isinstance(item.item(), (int, float))

    cut_off = 1.0
    num_description_explicit_cut_offs = network.numerical_network_description(
        trace_lengths_cut_off=cut_off, branch_lengths_cut_off=cut_off
    )
    for label in ("branch", "trace"):
        key = label + (
            " "
            + length_distributions.Dist.POWERLAW.value
            + " "
            + length_distributions.CUT_OFF
        )
        assert num_description_explicit_cut_offs[key] == 1.0

    network_target_areas = network.target_areas
    assert isinstance(network_target_areas, list)
    assert all(
        [isinstance(val, (Polygon, MultiPolygon)) for val in network_target_areas]
    )

    network_attributes = dict()
    for attribute in ("node_counts", "branch_counts"):
        # network_attributes[attribute] = getattr(network, attribute)
        for key, value in getattr(network, attribute).items():
            if not np.isnan(value):
                network_attributes[key] = int(value)

        for key, value in numerical_description.items():
            if isinstance(value, (float, int)):
                network_attributes[key] = round(
                    value.item() if hasattr(value, "item") else value, 2
                )

    data_regression.check(network_attributes)

    if determine_branches_nodes and network.branch_gdf.shape[0] < 500:

        sorted_branch_gdf = network.branch_gdf.sort_index()
        assert isinstance(sorted_branch_gdf, gpd.GeoDataFrame)
        # Do not check massive branch counts
        # TODO: Add sort_keys=True
        file_regression.check(sorted_branch_gdf.to_json(indent=1, sort_keys=True))
        network_extensive_testing(
            network=network, traces=traces, area=area, snap_threshold=snap_threshold
        )

    trace_intersects = network.trace_intersects_target_area_boundary
    assert isinstance(trace_intersects, np.ndarray)
    assert trace_intersects.dtype in ("int32", "int64")
    branch_intersects = network.branch_intersects_target_area_boundary
    assert isinstance(branch_intersects, np.ndarray)
    assert network.branch_intersects_target_area_boundary.dtype in ("int32", "int64")

    # Test export_network_analysis
    # But only with small amount of traces as processing is time-consuming
    if try_export_of_data:
        network.export_network_analysis(
            output_path=tmp_path, include_contour_grid=network.trace_gdf.shape[0] < 250
        )


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
    copy_trace_gdf = network.trace_data._line_gdf.copy()
    copy_branch_gdf = network.branch_data._line_gdf.copy()

    # Test resetting
    network.reset_length_data()
    assert_frame_equal(copy_trace_gdf, network.trace_data._line_gdf)
    assert_frame_equal(copy_branch_gdf, network.branch_data._line_gdf)

    network_anisotropy = network.anisotropy
    assert isinstance(network_anisotropy, tuple)
    assert isinstance(network_anisotropy[0], np.ndarray)
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
    plt.close("all")

    # Test plotting
    fig_returns = network.plot_xyi()
    assert fig_returns is not None
    fig, ax, tax = fig_returns
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert isinstance(tax, TernaryAxesSubplot)
    plt.close("all")

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
        "plot_trace_azimuth_set_lengths",
        "plot_branch_azimuth_set_lengths",
    ):
        # Test plotting
        fig_returns = getattr(network, plot)()
        assert fig_returns is not None
        if len(fig_returns) == 2:
            fig, ax = fig_returns
        elif len(fig_returns) == 3 and "length" in plot:
            other, fig, ax = fig_returns
            if not isinstance(other, powerlaw.Fit):
                # assume fits, figs, axes from plot_*_set_lengths
                assert isinstance(other, list)
                assert isinstance(other[0], powerlaw.Fit)
                # Check just the first value of returns
                assert len(other) == len(fig)
                assert len(other) == len(ax)
                assert len(other) == len(network.azimuth_set_names)
                other, fig, ax = other[0], fig[0], ax[0]
        elif len(fig_returns) == 3 and "azimuth" in plot:
            other, fig, ax = fig_returns
            assert isinstance(other, AzimuthBins)
        else:
            raise ValueError("Expected 3 max returns.")
        assert isinstance(fig, Figure)
        assert isinstance(ax, (Axes, PolarAxes))

        # Test different set-wise length distribution logics (sanity check only)
        for (
            azimuth_set_name,
            set_lengths,
        ) in network.trace_data.azimuth_set_length_arrays.items():
            fit = length_distributions.determine_fit(
                length_array=set_lengths, cut_off=None
            )
            description = length_distributions.describe_powerlaw_fit(
                fit, length_array=set_lengths
            )
            ld = network.trace_length_distribution(azimuth_set=azimuth_set_name)
            fit_ld = ld.automatic_fit
            description_ld = length_distributions.describe_powerlaw_fit(
                fit_ld, length_array=set_lengths
            )
            desc_srs = pd.Series(description)
            desc_ld_srs = pd.Series(description_ld)
            assert_series_equal(desc_srs, desc_ld_srs)

        plt.close("all")


def _test_network_kb11_manual():
    """
    Test Network analysis with KB11 data.

    Returns the Network.
    """
    trace_gdf = tests.kb11_traces
    area_gdf = tests.kb11_area
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


def test_network_kb11_manual():
    """
    Test Network analysis with KB11 data.

    Returns the Network.
    """
    _test_network_kb11_manual()


@pytest.mark.parametrize(
    "trace_gdf,area_gdf,name",
    tests.test_network_circular_target_area_params,
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
            # network_circular.trace_boundary_intersect_count,
            # network_circular.branch_boundary_intersect_count,
            network_circular.trace_data.boundary_intersect_count_desc(label="Trace"),
            network_circular.branch_data.boundary_intersect_count_desc(label="Branch"),
        ),
        (network_circular.trace_gdf, network_circular.branch_gdf),
        ("Trace", "Branch"),
    ):

        assert all(isinstance(val, str) for val in boundary_intersect_count)
        assert all(isinstance(val, int) for val in boundary_intersect_count.values())
        assert sum(boundary_intersect_count.values()) == gdf.shape[0]
        assert sum(gdf.geometry.intersects(area_gdf.geometry.iloc[0].boundary)) == sum(
            (
                boundary_intersect_count[f"{name} Boundary 1 Intersect Count"],
                boundary_intersect_count[f"{name} Boundary 2 Intersect Count"],
            )
        )


@pytest.mark.parametrize(
    "trace_gdf,area_gdf,network_name,snap_threshold",
    [
        (tests.kb7_traces, tests.kb7_area, "kb7", 0.001),
        (tests.kb11_traces, tests.kb11_area, "kb11", 0.001),
    ],
)
@pytest.mark.parametrize(
    "truncate_traces,circular_target_area",
    [
        (True, True),
        (False, False),
        (True, False),
    ],
)
def test_network_topology_reassignment(
    trace_gdf: gpd.GeoDataFrame,
    area_gdf: gpd.GeoDataFrame,
    tmp_path: Path,
    network_name: str,
    truncate_traces: bool,
    circular_target_area: bool,
    snap_threshold: float,
):
    """
    Test reassignment of Network branch_gdf and node_gdf.
    """
    network_params: Dict[str, Any] = dict(
        trace_gdf=trace_gdf,
        area_gdf=area_gdf,
        name=network_name,
        truncate_traces=truncate_traces,
        circular_target_area=circular_target_area,
        snap_threshold=snap_threshold,
    )
    network = Network(**network_params, determine_branches_nodes=True)

    original_description = network.numerical_network_description()

    # Explicitly name outputs
    branches_name = f"{network_name}_branches.geojson"
    nodes_name = f"{network_name}_nodes.geojson"

    # Save branches and nodes to tmp_path directory
    network.write_branches_and_nodes(
        output_dir_path=tmp_path, branches_name=branches_name, nodes_name=nodes_name
    )

    branches_path = tmp_path / branches_name
    nodes_path = tmp_path / nodes_name

    assert branches_path.exists()
    assert nodes_path.exists()

    branches_gdf = read_geofile(branches_path)
    nodes_gdf = read_geofile(nodes_path)

    assert isinstance(branches_gdf, gpd.GeoDataFrame)
    assert isinstance(nodes_gdf, gpd.GeoDataFrame)

    new_network = Network(
        **network_params,
        determine_branches_nodes=False,
        branch_gdf=branches_gdf,
        node_gdf=nodes_gdf,
    )

    new_description = new_network.numerical_network_description()

    original_df = pd.DataFrame([original_description])
    new_df = pd.DataFrame([new_description])
    assert_frame_equal(original_df, new_df)
    assert_frame_equal(new_network.branch_gdf, branches_gdf)


@pytest.mark.parametrize(
    (
        "trace_gdf,area_gdf,network_name,snap_threshold,"
        "truncate_traces,circular_target_area"
    ),
    [
        (tests.kb7_traces, tests.kb7_area, "kb7", 0.001, True, True),
        (tests.kb7_traces, tests.kb7_area, "kb7", 0.001, False, False),
        (tests.kb7_traces, tests.kb7_area, "kb7", 0.001, True, False),
        (tests.kb11_traces, tests.kb11_area, "kb11", 0.001, True, False),
    ],
)
def test_cached_network(
    trace_gdf: gpd.GeoDataFrame,
    area_gdf: gpd.GeoDataFrame,
    tmp_path: Path,
    network_name: str,
    truncate_traces: bool,
    circular_target_area: bool,
    snap_threshold: float,
):
    """
    Test caching of Network branch_gdf and node_gdf with CachedNetwork.
    """
    # Sanity check that tmp_path is empty
    assert len(list(tmp_path.iterdir())) == 0

    network_params: Dict[str, Any] = dict(
        trace_gdf=trace_gdf,
        area_gdf=area_gdf,
        name=network_name,
        truncate_traces=truncate_traces,
        circular_target_area=circular_target_area,
        snap_threshold=snap_threshold,
    )
    cached_network = CachedNetwork(
        **network_params, determine_branches_nodes=True, network_cache_path=tmp_path
    )

    assert not cached_network._cache_hit

    assert len(list(tmp_path.iterdir())) == 2

    # Test a crucial method that it works as expected
    original_description = cached_network.numerical_network_description()
    assert isinstance(original_description, dict)

    cached_network_again = CachedNetwork(
        **network_params, determine_branches_nodes=True, network_cache_path=tmp_path
    )

    assert cached_network_again._cache_hit

    # Test a crucial method that it works as expected
    new_description = cached_network_again.numerical_network_description()
    assert isinstance(new_description, dict)

    # Test that result is same from determined and cached
    assert_frame_equal(
        pd.DataFrame([original_description]), pd.DataFrame([new_description])
    )

    # Test that no additional caching has been done
    assert len(list(tmp_path.iterdir())) == 2
