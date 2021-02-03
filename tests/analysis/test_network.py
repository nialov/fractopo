from typing import Tuple, Union

from tests import Helpers

import pandas as pd
import pytest
from fractopo import SetRangeTuple
from fractopo.analysis.network import Network


def test_azimuth_set_relationships_regression(file_regression):
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
):
    """
    Test Network object creation with general datasets.
    """
    network = Network(
        trace_gdf=traces,
        area_geoseries=area,
        name=name,
        determine_branches_nodes=determine_branches_nodes,
        truncate_traces=truncate_traces,
        snap_threshold=snap_threshold,
    )

    file_regression.check(network.branch_gdf.sort_index().to_json())

    network_attributes = dict()
    for attribute in ("node_counts", "branch_counts"):
        # network_attributes[attribute] = getattr(network, attribute)
        for key, value in getattr(network, attribute).items():
            network_attributes[key] = int(value)

    network_attributes["trace_lengths_powerlaw_fit_cut_off"] = float(
        round(network.trace_lengths_powerlaw_fit().xmin, 4)
    )
    network_attributes["branch_lengths_powerlaw_fit_cut_off"] = float(
        round(network.branch_lengths_powerlaw_fit().xmin, 4)
    )
    network_attributes["trace_lengths_powerlaw_fit_alpha"] = float(
        round(network.trace_lengths_powerlaw_fit().alpha, 4)
    )
    network_attributes["branch_lengths_powerlaw_fit_alpha"] = float(
        round(network.branch_lengths_powerlaw_fit().alpha, 4)
    )

    data_regression.check(network_attributes)
