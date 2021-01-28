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
    ).azimuth_set_relationships
    file_regression.check(relations_df.to_string())


@pytest.mark.parametrize(
    "traces,area,name,determine_branches_nodes,truncate_traces",
    Helpers.test_network_params,
)
def test_network(traces, area, name, determine_branches_nodes, truncate_traces):
    """
    Test Network object creation with general datasets.
    """
    network = Network(
        trace_gdf=traces,
        area_geoseries=area,
        name=name,
        determine_branches_nodes=determine_branches_nodes,
        truncate_traces=truncate_traces,
    )
