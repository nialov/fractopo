from typing import Tuple, Union

import pytest
import pandas as pd

from fractopo.analysis.network import Network

from tests import Helpers


def test_azimuth_set_relationships_regression(file_regression):
    azimuth_set_ranges: Tuple[Tuple[float, float], ...] = (
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
    trace_length_set_ranges: Tuple[Tuple[float, float], ...] = (
        (0, 2),
        (2, 4),
        (4, 6),
    )
    trace_length_set_names: Tuple[str, ...] = ("a", "b", "c")
    # branch_length_set_ranges: Tuple[Tuple[float, float], ...] = (
    #     (0, 1),
    #     (1, 2),
    #     (2, 6),
    # )
    # branch_length_set_names: Tuple[str, ...] = ("A", "B", "C")
    relations_df: pd.DataFrame = Network(
        Helpers.kb7_traces,  # type: ignore
        Helpers.kb7_area,  # type: ignore
        name="kb7",
        determine_branches_nodes=True,
        trace_length_set_names=trace_length_set_names,
        trace_length_set_ranges=trace_length_set_ranges,
    ).azimuth_set_relationships
    file_regression.check(relations_df.to_string())
