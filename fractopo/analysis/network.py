"""
Handles a single target area or a single grouped area. Does not discriminate between a single target area and
grouped target areas.
"""

# Python Windows co-operation imports
from pathlib import Path
from textwrap import wrap
from dataclasses import dataclass, field

import geopandas as gpd
import matplotlib.patches as patches
import matplotlib

# Math and analysis imports
# Plotting imports
# DataFrame analysis imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ternary
import powerlaw

from scipy.interpolate import CubicSpline

# Own code imports
import fractopo.analysis.tools as tools
import fractopo.analysis.config as config
from fractopo.general import (
    determine_azimuth,
    determine_set,
    CLASS_COLUMN,
    CONNECTION_COLUMN,
)
from fractopo.branches_and_nodes import branches_and_nodes
from fractopo.analysis.line_data import LineData
from fractopo.analysis.parameters import plot_xyi_plot, plot_branch_plot

from fractopo.analysis.config import POWERLAW, LOGNORMAL, EXPONENTIAL
from typing import Dict, Tuple, Union, List, Optional, Literal, Callable, Any
import logging


@dataclass
class Network:
    """
    Trace network.

    Consists of at its simplest of validated traces. All other datasets are
    optional but most analyses are locked behind the addition of atleast the
    target area dataset.
    """

    # Base data
    trace_gdf: gpd.GeoDataFrame
    area_geoseries: Optional[Union[gpd.GeoSeries, gpd.GeoDataFrame]] = None

    # Name the network for e.g. plot titles
    name: str = ""

    # Azimuth sets
    azimuth_set_ranges: Tuple[Tuple[float, float], ...] = (
        (0, 60),
        (60, 120),
        (120, 180),
    )
    azimuth_set_names: Tuple[str, ...] = ("1", "2", "3")

    # Length sets

    # traces
    trace_length_set_names: Optional[Tuple[str, ...]] = None
    trace_length_set_ranges: Optional[Tuple[Tuple[float, float], ...]] = None

    # branches
    branch_length_set_names: Optional[Tuple[str, ...]] = None
    branch_length_set_ranges: Optional[Tuple[Tuple[float, float], ...]] = None

    # Branches and nodes
    branch_gdf: Optional[gpd.GeoDataFrame] = None
    node_gdf: Optional[gpd.GeoDataFrame] = None
    determine_branches_nodes: bool = False
    snap_threshold: float = 0.001

    # Length distributions
    trace_length_cut_off: Optional[float] = None
    branch_length_cut_off: Optional[float] = None

    @staticmethod
    def _default_length_set_ranges(count, min, max):
        arr = np.linspace(min, max, count + 1)
        starts = arr[0 : count + 1]
        ends = arr[1:]
        assert len(starts) == len(ends)
        as_gen = ((start, end) for start, end in zip(starts, ends))
        return tuple(as_gen)

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, (gpd.GeoSeries, gpd.GeoDataFrame)):
            self.__dict__[name] = value.copy()
            if name == "branch_gdf":
                self.branch_data = LineData(
                    self.branch_gdf,  # type: ignore
                    self.azimuth_set_ranges,
                    self.azimuth_set_names,
                    self.branch_length_set_ranges,
                    self.branch_length_set_names,
                )
        else:
            self.__dict__[name] = value

    def __post_init__(self):

        # Copy GeoDataFrames instead of changing inputs.
        # If the data is passed later to attribute __setattr__ will also
        # handle copying.
        # Traces
        self.trace_gdf = self.trace_gdf.copy()
        self.trace_data = LineData(
            self.trace_gdf,
            self.azimuth_set_ranges,
            self.azimuth_set_names,
            self.trace_length_set_ranges,
            self.trace_length_set_names,
        )
        # Area
        self.area_geoseries = (
            self.area_geoseries.copy() if self.area_geoseries is not None else None
        )
        # Branches
        self.branch_gdf = (
            self.branch_gdf.copy() if self.branch_gdf is not None else None
        )
        if self.branch_gdf is not None:
            self.branch_data = LineData(
                self.branch_gdf,
                self.azimuth_set_ranges,
                self.azimuth_set_names,
                self.branch_length_set_ranges,
                self.branch_length_set_names,
            )
        if self.determine_branches_nodes:
            self.assign_branches_nodes()

        # Nodes
        self.node_gdf = self.node_gdf.copy() if self.node_gdf is not None else None

    def _require_branches(self) -> bool:
        if self.branch_gdf is None:
            print(f"Expected branch_gdf to be defined.")
            return False
        return True

    @property
    def trace_azimuth_array(self) -> np.ndarray:
        return self.trace_data.azimuth_array

    @property
    def branch_azimuth_array(self) -> Optional[np.ndarray]:
        if not self._require_branches():
            return None
        return self.branch_data.azimuth_array

    @property
    def trace_length_array(self) -> np.ndarray:
        return self.trace_data.length_array

    @property
    def branch_length_array(self) -> Optional[np.ndarray]:
        if not self._require_branches():
            return None
        return self.branch_data.length_array

    @property
    def trace_azimuth_set_array(self) -> np.ndarray:
        return self.trace_data.azimuth_set_array

    @property
    def branch_azimuth_set_array(self) -> Optional[np.ndarray]:
        if not self._require_branches():
            return None
        return self.branch_data.azimuth_set_array

    @property
    def trace_length_set_array(self) -> Optional[np.ndarray]:
        return self.trace_data.length_set_array

    @property
    def branch_length_set_array(self) -> Optional[np.ndarray]:
        if not self._require_branches():
            return None
        return self.branch_data.length_set_array

    @property
    def node_types(self) -> Optional[np.ndarray]:
        if not self._require_branches():
            return None
        return self.node_gdf[CLASS_COLUMN].to_numpy()

    @property
    def branch_types(self) -> Optional[np.ndarray]:
        if not self._require_branches():
            return None
        return self.branch_gdf[CONNECTION_COLUMN].to_numpy()

    def assign_branches_nodes(self):
        if self.area_geoseries is not None:
            branches, nodes = branches_and_nodes(
                self.trace_gdf, self.area_geoseries, self.snap_threshold
            )
            if self.trace_gdf.crs is not None:
                branches.set_crs(self.trace_gdf.crs, inplace=True)
                nodes.set_crs(self.trace_gdf.crs, inplace=True)
            self.branch_gdf = branches
            self.node_gdf = nodes
            self.branch_data = LineData(
                self.branch_gdf,
                self.azimuth_set_ranges,
                self.azimuth_set_names,
                self.branch_length_set_ranges,
                self.branch_length_set_names,
            )
        else:
            print("Expected area_geoseries to be defined to assign branches and nodes.")

    def plot_trace_lengths(
        self, label: Optional[str] = None
    ) -> Tuple[powerlaw.Fit, matplotlib.figure.Figure, matplotlib.axes.Axes]:  # type: ignore
        if label is None:
            label = self.name
        return self.trace_data.plot_lengths(
            label=label, cut_off=self.trace_length_cut_off
        )

    def plot_trace_azimuth(
        self, label: Optional[str] = None
    ) -> Tuple[Dict[str, np.ndarray], matplotlib.figure.Figure, matplotlib.axes.Axes]:  # type: ignore
        if label is None:
            label = self.name
        return self.trace_data.plot_azimuth(label=label)

    def plot_xyi(self, label: Optional[str] = None):
        if label is None:
            label = self.name
        if self.node_types is None:
            print("Expected node_gdf to be defined for plot_xyi.")
            return
        return plot_xyi_plot(node_types_list=[self.node_types], labels=[label])

    def plot_branch(self, label: Optional[str] = None):
        if label is None:
            label = self.name
        if self.branch_types is None:
            print("Expected node_gdf to be defined for plot_xyi.")
            return
        return plot_branch_plot(branch_types=self.branch_types, label=label)
