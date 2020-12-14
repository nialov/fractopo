"""
Analyse and plot trace map data with Network.
"""

# Python Windows co-operation imports
from pathlib import Path
from dataclasses import dataclass

import geopandas as gpd
import matplotlib

# Math and analysis imports
# Plotting imports
# DataFrame analysis imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ternary
import powerlaw

# Own code imports
from fractopo.general import (
    determine_azimuth,
    determine_set,
    CLASS_COLUMN,
    CONNECTION_COLUMN,
POWERLAW, LOGNORMAL, EXPONENTIAL,
)
from fractopo.branches_and_nodes import branches_and_nodes
from fractopo.analysis.line_data import LineData
from fractopo.analysis.parameters import (
    plot_xyi_plot,
    plot_branch_plot,
    determine_node_classes,
    determine_branch_classes,
    determine_topology_parameters,
    plot_parameters_plot,
)
from fractopo.analysis.anisotropy import determine_anisotropy_sum, plot_anisotropy_plot
from fractopo.analysis.relationships import (
    determine_crosscut_abutting_relationships,
    plot_crosscut_abutting_relationships_plot,
)

from fractopo.analysis.config import 
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
    # =========
    trace_gdf: gpd.GeoDataFrame
    area_geoseries: Optional[Union[gpd.GeoSeries, gpd.GeoDataFrame]] = None

    # Name the network for e.g. plot titles
    name: str = "Network"

    # Azimuth sets
    # ============
    azimuth_set_ranges: Tuple[Tuple[float, float], ...] = (
        (0, 60),
        (60, 120),
        (120, 180),
    )
    azimuth_set_names: Tuple[str, ...] = ("1", "2", "3")

    # Length sets
    # ===========

    # Trace length
    trace_length_set_names: Optional[Tuple[str, ...]] = None
    trace_length_set_ranges: Optional[Tuple[Tuple[float, float], ...]] = None

    # Branch length
    branch_length_set_names: Optional[Tuple[str, ...]] = None
    branch_length_set_ranges: Optional[Tuple[Tuple[float, float], ...]] = None

    # Branches and nodes
    # ==================
    branch_gdf: Optional[gpd.GeoDataFrame] = None
    node_gdf: Optional[gpd.GeoDataFrame] = None
    determine_branches_nodes: bool = False
    snap_threshold: float = 0.001

    # Length distributions
    # ====================
    trace_length_cut_off: Optional[float] = None
    branch_length_cut_off: Optional[float] = None

    # Private caching attributes
    # ==========================
    _anisotropy: Optional[np.ndarray] = None
    _parameters: Optional[Dict[str, float]] = None
    _azimuth_set_relationships: Optional[pd.DataFrame] = None
    _trace_length_set_relationships: Optional[pd.DataFrame] = None

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

    def reset_length_data(self):
        self.trace_data = LineData(
            self.trace_gdf,
            self.azimuth_set_ranges,
            self.azimuth_set_names,
            self.trace_length_set_ranges,
            self.trace_length_set_names,
        )
        if self.branch_gdf is not None:
            self.branch_data = LineData(
                self.branch_gdf,
                self.azimuth_set_ranges,
                self.azimuth_set_names,
                self.branch_length_set_ranges,
                self.branch_length_set_names,
            )
            self._azimuth_set_relationships = None

    def _is_branch_gdf_defined(self) -> bool:
        """
        Is branch_gdf defined.

        TODO: Make more intelligent.
        """
        if self.branch_gdf is None:
            return False
        return True

    @property
    def trace_series(self) -> gpd.GeoSeries:
        return self.trace_data.line_gdf.geometry

    @property
    def node_series(self) -> Optional[gpd.GeoSeries]:
        if not self._is_branch_gdf_defined:
            return None
        return self.node_gdf.geometry

    @property
    def branch_series(self) -> Optional[gpd.GeoSeries]:
        if not self._is_branch_gdf_defined:
            return None
        return self.branch_data.line_gdf.geometry

    @property
    def trace_azimuth_array(self) -> np.ndarray:
        return self.trace_data.azimuth_array

    @property
    def branch_azimuth_array(self) -> Optional[np.ndarray]:
        if not self._is_branch_gdf_defined():
            return None
        return self.branch_data.azimuth_array

    @property
    def trace_length_array(self) -> np.ndarray:
        return self.trace_data.length_array

    @property
    def branch_length_array(self) -> Optional[np.ndarray]:
        if not self._is_branch_gdf_defined():
            return None
        return self.branch_data.length_array

    @property
    def trace_azimuth_set_array(self) -> np.ndarray:
        return self.trace_data.azimuth_set_array

    @property
    def branch_azimuth_set_array(self) -> Optional[np.ndarray]:
        if not self._is_branch_gdf_defined():
            return None
        return self.branch_data.azimuth_set_array

    @property
    def trace_length_set_array(self) -> Optional[np.ndarray]:
        return self.trace_data.length_set_array

    @property
    def branch_length_set_array(self) -> Optional[np.ndarray]:
        if not self._is_branch_gdf_defined():
            return None
        return self.branch_data.length_set_array

    @property
    def node_types(self) -> Optional[np.ndarray]:
        if not self._is_branch_gdf_defined():
            return None
        return self.node_gdf[CLASS_COLUMN].to_numpy()

    @property
    def node_counts(self) -> Optional[Dict[str, int]]:
        if not self._is_branch_gdf_defined():
            return None
        return determine_node_classes(self.node_types)

    @property
    def branch_types(self) -> Optional[np.ndarray]:
        if not self._is_branch_gdf_defined():
            return None
        return self.branch_gdf[CONNECTION_COLUMN].to_numpy()

    @property
    def branch_counts(self) -> Optional[Dict[str, int]]:
        if not self._is_branch_gdf_defined():
            return None
        return determine_branch_classes(self.branch_types)

    @property
    def total_area(self) -> float:
        return self.area_geoseries.geometry.area.sum()

    @property
    def parameters(self) -> Optional[Dict[str, float]]:
        if not self._is_branch_gdf_defined():
            return None
        # Cannot do simple cached_property because None might have been
        # returned previously.
        if self._parameters is None:
            self._parameters = determine_topology_parameters(
                trace_length_array=self.trace_length_array,
                branch_length_array=self.branch_length_array,
                node_counts=self.node_counts,  # type: ignore
                area=self.total_area,
            )
        return self._parameters

    @property
    def anisotropy(self) -> Optional[np.ndarray]:
        if not self._is_branch_gdf_defined:
            return None
        if self._anisotropy is None:
            self._anisotropy = determine_anisotropy_sum(
                azimuth_array=self.branch_azimuth_array,
                length_array=self.branch_length_array,
                branch_types=self.branch_types,
            )
        return self._anisotropy

    @property
    def azimuth_set_relationships(self) -> Optional[np.ndarray]:
        if not self._is_branch_gdf_defined:
            return None
        if self._azimuth_set_relationships is None:
            self._azimuth_set_relationships = determine_crosscut_abutting_relationships(
                trace_series=self.trace_series,
                node_series=self.node_series,  # type: ignore
                node_types=self.node_types,
                set_array=self.trace_azimuth_set_array,
                set_names=self.azimuth_set_names,
                buffer_value=0.001,
                label=self.name,
            )
        return self._azimuth_set_relationships

    @property
    def length_set_relationships(self) -> Optional[np.ndarray]:
        if not self._is_branch_gdf_defined:
            return None
        if self._trace_length_set_relationships is None:
            self._trace_length_set_relationships = (
                determine_crosscut_abutting_relationships(
                    trace_series=self.trace_series,
                    node_series=self.node_series,  # type: ignore
                    node_types=self.node_types,
                    set_array=self.trace_data.length_set_array,
                    set_names=self.trace_data.length_set_names,  # type: ignore
                    buffer_value=0.001,
                    label=self.name,
                )
            )
        return self._trace_length_set_relationships

    @property
    def trace_azimuth_set_counts(self) -> Dict[str, int]:
        return self.trace_data.azimuth_set_counts

    @property
    def trace_length_set_counts(self) -> Dict[str, int]:
        return self.trace_data.length_set_counts

    @property
    def branch_azimuth_set_counts(self) -> Dict[str, int]:
        return self.branch_data.azimuth_set_counts

    @property
    def branch_length_set_counts(self) -> Dict[str, int]:
        return self.branch_data.length_set_counts

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
            logging.error(
                "Expected area_geoseries to be defined to assign branches and nodes."
            )

    def plot_trace_lengths(
        self, label: Optional[str] = None
    ) -> Tuple[powerlaw.Fit, matplotlib.figure.Figure, matplotlib.axes.Axes]:  # type: ignore
        if label is None:
            label = self.name
        return self.trace_data.plot_lengths(
            label=label, cut_off=self.trace_length_cut_off
        )

    def plot_branch_lengths(
        self, label: Optional[str] = None
    ) -> Tuple[powerlaw.Fit, matplotlib.figure.Figure, matplotlib.axes.Axes]:  # type: ignore
        if label is None:
            label = self.name
        return self.branch_data.plot_lengths(
            label=label, cut_off=self.branch_length_cut_off
        )

    def plot_trace_azimuth(
        self, label: Optional[str] = None
    ) -> Tuple[Dict[str, np.ndarray], matplotlib.figure.Figure, matplotlib.axes.Axes]:  # type: ignore
        if label is None:
            label = self.name
        return self.trace_data.plot_azimuth(label=label)

    def plot_branch_azimuth(
        self, label: Optional[str] = None
    ) -> Tuple[Dict[str, np.ndarray], matplotlib.figure.Figure, matplotlib.axes.Axes]:  # type: ignore
        if label is None:
            label = self.name
        return self.branch_data.plot_azimuth(label=label)

    def plot_xyi(
        self, label: Optional[str] = None
    ) -> Optional[
        Tuple[
            matplotlib.figure.Figure,  # type: ignore
            matplotlib.axes.Axes,  # type: ignore
            ternary.ternary_axes_subplot.TernaryAxesSubplot,
        ]
    ]:
        if label is None:
            label = self.name
        if self.node_counts is None:
            logging.error("Expected node_gdf to be defined for plot_xyi.")
            return
        return plot_xyi_plot(node_counts_list=[self.node_counts], labels=[label])

    def plot_branch(
        self, label: Optional[str] = None
    ) -> Optional[
        Tuple[
            matplotlib.figure.Figure,  # type: ignore
            matplotlib.axes.Axes,  # type: ignore
            ternary.ternary_axes_subplot.TernaryAxesSubplot,
        ]
    ]:
        if label is None:
            label = self.name
        if self.branch_counts is None:
            logging.error("Expected branch_gdf to be defined for plot_xyi.")
            return
        return plot_branch_plot(branch_counts_list=[self.branch_counts], labels=[label])

    def plot_parameters(
        self, label: Optional[str] = None, color: Optional[str] = None
    ) -> Optional[Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]]:  # type: ignore
        if not self._is_branch_gdf_defined():
            return None
        if label is None:
            label = self.name
        if color is None:
            color = "black"
        assert self.parameters is not None
        figs, axes = plot_parameters_plot(
            topology_parameters_list=[self.parameters],  # type: ignore
            labels=[label],
            colors=[color],
        )
        return figs[0], axes[0]

    def plot_anisotropy(
        self, label: Optional[str] = None, color: Optional[str] = None
    ) -> Optional[Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]]:  # type: ignore
        if label is None:
            label = self.name
        if not self._is_branch_gdf_defined:
            return None
        if color is None:
            color = "black"
        anisotropy_sum = self.anisotropy[0]
        sample_intervals = self.anisotropy[1]
        fig, ax = plot_anisotropy_plot(
            anisotropy_sum=anisotropy_sum,
            sample_intervals=sample_intervals,
            label=label,  # type: ignore
            color=color,
        )
        return fig, ax

    def plot_azimuth_crosscut_abutting_relationships(
        self,
    ) -> Tuple[List[matplotlib.figure.Figure], List[np.ndarray]]:  # type: ignore
        return plot_crosscut_abutting_relationships_plot(
            relations_df=self.azimuth_set_relationships,  # type: ignore
            set_array=self.trace_azimuth_set_array,
            set_names=self.azimuth_set_names,
        )

    def plot_trace_length_crosscut_abutting_relationships(
        self,
    ) -> Tuple[List[matplotlib.figure.Figure], List[np.ndarray]]:  # type: ignore
        return plot_crosscut_abutting_relationships_plot(
            relations_df=self.length_set_relationships,  # type: ignore
            set_array=self.trace_data.length_set_array,
            set_names=self.trace_data.length_set_names,  # type: ignore
        )

    def plot_trace_azimuth_set_count(
        self, label: Optional[str] = None
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        if label is None:
            label = self.name
        return self.trace_data.plot_azimuth_set_count(label=label)

    def plot_branch_azimuth_set_count(
        self, label: Optional[str] = None
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        if label is None:
            label = self.name
        return self.branch_data.plot_azimuth_set_count(label=label)

    def plot_trace_length_set_count(
        self, label: Optional[str] = None
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        if label is None:
            label = self.name
        return self.trace_data.plot_length_set_count(label=label)

    def plot_branch_length_set_count(
        self, label: Optional[str] = None
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        if label is None:
            label = self.name
        return self.branch_data.plot_length_set_count(label=label)
