"""
Analyse and plot trace map data with Network.
"""

# Python Windows co-operation imports
import logging

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import powerlaw
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.projections import PolarAxes
from shapely.geometry import Point, MultiPolygon, Polygon
from ternary.ternary_axes_subplot import TernaryAxesSubplot

from fractopo import SetRangeTuple
from fractopo.analysis.anisotropy import determine_anisotropy_sum, plot_anisotropy_plot
from fractopo.analysis.line_data import LineData
from fractopo.analysis.parameters import (
    determine_branch_type_counts,
    determine_node_type_counts,
    determine_topology_parameters,
    plot_branch_plot,
    plot_parameters_plot,
    plot_xyi_plot,
    branches_intersect_boundary,
)
from fractopo.analysis.relationships import (
    determine_crosscut_abutting_relationships,
    plot_crosscut_abutting_relationships_plot,
)
from fractopo.branches_and_nodes import branches_and_nodes
from fractopo.general import (
    EE_branch,
    CLASS_COLUMN,
    CONNECTION_COLUMN,
    crop_to_target_areas,
    determine_boundary_intersecting_lines,
    bool_arrays_sum,
)


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
    area_gdf: Optional[Union[gpd.GeoSeries, gpd.GeoDataFrame]] = None

    # Name the network for e.g. plot titles
    name: str = "Network"

    # The traces can be cut to end at the boundary of the target area
    # Defaults to True
    truncate_traces: bool = True

    # Azimuth sets
    # ============
    azimuth_set_ranges: SetRangeTuple = (
        (0, 60),
        (60, 120),
        (120, 180),
    )
    azimuth_set_names: Tuple[str, ...] = ("1", "2", "3")

    # Length sets
    # ===========

    # Trace length
    trace_length_set_names: Optional[Tuple[str, ...]] = None
    trace_length_set_ranges: Optional[SetRangeTuple] = None

    # Branch length
    branch_length_set_names: Optional[Tuple[str, ...]] = None
    branch_length_set_ranges: Optional[SetRangeTuple] = None

    # Branches and nodes
    # ==================
    branch_gdf: Optional[gpd.GeoDataFrame] = None
    node_gdf: Optional[gpd.GeoDataFrame] = None
    determine_branches_nodes: bool = False
    snap_threshold: float = 0.01

    # Length distributions
    # ====================
    # trace_length_cut_off: Optional[float] = None
    # branch_length_cut_off: Optional[float] = None

    # Private caching attributes
    # ==========================
    _anisotropy: Optional[np.ndarray] = None
    _parameters: Optional[Dict[str, float]] = None
    _azimuth_set_relationships: Optional[pd.DataFrame] = None
    _trace_length_set_relationships: Optional[pd.DataFrame] = None
    _trace_intersects_target_area_boundary: Optional[np.ndarray] = None
    _branch_intersects_target_area_boundary: Optional[np.ndarray] = None

    # TODO: No Optional property return types.

    @staticmethod
    def _default_length_set_ranges(count, min, max):
        arr = np.linspace(min, max, count + 1)
        starts = arr[0 : count + 1]
        ends = arr[1:]
        assert len(starts) == len(ends)
        as_gen = ((start, end) for start, end in zip(starts, ends))
        return tuple(as_gen)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Override default __setattr__ to force DataFrame copying.

        Normally DataFrames are passed as references instead of passing
        reference allowing side-effects. Also resets LineData for branch
        data when setting it.
        """
        if isinstance(value, (gpd.GeoSeries, gpd.GeoDataFrame)):
            self.__dict__[name] = value.copy()
            if name == "branch_gdf":
                self.branch_data = LineData(
                    line_gdf=self.branch_gdf,
                    azimuth_set_ranges=self.azimuth_set_ranges,
                    azimuth_set_names=self.azimuth_set_names,
                    length_set_ranges=self.branch_length_set_ranges,
                    length_set_names=self.branch_length_set_names,
                    area_boundary_intersects=self.branch_intersects_target_area_boundary,
                )
        else:
            self.__dict__[name] = value

    def __post_init__(self):
        """
        Copy GeoDataFrames instead of changing inputs.

        If the data is passed later to attribute, __setattr__ will also
        handle copying.
        """
        # Traces
        self.trace_gdf = self.trace_gdf.copy()

        if self.truncate_traces:
            self.trace_gdf = gpd.GeoDataFrame(
                crop_to_target_areas(
                    self.trace_gdf,
                    self.area_gdf,
                    snap_threshold=self.snap_threshold,
                )
            )
            self.trace_gdf.reset_index(inplace=True, drop=True)
            if self.trace_gdf.shape[0] == 0:
                raise ValueError("Empty trace GeoDataFrame after crop_to_target_areas.")

        self.trace_data = LineData(
            line_gdf=self.trace_gdf,
            azimuth_set_ranges=self.azimuth_set_ranges,
            azimuth_set_names=self.azimuth_set_names,
            length_set_ranges=self.trace_length_set_ranges,
            length_set_names=self.trace_length_set_names,
            area_boundary_intersects=self.trace_intersects_target_area_boundary,
        )
        # Area
        self.area_gdf = self.area_gdf.copy() if self.area_gdf is not None else None
        # Branches
        self.branch_gdf = (
            self.branch_gdf.copy() if self.branch_gdf is not None else None
        )
        if self.branch_gdf is not None:
            self.branch_data = LineData(
                line_gdf=self.branch_gdf,
                azimuth_set_ranges=self.azimuth_set_ranges,
                azimuth_set_names=self.azimuth_set_names,
                length_set_ranges=self.branch_length_set_ranges,
                length_set_names=self.branch_length_set_names,
                area_boundary_intersects=self.branch_intersects_target_area_boundary,
            )
        if self.determine_branches_nodes:
            self.assign_branches_nodes()

        # Nodes
        self.node_gdf = self.node_gdf.copy() if self.node_gdf is not None else None

    def reset_length_data(self):
        """
        Reset LineData attributes.
        """
        self.trace_data = LineData(
            line_gdf=self.trace_gdf,
            azimuth_set_ranges=self.azimuth_set_ranges,
            azimuth_set_names=self.azimuth_set_names,
            length_set_ranges=self.trace_length_set_ranges,
            length_set_names=self.trace_length_set_names,
            area_boundary_intersects=self.trace_intersects_target_area_boundary,
        )
        if self.branch_gdf is not None:
            self.branch_data = LineData(
                line_gdf=self.branch_gdf,
                azimuth_set_ranges=self.azimuth_set_ranges,
                azimuth_set_names=self.azimuth_set_names,
                length_set_ranges=self.branch_length_set_ranges,
                length_set_names=self.branch_length_set_names,
                area_boundary_intersects=self.branch_intersects_target_area_boundary,
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
        if not self._is_branch_gdf_defined():
            return None
        return self.node_gdf.geometry

    @property
    def branch_series(self) -> Optional[gpd.GeoSeries]:
        if not self._is_branch_gdf_defined():
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
    def trace_length_array_non_weighted(self) -> np.ndarray:
        return self.trace_data.length_array_non_weighted

    @property
    def branch_length_array(self) -> Optional[np.ndarray]:
        if not self._is_branch_gdf_defined():
            return None
        return self.branch_data.length_array

    @property
    def branch_length_array_non_weighted(self) -> Optional[np.ndarray]:
        if not self._is_branch_gdf_defined():
            return None
        return self.branch_data.length_array_non_weighted

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
        return determine_node_type_counts(self.node_types)

    @property
    def branch_types(self) -> Optional[np.ndarray]:
        if not self._is_branch_gdf_defined():
            return None
        return self.branch_gdf[CONNECTION_COLUMN].to_numpy()

    @property
    def branch_counts(self) -> Optional[Dict[str, int]]:
        if not self._is_branch_gdf_defined():
            return None
        return determine_branch_type_counts(self.branch_types)

    @property
    def total_area(self) -> float:
        return self.area_gdf.geometry.area.sum()

    @property
    def parameters(self) -> Optional[Dict[str, float]]:
        if not self._is_branch_gdf_defined():
            return None
        # Cannot do simple cached_property because None might have been
        # returned previously.
        if self._parameters is None:
            self._parameters = determine_topology_parameters(
                trace_length_array=self.trace_length_array_non_weighted,
                node_counts=self.node_counts,  # type: ignore
                area=self.total_area,
            )
        return self._parameters

    @property
    def anisotropy(self) -> Optional[np.ndarray]:
        if not self._is_branch_gdf_defined():
            return None
        if self._anisotropy is None:
            self._anisotropy = determine_anisotropy_sum(
                azimuth_array=self.branch_azimuth_array,
                length_array=self.branch_length_array,
                branch_types=self.branch_types,
            )
        return self._anisotropy

    @property
    def azimuth_set_relationships(self) -> Optional[pd.DataFrame]:
        if not self._is_branch_gdf_defined():
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
    def length_set_relationships(self) -> Optional[pd.DataFrame]:
        if not self._is_branch_gdf_defined():
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

    @property
    def trace_lengths_powerlaw_fit_description(self) -> Dict[str, float]:
        """
        Short numerical description dict of trace length powerlaw fit.
        """
        return self.trace_data.describe_fit(label="trace")

    @property
    def branch_lengths_powerlaw_fit_description(self) -> Dict[str, float]:
        """
        Short numerical description dict of branch length powerlaw fit.
        """
        return self.branch_data.describe_fit(label="branch")

    @property
    def target_areas(self) -> List[Union[Polygon, MultiPolygon]]:
        """
        Get all target areas from area_gdf.
        """
        target_areas = []
        for target_area in self.area_gdf.geometry.values:
            if not isinstance(target_area, (Polygon, MultiPolygon)):
                raise TypeError("Expected (Multi)Polygon geometries in area_gdf.")
            target_areas.append(target_area)
        return target_areas

    @property
    def trace_intersects_target_area_boundary(self) -> np.ndarray:
        """
        Check traces for intersection with target area boundaries.

        Results are in integers:

        -  0 == No intersections
        -  1 == One intersection
        -  2 == Two intersections

        Does not discriminate between which target area (if multiple) the trace
        intersects. Intersection detection based on snap_threshold.
        """
        if self._trace_intersects_target_area_boundary is None:

            (
                intersecting_lines,
                cuts_through_lines,
            ) = determine_boundary_intersecting_lines(
                line_gdf=self.trace_gdf,
                area_gdf=self.area_gdf,
                snap_threshold=self.snap_threshold,
            )
            self._trace_intersects_target_area_boundary = bool_arrays_sum(
                intersecting_lines, cuts_through_lines
            )
        return self._trace_intersects_target_area_boundary

    @property
    def branch_intersects_target_area_boundary(self) -> Optional[np.ndarray]:
        """
        Get array of E-component count.
        """
        if (
            self._branch_intersects_target_area_boundary is None
            and self._is_branch_gdf_defined
        ):
            intersecting_lines = branches_intersect_boundary(self.branch_types)
            cuts_through_lines = np.array(
                [branch_type == EE_branch for branch_type in self.branch_types]
            )
            self._branch_intersects_target_area_boundary = bool_arrays_sum(
                intersecting_lines, cuts_through_lines
            )

        return self._branch_intersects_target_area_boundary

    def numerical_network_description(self) -> Dict[str, Union[float, int]]:
        """
        Collect numerical network attributes and return them as a dictionary.
        """
        parameters = self.parameters if self.parameters is not None else {}
        branch_counts = self.branch_counts if self.branch_counts is not None else {}
        node_counts = self.node_counts if self.node_counts is not None else {}
        description = {
            **node_counts,
            **branch_counts,
            **self.trace_lengths_powerlaw_fit_description,
            **self.branch_lengths_powerlaw_fit_description,
            **parameters,
        }

        func_descriptors = [
            "trace_lengths_cut_off_proportion",
            "branch_lengths_cut_off_proportion",
        ]

        for descriptor in func_descriptors:
            description[descriptor.replace("_", " ")] = getattr(self, descriptor)()

        return description

    def representative_points(self) -> List[Point]:
        """
        Get representative point(s) of target area(s).
        """
        return self.area_gdf.representative_point().to_list()

    def trace_lengths_powerlaw_fit(
        self, cut_off: Optional[float] = None
    ) -> powerlaw.Fit:
        """
        Determine powerlaw fit for trace lengths.
        """
        return (
            self.trace_data.automatic_fit
            if cut_off is None
            else self.trace_data.determine_manual_fit(cut_off=cut_off)
        )

    def trace_lengths_cut_off_proportion(
        self, fit: Optional[powerlaw.Fit] = None
    ) -> float:
        """
        Get proportion of trace data cut off by cut off.
        """
        return self.trace_data.cut_off_proportion_of_data(fit=fit)

    def branch_lengths_powerlaw_fit(
        self, cut_off: Optional[float] = None
    ) -> powerlaw.Fit:
        """
        Determine powerlaw fit for branch lengths.
        """
        return (
            self.branch_data.automatic_fit
            if cut_off is None
            else self.branch_data.determine_manual_fit(cut_off=cut_off)
        )

    def branch_lengths_cut_off_proportion(
        self, fit: Optional[powerlaw.Fit] = None
    ) -> float:
        """
        Get proportion of branch data cut off by cut off.
        """
        return self.branch_data.cut_off_proportion_of_data(fit=fit)

    def assign_branches_nodes(self):
        """
        Determine and assign branches and nodes as attributes.
        """
        if self.area_gdf is not None:
            branches, nodes = branches_and_nodes(
                self.trace_gdf,
                self.area_gdf,
                self.snap_threshold,
                already_clipped=self.truncate_traces,
            )
            if self.trace_gdf.crs is not None:
                branches.set_crs(self.trace_gdf.crs, inplace=True)
                nodes.set_crs(self.trace_gdf.crs, inplace=True)
            self.branch_gdf = branches
            self.node_gdf = nodes
            self.branch_data = LineData(
                line_gdf=self.branch_gdf,
                azimuth_set_ranges=self.azimuth_set_ranges,
                azimuth_set_names=self.azimuth_set_names,
                length_set_ranges=self.branch_length_set_ranges,
                length_set_names=self.branch_length_set_names,
                area_boundary_intersects=self.branch_intersects_target_area_boundary,
            )
        else:
            logging.error(
                "Expected area_geoseries to be defined to assign branches and nodes."
            )

    def plot_trace_lengths(
        self, label: Optional[str] = None, fit: Optional[powerlaw.Fit] = None
    ) -> Tuple[powerlaw.Fit, Figure, Axes]:  # type: ignore
        """
        Plot trace length distribution with `powerlaw` fits.
        """
        label = self.name if label is None else label
        return self.trace_data.plot_lengths(label=label, fit=fit)

    def plot_branch_lengths(
        self, label: Optional[str] = None, fit: Optional[powerlaw.Fit] = None
    ) -> Tuple[powerlaw.Fit, Figure, Axes]:  # type: ignore
        """
        Plot branch length distribution with `powerlaw` fits.
        """
        if label is None:
            label = self.name
        return self.branch_data.plot_lengths(
            label=label,
            fit=fit,
        )

    def plot_trace_azimuth(
        self, label: Optional[str] = None
    ) -> Tuple[Dict[str, np.ndarray], Figure, PolarAxes]:
        if label is None:
            label = self.name
        return self.trace_data.plot_azimuth(label=label)

    def plot_branch_azimuth(
        self, label: Optional[str] = None
    ) -> Tuple[Dict[str, np.ndarray], Figure, PolarAxes]:
        if label is None:
            label = self.name
        return self.branch_data.plot_azimuth(label=label)

    def plot_xyi(
        self, label: Optional[str] = None
    ) -> Optional[Tuple[Figure, Axes, TernaryAxesSubplot]]:
        """
        Plot ternary plot of node types.
        """
        if label is None:
            label = self.name
        if self.node_counts is None:
            logging.error("Expected node_gdf to be defined for plot_xyi.")
            return
        return plot_xyi_plot(node_counts_list=[self.node_counts], labels=[label])

    def plot_branch(
        self, label: Optional[str] = None
    ) -> Optional[Tuple[Figure, Axes, TernaryAxesSubplot]]:
        """
        Plot ternary plot of branch types.
        """
        if label is None:
            label = self.name
        if self.branch_counts is None:
            logging.error("Expected branch_gdf to be defined for plot_xyi.")
            return
        return plot_branch_plot(branch_counts_list=[self.branch_counts], labels=[label])

    def plot_parameters(
        self, label: Optional[str] = None, color: Optional[str] = None
    ) -> Optional[Tuple[Figure, Axes]]:
        """
        Plot geometric and topological parameters.
        """
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
    ) -> Optional[Tuple[Figure, Axes]]:
        if label is None:
            label = self.name
        if not self._is_branch_gdf_defined():
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
    ) -> Tuple[List[Figure], List[np.ndarray]]:
        return plot_crosscut_abutting_relationships_plot(
            relations_df=self.azimuth_set_relationships,  # type: ignore
            set_array=self.trace_azimuth_set_array,
            set_names=self.azimuth_set_names,
        )

    def plot_trace_length_crosscut_abutting_relationships(
        self,
    ) -> Tuple[List[Figure], List[np.ndarray]]:
        return plot_crosscut_abutting_relationships_plot(
            relations_df=self.length_set_relationships,  # type: ignore
            set_array=self.trace_data.length_set_array,
            set_names=self.trace_data.length_set_names,  # type: ignore
        )

    def plot_trace_azimuth_set_count(
        self, label: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        if label is None:
            label = self.name
        return self.trace_data.plot_azimuth_set_count(label=label)

    def plot_branch_azimuth_set_count(
        self, label: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        if label is None:
            label = self.name
        return self.branch_data.plot_azimuth_set_count(label=label)

    def plot_trace_length_set_count(
        self, label: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        if label is None:
            label = self.name
        return self.trace_data.plot_length_set_count(label=label)

    def plot_branch_length_set_count(
        self, label: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        if label is None:
            label = self.name
        return self.branch_data.plot_length_set_count(label=label)
