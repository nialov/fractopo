"""
Analyse and plot trace map data with Network.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import powerlaw
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.projections import PolarAxes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon
from ternary.ternary_axes_subplot import TernaryAxesSubplot

from fractopo.analysis.anisotropy import determine_anisotropy_sum, plot_anisotropy_plot
from fractopo.analysis.azimuth import AzimuthBins
from fractopo.analysis.contour_grid import run_grid_sampling
from fractopo.analysis.line_data import LineData
from fractopo.analysis.parameters import (
    branches_intersect_boundary,
    determine_branch_type_counts,
    determine_node_type_counts,
    determine_topology_parameters,
    plot_branch_plot,
    plot_parameters_plot,
    plot_xyi_plot,
)
from fractopo.analysis.relationships import (
    determine_crosscut_abutting_relationships,
    plot_crosscut_abutting_relationships_plot,
)
from fractopo.branches_and_nodes import branches_and_nodes
from fractopo.general import (
    CENSORING,
    CLASS_COLUMN,
    CONNECTION_COLUMN,
    NAME,
    RADIUS,
    RELATIVE_CENSORING,
    REPRESENTATIVE_POINT,
    EE_branch,
    Number,
    Param,
    SetRangeTuple,
    bool_arrays_sum,
    calc_circle_radius,
    crop_to_target_areas,
    determine_boundary_intersecting_lines,
    pygeos_spatial_index,
    raise_determination_error,
    spatial_index_intersection,
    total_bounds,
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
    area_gdf: Optional[gpd.GeoDataFrame] = None

    # Name the network for e.g. plot titles
    name: str = "Network"

    # The traces can be cut to end at the boundary of the target area
    # Defaults to True
    truncate_traces: bool = True

    # Whether to apply boundary line length weighting
    # Applies to both traces and branches
    # Length of lines that intersect the boundary once are multiplied by two,
    # and double intersections with 0, non-intersecting are multiplied by one
    # (no change).
    circular_target_area: bool = False

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
    snap_threshold: float = 0.001
    # If unary_union fails try lower and higher values
    unary_size_threshold: int = 5000

    # Length distributions
    # ====================
    # trace_length_cut_off: Optional[float] = None
    # branch_length_cut_off: Optional[float] = None

    censoring_area: Union[
        Polygon, MultiPolygon, gpd.GeoSeries, gpd.GeoDataFrame, None
    ] = None

    # Private caching attributes
    # ==========================
    _anisotropy: Optional[Tuple[np.ndarray, np.ndarray]] = field(
        default=None, repr=False
    )
    _parameters: Optional[Dict[str, float]] = field(default=None, repr=False)
    _azimuth_set_relationships: Optional[pd.DataFrame] = field(default=None, repr=False)
    _trace_length_set_relationships: Optional[pd.DataFrame] = field(
        default=None, repr=False
    )
    _trace_intersects_target_area_boundary: Optional[np.ndarray] = field(
        default=None, repr=False
    )
    _branch_intersects_target_area_boundary: Optional[np.ndarray] = field(
        default=None, repr=False
    )

    @staticmethod
    def _default_length_set_ranges(count, min_value, max_value):
        """
        Get default lengt set ranges.
        """
        arr = np.linspace(min_value, max_value, count + 1)
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
            if name == "branch_gdf" and self.branch_gdf is not None:

                self.branch_data = LineData(
                    line_gdf=self.get_branch_gdf().copy(),
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
                    self.get_area_gdf(),
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

    def get_area_gdf(self) -> gpd.GeoDataFrame:
        """
        Get area_gdf if it is given.
        """
        if self.area_gdf is None:
            raise_determination_error("area", verb="initilization", determine_target="")
        return self.area_gdf

    def get_branch_gdf(self) -> gpd.GeoDataFrame:
        """
        Get branch_gdf if it is determined.
        """
        if self.branch_gdf is None:
            raise_determination_error("branches")
        return self.branch_gdf

    def get_node_gdf(self) -> gpd.GeoDataFrame:
        """
        Get node_gdf if it is determined.
        """
        if self.node_gdf is None:
            raise_determination_error("nodes")
        return self.node_gdf

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
        """
        Get trace geometries as GeoSeries.
        """
        return self.trace_data.line_gdf.geometry

    @property
    def node_series(self) -> gpd.GeoSeries:
        """
        Get node geometries as GeoSeries.
        """
        if not self._is_branch_gdf_defined():
            raise_determination_error("node_series")
        return self.get_node_gdf().geometry

    @property
    def branch_series(self) -> gpd.GeoSeries:
        """
        Get branch geometries as GeoSeries.
        """
        if not self._is_branch_gdf_defined():
            raise_determination_error("branch_series")
        return self.branch_data.line_gdf.geometry

    @property
    def trace_azimuth_array(self) -> np.ndarray:
        """
        Get trace azimuths as array.
        """
        return self.trace_data.azimuth_array

    @property
    def branch_azimuth_array(self) -> np.ndarray:
        """
        Get branch azimuths as array.
        """
        if not self._is_branch_gdf_defined():
            raise_determination_error("branch_azimuth_array")
        return self.branch_data.azimuth_array

    @property
    def trace_length_array(self) -> np.ndarray:
        """
        Get trace lengths as array.
        """
        return self.trace_data.length_array

    @property
    def trace_length_array_non_weighted(self) -> np.ndarray:
        """
        Get non-boundary-weighted trace lengths as array.
        """
        return self.trace_data.length_array_non_weighted

    @property
    def branch_length_array(self) -> np.ndarray:
        """
        Get branch lengths as array.
        """
        if not self._is_branch_gdf_defined():
            raise_determination_error("branch_length_array")
        return self.branch_data.length_array

    @property
    def branch_length_array_non_weighted(self) -> np.ndarray:
        """
        Get non-boundary-weighted branch lengths as array.
        """
        if not self._is_branch_gdf_defined():
            raise_determination_error("branch_length_array_non_weighted")
        return self.branch_data.length_array_non_weighted

    @property
    def trace_azimuth_set_array(self) -> np.ndarray:
        """
        Get azimuth set for each trace.
        """
        return self.trace_data.azimuth_set_array

    @property
    def branch_azimuth_set_array(self) -> np.ndarray:
        """
        Get azimuth set for each branch.
        """
        if not self._is_branch_gdf_defined():
            raise_determination_error("branch_azimuth_set_array")
        return self.branch_data.azimuth_set_array

    @property
    def trace_length_set_array(self) -> np.ndarray:
        """
        Get length set for each trace.
        """
        return self.trace_data.length_set_array

    @property
    def branch_length_set_array(self) -> np.ndarray:
        """
        Get length set for each branch.
        """
        if not self._is_branch_gdf_defined():
            raise_determination_error("branch_length_set_array")
        return self.branch_data.length_set_array

    @property
    def node_types(self) -> np.ndarray:
        """
        Get node type of each node.
        """
        if not self._is_branch_gdf_defined():
            raise_determination_error("node_types")
        node_class_series = self.get_node_gdf()[CLASS_COLUMN]
        assert isinstance(node_class_series, pd.Series)
        return node_class_series.to_numpy()

    @property
    def node_counts(self) -> Dict[str, Number]:
        """
        Get node counts.
        """
        return determine_node_type_counts(
            self.node_types, branches_defined=self._is_branch_gdf_defined()
        )

    @property
    def branch_types(self) -> np.ndarray:
        """
        Get branch type of each branch.
        """
        if not self._is_branch_gdf_defined():
            raise_determination_error("branch_types")
        branch_connection_series = self.get_branch_gdf()[CONNECTION_COLUMN]
        assert isinstance(branch_connection_series, pd.Series)
        return branch_connection_series.to_numpy()

    @property
    def branch_counts(self) -> Dict[str, Number]:
        """
        Get branch counts.
        """
        return determine_branch_type_counts(
            self.branch_types, branches_defined=self._is_branch_gdf_defined()
        )

    @property
    def total_area(self) -> float:
        """
        Get total area.
        """
        return self.get_area_gdf().geometry.area.sum()

    @property
    def parameters(self) -> Dict[str, float]:
        """
        Get numerical geometric and topological parameters.
        """
        # Cannot do simple cached_property because None might have been
        # returned previously.
        if self._parameters is None:
            self._parameters = determine_topology_parameters(
                trace_length_array=self.trace_length_array_non_weighted,
                node_counts=self.node_counts,  # type: ignore
                area=self.total_area,
                branches_defined=self._is_branch_gdf_defined(),
            )
        return self._parameters

    @property
    def anisotropy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine anisotropy of connectivity.
        """
        if not self._is_branch_gdf_defined():
            raise_determination_error("anisotropy")
        if self._anisotropy is None:
            self._anisotropy = determine_anisotropy_sum(
                azimuth_array=self.branch_azimuth_array,
                length_array=self.branch_length_array,
                branch_types=self.branch_types,
            )
        return self._anisotropy

    @property
    def azimuth_set_relationships(self) -> pd.DataFrame:
        """
        Determine azimuth set relationships.
        """
        if not self._is_branch_gdf_defined():
            raise_determination_error("azimuth_set_relationships")
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
    def length_set_relationships(self) -> pd.DataFrame:
        """
        Determine length set relationships.
        """
        if not self._is_branch_gdf_defined():
            raise_determination_error("length_set_relationships")
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
        """
        Get trace azimuth set counts.
        """
        return self.trace_data.azimuth_set_counts

    @property
    def trace_length_set_counts(self) -> Dict[str, int]:
        """
        Get trace length set counts.
        """
        return self.trace_data.length_set_counts

    @property
    def branch_azimuth_set_counts(self) -> Dict[str, int]:
        """
        Get branch azimuth set counts.
        """
        return self.branch_data.azimuth_set_counts

    @property
    def branch_length_set_counts(self) -> Dict[str, int]:
        """
        Get branch length set counts.
        """
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
        for target_area in self.get_area_gdf().geometry.values:
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

            if self.circular_target_area:

                (
                    intersecting_lines,
                    cuts_through_lines,
                ) = determine_boundary_intersecting_lines(
                    line_gdf=self.trace_gdf,
                    area_gdf=self.get_area_gdf(),
                    snap_threshold=self.snap_threshold,
                )
                self._trace_intersects_target_area_boundary = bool_arrays_sum(
                    intersecting_lines, cuts_through_lines
                )
            else:
                self._trace_intersects_target_area_boundary = np.array(
                    [0] * self.trace_gdf.shape[0]
                )
        return self._trace_intersects_target_area_boundary

    @property
    def branch_intersects_target_area_boundary(self) -> np.ndarray:
        """
        Get array of E-component count.
        """
        if not self._is_branch_gdf_defined():
            raise_determination_error("branch_intersects_target_area_boundary")
        if self._branch_intersects_target_area_boundary is None:
            if self.circular_target_area:
                intersecting_lines = branches_intersect_boundary(self.branch_types)
                cuts_through_lines = np.array(
                    [branch_type == EE_branch for branch_type in self.branch_types]
                )
                self._branch_intersects_target_area_boundary = bool_arrays_sum(
                    intersecting_lines, cuts_through_lines
                )
            else:
                self._branch_intersects_target_area_boundary = np.array(
                    [0] * len(self.branch_types)
                )
        return self._branch_intersects_target_area_boundary

    @property
    def trace_boundary_intersect_count(self) -> Dict[str, int]:
        """
        Get counts of trace intersects with boundary.
        """
        key_counts = self.trace_data.boundary_intersect_count
        trace_key_counts = dict()
        for key, item in key_counts.items():
            trace_key_counts[f"Trace Boundary {key} Intersect Count"] = item
        return trace_key_counts

    @property
    def branch_boundary_intersect_count(self) -> Dict[str, int]:
        """
        Get counts of branch intersects with boundary.
        """
        key_counts = self.branch_data.boundary_intersect_count
        branch_key_counts = dict()
        for key, item in key_counts.items():
            branch_key_counts[f"Branch Boundary {key} Intersect Count"] = item
        return branch_key_counts

    def numerical_network_description(self) -> Dict[str, Union[Number, str]]:
        """
        Collect numerical network attributes and return them as a dictionary.
        """
        parameters = self.parameters
        branch_counts = self.branch_counts
        node_counts = self.node_counts
        trace_boundary_intersect_count = self.trace_boundary_intersect_count
        branch_boundary_intersect_count = self.branch_boundary_intersect_count
        radius = {
            RADIUS: (
                calc_circle_radius(parameters[Param.AREA.value])
                if self.circular_target_area
                else np.nan
            )
        }
        censoring_value = (
            self.estimate_censoring() if self.censoring_area is not None else np.nan
        )
        censoring_and_relative = {
            CENSORING: censoring_value,
            RELATIVE_CENSORING: censoring_value / parameters[Param.AREA.value],
        }
        description = {
            **trace_boundary_intersect_count,
            **branch_boundary_intersect_count,
            **branch_counts,
            **node_counts,
            **parameters,
            **radius,
            **self.branch_lengths_powerlaw_fit_description,
            **self.trace_lengths_powerlaw_fit_description,
            NAME: self.name,
            REPRESENTATIVE_POINT: MultiPoint(self.representative_points()).centroid.wkt,
            **censoring_and_relative,
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
        return self.get_area_gdf().representative_point().to_list()

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
                unary_size_threshold=self.unary_size_threshold,
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
            raise AttributeError(
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
    ) -> Tuple[AzimuthBins, Figure, PolarAxes]:
        """
        Plot trace azimuth rose plot.
        """
        if label is None:
            label = self.name
        return self.trace_data.plot_azimuth(label=label)

    def plot_branch_azimuth(
        self, label: Optional[str] = None
    ) -> Tuple[AzimuthBins, Figure, PolarAxes]:
        """
        Plot branch azimuth rose plot.
        """
        if label is None:
            label = self.name
        return self.branch_data.plot_azimuth(label=label)

    def plot_xyi(
        self, label: Optional[str] = None
    ) -> Tuple[Figure, Axes, TernaryAxesSubplot]:
        """
        Plot ternary plot of node types.
        """
        if label is None:
            label = self.name
        if self.node_counts is None:
            raise AttributeError("Expected node_gdf to be defined for plot_xyi.")
        if any(np.isnan(list(self.node_counts.values()))):
            raise ValueError(f"Expected no nan in node_counts: {self.node_counts}")
        node_counts_ints = {
            key: int(value)
            for key, value in self.node_counts.items()
            if not np.isnan(value)
        }
        if len(node_counts_ints) != len(self.node_counts):
            raise ValueError(f"Expected no nan in node_counts: {self.node_counts}")
        return plot_xyi_plot(node_counts_list=[node_counts_ints], labels=[label])

    def plot_branch(
        self, label: Optional[str] = None
    ) -> Tuple[Figure, Axes, TernaryAxesSubplot]:
        """
        Plot ternary plot of branch types.
        """
        if label is None:
            label = self.name
        if self.branch_counts is None:
            raise AttributeError("Expected branch_gdf to be defined for plot_xyi.")
        branch_counts_ints = {
            key: int(value)
            for key, value in self.branch_counts.items()
            if not np.isnan(value)
        }
        if len(branch_counts_ints) != len(self.branch_counts):
            raise ValueError(f"Expected no nan in branch_counts: {self.branch_counts}")
        return plot_branch_plot(branch_counts_list=[branch_counts_ints], labels=[label])

    def plot_parameters(
        self, label: Optional[str] = None, color: Optional[str] = None
    ) -> Optional[Tuple[Figure, Axes]]:
        """
        Plot geometric and topological parameters.
        """
        if not self._is_branch_gdf_defined():
            raise_determination_error("parameters")
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
        """
        Plot anisotropy of connectivity plot.
        """
        if label is None:
            label = self.name
        if not self._is_branch_gdf_defined():
            raise_determination_error("anisotropy")
        if color is None:
            color = "black"
        anisotropy_sum = self.anisotropy[0]
        sample_intervals = self.anisotropy[1]
        fig, ax = plot_anisotropy_plot(
            anisotropy_sum=anisotropy_sum,
            sample_intervals=sample_intervals,
            # label=label,  # type: ignore
            # color=color,
        )
        return fig, ax

    def plot_azimuth_crosscut_abutting_relationships(
        self,
    ) -> Tuple[List[Figure], List[np.ndarray]]:
        """
        Plot azimuth set crosscutting and abutting relationships.
        """
        return plot_crosscut_abutting_relationships_plot(
            relations_df=self.azimuth_set_relationships,  # type: ignore
            set_array=self.trace_azimuth_set_array,
            set_names=self.azimuth_set_names,
        )

    def plot_trace_length_crosscut_abutting_relationships(
        self,
    ) -> Tuple[List[Figure], List[np.ndarray]]:
        """
        Plot length set crosscutting and abutting relationships.
        """
        return plot_crosscut_abutting_relationships_plot(
            relations_df=self.length_set_relationships,  # type: ignore
            set_array=self.trace_data.length_set_array,
            set_names=self.trace_data.length_set_names,  # type: ignore
        )

    def plot_trace_azimuth_set_count(
        self, label: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot trace azimuth set counts.
        """
        if label is None:
            label = self.name
        return self.trace_data.plot_azimuth_set_count(label=label)

    def plot_branch_azimuth_set_count(
        self, label: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot branch azimuth set counts.
        """
        if label is None:
            label = self.name
        return self.branch_data.plot_azimuth_set_count(label=label)

    def plot_trace_length_set_count(
        self, label: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot trace length set counts.
        """
        if label is None:
            label = self.name
        return self.trace_data.plot_length_set_count(label=label)

    def plot_branch_length_set_count(
        self, label: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot branch length set counts.
        """
        if label is None:
            label = self.name
        return self.branch_data.plot_length_set_count(label=label)

    def contour_grid(
        self,
        cell_width: Optional[float] = None,
        bounds_divider: float = 20.0,
        precursor_grid: Optional[gpd.GeoDataFrame] = None,
    ):
        """
        Sample the network with a contour grid.

        If ``cell_width`` is passed it is used as the cell width. Otherwise
        a cell width is calculated using the network branch bounds using
        the passed ``bounds_divider`` or its default value.

        If ``precursor_grid`` is passed it is used as the grid in which
        each Polygon cell is filled with calculated network parameter values.
        """
        if cell_width is None:
            # Use trace bounds to calculate a width
            min_x, min_y, max_x, max_y = total_bounds(self.get_branch_gdf())
            x_diff = max_x - min_x
            y_diff = max_y - min_y
            if x_diff < y_diff:
                cell_width = y_diff / bounds_divider
            else:
                cell_width = x_diff / bounds_divider

        assert isinstance(cell_width, (float, int))

        sampled_grid = run_grid_sampling(
            traces=self.trace_gdf,
            branches=self.get_branch_gdf(),
            nodes=self.get_node_gdf(),
            cell_width=cell_width,
            snap_threshold=self.snap_threshold,
            precursor_grid=precursor_grid,
            resolve_branches_and_nodes=False,
        )
        return sampled_grid

    def plot_contour(
        self, parameter: str, sampled_grid: gpd.GeoDataFrame
    ) -> Tuple[Figure, Axes]:
        """
        Plot contour plot of a geometric or topological parameter.

        Creating the contour grid is expensive so the ``sampled_grid`` must
        be first created with ``Network.contour_grid`` method and then passed
        to this one for plotting.
        """
        assert all(isinstance(val, Polygon) for val in sampled_grid.geometry.values)
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        sampled_grid.plot(
            column=parameter,
            legend=True,
            cax=cax,
            ax=ax,
            legend_kwds={"label": parameter},
        )
        ax.set_title(self.name)
        return fig, ax

    def estimate_censoring(
        self,
        censoring_area: Union[
            Polygon, MultiPolygon, gpd.GeoSeries, gpd.GeoDataFrame, None
        ] = None,
    ):
        """
        Estimate the amount of censoring caused by e.g. vegetation.

        Requires that ``Network`` is initialized with ``censoring_gdf`` or that
        its passed here. If passed here the passed area is always used
        overriding the one given at ``Network`` initilization.
        """
        # Either censoring_area is passed directly or its passed in Network
        # creation. Otherwise raise ValueError.
        if censoring_area is None:
            if self.censoring_area is None:
                raise ValueError(
                    "Expected censoring_area as an argument or initialized"
                    f" as Network (name:{self.name}) attribute."
                )
            censoring_area = self.censoring_area

        # Gather into GeoSeries or GeoDataFrame, if not already.
        if not isinstance(censoring_area, (gpd.GeoSeries, gpd.GeoDataFrame)):
            polygons: List[Polygon] = []
            if isinstance(censoring_area, Polygon):
                polygons = [censoring_area]
            elif isinstance(censoring_area, MultiPolygon):
                polygons = list(censoring_area.geoms)
            censoring_geodata = gpd.GeoSeries(polygons, crs=self.trace_gdf.crs)
        else:
            censoring_geodata = censoring_area

        # Determine bounds of Network.area_gdf
        network_area_bounds_arr = self.get_area_gdf().total_bounds
        network_area_bounds: Tuple[float, float, float, float] = (
            network_area_bounds_arr[0],
            network_area_bounds_arr[1],
            network_area_bounds_arr[2],
            network_area_bounds_arr[3],
        )

        # Use spatial index to filter censoring polygons that are not near the
        # network
        sindex = pygeos_spatial_index(censoring_geodata)
        index_intersection = spatial_index_intersection(
            spatial_index=sindex, coordinates=network_area_bounds
        )
        candidate_idxs = list(
            index_intersection if index_intersection is not None else []
        )
        if len(candidate_idxs) == 0:
            return 0.0
        candidates = censoring_geodata.iloc[candidate_idxs]

        # Clip the censoring areas with the network area and calculate the area
        # sum of leftover area.
        censoring_value = gpd.clip(candidates, self.get_area_gdf()).area.sum()
        unpacked_value = (
            censoring_value.item()
            if hasattr(censoring_value, "item")
            else censoring_value
        )
        assert isinstance(unpacked_value, float)
        assert unpacked_value >= 0.0
        return unpacked_value
