"""
Analyse and plot trace map data with Network.
"""
import logging
from dataclasses import dataclass, field
from functools import wraps
from hashlib import sha256
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

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
    convert_counts,
    determine_branch_type_counts,
    determine_node_type_counts,
    determine_topology_parameters,
    plot_parameters_plot,
    plot_ternary_plot,
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
    numpy_to_python_type,
    pygeos_spatial_index,
    raise_determination_error,
    read_geofile,
    spatial_index_intersection,
    total_bounds,
    write_geodata,
)

DEFAULT_NETWORK_CACHE_PATH = Path(".fractopo_cache")


def requires_topology(func: Callable):
    """
    Wrap methods that require determined topology.

    Raises an error if trying to call them without determined topology.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if "network" in kwargs:
            network = kwargs.pop("network")
        else:
            network = args[0]
            args = args[1:] if len(args) > 1 else []
        if not network.topology_determined:
            raise_determination_error(func.__name__)
        return func(network, *args, **kwargs)

    return wrapper


@dataclass
class Network:

    """
    Trace network.

    Consists of at its simplest of validated traces and a target area that
    delineates the traces.

    :param trace_gdf: ``GeoDataFrame`` containing trace data
        i.e. ``shapely.geometry.LineString's``.
    :param area_gdf: ``GeoDataFrame`` containing
        target area data i.e. ``(Multi)Polygon's``.
    :param name: Name the Network.
    :param determine_branches_nodes: Whether to determine branches and nodes.
    :param snap_threshold: The snapping distance threshold to identify
        snapped traces.
    :param truncate_traces: Whether to crop the traces at the target area
        boundary.
    :param circular_target_area: Is the target are a circle.
    :param azimuth_set_names: Names of each azimuth set.
    :param azimuth_set_ranges: Ranges of each azimuth set.
    :param trace_length_set_names: Names of each trace length set.
    :param trace_length_set_ranges: Ranges of each trace length set.
    :param branch_length_set_names: Names of each branch length set.
    :param branch_length_set_ranges: Ranges of each branch length set.
    :param branch_gdf: ``GeoDataFrame`` containing branch data.
        It is recommended to let ``fractopo.Network`` determine both
        branches and nodes instead of passing them here.
    :param node_gdf: GeoDataFrame containing node data.
        It is recommended to let ``fractopo.Network`` determine both
        branches and nodes instead of passing them here.
    :param censoring_area: Geometry that delineates the area in which trace
        digitization was uncertain due to censoring caused by e.g. vegetation.
    """

    # Base data
    # =========
    trace_gdf: gpd.GeoDataFrame
    area_gdf: gpd.GeoDataFrame

    # Name the network for e.g. plot titles
    name: str = "Network"

    determine_branches_nodes: bool = False
    snap_threshold: float = 0.001

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
    azimuth_set_names: Tuple[str, ...] = ("1", "2", "3")
    azimuth_set_ranges: SetRangeTuple = (
        (0, 60),
        (60, 120),
        (120, 180),
    )

    # Length sets
    # ===========

    # Trace length
    trace_length_set_names: Tuple[str, ...] = ()
    trace_length_set_ranges: SetRangeTuple = ()

    # Branch length
    branch_length_set_names: Tuple[str, ...] = ()
    branch_length_set_ranges: SetRangeTuple = ()

    # Branches and nodes
    # ==================
    branch_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame()
    node_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame()

    censoring_area: gpd.GeoDataFrame = gpd.GeoDataFrame()

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
        Get default length set ranges.

        TODO: Currently not used.
        """
        arr = np.linspace(min_value, max_value, count + 1)
        starts = arr[0 : count + 1]
        ends = arr[1:]
        assert len(starts) == len(ends)
        as_gen = ((start, end) for start, end in zip(starts, ends))
        return tuple(as_gen)

    def __post_init__(self):
        """
        Copy GeoDataFrames instead of changing inputs.

        If the data is passed later to attribute, __setattr__ will also
        handle copying.

        :raises ValueError: If trace ``GeoDataFrame`` is empty after
            ``crop_to_target_areas``.
        """
        self.topology_determined = False
        if self.area_gdf.empty or self.area_gdf.geometry.iloc[0].is_empty:
            raise ValueError(
                "Passed area_gdf or geometry at index 0 is empty.\n"
                "Either pass a non-empty area_gdf or alternatively use\n"
                "fractopo.general.bounding_polygon to create an enveloping\n"
                "non-intersecting Polygon around your trace_gdf."
            )
        if self.circular_target_area:
            if not self.truncate_traces:
                raise ValueError(
                    "Traces must be truncated to the target area"
                    " to perform circular area trace weighting. "
                    "\n(To fix: pass truncate_traces=True.)"
                )
        # Copy geodataframes instead of using pointers
        # Traces
        self.trace_gdf = self.trace_gdf.copy()
        # Area
        self.area_gdf = self.area_gdf.copy()
        # Branches
        self.branch_gdf = self.branch_gdf.copy()
        # Branches
        self.node_gdf = self.node_gdf.copy()

        if self.truncate_traces:
            self.trace_gdf = gpd.GeoDataFrame(
                crop_to_target_areas(
                    self.trace_gdf,
                    self.area_gdf,
                    keep_column_data=True,
                )
            )
            self.trace_gdf.reset_index(inplace=True, drop=True)
            if self.trace_gdf.shape[0] == 0:
                raise ValueError("Empty trace GeoDataFrame after crop_to_target_areas.")

        self.trace_data = LineData(
            _line_gdf=self.trace_gdf,
            azimuth_set_ranges=self.azimuth_set_ranges,
            azimuth_set_names=self.azimuth_set_names,
            length_set_ranges=self.trace_length_set_ranges,
            length_set_names=self.trace_length_set_names,
            area_boundary_intersects=self.trace_intersects_target_area_boundary,
        )

        empty_branches_and_nodes = self.branch_gdf.empty and self.node_gdf.empty
        if self.determine_branches_nodes and empty_branches_and_nodes:
            logging.info(
                "Determining branches and nodes.",
                extra=dict(
                    empty_branches_and_nodes=empty_branches_and_nodes,
                    network_name=self.name,
                ),
            )
            self.assign_branches_nodes()
        elif not empty_branches_and_nodes:
            logging.info(
                "Found branch_gdf and node_gdf in inputs. Using them.",
                extra=dict(
                    empty_branches_and_nodes=empty_branches_and_nodes,
                    network_name=self.name,
                ),
            )
            self.assign_branches_nodes(branches=self.branch_gdf, nodes=self.node_gdf)
            self.topology_determined = True

        logging.info(
            "Created and initialized Network instance.",
            # Network has .name attribute which will overwrite logging
            # attribute!!!
            extra={f"network_{key}": value for key, value in self.__dict__.items()},
        )

    def __hash__(self) -> int:
        """
        Implement Network hashing.

        Deprecated. Memory caching of ``GeoDataFrame``s is unstable.
        """

        def convert_gdf(
            gdf: Union[gpd.GeoDataFrame, gpd.GeoSeries, None, Polygon, MultiPolygon]
        ) -> Optional[str]:
            """
            Convert GeoDataFrame or geometry to (json) str.

            """
            if gdf is None:
                return None
            if isinstance(gdf, (gpd.GeoSeries, gpd.GeoDataFrame)):
                return gdf.geometry.to_json()
            return gdf.wkt

        traces_geojson = convert_gdf(self.trace_gdf)
        area_geojson = convert_gdf(self.area_gdf)

        hash_args = (
            traces_geojson,
            area_geojson,
            self.name,
            self.determine_branches_nodes,
            self.snap_threshold,
            self.truncate_traces,
            self.circular_target_area,
            self.azimuth_set_names,
            self.azimuth_set_ranges,
            self.trace_length_set_names,
            self.trace_length_set_ranges,
            self.branch_length_set_names,
            self.branch_length_set_ranges,
            convert_gdf(self.branch_gdf),
            convert_gdf(self.node_gdf),
            convert_gdf(self.censoring_area),
        )

        return hash(hash_args)

    def reset_length_data(self):
        """
        Reset LineData attributes.

        WARNING: Mostly untested.
        """
        logging.warning(
            "Method Network.reset_length_data is mostly untested. Use with"
            "caution. It is safer to recreate the Network instance.",
            extra=dict(network_name=self.name),
        )
        self.trace_data = LineData(
            _line_gdf=self.trace_gdf,
            azimuth_set_ranges=self.azimuth_set_ranges,
            azimuth_set_names=self.azimuth_set_names,
            length_set_ranges=self.trace_length_set_ranges,
            length_set_names=self.trace_length_set_names,
            area_boundary_intersects=self.trace_intersects_target_area_boundary,
        )
        if not self.branch_gdf.empty:
            self.branch_data = LineData(
                _line_gdf=self.branch_gdf,
                azimuth_set_ranges=self.azimuth_set_ranges,
                azimuth_set_names=self.azimuth_set_names,
                length_set_ranges=self.branch_length_set_ranges,
                length_set_names=self.branch_length_set_names,
                area_boundary_intersects=self.branch_intersects_target_area_boundary,
            )
            self._azimuth_set_relationships = None

    @property
    def trace_series(self) -> gpd.GeoSeries:
        """
        Get trace geometries as GeoSeries.
        """
        return self.trace_data.geometry

    @property
    @requires_topology
    def node_series(self) -> gpd.GeoSeries:
        """
        Get node geometries as GeoSeries.
        """
        return self.node_gdf.geometry

    @property
    @requires_topology
    def branch_series(self) -> gpd.GeoSeries:
        """
        Get branch geometries as GeoSeries.
        """
        return self.branch_data.geometry

    @property
    def trace_azimuth_array(self) -> np.ndarray:
        """
        Get trace azimuths as array.
        """
        return self.trace_data.azimuth_array

    @property
    @requires_topology
    def branch_azimuth_array(self) -> np.ndarray:
        """
        Get branch azimuths as array.
        """
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
    @requires_topology
    def branch_length_array(self) -> np.ndarray:
        """
        Get branch lengths as array.
        """
        return self.branch_data.length_array

    @property
    @requires_topology
    def branch_length_array_non_weighted(self) -> np.ndarray:
        """
        Get non-boundary-weighted branch lengths as array.
        """
        return self.branch_data.length_array_non_weighted

    @property
    def trace_azimuth_set_array(self) -> np.ndarray:
        """
        Get azimuth set for each trace.
        """
        return self.trace_data.azimuth_set_array

    @property
    @requires_topology
    def branch_azimuth_set_array(self) -> np.ndarray:
        """
        Get azimuth set for each branch.
        """
        return self.branch_data.azimuth_set_array

    @property
    def trace_length_set_array(self) -> np.ndarray:
        """
        Get length set for each trace.
        """
        return self.trace_data.length_set_array

    @property
    @requires_topology
    def branch_length_set_array(self) -> np.ndarray:
        """
        Get length set for each branch.
        """
        return self.branch_data.length_set_array

    @property
    @requires_topology
    def node_types(self) -> np.ndarray:
        """
        Get node type of each node.
        """
        node_class_series = self.node_gdf[CLASS_COLUMN]
        assert isinstance(node_class_series, pd.Series)
        return node_class_series.to_numpy()

    @property
    def node_counts(self) -> Dict[str, int]:
        """
        Get node counts.
        """
        return convert_counts(
            determine_node_type_counts(
                self.node_types, branches_defined=self.topology_determined
            )
        )

    @property
    @requires_topology
    def branch_types(self) -> np.ndarray:
        """
        Get branch type of each branch.
        """
        branch_connection_series = self.branch_gdf[CONNECTION_COLUMN]
        assert isinstance(branch_connection_series, pd.Series)
        as_array = branch_connection_series.to_numpy()
        assert isinstance(as_array, np.ndarray)
        return as_array

    @property
    @requires_topology
    def branch_counts(self) -> Dict[str, int]:
        """
        Get branch counts.
        """
        return convert_counts(
            determine_branch_type_counts(
                self.branch_types, branches_defined=self.topology_determined
            )
        )

    @property
    def total_area(self) -> float:
        """
        Get total area.
        """
        area_sum = numpy_to_python_type(self.area_gdf.geometry.area.sum())
        assert isinstance(area_sum, float)
        assert area_sum >= 0.0
        return area_sum

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
                node_counts=self.node_counts,
                area=self.total_area,
                branches_defined=self.topology_determined,
                correct_mauldon=self.circular_target_area,
            )
        return self._parameters

    @property
    @requires_topology
    def anisotropy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine anisotropy of connectivity.
        """
        if self._anisotropy is None:
            self._anisotropy = determine_anisotropy_sum(
                azimuth_array=self.branch_azimuth_array,
                length_array=self.branch_length_array,
                branch_types=self.branch_types,
            )
        return self._anisotropy

    @property
    @requires_topology
    def azimuth_set_relationships(self) -> pd.DataFrame:
        """
        Determine azimuth set relationships.
        """
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
    @requires_topology
    def length_set_relationships(self) -> pd.DataFrame:
        """
        Determine length set relationships.
        """
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
        for target_area in self.area_gdf.geometry.values:
            if not isinstance(target_area, (Polygon, MultiPolygon)):
                raise TypeError("Expected (Multi)Polygon geometries in area_gdf.")
            if target_area.is_empty:
                raise ValueError("Expected non-empty geometries for target areas.")
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
                    area_gdf=self.area_gdf,
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
    @requires_topology
    def branch_intersects_target_area_boundary(self) -> np.ndarray:
        """
        Get array of E-component count.
        """
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

    def numerical_network_description(
        self,
        trace_lengths_cut_off: Optional[float] = None,
        branch_lengths_cut_off: Optional[float] = None,
    ) -> Dict[str, Union[Number, str]]:
        """
        Collect numerical network attributes and return them as a dictionary.
        """
        parameters = self.parameters
        branch_counts = self.branch_counts
        node_counts = self.node_counts
        trace_boundary_intersect_count = self.trace_data.boundary_intersect_count_desc(
            label="Trace"
        )
        branch_boundary_intersect_count = (
            self.branch_data.boundary_intersect_count_desc(label="Branch")
        )
        radius = {
            RADIUS: (
                calc_circle_radius(parameters[Param.AREA.value.name])
                if self.circular_target_area
                else np.nan
            )
        }
        censoring_value = self.estimate_censoring()
        censoring_and_relative = {
            CENSORING: censoring_value,
            RELATIVE_CENSORING: censoring_value / parameters[Param.AREA.value.name],
        }
        description = {
            **trace_boundary_intersect_count,
            **branch_boundary_intersect_count,
            **branch_counts,
            **node_counts,
            **parameters,
            **radius,
            **self.trace_data.describe_fit(
                label="trace", cut_off=trace_lengths_cut_off
            ),
            **self.branch_data.describe_fit(
                label="branch", cut_off=branch_lengths_cut_off
            ),
            NAME: self.name,
            REPRESENTATIVE_POINT: MultiPoint(self.representative_points()).centroid.wkt,
            **censoring_and_relative,
        }

        return description

    def representative_points(self) -> List[Point]:
        """
        Get representative point(s) of target area(s).
        """
        point_list = self.area_gdf.representative_point().to_list()
        assert isinstance(point_list, list)
        assert all(isinstance(point, Point) for point in point_list)
        return point_list

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

    @requires_topology
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

    def assign_branches_nodes(
        self,
        branches: Optional[gpd.GeoDataFrame] = None,
        nodes: Optional[gpd.GeoDataFrame] = None,
    ):
        """
        Determine and assign branches and nodes as attributes.
        """
        if branches is None or nodes is None:
            branches, nodes = branches_and_nodes(
                self.trace_gdf,
                self.area_gdf,
                self.snap_threshold,
                already_clipped=self.truncate_traces,
                # unary_size_threshold=self.unary_size_threshold,
            )
        else:
            branches = branches.copy()
            nodes = nodes.copy()
        # if self.trace_gdf.crs is not None:
        #     branches.set_crs(self.trace_gdf.crs, inplace=True)
        #     nodes.set_crs(self.trace_gdf.crs, inplace=True)
        self.branch_gdf = branches
        self.node_gdf = nodes
        self.topology_determined = True
        self.branch_data = LineData(
            _line_gdf=self.branch_gdf,
            azimuth_set_ranges=self.azimuth_set_ranges,
            azimuth_set_names=self.azimuth_set_names,
            length_set_ranges=self.branch_length_set_ranges,
            length_set_names=self.branch_length_set_names,
            area_boundary_intersects=self.branch_intersects_target_area_boundary,
        )

    def plot_trace_lengths(
        self, label: Optional[str] = None, fit: Optional[powerlaw.Fit] = None
    ) -> Tuple[powerlaw.Fit, Figure, Axes]:  # type: ignore
        """
        Plot trace length distribution with `powerlaw` fits.
        """
        label = self.name if label is None else label
        return self.trace_data.plot_lengths(label=label, fit=fit)

    @requires_topology
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
        self, label: Optional[str] = None, append_azimuth_set_text: bool = False
    ) -> Tuple[AzimuthBins, Figure, PolarAxes]:
        """
        Plot trace azimuth rose plot.
        """
        if label is None:
            label = self.name
        return self.trace_data.plot_azimuth(
            label=label, append_azimuth_set_text=append_azimuth_set_text
        )

    @requires_topology
    def plot_branch_azimuth(
        self, label: Optional[str] = None, append_azimuth_set_text: bool = False
    ) -> Tuple[AzimuthBins, Figure, PolarAxes]:
        """
        Plot branch azimuth rose plot.
        """
        if label is None:
            label = self.name
        return self.branch_data.plot_azimuth(
            label=label, append_azimuth_set_text=append_azimuth_set_text
        )

    @requires_topology
    def plot_xyi(
        self, label: Optional[str] = None
    ) -> Tuple[Figure, Axes, TernaryAxesSubplot]:
        """
        Plot ternary plot of node types.
        """
        if label is None:
            label = self.name
        return plot_ternary_plot(
            counts_list=[self.node_counts], labels=[label], is_nodes=True
        )

    @requires_topology
    def plot_branch(
        self, label: Optional[str] = None
    ) -> Tuple[Figure, Axes, TernaryAxesSubplot]:
        """
        Plot ternary plot of branch types.
        """
        if label is None:
            label = self.name
        return plot_ternary_plot(
            counts_list=[self.branch_counts], labels=[label], is_nodes=False
        )

    def plot_parameters(
        self, label: Optional[str] = None, color: Optional[str] = None
    ) -> Optional[Tuple[Figure, Axes]]:
        """
        Plot geometric and topological parameters.
        """
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

    @requires_topology
    def plot_anisotropy(
        self, label: Optional[str] = None, color: Optional[str] = None
    ) -> Optional[Tuple[Figure, Axes]]:
        """
        Plot anisotropy of connectivity plot.
        """
        if label is None:
            label = self.name
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

    @requires_topology
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

    @requires_topology
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

    @requires_topology
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

    def plot_trace_azimuth_set_lengths(
        self,
    ) -> Tuple[List[powerlaw.Fit], List[Figure], List[Axes]]:
        """
        Plot trace azimuth set lengths with fits.
        """
        return self.trace_data.plot_azimuth_set_lengths()

    @requires_topology
    def plot_branch_azimuth_set_lengths(
        self,
    ) -> Tuple[List[powerlaw.Fit], List[Figure], List[Axes]]:
        """
        Plot branch azimuth set lengths with fits.
        """
        return self.branch_data.plot_azimuth_set_lengths()

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
            min_x, min_y, max_x, max_y = total_bounds(self.branch_gdf)
            x_diff = max_x - min_x
            y_diff = max_y - min_y
            if x_diff < y_diff:
                cell_width = y_diff / bounds_divider
            else:
                cell_width = x_diff / bounds_divider

        assert isinstance(cell_width, (float, int))

        sampled_grid = run_grid_sampling(
            traces=self.trace_gdf,
            branches=self.branch_gdf,
            nodes=self.node_gdf,
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
        ax.set_title(f"{ self.name } - {parameter}")
        return fig, ax

    def estimate_censoring(
        self,
    ) -> float:
        """
        Estimate the amount of censoring as area float value.

        Censoring is caused by e.g. vegetation.

        Returns np.nan if no ``censoring_area`` is passed by the user into
        ``Network`` creation or if the passed GeoDataFrame is empty.
        """
        # Either censoring_area is passed directly or its passed in Network
        # creation. Otherwise raise ValueError.
        if self.censoring_area is None:
            raise ValueError(
                "Expected censoring_area as an argument or initialized"
                f" as Network (name:{self.name}) attribute."
            )
        if not isinstance(self.censoring_area, gpd.GeoDataFrame):
            raise TypeError(
                "Expected censoring_area to be of type gpd.GeoDataFrame."
                f" Got type={type(self.censoring_area)}"
            )
        if self.censoring_area.empty:
            return np.nan

        # Determine bounds of Network.area_gdf
        network_area_bounds = total_bounds(self.area_gdf)

        # Use spatial index to filter censoring polygons that are not near the
        # network
        sindex = pygeos_spatial_index(self.censoring_area)
        index_intersection = spatial_index_intersection(
            spatial_index=sindex, coordinates=network_area_bounds
        )
        candidate_idxs = list(
            index_intersection if index_intersection is not None else []
        )
        if len(candidate_idxs) == 0:
            return 0.0
        candidates = self.censoring_area.iloc[candidate_idxs]

        # Clip the censoring areas with the network area and calculate the area
        # sum of leftover area.
        clipped = gpd.clip(candidates, self.area_gdf)
        if not isinstance(clipped, (gpd.GeoDataFrame, gpd.GeoSeries)):
            vals = type(clipped), clipped
            raise TypeError(
                f"Expected that clipped is of geopandas data type. Got: {vals}."
            )
        censoring_value = clipped.area.sum()
        unpacked_value = numpy_to_python_type(censoring_value)
        assert isinstance(unpacked_value, float)
        assert unpacked_value >= 0.0
        return unpacked_value

    @requires_topology
    def write_branches_and_nodes(
        self,
        output_dir_path: Path,
        branches_name: Optional[str] = None,
        nodes_name: Optional[str] = None,
    ):
        """
        Write branches and nodes to disk.

        Enables reuse of the same data in analysis of the same data to skip
        topology determination which is computationally expensive.

        Writes only with the GeoJSON driver as there are differences between
        different spatial filetypes. Only GeoJSON is supported to avoid unexpected
        errors.
        """
        for topo_name, topo_type, gdf in zip(
            (branches_name, nodes_name),
            ("branches", "nodes"),
            (self.branch_gdf, self.node_gdf),
        ):

            topo_path = output_dir_path / (
                f"{self.name}_{topo_type}.geojson" if topo_name is None else topo_name
            )
            write_geodata(gdf=gdf, path=topo_path, allow_list_column_transform=False)
            logging.info(
                "Wrote topological data to disk.",
                extra=dict(
                    network_name=self.name,
                    topo_path=topo_path,
                    topo_type=topo_type,
                ),
            )


@dataclass
class CachedNetwork(Network):

    """
    A naive implementation of a cache for the topology of a ``Network``.
    """

    network_cache_path: Path = DEFAULT_NETWORK_CACHE_PATH
    determine_branches_nodes: bool = True
    _cache_hit: bool = False

    def __post_init__(self):
        """
        Overload ``__post_init__`` to handle caching.

        Handle caching by loading branch and node data from
        ``network_cache_path`` if they exist there.

        Uses ``sha256`` hexdigest to hash the network data.
        """
        branch_gdf_empty = self.branch_gdf.empty
        node_gdf_empty = self.node_gdf.empty
        if not branch_gdf_empty or not node_gdf_empty:
            error = "Do not pass branch and node GeoDataFrames to CachedNetwork."
            logging.error(
                error,
                extra=dict(
                    branch_gdf_empty=branch_gdf_empty,
                    node_gdf_empty=node_gdf_empty,
                    network_name=self.name,
                ),
            )
            raise ValueError(error)
        if not self.determine_branches_nodes:
            error = (
                "CachedNetwork has no utility if branches and nodes are not determined."
            )
            logging.error(
                error,
                extra=dict(
                    determine_branches_nodes=self.determine_branches_nodes,
                    network_name=self.name,
                ),
            )
            raise ValueError(error)

        try:
            # Combine jsons of trace_gdf and area_gdf + other relevant network
            # data
            network_data_as_string = (
                str(self.trace_gdf.to_json())
                + str(self.area_gdf.to_json())
                + str(self.circular_target_area)
                + str(self.snap_threshold)
            )

            # Encode the string to bytes
            encoded = network_data_as_string.encode()

            # Create sha256 hexdigest of the bytes
            sha256_hexdigest = sha256(encoded).hexdigest()
        except Exception:

            # Log the exception
            logging.error(
                "Failed to sha256 hash trace and area GeoDataFrames."
                " If this error persists using the regular ``Network``"
                " instance is recommended.",
                exc_info=True,
            )
            # If hashing cannot be done no caching can be done
            raise
            # Continue with regular Network initialization

            # return super().__post_init__()

        branch_path = self.network_cache_path / f"{sha256_hexdigest}_branches.geojson"
        node_path = self.network_cache_path / f"{sha256_hexdigest}_nodes.geojson"

        if branch_path.exists() and node_path.exists():
            self._cache_hit = True
            # Cache hit -> Load branch and node data
            self.branch_gdf = read_geofile(branch_path)
            self.node_gdf = read_geofile(node_path)
            logging.info(
                "Hit cache for branch and node data. Loading.",
                extra=dict(
                    network_name=self.name, branch_path=branch_path, node_path=node_path
                ),
            )

            # CRS should be the same. It is set here explicitly to confirm
            # that.
            if (
                self.branch_gdf.crs != self.trace_gdf.crs
                or self.node_gdf.crs != self.trace_gdf.crs
            ):
                logging.info(
                    "Cache loaded branches and nodes did not have same crs as traces.",
                    extra=dict(
                        branch_gdf_crs=self.branch_gdf.crs,
                        node_gdf_crs=self.node_gdf.crs,
                        trace_gdf_crs=self.trace_gdf.crs,
                        network_name=self.name,
                    ),
                )
                self.branch_gdf.crs = self.trace_gdf.crs
                self.node_gdf.crs = self.trace_gdf.crs
            assert self.branch_gdf.crs == self.trace_gdf.crs
            assert self.node_gdf.crs == self.trace_gdf.crs
        else:
            logging.info(
                "No cache hit for branch and node data. Determining.",
                extra=dict(network_name=self.name),
            )
            # No cache hit, determine branches and nodes
            self.assign_branches_nodes()

            # Create cache directory
            self.network_cache_path.mkdir(exist_ok=True)

            # Cache the determined branches and nodes
            write_geodata(
                gdf=self.branch_gdf, path=branch_path, allow_list_column_transform=False
            )
            write_geodata(
                gdf=self.node_gdf, path=node_path, allow_list_column_transform=False
            )
            logging.info(
                "Caching determined branches and nodes.",
                extra=dict(
                    network_name=self.name, branch_path=branch_path, node_path=node_path
                ),
            )
        # Continue with normal Network initialization
        super().__post_init__()
