"""
Contains main entrypoint class for validating trace data, Validation.

Create Validation objects from traces and their target areas to validate
the traces for further analysis (branch and node determination).
"""
import logging
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Set, Tuple, Type, Union

import geopandas as gpd
from fractopo.general import determine_general_nodes
from fractopo.tval import trace_validators
from fractopo.tval.trace_validation_utils import determine_trace_candidates
from fractopo.tval.trace_validators import (
    ALL_VALIDATORS,
    MAJOR_ERRORS,
    MAJOR_VALIDATORS,
    ValidatorClass,
)
from geopandas.sindex import PyGEOSSTRTreeIndex
from shapely.geometry import LineString, MultiLineString, Point


logging.basicConfig(
    level=logging.WARNING, format="%(process)d-%(levelname)s-%(message)s"
)


@dataclass
class Validation:
    """
    Validate traces data delineated by target area(s).

    If allow_fix is True, some automatic fixing will be done to e.g. convert
    MultiLineStrings to LineStrings.
    """

    traces: gpd.GeoDataFrame
    area: gpd.GeoDataFrame
    name: str
    allow_fix: bool

    # Default thresholds
    SNAP_THRESHOLD: float = 0.01
    SNAP_THRESHOLD_ERROR_MULTIPLIER: float = 1.1
    AREA_EDGE_SNAP_MULTIPLIER: float = 2.5
    TRIANGLE_ERROR_SNAP_MULTIPLIER: float = 10.0
    OVERLAP_DETECTION_MULTIPLIER: float = 50.0
    # TODO: Separate SHARP TURNS to major and minor
    SHARP_AVG_THRESHOLD: float = 135.0
    SHARP_PREV_SEG_THRESHOLD: float = 100.0
    ERROR_COLUMN: str = "VALIDATION_ERRORS"
    GEOMETRY_COLUMN: Final[str] = "geometry"

    def __post_init__(self):
        # Private caching attributes
        self._endpoint_nodes: Optional[List[Tuple[Point, ...]]] = None
        self._intersect_nodes: Optional[List[Tuple[Point, ...]]] = None
        self._spatial_index: Optional[PyGEOSSTRTreeIndex] = None
        self._faulty_junctions: Optional[Set[int]] = None
        self._vnodes: Optional[Set[int]] = None

        # Validate trace and area inputs
        # for geom in self.area.geometry.values:
        #     if not isinstance(geom, Polygon)

    def set_general_nodes(self):
        """
        Set _intersect_nodes and _endpoint_nodes attributes.
        """
        self._intersect_nodes, self._endpoint_nodes = determine_general_nodes(
            self.traces, self.SNAP_THRESHOLD
        )

    @property
    def endpoint_nodes(self) -> List[Tuple[Point, ...]]:
        """
        Get endpoints of all traces.

        Returned as a list of tuples wherein each tuple represents the nodes
        of a trace in traces i.e. endpoint_nodes[index] are the nodes for
        traces[index].
        """
        if self._endpoint_nodes is None:
            self.set_general_nodes()
        if not self._endpoint_nodes is None:
            return self._endpoint_nodes
        else:
            raise TypeError("Expected self._endpoint_nodes to not be None.")

    @property
    def intersect_nodes(self) -> List[Tuple[Point, ...]]:
        """
        Get intersection nodes of all traces.

        Returned as a list of tuples wherein each tuple represents the nodes
        of a trace in traces i.e. intersect_nodes[index] are the nodes for
        traces[index].
        """
        if self._intersect_nodes is None:
            self.set_general_nodes()
        if not self._intersect_nodes is None:
            return self._intersect_nodes
        else:
            raise TypeError("Expected self._intersect_nodes to not be None.")

    @property
    def spatial_index(self) -> Optional[PyGEOSSTRTreeIndex]:
        """
        Get geopandas/pygeos spatial_index of traces.
        """
        if self._spatial_index is None:
            spatial_index = self.traces.sindex
            if (
                not isinstance(spatial_index, PyGEOSSTRTreeIndex)
                or len(spatial_index) == 0
            ):
                logging.warning(
                    "Expected sindex property to be of type: PyGEOSSTRTreeIndex \n"
                    "and non-empty."
                )
                self._spatial_index = None
                return self._spatial_index
            self._spatial_index = spatial_index

        return self._spatial_index

    @property
    def faulty_junctions(self) -> Set[int]:
        """
        Determine indexes with Multi Junctions.
        """
        if self._faulty_junctions is None:
            all_nodes = [
                tuple(chain(first, second))
                for first, second in zip(self.intersect_nodes, self.endpoint_nodes)
            ]
            self._faulty_junctions = trace_validators.MultiJunctionValidator.determine_faulty_junctions(
                all_nodes,
                snap_threshold=self.SNAP_THRESHOLD,
                snap_threshold_error_multiplier=self.SNAP_THRESHOLD_ERROR_MULTIPLIER,
            )
        return self._faulty_junctions

    @property
    def vnodes(self) -> Set[int]:
        """
        Determine indexes with V-Nodes.
        """
        if self._vnodes is None:
            self._vnodes = trace_validators.VNodeValidator.determine_v_nodes(
                endpoint_nodes=self.endpoint_nodes,
                snap_threshold=self.SNAP_THRESHOLD,
                snap_threshold_error_multiplier=self.SNAP_THRESHOLD_ERROR_MULTIPLIER,
            )
        return self._vnodes

    def run_validation(self, first_pass=True) -> gpd.GeoDataFrame:
        """
        Main entrypoint for validation.

        Returns validated traces GeoDataFrame.
        """

        # Validations that if are invalid will break all other validation:
        # - GeomNullValidator (also checks for empty)
        # - GeomTypeValidator
        # If these pass the geometry is LineString

        # Non-invasive errors:
        # - SimpleGeometryValidator
        # - MultiJunctionValidator
        # - VNodeValidator
        # - MultipleCrosscutValidator
        # - UnderlappingSnapValidator
        # - TargetAreaSnapValidator
        # - StackedTracesValidator
        # - SharpCornerValidator
        all_errors: List[List[str]] = []
        all_geoms: List[LineString] = []
        for idx, geom in enumerate(self.traces.geometry.values):
            # Collect errors from each validator for each geom
            current_errors: List[str] = []
            # If geom contains validation error that will cause issues in later
            # validation -> ignore the geom and break out of validation loop
            # for current geom.
            ignore_geom: bool = False
            trace_candidates: Optional[gpd.GeoSeries] = None
            # validation loop
            validators = MAJOR_VALIDATORS if first_pass else ALL_VALIDATORS
            for validator in validators:
                if ignore_geom:
                    # Break out of validation loop. See above comments
                    break
                delicate_kwargs = dict()
                if isinstance(geom, LineString) and not geom.is_empty:
                    # Some conditionals to avoid try-except loop
                    # trace candidates that are nearby to geom based on spatial index
                    trace_candidates = (
                        determine_trace_candidates(
                            geom, idx, self.traces, spatial_index=self.spatial_index
                        )
                        if trace_candidates is None
                        else trace_candidates
                    )

                # Overwrites geom if fix was executed
                # current_errors either contains new error or is unchanged
                # ignore_geom overwritten with True when geom must be ignored.
                geom, current_errors, ignore_geom = self._validate(
                    geom=geom,
                    validator=validator,
                    current_errors=current_errors,
                    allow_fix=self.allow_fix,
                    idx=idx,
                    snap_threshold=self.SNAP_THRESHOLD,
                    snap_threshold_error_multiplier=self.SNAP_THRESHOLD_ERROR_MULTIPLIER,
                    overlap_detection_multiplier=self.OVERLAP_DETECTION_MULTIPLIER,
                    triangle_error_snap_multiplier=self.TRIANGLE_ERROR_SNAP_MULTIPLIER,
                    trace_candidates=trace_candidates,
                    sharp_avg_threshold=self.SHARP_AVG_THRESHOLD,
                    sharp_prev_seg_threshold=self.SHARP_PREV_SEG_THRESHOLD,
                    area=self.area,
                    area_edge_snap_multiplier=self.AREA_EDGE_SNAP_MULTIPLIER,
                    spatial_index=self.spatial_index,
                    vnodes=self.vnodes,
                    faulty_junctions=self.faulty_junctions,
                )
            all_errors.append(current_errors)
            all_geoms.append(geom)

        assert len(all_errors) == len(all_geoms)
        validated_gdf = self.traces.copy()
        validated_gdf[self.ERROR_COLUMN] = all_errors
        validated_gdf[self.GEOMETRY_COLUMN] = all_geoms
        if first_pass:
            self.traces = validated_gdf
            validated_gdf = self.run_validation(first_pass=False)

        return validated_gdf

    @staticmethod
    def _validate(
        geom: Any,
        validator: ValidatorClass,
        current_errors: List[str],
        allow_fix: bool,
        **kwargs,
    ) -> Tuple[Any, List[str], bool]:
        """
        Validate geom with validator.

        Returns possibly fixed geom (if allow_fix is True and validator handles
        fixing), updated current_errors list for geom and whether to ignore
        the geom in later validations (e.g. when MultiLineString could not
        be merged by GeomTypeValidator fix.).

        Some validators require many additional kwargs.

        >>> geom = MultiLineString([((0, 0), (1, 1)), ((1, 1), (2, 2))])
        >>> validator = trace_validators.GeomTypeValidator
        >>> current_errors = []
        >>> allow_fix = True
        >>> fixed_geom, updated_errors, ignore_geom = Validation._validate(
        ...     geom=geom,
        ...     validator=validator,
        ...     current_errors=current_errors,
        ...     allow_fix=allow_fix
        ... )
        >>> fixed_geom.wkt, updated_errors, ignore_geom
        ('LINESTRING (0 0, 1 1, 2 2)', [], False)

        """
        ignore_geom = False
        fixed = None
        if validator.LINESTRING_ONLY and not isinstance(geom, LineString):
            # Do not pass invalid geometry types to most validators. There's
            # already a error string in current_errors for e.g. MultiLineString
            # or empty geom rows.
            return geom, current_errors, True
        elif (
            not validator.validation_method(geom, **kwargs)
            and validator.ERROR not in current_errors
        ):
            # geom is invalid
            current_errors.append(validator.ERROR)
            if allow_fix:
                # Try to fix it
                try:
                    # fixed is None if fix is not succesful but no error
                    # is raised
                    fixed = validator.fix_method(geom, **kwargs)
                except NotImplementedError:
                    # No fix implemented for validator
                    fixed = None
                if fixed is not None:
                    # Fix succesful
                    current_errors.remove(validator.ERROR)
                    # geom is passed to later validators
                    geom = fixed
            if validator.ERROR in MAJOR_ERRORS and fixed is None:
                # If error was not fixed and its part of major errors ->
                # the geom must be ignore by other validators.
                ignore_geom = True

        return geom, current_errors, ignore_geom

