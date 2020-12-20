from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Final, Any, Type, Union, Set
from datetime import datetime
from pathlib import Path
import time
import logging
from itertools import chain

import geopandas as gpd
from shapely.geometry import Point, LineString
from geopandas.sindex import PyGEOSSTRTreeIndex

from fractopo.tval import trace_validator
import fractopo.tval.trace_validation as trace_validation
from fractopo.tval.trace_validation import (
    ALL_VALIDATORS,
    ValidatorClass,
    MAJOR_ERRORS,
)

from fractopo.general import determine_general_nodes
from fractopo.tval.trace_validation_utils import determine_trace_candidates

logging.basicConfig(
    level=logging.WARNING, format="%(process)d-%(levelname)s-%(message)s"
)


@dataclass
class Validation:

    traces: gpd.GeoDataFrame
    area: gpd.GeoDataFrame
    name: str
    allow_fix: bool

    # Default thresholds
    SNAP_THRESHOLD: Final[float] = 0.01
    SNAP_THRESHOLD_ERROR_MULTIPLIER: Final[float] = 1.1
    AREA_EDGE_SNAP_MULTIPLIER: Final[float] = 5.0
    TRIANGLE_ERROR_SNAP_MULTIPLIER: Final[float] = 10.0
    OVERLAP_DETECTION_MULTIPLIER: Final[float] = 50.0
    SHARP_AVG_THRESHOLD: Final[float] = 80.0
    SHARP_PREV_SEG_THRESHOLD: Final[float] = 70.0
    ERROR_COLUMN = "VALIDATION_ERRORS"
    GEOMETRY_COLUMN = "geometry"

    # Private caching attributes
    _endpoint_nodes: Optional[List[Tuple[Point, ...]]] = None
    _intersect_nodes: Optional[List[Tuple[Point, ...]]] = None
    _spatial_index: Optional[PyGEOSSTRTreeIndex] = None
    _faulty_junctions: Optional[Set[int]] = None
    _vnodes: Optional[Set[int]] = None

    def set_general_nodes(self):
        self._intersect_nodes, self._endpoint_nodes = determine_general_nodes(
            self.traces, self.SNAP_THRESHOLD
        )

    @property
    def endpoint_nodes(self) -> List[Tuple[Point, ...]]:
        if self._endpoint_nodes is None:
            self.set_general_nodes()
        if not self._endpoint_nodes is None:
            return self._endpoint_nodes
        else:
            raise TypeError("Expected self._endpoint_nodes to not be None.")

    @property
    def intersect_nodes(self) -> List[Tuple[Point, ...]]:
        if self._intersect_nodes is None:
            self.set_general_nodes()
        if not self._intersect_nodes is None:
            return self._intersect_nodes
        else:
            raise TypeError("Expected self._intersect_nodes to not be None.")

    @property
    def spatial_index(self) -> PyGEOSSTRTreeIndex:
        if self._spatial_index is None:
            spatial_index = self.traces.sindex
            if not isinstance(spatial_index, PyGEOSSTRTreeIndex):
                raise TypeError(
                    "Expected sindex property to be of type: PyGEOSSTRTreeIndex"
                )
            self._spatial_index = spatial_index

        return self._spatial_index

    @property
    def faulty_junctions(self):
        if self._faulty_junctions is None:
            all_nodes = [
                tuple(chain(first, second))
                for first, second in zip(self.intersect_nodes, self.endpoint_nodes)
            ]
            self._faulty_junctions = trace_validation.MultiJunctionValidator.determine_faulty_junctions(
                all_nodes,
                snap_threshold=self.SNAP_THRESHOLD,
                snap_threshold_error_multiplier=self.SNAP_THRESHOLD_ERROR_MULTIPLIER,
            )
        return self._faulty_junctions

    @property
    def vnodes(self):
        if self._vnodes is None:
            self._vnodes = trace_validation.VNodeValidator.determine_v_nodes(
                endpoint_nodes=self.endpoint_nodes,
                snap_threshold=self.SNAP_THRESHOLD,
                snap_threshold_error_multiplier=self.SNAP_THRESHOLD_ERROR_MULTIPLIER,
            )
        return self._vnodes

    def run_validation(self):

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
            for validator in ALL_VALIDATORS:
                if ignore_geom:
                    # Break out of validation loop. See above comments
                    break
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
                    spatial_index=self.spatial_index,
                    idx=idx,
                    vnodes=self.vnodes,
                    faulty_junctions=self.faulty_junctions,
                    snap_threshold=self.SNAP_THRESHOLD,
                    snap_threshold_error_multiplier=self.SNAP_THRESHOLD_ERROR_MULTIPLIER,
                    overlap_detection_multiplier=self.OVERLAP_DETECTION_MULTIPLIER,
                    triangle_error_snap_multiplier=self.TRIANGLE_ERROR_SNAP_MULTIPLIER,
                    trace_candidates=trace_candidates,
                    sharp_avg_threshold=self.SHARP_AVG_THRESHOLD,
                    sharp_prev_seg_threshold=self.SHARP_PREV_SEG_THRESHOLD,
                    area=self.area,
                    area_edge_snap_multiplier=self.AREA_EDGE_SNAP_MULTIPLIER,
                )
            all_errors.append(current_errors)
            all_geoms.append(geom)

        assert len(all_errors) == len(all_geoms)
        validated_gdf = self.traces.copy()
        validated_gdf[self.ERROR_COLUMN] = all_errors
        validated_gdf[self.GEOMETRY_COLUMN] = all_geoms

        return validated_gdf

    @staticmethod
    def _validate(
        geom: Any,
        validator: ValidatorClass,
        current_errors: List[str],
        allow_fix: bool,
        **kwargs,
    ) -> Tuple[Any, List[str], bool]:
        ignore_geom = False
        fixed = None
        if (
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
                    geom = fixed
            if validator.ERROR in MAJOR_ERRORS and fixed is None:
                # If error was not fixed and its part of major errors ->
                # the geom must be ignore by other validators.
                ignore_geom = True

        return geom, current_errors, ignore_geom
