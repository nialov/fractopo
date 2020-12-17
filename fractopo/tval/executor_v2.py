from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Final, Any, Type, Union, Set
from datetime import datetime
from pathlib import Path
import time
import logging

import geopandas as gpd
from shapely.geometry import Point, LineString
from geopandas.sindex import PyGEOSSTRTreeIndex

from fractopo.tval import trace_validator
import fractopo.tval.trace_validation as trace_validation
from fractopo.tval.trace_validation import ALL_VALIDATORS, ValidatorClass, MAJOR_ERRORS

from fractopo.general import determine_general_nodes

logging.basicConfig(
    level=logging.WARNING, format="%(process)d-%(levelname)s-%(message)s"
)


@dataclass
class Validation:

    traces: gpd.GeoDataFrame
    area: gpd.GeoDataFrame
    name: str
    auto_fix: bool

    # Default thresholds
    SNAP_THRESHOLD: Final[float] = 0.01
    SNAP_THRESHOLD_ERROR_MULTIPLIER: Final[float] = 1.1
    AREA_EDGE_SNAP_MULTIPLIER: Final[float] = 1.0
    TRIANGLE_ERROR_SNAP_MULTIPLIER: Final[float] = 10.0
    OVERLAP_DETECTION_MULTIPLIER: Final[float] = 50.0
    SHARP_AVG_THRESHOLD: Final[float] = 80.0
    SHARP_PREV_SEG_THRESHOLD: Final[float] = 70.0

    # Private caching attributes
    _endpoint_nodes: Optional[List[Tuple[Point, ...]]] = None
    _intersect_nodes: Optional[List[Tuple[Point, ...]]] = None
    _spatial_index: Optional[PyGEOSSTRTreeIndex] = None
    _faulty_junctions: Optional[Set[int]] = None

    @property
    def endpoint_nodes(self):
        if self._endpoint_nodes is None:
            self._endpoint_nodes, self._intersect_nodes = determine_general_nodes(
                self.traces, self.SNAP_THRESHOLD
            )
        return self._endpoint_nodes

    @property
    def intersect_nodes(self):
        if self._intersect_nodes is None:
            self._endpoint_nodes, self._intersect_nodes = determine_general_nodes(
                self.traces, self.SNAP_THRESHOLD
            )
        return self._intersect_nodes

    @property
    def spatial_index(self):
        if self._spatial_index is None:
            self._spatial_index = self.traces.sindex
        return self._spatial_index

    @property
    def faulty_junctions(self):
        if self._faulty_junctions is None:
            self._faulty_junctions = trace_validation.MultiJunctionValidator.determine_faulty_junctions(
                self.intersect_nodes,
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

    def run_validation(self, allow_fix: bool):

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
        fixes: List[LineString] = []
        for idx, geom in enumerate(self.traces.geometry.values):
            # Collect errors from each validator for each geom
            current_errors: List[str] = []
            # If geom contains validation error that will cause issues in later
            # validation -> ignore the geom and break out of validation loop
            # for current geom.
            ignore_geom: bool = False
            # validation loop
            for validator in ALL_VALIDATORS:
                if ignore_geom:
                    # Break out of validation loop. See above comments
                    break
                # Overwrites geom if fix was executed
                # current_errors either contains new error or is unchanged
                # ignore_geom overwritten with True when geom must be ignored.
                geom, current_errors, ignore_geom = self._validate(
                    geom,
                    validator,
                    current_errors,
                    allow_fix,
                    spatial_index=self.spatial_index,
                    idx=idx,
                    vnodes=self.vnodes,
                    faulty_junctions=self.faulty_junctions,
                )

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
