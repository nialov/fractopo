"""
Contains Validator classes which each have their own error to handle and mark.
"""

import logging

import geopandas as gpd
import numpy as np
from beartype import beartype
from beartype.typing import Any, List, Optional, Set, Tuple, Type, Union
from shapely.affinity import scale
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, split

from fractopo.general import (
    Number,
    compare_unit_vector_orientation,
    create_unit_vector,
    determine_node_junctions,
    get_trace_coord_points,
    get_trace_endpoints,
)
from fractopo.tval.trace_validation_utils import (
    is_underlapping,
    segment_within_buffer,
    split_to_determine_triangle_errors,
)

log = logging.getLogger(__name__)


class BaseValidator:
    """
    Base validator that all classes inherit.

    TODO: Validation requires overhaul at some point.
    """

    ERROR = "BASE ERROR"
    INTERACTION_NODES_COLUMN = "IP"
    LINESTRING_ONLY = True

    @staticmethod
    def fix_method(**_) -> Optional[LineString]:
        """
        Abstract fix method.
        """
        raise NotImplementedError("fix_method not implemented.")

    @staticmethod
    def validation_method(**_) -> bool:
        """
        Abstract validation method.
        """
        raise NotImplementedError("validation_method not implemented.")


class GeomTypeValidator(BaseValidator):
    """
    Validates the geometry type.

    Validates that all traces are LineStrings. Tries to use shapely.ops.linemerge
    to merge MultiLineStrings into LineStrings.
    """

    ERROR = "GEOM TYPE MULTILINESTRING"
    LINESTRING_ONLY = False

    @staticmethod
    @beartype
    def fix_method(geom: Any, **_) -> Optional[LineString]:
        """
        Fix mergeable MultiLineStrings to LineStrings.

        E.g. mergeable MultiLineString

        >>> mls = MultiLineString([((0, 0), (1, 1)), ((1, 1), (2, 2))])
        >>> GeomTypeValidator.fix_method(geom=mls).wkt
        'LINESTRING (0 0, 1 1, 2 2)'

        Unhandled types will just return as None.

        >>> GeomTypeValidator.fix_method(geom=Point(1, 1)) is None
        True

        """
        if isinstance(geom, LineString):
            return geom
        if not isinstance(geom, MultiLineString):
            return None
        fixed_geom = linemerge(geom)
        return fixed_geom if isinstance(fixed_geom, LineString) else None

    @staticmethod
    @beartype
    def validation_method(geom: Any, **_) -> bool:
        """
        Validate geometries.

        E.g. Anything but LineString

        >>> GeomTypeValidator.validation_method(Point(1, 1))
        False

        With LineString:

        >>> GeomTypeValidator.validation_method(LineString([(0, 0), (1, 1)]))
        True

        """
        return isinstance(geom, LineString)


class MultiJunctionValidator(BaseValidator):
    """
    Validates that junctions consists of a maximum of two lines crossing.
    """

    ERROR = "MULTI JUNCTION"

    @staticmethod
    @beartype
    def validation_method(idx: int, faulty_junctions: Set[int], **_) -> bool:
        """
        Validate for multi junctions between traces.

        >>> MultiJunctionValidator.validation_method(
        ...     idx=1, faulty_junctions=set([1, 2])
        ... )
        False

        """
        return idx not in faulty_junctions

    @staticmethod
    @beartype
    def determine_faulty_junctions(
        all_nodes: List[Tuple[Point, ...]],
        snap_threshold: Number,
        snap_threshold_error_multiplier: Number,
    ) -> Set[int]:
        """
        Determine when a point of interest represents a multi junction.

        I.e. when there are more than 2 points within the buffer distance of
        each other.

        Two points is the limit because an intersection point between two traces
        is added to the list of interest points twice, once for each trace.

        Faulty junction can also represent a slightly overlapping trace i.e.
        a snapping error.

        >>> all_nodes = [
        ...     (Point(0, 0), Point(1, 1)),
        ...     (Point(1, 1),),
        ...     (Point(5, 5),),
        ...     (Point(0, 0), Point(1, 1)),
        ... ]
        >>> snap_threshold = 0.01
        >>> snap_threshold_error_multiplier = 1.1
        >>> MultiJunctionValidator.determine_faulty_junctions(
        ...     all_nodes, snap_threshold, snap_threshold_error_multiplier
        ... )
        {0, 1, 3}

        """
        return determine_node_junctions(
            nodes=all_nodes,
            snap_threshold=snap_threshold,
            snap_threshold_error_multiplier=snap_threshold_error_multiplier,
            error_threshold=2,
        )


class VNodeValidator(BaseValidator):
    """
    Finds V-nodes within trace data.
    """

    ERROR = "V NODE"

    @staticmethod
    @beartype
    def validation_method(idx: int, vnodes: Set[int], **_) -> bool:
        """
        Validate for V-nodes.

        >>> VNodeValidator.validation_method(idx=1, vnodes=set([1, 2]))
        False

        >>> VNodeValidator.validation_method(idx=5, vnodes=set([1, 2]))
        True

        """
        return idx not in vnodes

    @staticmethod
    @beartype
    def determine_v_nodes(
        endpoint_nodes: List[Tuple[Point, ...]],
        snap_threshold: Number,
        snap_threshold_error_multiplier: Number,
    ) -> Set[int]:
        """
        Determine V-node errors.

        >>> endpoint_nodes = [
        ...     (Point(0, 0), Point(1, 1)),
        ...     (Point(1, 1),),
        ... ]
        >>> snap_threshold = 0.01
        >>> snap_threshold_error_multiplier = 1.1
        >>> VNodeValidator.determine_v_nodes(
        ...     endpoint_nodes, snap_threshold, snap_threshold_error_multiplier
        ... )
        {0, 1}

        """
        return determine_node_junctions(
            nodes=endpoint_nodes,
            snap_threshold=snap_threshold,
            snap_threshold_error_multiplier=snap_threshold_error_multiplier,
            error_threshold=1,
        )


class MultipleCrosscutValidator(BaseValidator):
    """
    Find traces that cross-cut each other multiple times.

    This also indicates the possibility of duplicate traces.
    """

    ERROR = "MULTIPLE CROSSCUTS"

    @staticmethod
    @beartype
    def validation_method(
        geom: LineString,
        trace_candidates: gpd.GeoSeries,
        **_,
    ) -> bool:
        """
        Validate for multiple crosscuts.

        >>> geom = LineString([Point(-3, -4), Point(-3, -1)])
        >>> trace_candidates = gpd.GeoSeries(
        ...     [
        ...         LineString(
        ...             [Point(-4, -3), Point(-2, -3), Point(-4, -2), Point(-2, -1)]
        ...         )
        ...     ]
        ... )
        >>> MultipleCrosscutValidator.validation_method(geom, trace_candidates)
        False

        >>> geom = LineString([Point(-3, -4), Point(-3, -4.5)])
        >>> trace_candidates = gpd.GeoSeries(
        ...     [
        ...         LineString(
        ...             [Point(-4, -3), Point(-2, -3), Point(-4, -2), Point(-2, -1)]
        ...         )
        ...     ]
        ... )
        >>> MultipleCrosscutValidator.validation_method(geom, trace_candidates)
        True

        """
        if not any(trace_candidates.intersects(geom)):
            return True
        intersection_geoms = trace_candidates.intersection(geom)
        return not any(
            len(list(geom.geoms)) > 2
            for geom in intersection_geoms
            if isinstance(geom, MultiPoint)
        )


class UnderlappingSnapValidator(BaseValidator):
    """
    Find snapping errors of underlapping traces.

    Uses a multiple of the given snap_threshold.
    """

    ERROR = "UNDERLAPPING SNAP"
    _OVERLAPPING = "OVERLAPPING SNAP"
    _UNDERLAPPING = "UNDERLAPPING SNAP"

    @classmethod
    @beartype
    def validation_method(
        cls,
        geom: LineString,
        trace_candidates: gpd.GeoSeries,
        snap_threshold: Number,
        snap_threshold_error_multiplier: Number,
        **_,
    ) -> bool:
        """
        Validate for UnderlappingSnapValidator errors.

        >>> snap_threshold = 0.01
        >>> snap_threshold_error_multiplier = 1.1
        >>> geom = LineString(
        ...     [
        ...         (0, 0),
        ...         (0, 1 + snap_threshold * snap_threshold_error_multiplier * 0.99),
        ...     ]
        ... )
        >>> trace_candidates = gpd.GeoSeries([LineString([(-1, 1), (1, 1)])])
        >>> UnderlappingSnapValidator.validation_method(
        ...     geom, trace_candidates, snap_threshold, snap_threshold_error_multiplier
        ... )
        False

        >>> snap_threshold = 0.01
        >>> snap_threshold_error_multiplier = 1.1
        >>> geom = LineString([(0, 0), (0, 1)])
        >>> trace_candidates = gpd.GeoSeries([LineString([(-1, 1), (1, 1)])])
        >>> UnderlappingSnapValidator.validation_method(
        ...     geom, trace_candidates, snap_threshold, snap_threshold_error_multiplier
        ... )
        True

        """
        if len(trace_candidates) == 0:
            return True

        endpoints = get_trace_endpoints(geom)
        for endpoint in endpoints:
            # Check that if endpoint is well snapped to one trace another trace
            # won't have chance to cause a validation error (false positive).
            if any(trace_candidates.distance(endpoint) < snap_threshold):
                continue
            trace: LineString
            for trace in trace_candidates.geometry.values:
                if (
                    snap_threshold
                    < trace.distance(endpoint)
                    < snap_threshold * snap_threshold_error_multiplier
                ):
                    # Determine if its underlappings or overlapping
                    is_ul_result = is_underlapping(
                        geom,
                        trace,
                        endpoint,
                        snap_threshold,
                        snap_threshold_error_multiplier,
                    )
                    if is_ul_result is None:
                        if geom.overlaps(trace):
                            cls.ERROR = StackedTracesValidator.ERROR
                        else:
                            raise ValueError(
                                "Expected None is_ul_result to indicate overlapping traces."
                            )
                    elif is_ul_result:
                        # Underlapping
                        cls.ERROR = cls._UNDERLAPPING
                    else:
                        cls.ERROR = cls._OVERLAPPING
                    return False

        return True


class TargetAreaSnapValidator(BaseValidator):
    """
    Validator for traces that underlap the target area.
    """

    ERROR = "TRACE UNDERLAPS TARGET AREA"

    @staticmethod
    @beartype
    def validation_method(
        geom: LineString,
        area: Union[gpd.GeoDataFrame, gpd.GeoSeries],
        snap_threshold: Number,
        snap_threshold_error_multiplier: Number,
        area_edge_snap_multiplier: Number,
        **_,
    ) -> bool:
        """
        Validate for trace underlaps.

        >>> geom = LineString([(0, 0), (0, 1)])
        >>> area = gpd.GeoDataFrame(
        ...     geometry=[Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])]
        ... )
        >>> snap_threshold = 0.01
        >>> snap_threshold_error_multiplier = 1.1
        >>> area_edge_snap_multiplier = 1.0
        >>> TargetAreaSnapValidator.validation_method(
        ...     geom,
        ...     area,
        ...     snap_threshold,
        ...     snap_threshold_error_multiplier,
        ...     area_edge_snap_multiplier,
        ... )
        True

        >>> geom = LineString([(0.5, 0.5), (0.5, 0.98)])
        >>> area = gpd.GeoDataFrame(
        ...     geometry=[Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])]
        ... )
        >>> snap_threshold = 0.01
        >>> snap_threshold_error_multiplier = 1.1
        >>> area_edge_snap_multiplier = 10
        >>> TargetAreaSnapValidator.validation_method(
        ...     geom,
        ...     area,
        ...     snap_threshold,
        ...     snap_threshold_error_multiplier,
        ...     area_edge_snap_multiplier,
        ... )
        False

        """
        endpoints = get_trace_endpoints(geom)
        for endpoint in endpoints:
            for area_polygon in area.geometry.values:
                if TargetAreaSnapValidator.is_candidate_underlapping(
                    endpoint,
                    geom,
                    area_polygon,
                    snap_threshold=snap_threshold,
                ):
                    # if endpoint.within(area_polygon) and geom.within(area_polygon):
                    # Point and trace completely within the area, does not
                    # intersect its edge.
                    if (
                        snap_threshold
                        <= endpoint.distance(area_polygon.boundary)
                        < snap_threshold
                        * snap_threshold_error_multiplier
                        * area_edge_snap_multiplier
                    ):
                        return False
        return True

    @staticmethod
    @beartype
    def simple_underlapping_checks(
        endpoint: Point,
        geom: LineString,
        area_polygon: Union[Polygon, MultiPolygon],
        snap_threshold: Number,
    ) -> Optional[bool]:
        """
        Perform simple underlapping checks.
        """
        endpoint_within = endpoint.within(area_polygon)
        # If not even inside target area, not a candidate
        if not endpoint_within:
            return False

        # Easy case: endpoint within target area and trace completely inside
        # target area
        if endpoint_within and geom.within(area_polygon):
            return True

        # Does an affine scale to catch traces that intersect the target area
        # edge at their other endpoint. This check overlaps with previous but
        # shapely.affinity.scale is not completely 'stable'
        if endpoint_within and geom.within(
            scale(area_polygon, xfact=1 + snap_threshold, yfact=1 + snap_threshold)
        ):
            return True
        return None

    @staticmethod
    @beartype
    def is_candidate_underlapping(
        endpoint: Point,
        geom: LineString,
        area_polygon: Union[Polygon, MultiPolygon],
        snap_threshold: Number,
    ) -> bool:
        """
        Determine if endpoint is candidate for Underlapping error.
        """
        is_simple_underlapping = TargetAreaSnapValidator.simple_underlapping_checks(
            endpoint=endpoint,
            geom=geom,
            area_polygon=area_polygon,
            snap_threshold=snap_threshold,
        )

        if is_simple_underlapping is not None:
            return is_simple_underlapping

        # Split trace with area polygon
        geom_split = list(split(geom, area_polygon).geoms)

        # If not splittable, not a candidate
        if len(geom_split) == 1:
            return False

        # Filter segments that do not intersect endpoint
        endpoint_segments = [
            segment
            for segment in geom_split
            if segment.distance(endpoint) < snap_threshold
        ]

        # More than 1 shouldn't intersect one endpoint
        if len(endpoint_segments) != 1:
            log.warning("Expected only one segment to be close to endpoint.")
            return False
        endpoint_segment = endpoint_segments[0]

        # Make sure result of split is LineString
        if not isinstance(endpoint_segment, LineString):
            log.warning("Expected endpoint_segment to be of type LineString.")
            return False

        # If the LineString intersects polygon boundary -> not a candidate
        if endpoint_segment.distance(area_polygon.boundary) < snap_threshold:
            return False
        return True


class GeomNullValidator(BaseValidator):
    """
    Validate the geometry for NULL GEOMETRY errors.
    """

    ERROR = "NULL GEOMETRY"
    LINESTRING_ONLY = False

    @staticmethod
    def validation_method(geom: Any, **_) -> bool:
        """
        Validate for empty and null geometries.

        E.g. some validations are handled by GeomTypeValidator

        >>> GeomNullValidator.validation_method(Point(1, 1))
        True

        Empty geometries are not valid.

        >>> GeomNullValidator.validation_method(LineString())
        False

        >>> GeomNullValidator.validation_method(LineString([(-1, 1), (1, 1)]))
        True

        """
        if not isinstance(geom, BaseGeometry):
            return False
        if geom is None:
            return False
        if geom.is_empty:
            return False
        return True


class StackedTracesValidator(BaseValidator):
    """
    Find stacked traces and small triangle intersections.
    """

    ERROR = "STACKED TRACES"

    @staticmethod
    @beartype
    def validation_method(
        geom: LineString,
        trace_candidates: gpd.GeoSeries,
        snap_threshold: Number,
        snap_threshold_error_multiplier: Number,
        overlap_detection_multiplier: Number,
        triangle_error_snap_multiplier: Number,
        stacked_detector_buffer_multiplier: Number,
        **_,
    ) -> bool:
        """
        Validate for stacked traces and small triangle intersections.

        >>> geom = LineString([(0, 0), (0, 1)])
        >>> trace_candidates = gpd.GeoSeries([LineString([(0, -1), (0, 2)])])
        >>> snap_threshold = 0.01
        >>> snap_threshold_error_multiplier = 1.1
        >>> overlap_detection_multiplier = 50
        >>> triangle_error_snap_multiplier = 10
        >>> stacked_detector_buffer_multiplier = 5
        >>> StackedTracesValidator.validation_method(
        ...     geom,
        ...     trace_candidates,
        ...     snap_threshold,
        ...     snap_threshold_error_multiplier,
        ...     overlap_detection_multiplier,
        ...     triangle_error_snap_multiplier,
        ...     stacked_detector_buffer_multiplier,
        ... )
        False

        >>> geom = LineString([(10, 0), (10, 1)])
        >>> trace_candidates = gpd.GeoSeries([LineString([(0, -1), (0, 2)])])
        >>> snap_threshold = 0.01
        >>> snap_threshold_error_multiplier = 1.1
        >>> overlap_detection_multiplier = 50
        >>> triangle_error_snap_multiplier = 10
        >>> stacked_detector_buffer_multiplier = 10
        >>> StackedTracesValidator.validation_method(
        ...     geom,
        ...     trace_candidates,
        ...     snap_threshold,
        ...     snap_threshold_error_multiplier,
        ...     overlap_detection_multiplier,
        ...     triangle_error_snap_multiplier,
        ...     stacked_detector_buffer_multiplier,
        ... )
        True

        """
        # Cannot overlap if nothing to overlap
        if len(trace_candidates) == 0:
            return True

        trace_candidates_multils = MultiLineString(
            [
                tc
                for tc in trace_candidates.geometry.values
                if isinstance(tc, LineString)
                and tc.buffer(
                    snap_threshold
                    * overlap_detection_multiplier
                    * snap_threshold_error_multiplier
                ).intersects(geom)
            ]
        )
        # Test for overlapping traces.
        if segment_within_buffer(
            geom,
            trace_candidates_multils,
            snap_threshold=snap_threshold,
            snap_threshold_error_multiplier=snap_threshold_error_multiplier,
            overlap_detection_multiplier=overlap_detection_multiplier,
            stacked_detector_buffer_multiplier=stacked_detector_buffer_multiplier,
        ):
            return False

        # Test for small triangles created by the intersection of two traces.
        splitter_trace: LineString
        for splitter_trace in trace_candidates.geometry.values:
            if split_to_determine_triangle_errors(
                geom,
                splitter_trace,
                snap_threshold=snap_threshold,
                triangle_error_snap_multiplier=triangle_error_snap_multiplier,
            ):
                return False

        return True


class SimpleGeometryValidator(BaseValidator):
    """
    Use shapely is_simple and is_ring attributes to validate LineString.

    Checks that trace does not cut itself.
    """

    ERROR = "CUTS ITSELF"

    @staticmethod
    @beartype
    def validation_method(geom: LineString, **_) -> bool:
        """
        Validate for self-intersections.
        """
        return geom.is_simple and not geom.is_ring


class SharpCornerValidator(BaseValidator):
    """
    Find sharp cornered traces.
    """

    ERROR = "SHARP TURNS"

    @staticmethod
    @beartype
    def validation_method(
        geom: LineString,
        sharp_avg_threshold: Number,
        sharp_prev_seg_threshold: Number,
        **_,
    ) -> bool:
        """
        Validate for sharp cornered traces.
        """
        geom_coords = get_trace_coord_points(geom)
        if len(geom_coords) == 2:
            # If LineString consists of two Points -> No sharp corners.
            return True
        trace_unit_vector = create_unit_vector(geom_coords[0], geom_coords[-1])
        if any(np.isnan(trace_unit_vector)):
            return False
        for idx, segment_start in enumerate(geom_coords):
            if idx == len(geom_coords) - 1:
                break
            segment_end: Point = geom_coords[idx + 1]
            segment_unit_vector = create_unit_vector(segment_start, segment_end)
            # Compare the two unit vectors
            if any(
                np.isnan(segment_unit_vector)
            ) or not compare_unit_vector_orientation(
                trace_unit_vector, segment_unit_vector, sharp_avg_threshold
            ):
                return False
            # Check how much of a change compared to previous segment.
            if idx != 0:
                # Cannot get previous if node is first
                prev_segment_unit_vector = create_unit_vector(
                    geom_coords[idx - 1], geom_coords[idx]
                )
                if any(
                    np.isnan(prev_segment_unit_vector)
                ) or not compare_unit_vector_orientation(
                    segment_unit_vector,
                    prev_segment_unit_vector,
                    sharp_prev_seg_threshold,
                ):
                    return False

        return True


class EmptyTargetAreaValidator(BaseValidator):
    """
    Stub validator for empty target area.

    Validation for this error occurs at the start of run_validation.
    """

    ERROR = "EMPTY TARGET AREA"


MINOR_VALIDATORS = (
    SimpleGeometryValidator,
    MultiJunctionValidator,
    VNodeValidator,
    MultipleCrosscutValidator,
    UnderlappingSnapValidator,
    TargetAreaSnapValidator,
    StackedTracesValidator,
    SharpCornerValidator,
)

MAJOR_VALIDATORS = (GeomNullValidator, GeomTypeValidator)

ALL_VALIDATORS = MAJOR_VALIDATORS + MINOR_VALIDATORS

MAJOR_ERRORS = (GeomTypeValidator.ERROR, GeomNullValidator.ERROR)
ValidatorClass = Type[BaseValidator]

VALIDATION_REQUIRES_NODES = (MultiJunctionValidator, VNodeValidator)
