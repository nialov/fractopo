import ast
from typing import Union, Tuple, List, Optional, Any, Set, Type
from abc import abstractmethod
import logging
from itertools import chain, accumulate, zip_longest
from bisect import bisect

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry
from shapely.geometry import (
    MultiPoint,
    Point,
    LineString,
    MultiLineString,
    Polygon,
)
from geopandas.sindex import PyGEOSSTRTreeIndex
from shapely.ops import split, linemerge
import numpy as np


from fractopo.general import (
    get_trace_coord_points,
    point_to_xy,
    get_trace_endpoints,
    flatten_tuples,
    determine_node_junctions,
    zip_equal,
    compare_unit_vector_orientation,
    create_unit_vector,
)
from fractopo.tval.trace_validation_utils import (
    segment_within_buffer,
    split_to_determine_triangle_errors,
)


class BaseValidator:
    """
    Base validator that all classes inherit.
    """

    ERROR = "BASE ERROR"
    INTERACTION_NODES_COLUMN = "IP"
    # Default snap threshold
    SNAP_THRESHOLD = 0.01
    SNAP_THRESHOLD_ERROR_MULTIPLIER = 1.1
    AREA_EDGE_SNAP_MULTIPLIER = 1.0
    TRIANGLE_ERROR_SNAP_MULTIPLIER = 10.0
    OVERLAP_DETECTION_MULTIPLIER = 50.0
    SHARP_AVG_THRESHOLD = 80
    SHARP_PREV_SEG_THRESHOLD = 70

    @staticmethod
    @abstractmethod
    def fix_method(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validation_method(*args, **kwargs):
        raise NotImplementedError


class GeomTypeValidator(BaseValidator):
    """
    Validates the geometry.
    Validates that all traces are LineStrings. Tries to use shapely.ops.linemerge
    to merge MultiLineStrings into LineStrings.
    """

    ERROR = "GEOM TYPE MULTILINESTRING"

    @staticmethod
    def fix_method(geom: Any) -> Optional[LineString]:
        """
        E.g. mergeable MultiLineString

        >>> mls = MultiLineString(
        ...         [((0, 0), (1, 1)), ((1, 1), (2, 2))]
        ... )
        >>> GeomTypeValidator.fix_method(mls).wkt
        'LINESTRING (0 0, 1 1, 2 2)'

        Unhandled types will just return as None.

        >>> GeomTypeValidator.fix_method(Point(1,1)) is None
        True

        """
        if isinstance(geom, LineString):
            return geom
        if not isinstance(geom, MultiLineString):
            return None
        fixed_geom = linemerge(geom)
        return fixed_geom if isinstance(fixed_geom, LineString) else None

    @classmethod
    def validation_method(cls, geom: Any, **kwargs) -> bool:
        """
        E.g. Anything but LineString

        >>> GeomTypeValidator.validation_method(Point(1,1))
        False

        With LineString:

        >>> GeomTypeValidator.validation_method(LineString([(0, 0), (1, 1)]))
        True

        """
        return isinstance(geom, LineString)


class MultiJunctionValidator(BaseValidator):
    """
    Validates that junctions consists of a maximum of two lines crossing,
    never more.
    """

    ERROR = "MULTI JUNCTION"

    @classmethod
    def validation_method(cls, *args, idx: int, faulty_junctions: Set[int], **kwargs):
        """

        >>> MultiJunctionValidator.validation_method(idx=1, faulty_junctions=set([1, 2]))
        False

        """
        return idx not in faulty_junctions

    @staticmethod
    def determine_faulty_junctions(
        intersect_nodes: List[Tuple[Point, ...]],
        snap_threshold: float,
        snap_threshold_error_multiplier: float,
    ) -> Set[int]:
        """
        Determines when a point of interest represents a multi junction i.e.
        when there are more than 2 points within the buffer distance of each other.

        Two points is the limit because an intersection point between two traces
        is added to the list of interest points twice, once for each trace.

        Faulty junction can also represent a slightly overlapping trace i.e.
        a snapping error.

        >>> intersect_nodes = [
        ...     (Point(0, 0), Point(1, 1)),
        ...     (Point(1, 1),),
        ...     (Point(5, 5),),
        ...     (Point(0, 0), Point(1, 1)),
        ... ]
        >>> snap_threshold = 0.01
        >>> snap_threshold_error_multiplier = 1.1
        >>> MultiJunctionValidator.determine_faulty_junctions(
        ...     intersect_nodes, snap_threshold, snap_threshold_error_multiplier
        ... )
        {0, 1, 3}

        """
        return determine_node_junctions(
            nodes=intersect_nodes,
            snap_threshold=snap_threshold,
            snap_threshold_error_multiplier=snap_threshold_error_multiplier,
            error_threshold=2,
        )


class VNodeValidator(BaseValidator):
    """
    Finds V-nodes within trace data.
    """

    ERROR = "V NODE"

    @classmethod
    def validation_method(cls, *args, idx: int, vnodes: Set[int], **kwargs):
        """

        >>> VNodeValidator.validation_method(idx=1, vnodes=set([1, 2]))
        False

        >>> VNodeValidator.validation_method(idx=5, vnodes=set([1, 2]))
        True

        """
        return idx not in vnodes

    @staticmethod
    def determine_v_nodes(
        endpoint_nodes: List[Tuple[Point, ...]],
        snap_threshold: float,
        snap_threshold_error_multiplier: float,
    ) -> Set[int]:
        """

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
    def validation_method(
        geom: LineString,
        trace_candidates: gpd.GeoSeries,
        **kwargs,
    ):
        """

        >>> geom = LineString([Point(-3, -4), Point(-3, -1)])
        >>> trace_candidates = gpd.GeoSeries(
        ...     [LineString([Point(-4, -3), Point(-2, -3), Point(-4, -2), Point(-2, -1)])]
        ... )
        >>> MultipleCrosscutValidator.validation_method(geom, trace_candidates)
        False

        >>> geom = LineString([Point(-3, -4), Point(-3, -4.5)])
        >>> trace_candidates = gpd.GeoSeries(
        ...     [LineString([Point(-4, -3), Point(-2, -3), Point(-4, -2), Point(-2, -1)])]
        ... )
        >>> MultipleCrosscutValidator.validation_method(geom, trace_candidates)
        True

        """
        intersection_geoms = trace_candidates.intersection(geom)
        if any(
            [
                len(list(geom.geoms)) > 2
                for geom in intersection_geoms
                if isinstance(geom, MultiPoint)
            ]
        ):
            return False
        return True


class UnderlappingSnapValidator(BaseValidator):
    """
    Find snapping errors of
    underlapping traces by using a multiple of the given snap_threshold

    Uses validation_method from MultipleCrosscutValidator
    """

    ERROR = "UNDERLAPPING SNAP"

    @staticmethod
    def validation_method(
        geom: LineString,
        trace_candidates: gpd.GeoSeries,
        snap_threshold: float,
        snap_threshold_error_multiplier: float,
        **kwargs,
    ):
        """

        >>> snap_threshold = 0.01
        >>> snap_threshold_error_multiplier = 1.1
        >>> geom = LineString(
        ...     [(0, 0), (0, 1 + snap_threshold * snap_threshold_error_multiplier * 0.99)]
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
            if any(
                [
                    snap_threshold
                    < trace.distance(endpoint)
                    < snap_threshold * snap_threshold_error_multiplier
                    for trace in trace_candidates.geometry.values
                ]
            ):
                return False
        return True


class TargetAreaSnapValidator(MultipleCrosscutValidator):

    ERROR = "TRACE UNDERLAPS TARGET AREA"

    @staticmethod
    def validation_method(
        geom: LineString,
        area: gpd.GeoDataFrame,
        snap_threshold: float,
        snap_threshold_error_multiplier: float,
        area_edge_snap_multiplier: float,
        **kwargs,
    ):
        """

        >>> geom = LineString([(0, 0), (0, 1)])
        >>> area = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
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
        >>> area = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
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
                if endpoint.within(area_polygon):
                    # Point is completely within the area, does not intersect
                    # its edge.
                    if (
                        snap_threshold
                        <= endpoint.distance(area_polygon.boundary)
                        < snap_threshold
                        * snap_threshold_error_multiplier
                        * area_edge_snap_multiplier
                    ):
                        return False
        return True


class GeomNullValidator(BaseValidator):
    """
    Validate the geometry for NULL GEOMETRY errors.
    """

    ERROR = "NULL GEOMETRY"

    @classmethod
    def validation_method(cls, geom: Any, **kwargs) -> bool:
        """

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
        elif geom.is_empty:
            return False
        else:
            return True


class StackedTracesValidator(MultipleCrosscutValidator):
    """
    Find stacked traces and small triangle intersections.
    """

    ERROR = "STACKED TRACES"

    @staticmethod
    def validation_method(
        geom: LineString,
        trace_candidates: gpd.GeoSeries,
        snap_threshold: float,
        snap_threshold_error_multiplier: float,
        overlap_detection_multiplier: float,
        triangle_error_snap_multiplier: float,
        **kwargs,
    ):
        """

        >>> geom = LineString([(0, 0), (0, 1)])
        >>> trace_candidates = gpd.GeoSeries([LineString([(0, -1), (0, 2)])])
        >>> snap_threshold = 0.01
        >>> snap_threshold_error_multiplier = 1.1
        >>> overlap_detection_multiplier = 50
        >>> triangle_error_snap_multiplier = 10
        >>> StackedTracesValidator.validation_method(
        ...     geom,
        ...     trace_candidates,
        ...     snap_threshold,
        ...     snap_threshold_error_multiplier,
        ...     overlap_detection_multiplier,
        ...     triangle_error_snap_multiplier,
        ... )
        False

        >>> geom = LineString([(10, 0), (10, 1)])
        >>> trace_candidates = gpd.GeoSeries([LineString([(0, -1), (0, 2)])])
        >>> snap_threshold = 0.01
        >>> snap_threshold_error_multiplier = 1.1
        >>> overlap_detection_multiplier = 50
        >>> triangle_error_snap_multiplier = 10
        >>> StackedTracesValidator.validation_method(
        ...     geom,
        ...     trace_candidates,
        ...     snap_threshold,
        ...     snap_threshold_error_multiplier,
        ...     overlap_detection_multiplier,
        ...     triangle_error_snap_multiplier,
        ... )
        True

        """
        if len(trace_candidates) == 0:
            return True

        trace_candidates_multils = MultiLineString(
            [tc for tc in trace_candidates.geometry.values]
        )
        # Test for overlapping traces.
        if segment_within_buffer(
            geom,
            trace_candidates_multils,
            snap_threshold=snap_threshold,
            snap_threshold_error_multiplier=snap_threshold_error_multiplier,
            overlap_detection_multiplier=overlap_detection_multiplier,
        ):
            return False

        # Test for small triangles created by the intersection of two traces.
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
    Use shapely is_simple and is_ring attributes to check that LineString does
    not cut itself.
    """

    ERROR = "CUTS ITSELF"

    @staticmethod
    def validation_method(geom: LineString, **kwargs) -> bool:
        return geom.is_simple and not geom.is_ring


class SharpCornerValidator(BaseValidator):
    """
    Finds sharp cornered traces.
    """

    ERROR = "SHARP TURNS"

    @staticmethod
    def validation_method(
        geom: LineString,
        sharp_avg_threshold: float,
        sharp_prev_seg_threshold: float,
        **kwargs,
    ) -> bool:
        geom_coords = get_trace_coord_points(geom)
        if len(geom_coords) == 2:
            # If LineString consists of two Points -> No sharp corners.
            return True
        trace_unit_vector = create_unit_vector(geom_coords[0], geom_coords[-1])
        for idx, segment_start in enumerate(geom_coords):
            if idx == len(geom_coords) - 1:
                break
            segment_end: Point = geom_coords[idx + 1]
            segment_unit_vector = create_unit_vector(segment_start, segment_end)
            # Compare the two unit vectors
            if not compare_unit_vector_orientation(
                trace_unit_vector, segment_unit_vector, sharp_avg_threshold
            ):
                return False
            # Check how much of a change compared to previous segment.
            if idx != 0:
                # Cannot get previous if node is first
                prev_segment_unit_vector = create_unit_vector(
                    geom_coords[idx - 1], geom_coords[idx]
                )
                if not compare_unit_vector_orientation(
                    segment_unit_vector,
                    prev_segment_unit_vector,
                    sharp_prev_seg_threshold,
                ):
                    return False

        return True


ALL_VALIDATORS = (
    GeomNullValidator,
    GeomTypeValidator,
    SimpleGeometryValidator,
    MultiJunctionValidator,
    VNodeValidator,
    MultipleCrosscutValidator,
    UnderlappingSnapValidator,
    TargetAreaSnapValidator,
    StackedTracesValidator,
    SharpCornerValidator,
)

MAJOR_ERRORS = (GeomTypeValidator.ERROR, GeomNullValidator.ERROR)
ValidatorClass = Type[BaseValidator]
