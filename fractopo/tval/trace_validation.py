import ast
from typing import Union, Tuple, List, Optional, Any, Set
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
    flatten_node_tuples,
    determine_node_junctions,
)


class BaseValidator:
    """
    Base validator that all classes inherit.
    """

    ERROR = "BASE ERROR"
    ERROR_COLUMN = "VALIDATION_ERRORS"
    GEOMETRY_COLUMN = "geometry"
    INTERACTION_NODES_COLUMN = "IP"
    # Default snap threshold
    SNAP_THRESHOLD = 0.01
    SNAP_THRESHOLD_ERROR_MULTIPLIER = 1.1
    AREA_EDGE_SNAP_MULTIPLIER = 1.0
    TRIANGLE_ERROR_SNAP_MULTIPLIER = 10.0
    OVERLAP_DETECTION_MULTIPLIER = 50.0
    SHARP_AVG_THRESHOLD = 80
    SHARP_PREV_SEG_THRESHOLD = 70

    @classmethod
    def execute(
        cls,
        trace_geodataframe: gpd.GeoDataFrame,
        area_geodataframe: Union[gpd.GeoDataFrame, None],
        auto_fix: bool,
        parallel: bool,
    ):
        # Copy the GeoDataFrame instead of modifying the original.
        trace_geodataframe = trace_geodataframe.copy()
        trace_geodataframe = cls.handle_error_column(trace_geodataframe)
        trace_geodataframe = cls.validate(
            trace_geodataframe, area_geodataframe, parallel
        )
        if auto_fix:
            try:
                fixed_trace_geodataframe = cls.fix(trace_geodataframe.copy())
                trace_geodataframe = fixed_trace_geodataframe
            except NotImplementedError:
                logging.info(f"{cls.__name__} cannot automatically fix {cls.ERROR}.")

        return trace_geodataframe

    @classmethod
    @abstractmethod
    def validate(
        cls,
        trace_geodataframe: gpd.GeoDataFrame,
        area_geodataframe: Union[gpd.GeoDataFrame, None],
        parallel=False,
    ) -> gpd.GeoDataFrame:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def fix(cls, trace_geodataframe: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def fix_method(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def handle_error_column(
        cls, trace_geodataframe: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        if cls.ERROR_COLUMN not in trace_geodataframe.columns:
            trace_geodataframe[cls.ERROR_COLUMN] = pd.Series(
                [[] for _ in trace_geodataframe.index.values], dtype=object
            )
        validated_error_values: List[list]
        validated_error_values = []
        for idx, row in trace_geodataframe.iterrows():
            err = row[cls.ERROR_COLUMN]
            if isinstance(err, float):
                validated_error_values.append([])
            elif isinstance(err, list):
                validated_error_values.append(err)
            elif isinstance(err, str):
                try:
                    # Try to safely evaluate as a Python list object
                    err_eval = ast.literal_eval(err)
                    if isinstance(err_eval, list):
                        validated_error_values.append(err_eval)
                    else:
                        validated_error_values.append([])
                except (SyntaxError, ValueError):
                    validated_error_values.append([])
            else:
                validated_error_values.append([])

        assert len(validated_error_values) == len(trace_geodataframe)
        trace_geodataframe[cls.ERROR_COLUMN] = pd.Series(
            validated_error_values, dtype=object
        )

        return trace_geodataframe

    @classmethod
    def error_test(cls, err):
        # If it is invalid -> it can be an empty list.
        if isinstance(err, list):
            return err
        elif isinstance(err, str):
            try:
                err_eval = ast.literal_eval(err)
                if isinstance(err_eval, list):
                    return err_eval
                else:
                    return []
            except (SyntaxError, ValueError):
                return []

        elif isinstance(err, float):
            return []
        else:
            return []

    @staticmethod
    def zip_equal(*iterables):
        sentinel = object()
        for combo in zip_longest(*iterables, fillvalue=sentinel):
            if sentinel in combo:
                raise ValueError("Iterables have different lengths")
            yield combo


ValidatorClass = Type[BaseValidator]


class GeomTypeValidator(BaseValidator):
    """
    Validates the geometry.
    Validates that all traces are LineStrings. Tries to use shapely.ops.linemerge
    to merge MultiLineStrings into LineStrings.
    """

    ERROR = "GEOM TYPE MULTILINESTRING"

    @staticmethod
    def fix_method(geom: MultiLineString) -> Optional[LineString]:
        fixed_geom = linemerge(geom)
        return fixed_geom if isinstance(fixed_geom, LineString) else None

    @classmethod
    def validation_method(cls, geom: Any, **kwargs) -> bool:
        return isinstance(geom, LineString)

    @classmethod
    def validate(
        cls,
        trace_geodataframe: gpd.GeoDataFrame,
        area_geodataframe=None,
        parallel=False,
    ) -> gpd.GeoDataFrame:
        trace_geodataframe[cls.ERROR_COLUMN] = pd.Series(
            [
                cls.error_test(error) + [cls.ERROR]
                if not cls.validation_method(geom)
                and cls.ERROR not in cls.error_test(error)
                else cls.error_test(error)
                for geom, error in cls.zip_equal(
                    trace_geodataframe[cls.GEOMETRY_COLUMN],
                    trace_geodataframe[cls.ERROR_COLUMN],
                )
            ],
            dtype=object,
        )
        # TODO: There's no reason this needs to called. Find the error.
        trace_geodataframe = cls.handle_error_column(trace_geodataframe)
        return trace_geodataframe

    # @classmethod
    # def fix_method(cls, row: pd.Series) -> pd.Series:
    #     if isinstance(row[cls.GEOMETRY_COLUMN], LineString):
    #         return row
    #     elif row[cls.GEOMETRY_COLUMN] is None:
    #         logging.error("row[cls.GEOMETRY_COLUMN] is None")
    #         return row
    #     # Fix will not throw error if merge cannot happen. It will simply
    #     # return a MultiLineString instead of a LineString.
    #     fixed_geom = linemerge(row[cls.GEOMETRY_COLUMN])
    #     # If linemerge results in LineString:
    #     if isinstance(fixed_geom, LineString):
    #         try:
    #             removed_error = row[cls.ERROR_COLUMN].remove(cls.ERROR)
    #         except ValueError:
    #             # TODO: Error not in row for some reason...
    #             removed_error = row[cls.ERROR_COLUMN]
    #         removed_error = [] if removed_error is None else removed_error
    #         # Update input row with fixed geometry and with error removed
    #         # from list
    #         row[cls.GEOMETRY_COLUMN] = fixed_geom
    #         row[cls.ERROR_COLUMN] = removed_error
    #         return row
    #     else:
    #         # Fix was not succesful, keep error message in column.
    #         logging.info(
    #             "Unable to convert MultiLineString to LineString. "
    #             "MultiLineString probably consists of disjointed segments."
    #         )
    #         return row

    #     return row

    @classmethod
    def fix(cls, trace_geodataframe: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        trace_geodataframe = trace_geodataframe.apply(cls.fix_method, axis=1)
        return trace_geodataframe


class MultiJunctionValidator(BaseValidator):
    """
    Validates that junctions consists of a maximum of two lines crossing,
    never more.
    """

    ERROR = "MULTI JUNCTION"

    @classmethod
    def validation_method(cls, *args, idx: int, faulty_junctions: Set[int], **kwargs):
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
        return idx not in vnodes

    @staticmethod
    def determine_v_nodes(
        endpoint_nodes: List[Tuple[Point, ...]],
        snap_threshold: float,
        snap_threshold_error_multiplier: float,
    ) -> Set[int]:
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
        idx: int,
        spatial_index: PyGEOSSTRTreeIndex,
        traces: gpd.GeoDataFrame,
        **kwargs,
    ):
        trace_candidates_idx = list(spatial_index.intersection(geom.bounds))
        trace_candidates_idx.remove(idx)
        trace_candidates = traces.geometry.iloc[trace_candidates_idx]

        intersection_geoms = trace_candidates.intersection(geom)
        # Stacked traces are defined as traces with more than two intersections
        # with each other.
        # the if-statement checks for MultiPoints in intersection_traces
        # which indicates atleast two intersection between traces.
        # if there are more than two points in a MultiPoint traceetry
        # -> stacked traces
        if any(
            [
                len(list(geom.geoms)) > 2
                for geom in intersection_geoms
                if isinstance(geom, MultiPoint)
            ]
        ):
            return False
        return True


class UnderlappingSnapValidator(MultipleCrosscutValidator):
    """
    Find snapping errors of
    underlapping traces by using a multiple of the given snap_threshold

    Uses validation_method from MultipleCrosscutValidator
    """

    ERROR = "UNDERLAPPING SNAP"

    @classmethod
    def validate(
        cls,
        trace_geodataframe: gpd.GeoDataFrame,
        area_geodataframe=None,
        parallel=False,
    ) -> gpd.GeoDataFrame:
        rows_with_underlapping = cls.determine_underlapping(trace_geodataframe)

        trace_geodataframe[cls.ERROR_COLUMN] = pd.Series(
            [
                cls.error_test(error) + [cls.ERROR]
                if not cls.validation_method(idx, rows_with_underlapping)
                and cls.ERROR not in cls.error_test(error)
                else cls.error_test(error)
                for idx, error in cls.zip_equal(
                    trace_geodataframe.index,
                    trace_geodataframe[cls.ERROR_COLUMN],
                )
            ],
            dtype=object,
        )
        trace_geodataframe = cls.handle_error_column(trace_geodataframe)
        return trace_geodataframe

    @classmethod
    def determine_underlapping(cls, trace_geodataframe):
        rows_with_underlapping = []
        for idx, row in trace_geodataframe.iterrows():
            try:
                row_endpoints = get_trace_endpoints(row.geometry)
            except TypeError:
                # Multipart geometry encountered -> ignore row
                continue
            for endpoint in row_endpoints:
                # Three cases:
                # 1. Endpoint intersects a trace and the two are within
                # snapping distance -> No error
                # 2. Endpoint intersects a trace but the two are not within
                # snapping distance -> Error
                # 3. No intersects -> No error
                if any(
                    trace_geodataframe.geometry.drop(idx).intersects(
                        endpoint.buffer(cls.SNAP_THRESHOLD)
                    )
                ):
                    pass

                elif any(
                    trace_geodataframe.geometry.drop(idx).intersects(
                        endpoint.buffer(
                            cls.SNAP_THRESHOLD * cls.SNAP_THRESHOLD_ERROR_MULTIPLIER
                        )
                    )
                ):
                    rows_with_underlapping.append(idx)
        return rows_with_underlapping


class TargetAreaSnapValidator(MultipleCrosscutValidator):

    ERROR = "TRACE UNDERLAPS TARGET AREA"

    @classmethod
    def validate(
        cls,
        trace_geodataframe: gpd.GeoDataFrame,
        area_geodataframe=None,
        parallel=False,
    ) -> gpd.GeoDataFrame:
        rows_with_underlapping = cls.determine_area_underlapping(
            trace_geodataframe, area_geodataframe
        )

        trace_geodataframe[cls.ERROR_COLUMN] = pd.Series(
            [
                cls.error_test(error) + [cls.ERROR]
                if not cls.validation_method(idx, rows_with_underlapping)
                and cls.ERROR not in cls.error_test(error)
                else cls.error_test(error)
                for idx, error in cls.zip_equal(
                    trace_geodataframe.index,
                    trace_geodataframe[cls.ERROR_COLUMN],
                )
            ],
            dtype=object,
        )
        trace_geodataframe = cls.handle_error_column(trace_geodataframe)
        return trace_geodataframe

    @classmethod
    def determine_area_underlapping(cls, trace_geodataframe, area_geodataframe):
        rows_with_underlapping = []
        for idx, row in trace_geodataframe.iterrows():
            try:
                row_endpoints = get_trace_endpoints(row.geometry)
            except TypeError:
                # Multipart geometry encountered -> ignore row
                continue
            for endpoint in row_endpoints:
                if idx in rows_with_underlapping:
                    # No reason to check same trace twice if already marked
                    continue
                # Overlapping does not cause an error in topological processing
                # Underlapping does. Snapping to target area edges is not
                # currently implemented in topological analysis
                # -> Even if point is within snap_threshold, it will not
                # be interpreted as a snap to area edge.
                for area in area_geodataframe.geometry:
                    if endpoint.within(area):
                        # Point is completely within the area, does not intersect
                        # its edge.
                        if endpoint.buffer(
                            cls.SNAP_THRESHOLD
                            * cls.SNAP_THRESHOLD_ERROR_MULTIPLIER
                            * cls.AREA_EDGE_SNAP_MULTIPLIER
                        ).intersects(area.boundary):
                            rows_with_underlapping.append(idx)

        return rows_with_underlapping


class GeomNullValidator(BaseValidator):
    """
    Validate the geometry for NULL GEOMETRY errors.
    """

    ERROR = "NULL GEOMETRY"

    @classmethod
    def validation_method(cls, geom: Any, **kwargs) -> bool:
        if geom is None:
            return False
        elif geom.is_empty:
            return False
        else:
            return True

    @classmethod
    def validate(
        cls,
        trace_geodataframe: gpd.GeoDataFrame,
        area_geodataframe=None,
        parallel=False,
    ) -> gpd.GeoDataFrame:
        original_length = len(trace_geodataframe)
        valid_geoms = []
        for geom in trace_geodataframe.geometry:
            if geom is None:
                valid_geoms.append(False)
            elif geom.is_empty:
                valid_geoms.append(False)
            else:
                valid_geoms.append(True)

        trace_geodataframe = trace_geodataframe.loc[valid_geoms]
        trace_geodataframe.reset_index(drop=True, inplace=True)
        logging.info(
            f"There were {original_length - sum(valid_geoms)}"
            " rows with empty geometries. Rows containing these were removed."
        )
        return trace_geodataframe


class StackedTracesValidator(MultipleCrosscutValidator):
    """
    Finds stacked traces and small triangle intersections.
    """

    ERROR = "STACKED TRACES"

    @classmethod
    def validate(
        cls,
        trace_geodataframe: gpd.GeoDataFrame,
        area_geodataframe=None,
        parallel=False,
    ) -> gpd.GeoDataFrame:
        rows_with_overlapping = cls.determine_overlapping_traces(trace_geodataframe)

        trace_geodataframe[cls.ERROR_COLUMN] = pd.Series(
            [
                cls.error_test(error) + [cls.ERROR]
                if not cls.validation_method(idx, rows_with_overlapping)
                and cls.ERROR not in cls.error_test(error)
                else cls.error_test(error)
                for idx, error in cls.zip_equal(
                    trace_geodataframe.index,
                    trace_geodataframe[cls.ERROR_COLUMN],
                )
            ],
            dtype=object,
        )
        trace_geodataframe = cls.handle_error_column(trace_geodataframe)
        return trace_geodataframe

    @classmethod
    def determine_overlapping_traces(cls, trace_geodataframe):
        rows_with_overlapping_traces = []
        # Must use a buffered linestring to catch traces that might not intersect
        traces_sindex = trace_geodataframe.geometry.buffer(
            cls.SNAP_THRESHOLD * cls.SNAP_THRESHOLD_ERROR_MULTIPLIER
        ).sindex
        for idx, row in trace_geodataframe.iterrows():
            trace = row.geometry
            trace_candidates_idx = list(traces_sindex.intersection(trace.bounds))
            trace_candidates_idx.remove(idx)
            trace_candidates = trace_geodataframe.geometry.iloc[trace_candidates_idx]
            if len(trace_candidates) == 0:
                continue
            trace_candidates_multils = MultiLineString([tc for tc in trace_candidates])
            # Test for overlapping traces.
            if cls.segment_within_buffer(
                trace,
                trace_candidates_multils,
            ):
                rows_with_overlapping_traces.append(idx)
                continue

            # Test for small triangles created by the intersection of two traces.
            for splitter_trace in trace_candidates:
                if cls.split_to_determine_triangle_errors(trace, splitter_trace):
                    rows_with_overlapping_traces.append(idx)
                    break

        return rows_with_overlapping_traces

    @classmethod
    def determine_middle_in_triangle(
        cls, segments: List[LineString]
    ) -> Optional[LineString]:
        """
        Determines the middle segment within a triangle error. The middle
        segment always intersects the other two.
        """
        buffered = [
            ls.buffer(cls.SNAP_THRESHOLD * cls.SNAP_THRESHOLD_ERROR_MULTIPLIER)
            for ls in segments
        ]
        for idx, buffer in enumerate(buffered):
            others = buffered.copy()
            others.pop(idx)
            if sum([buffer.intersects(other) for other in others]) >= 2:
                return segments[idx]
        return None

    @classmethod
    def split_to_determine_triangle_errors(cls, trace, splitter_trace):
        assert isinstance(trace, LineString)
        assert isinstance(splitter_trace, LineString)
        try:
            segments = split(trace, splitter_trace)
        except ValueError:
            # split not possible, the traces overlap
            return True
        if len(segments) > 2:
            middle = StackedTracesValidator.determine_middle_in_triangle(
                [ls for ls in segments.geoms]
            )
            if middle is not None:
                seg_lengths: List[float] = [middle.length]
            else:
                seg_lengths: List[float] = [seg.length for seg in segments]
            for seg_length in seg_lengths:
                if (
                    cls.SNAP_THRESHOLD
                    < seg_length
                    < cls.SNAP_THRESHOLD * cls.TRIANGLE_ERROR_SNAP_MULTIPLIER
                ):
                    return True
        return False

    @classmethod
    def segment_within_buffer(
        cls, linestring: LineString, multilinestring: MultiLineString
    ):
        """
        First checks if given linestring completely overlaps any part of
        multilinestring and if it does, returns True.

        Next it starts to segmentize the multilinestring to smaller
        linestrings and consequently checks if these segments are completely
        within a buffer made of the given linestring.
        It also checks that the segment size is reasonable.
        """
        buffered_linestring = linestring.buffer(
            cls.SNAP_THRESHOLD * cls.SNAP_THRESHOLD_ERROR_MULTIPLIER
        )
        assert isinstance(linestring, LineString)
        assert isinstance(buffered_linestring, Polygon)
        assert isinstance(multilinestring, MultiLineString)
        assert buffered_linestring.area > 0
        # Test for a single segment overlap
        if linestring.overlaps(multilinestring):
            return True
        # Test for overlap with a buffered linestring
        all_segments: List[LineString]
        all_segments = []
        ls: LineString
        for ls in multilinestring:
            all_segments.extend(cls.segmentize_linestring(ls, 1))
            all_segments.extend(cls.segmentize_linestring(ls, 2))
            all_segments.extend(cls.segmentize_linestring(ls, 3))
            all_segments.extend(cls.segmentize_linestring(ls, 4))
        seg: LineString
        for seg in all_segments:
            if (
                seg.within(buffered_linestring)
                and seg.length > cls.SNAP_THRESHOLD * cls.OVERLAP_DETECTION_MULTIPLIER
            ):
                return True
        return False

    @classmethod
    def segmentize_linestring(
        cls, linestring: LineString, amount: int
    ) -> List[LineString]:
        assert isinstance(linestring, LineString)
        points: List[Point] = [Point(c) for c in linestring.coords]
        segmentized: List[Tuple[Point, Point, Point]] = []
        p: Point
        for idx, p in enumerate(points):
            if idx == len(points) - 1:
                break
            else:
                # Add additional point to find even smaller segments that overlap
                x1 = tuple(*p.coords)[0]
                x2 = tuple(*points[idx + 1].coords)[0]
                y1 = tuple(*p.coords)[1]
                y2 = tuple(*points[idx + 1].coords)[1]
                additional_point = Point((x1 + x2) / 2, (y1 + y2) / 2)
                segmentized.append((p, additional_point, points[idx + 1]))
        if amount == 1:
            return [LineString(points) for points in segmentized]
        else:
            ls = LineString([p for points in segmentized for p in points])
            return cls.segmentize_linestring(ls, amount - 1)


class SimpleGeometryValidator(BaseValidator):
    """
    Uses shapely is_simple to check that LineString does not cut itself.
    """

    ERROR = "CUTS ITSELF"

    @classmethod
    def validation_method(cls, geom: LineString, **kwargs) -> bool:
        return geom.is_simple and not geom.is_ring

    @classmethod
    def validate(
        cls,
        trace_geodataframe: gpd.GeoDataFrame,
        area_geodataframe=None,
        parallel=False,
    ) -> gpd.GeoDataFrame:
        trace_geodataframe[cls.ERROR_COLUMN] = pd.Series(
            [
                cls.error_test(error) + [cls.ERROR]
                if not cls.validation_method(geom)
                and cls.ERROR not in cls.error_test(error)
                else cls.error_test(error)
                for geom, error in cls.zip_equal(
                    trace_geodataframe[cls.GEOMETRY_COLUMN],
                    trace_geodataframe[cls.ERROR_COLUMN],
                )
            ],
            dtype=object,
        )
        # TODO: There's no reason this needs to called. Find the error.
        trace_geodataframe = cls.handle_error_column(trace_geodataframe)
        return trace_geodataframe


class EmptyGeometryValidator(BaseValidator):
    """
    Uses shapely is_empty to check that LineString is not empty.
    """

    ERROR = "IS EMPTY"

    @classmethod
    def validation_method(
        cls, geom: Union[LineString, MultiLineString], **kwargs
    ) -> bool:
        return not geom.is_empty

    @classmethod
    def validate(
        cls,
        trace_geodataframe: gpd.GeoDataFrame,
        area_geodataframe=None,
        parallel=False,
    ) -> gpd.GeoDataFrame:
        trace_geodataframe[cls.ERROR_COLUMN] = pd.Series(
            [
                cls.error_test(error) + [cls.ERROR]
                if not cls.validation_method(geom)
                and cls.ERROR not in cls.error_test(error)
                else cls.error_test(error)
                for geom, error in cls.zip_equal(
                    trace_geodataframe[cls.GEOMETRY_COLUMN],
                    trace_geodataframe[cls.ERROR_COLUMN],
                )
            ],
            dtype=object,
        )
        # TODO: There's no reason this needs to called. Find the error.
        trace_geodataframe = cls.handle_error_column(trace_geodataframe)
        return trace_geodataframe


class SharpCornerValidator(BaseValidator):
    """
    Finds sharp cornered traces.
    """

    ERROR = "SHARP TURNS"

    @classmethod
    def validation_method(
        cls, trace: Union[LineString, MultiLineString], **kwargs
    ) -> bool:
        geom_coords = get_trace_coord_points(trace)
        if len(geom_coords) == 2:
            # If LineString consists of two Points -> No sharp corners.
            return True
        trace_unit_vector = cls.create_unit_vector(geom_coords[0], geom_coords[-1])
        for idx, segment_start in enumerate(geom_coords):
            if idx == len(geom_coords) - 1:
                break
            segment_end: Point = geom_coords[idx + 1]
            segment_unit_vector = cls.create_unit_vector(segment_start, segment_end)
            # Compare the two unit vectors
            if not cls.compare_unit_vector_orientation(
                trace_unit_vector, segment_unit_vector, cls.SHARP_AVG_THRESHOLD
            ):
                return False
            # Check how much of a change compared to previous segment.
            if idx != 0:
                # Cannot get previous if node is first
                prev_segment_unit_vector = cls.create_unit_vector(
                    geom_coords[idx - 1], geom_coords[idx]
                )
                if not cls.compare_unit_vector_orientation(
                    segment_unit_vector,
                    prev_segment_unit_vector,
                    cls.SHARP_PREV_SEG_THRESHOLD,
                ):
                    return False

        return True

    @classmethod
    def validate(
        cls,
        trace_geodataframe: gpd.GeoDataFrame,
        area_geodataframe=None,
        parallel=False,
    ) -> gpd.GeoDataFrame:
        trace_geodataframe[cls.ERROR_COLUMN] = pd.Series(
            [
                cls.error_test(error) + [cls.ERROR]
                if not cls.validation_method(geom)
                and cls.ERROR not in cls.error_test(error)
                else cls.error_test(error)
                for geom, error in cls.zip_equal(
                    trace_geodataframe[cls.GEOMETRY_COLUMN],
                    trace_geodataframe[cls.ERROR_COLUMN],
                )
            ],
            dtype=object,
        )
        # TODO: There's no reason this needs to called. Find the error.
        trace_geodataframe = cls.handle_error_column(trace_geodataframe)
        return trace_geodataframe

    @staticmethod
    def create_unit_vector(start_point: Point, end_point: Point) -> np.ndarray:
        """
        Create numpy unit vector from two shapely Points.
        """
        # Convert Point coordinates to (x, y)
        segment_start = point_to_xy(start_point)
        segment_end = point_to_xy(end_point)
        segment_vector = np.array(
            [segment_end[0] - segment_start[0], segment_end[1] - segment_start[1]]
        )
        segment_unit_vector = segment_vector / np.linalg.norm(segment_vector)
        return segment_unit_vector

    @staticmethod
    def compare_unit_vector_orientation(vec_1, vec_2, threshold_angle):
        """
        If vec_1 and vec_2 are too different in orientation, will return False.
        """
        if np.linalg.norm(vec_1 + vec_2) < np.sqrt(2):
            # If they face opposite side -> False
            return False
        dot_product = np.dot(vec_1, vec_2)
        if np.isclose(dot_product, 1):
            return True
        if 1 < dot_product or dot_product < -1 or np.isnan(dot_product):
            return False
        rad_angle = np.arccos(dot_product)
        deg_angle = np.rad2deg(rad_angle)
        if deg_angle > threshold_angle:
            # If angle between more than threshold_angle -> False
            return False
        return True


# Order is important
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
