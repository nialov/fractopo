import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import MultiPoint, Point, LineString, MultiLineString, Polygon
from shapely.ops import split
import ast
from pygeos import GEOSException
import numpy as np

import math
from typing import Union, Tuple, List
from abc import abstractmethod
from itertools import zip_longest
import logging


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
    SNAP_THRESHOLD_ERROR_MULTIPLIER = 10.0
    AREA_EDGE_SNAP_MULTIPLIER = 1.0
    TRIANGLE_ERROR_SNAP_MULTIPLIER = 10.0
    OVERLAP_DETECTION_MULTIPLIER = 50.0
    SHARP_AVG_THRESHOLD = 80
    SHARP_PREV_SEG_THRESHOLD = 70
    nodes_calculated = False
    (
        nodes_of_interaction_both,
        node_id_data_both,
        nodes_of_interaction_endpoints,
        node_id_data_endpoints,
        nodes_of_interaction_interactions,
        node_id_data_interactions,
    ) = (None, None, None, None, None, None)

    @classmethod
    def determined_node_data(cls):
        for data in (
            cls.nodes_of_interaction_both,
            cls.node_id_data_both,
            cls.nodes_of_interaction_endpoints,
            cls.node_id_data_endpoints,
            cls.nodes_of_interaction_interactions,
            cls.node_id_data_interactions,
        ):
            yield data

    @classmethod
    def empty_node_data(cls):
        cls.nodes_of_interaction_both = None
        cls.node_id_data_both = None
        cls.nodes_of_interaction_endpoints = None
        cls.node_id_data_endpoints = None
        cls.nodes_of_interaction_interactions = None
        cls.node_id_data_interactions = None
        cls.nodes_calculated = False

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
    def set_snap_threshold_and_multipliers(
        cls,
        snap_threshold: Union[int, float],
        snap_threshold_error_multiplier: Union[int, float],
        area_edge_snap_multiplier: Union[int, float],
    ) -> None:
        if not all(
            [
                isinstance(arg, float) or isinstance(arg, int)
                for arg in (
                    snap_threshold,
                    snap_threshold_error_multiplier,
                    area_edge_snap_multiplier,
                )
            ]
        ):
            args = {
                "snap_threshold": snap_threshold,
                "snap_threshold_error_multiplier": snap_threshold_error_multiplier,
                "area_edge_snap_multiplier": area_edge_snap_multiplier,
            }
            raise TypeError(
                f"Arguments of set_snap_threshold_and_multipliers"
                f" must be either floats or ints. Passed args:\n"
                f"{args}"
            )
        if snap_threshold <= 0:
            raise ValueError("Snap threshold cannot be negative or zero.")
        # TODO: Check that this effects all classes.
        cls.SNAP_THRESHOLD = snap_threshold
        cls.SNAP_THRESHOLD_ERROR_MULTIPLIER = snap_threshold_error_multiplier
        cls.AREA_EDGE_SNAP_MULTIPLIER = area_edge_snap_multiplier

    @classmethod
    def handle_error_column(
        cls, trace_geodataframe: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        if cls.ERROR_COLUMN not in trace_geodataframe.columns:
            trace_geodataframe[cls.ERROR_COLUMN] = pd.Series(
                [[] for _ in trace_geodataframe.index], dtype=object
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
    def get_trace_endpoints(
        cls, trace: Union[shapely.geometry.LineString]
    ) -> List[shapely.geometry.Point]:
        """
        Returns endpoints (shapely.geometry.Point) of a given LineString
        """
        if not isinstance(trace, shapely.geometry.LineString):
            raise TypeError(
                "Non LineString geometry passed into get_trace_endpoints.\n"
                f"trace: {trace.wkt}"
            )
        return [
            endpoint
            for endpoint in (
                shapely.geometry.Point(trace.coords[0]),
                shapely.geometry.Point(trace.coords[-1]),
            )
        ]

    @classmethod
    def determine_nodes_old(
        cls, trace_geodataframe: gpd.GeoDataFrame, interactions=True, endpoints=True
    ) -> Tuple[List[shapely.geometry.Point], List[Tuple[int, ...]]]:
        """
        Determines points of interest between traces.

        The points are linked to the indexes of the traces if
        trace_geodataframe with the returned node_id_data list. node_id_data
        contains tuples of ids. The ids represent indexes of
        nodes_of_interaction. The order of the node_id_data tuples is
        equivalent to the trace_geodataframe trace indexes.

        Conditionals interactions and endpoints allow for choosing what
        to return specifically.
        """

        # nodes_of_interaction contains all intersection points between
        # trace_geodataframe traces.
        nodes_of_interaction: List[shapely.geometry.Point]
        nodes_of_interaction = []
        # node_id_data contains all ids that correspond to points in
        # nodes_of_interaction
        node_id_data: List[Tuple[int, ...]]
        node_id_data = []
        for idx, row in trace_geodataframe.iterrows():
            start_length = len(nodes_of_interaction)
            intersection_geoms = trace_geodataframe.geometry.drop(idx).intersection(
                row.geometry
            )
            if interactions:
                nodes_of_interaction.extend(
                    [
                        geom
                        for geom in intersection_geoms
                        # Checking for instance avoids errors with stacked traces
                        # -> intersection possible linestring
                        # and multiple intersections between two traces
                        # -> interaction is a multipoint
                        if isinstance(geom, shapely.geometry.Point)
                    ]
                )
            # Add trace endpoints to nodes_of_interaction
            if endpoints:
                try:
                    nodes_of_interaction.extend(
                        [
                            endpoint
                            for endpoint in cls.get_trace_endpoints(row.geometry)
                            if not any(
                                intersection_geoms.intersects(
                                    endpoint.buffer(cls.SNAP_THRESHOLD)
                                )
                            )
                            # Checking that endpoint is also not a point of
                            # interaction is not done if interactions are not
                            # determined.
                            or not interactions
                        ]
                    )
                except TypeError:
                    # Error is raised when MultiLineString is passed. They do
                    # not provide a coordinate sequence.
                    pass
            end_length = len(nodes_of_interaction)
            node_id_data.append(tuple([i for i in range(start_length, end_length)]))

        if len(nodes_of_interaction) == 0 or len(node_id_data) == 0:
            # TODO
            logging.error("Both nodes_of_interaction and node_id_data are empty...")
        return nodes_of_interaction, node_id_data

    @classmethod
    def get_nodes(
        cls,
        trace_geodataframe: gpd.GeoDataFrame,
        interactions=True,
        endpoints=True,
        parallel=False,
    ) -> Union[
        Tuple[List[shapely.geometry.Point], List[Tuple[int, ...]]], Tuple[None, None]
    ]:
        """
        To reduce on unnecessary calculations, determined nodes of interaction
        and endpoints of lines are saved to BaseValidator as a class attribute.
        """
        if (
            BaseValidator.nodes_calculated
            and parallel
            and all([saved is not None for saved in cls.determined_node_data()])
        ):
            if interactions and endpoints:
                return cls.nodes_of_interaction_both, cls.node_id_data_both
            elif endpoints:
                return cls.nodes_of_interaction_endpoints, cls.node_id_data_endpoints
            elif interactions:
                return (
                    cls.nodes_of_interaction_interactions,
                    cls.node_id_data_interactions,
                )
            else:
                raise ValueError(
                    "determine_nodes called with both"
                    "interactions and endpoints as False."
                )
        elif not parallel:
            BaseValidator.nodes_calculated = False
            return cls.determine_nodes(trace_geodataframe, interactions, endpoints)
        else:
            (nodes_of_interaction_both, node_id_data_both,) = cls.determine_nodes(
                trace_geodataframe, interactions=True, endpoints=True
            )
            BaseValidator.nodes_of_interaction_both = nodes_of_interaction_both
            BaseValidator.node_id_data_both = node_id_data_both
            (
                nodes_of_interaction_endpoints,
                node_id_data_endpoints,
            ) = cls.determine_nodes(
                trace_geodataframe, interactions=False, endpoints=True
            )
            BaseValidator.nodes_of_interaction_endpoints = (
                nodes_of_interaction_endpoints
            )
            BaseValidator.node_id_data_endpoints = node_id_data_endpoints
            (
                nodes_of_interaction_interactions,
                node_id_data_interactions,
            ) = cls.determine_nodes(
                trace_geodataframe, interactions=True, endpoints=False
            )
            BaseValidator.nodes_of_interaction_interactions = (
                nodes_of_interaction_interactions
            )
            BaseValidator.node_id_data_interactions = node_id_data_interactions
            BaseValidator.nodes_calculated = True
            return cls.get_nodes(trace_geodataframe, interactions, endpoints, parallel)

    @staticmethod
    def determine_valid_interaction_points(
        intersection_geoms: gpd.GeoSeries,
    ) -> List[Point]:
        assert isinstance(intersection_geoms, gpd.GeoSeries)
        valid_interaction_points = []
        for geom in intersection_geoms:
            if isinstance(geom, Point):
                valid_interaction_points.append(geom)
            elif isinstance(geom, MultiPoint):
                valid_interaction_points.extend([p for p in geom])
            else:
                pass
        assert all([isinstance(p, Point) for p in valid_interaction_points])
        return valid_interaction_points

    @classmethod
    def determine_nodes(
        cls, trace_geodataframe: gpd.GeoDataFrame, interactions=True, endpoints=True
    ) -> Tuple[List[shapely.geometry.Point], List[Tuple[int, ...]]]:
        """
        Determines points of interest between traces.

        The points are linked to the indexes of the traces if
        trace_geodataframe with the returned node_id_data list. node_id_data
        contains tuples of ids. The ids represent indexes of
        nodes_of_interaction. The order of the node_id_data tuples is
        equivalent to the trace_geodataframe trace indexes.

        Conditionals interactions and endpoints allow for choosing what
        to return specifically.
        """

        # nodes_of_interaction contains all intersection points between
        # trace_geodataframe traces.
        nodes_of_interaction: List[shapely.geometry.Point]
        nodes_of_interaction = []
        # node_id_data contains all ids that correspond to points in
        # nodes_of_interaction
        node_id_data: List[Tuple[int, ...]]
        node_id_data = []
        trace_geodataframe.reset_index(drop=True, inplace=True)
        spatial_index = trace_geodataframe.geometry.sindex
        for idx, geom in enumerate(trace_geodataframe.geometry):
            # for idx, row in trace_geodataframe.iterrows():
            start_length = len(nodes_of_interaction)

            trace_candidates_idx = list(spatial_index.intersection(geom.bounds))
            assert idx in trace_candidates_idx
            # Remove current geometry from candidates
            trace_candidates_idx.remove(idx)
            assert idx not in trace_candidates_idx
            trace_candidates = trace_geodataframe.geometry.iloc[trace_candidates_idx]
            intersection_geoms = trace_candidates.intersection(geom)
            if interactions:
                nodes_of_interaction.extend(
                    cls.determine_valid_interaction_points(intersection_geoms)
                )
            # Add trace endpoints to nodes_of_interaction
            if endpoints:
                try:
                    nodes_of_interaction.extend(
                        [
                            endpoint
                            for endpoint in cls.get_trace_endpoints(geom)
                            if not any(
                                intersection_geoms.intersects(
                                    endpoint.buffer(cls.SNAP_THRESHOLD)
                                )
                            )
                            # Checking that endpoint is also not a point of
                            # interaction is not done if interactions are not
                            # determined.
                            or not interactions
                        ]
                    )
                except TypeError:
                    # Error is raised when MultiLineString is passed. They do
                    # not provide a coordinate sequence.
                    pass
            end_length = len(nodes_of_interaction)
            node_id_data.append(tuple([i for i in range(start_length, end_length)]))

        if len(nodes_of_interaction) == 0 or len(node_id_data) == 0:
            # TODO
            logging.error("Both nodes_of_interaction and node_id_data are empty...")
        return nodes_of_interaction, node_id_data

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


class GeomTypeValidator(BaseValidator):
    """
    Validates the geometry.
    Validates that all traces are LineStrings. Tries to use shapely.ops.linemerge
    to merge MultiLineStrings into LineStrings.
    """

    ERROR = "GEOM TYPE MULTILINESTRING"

    @classmethod
    def validation_function(
        cls, geom: Union[shapely.geometry.LineString, shapely.geometry.MultiLineString]
    ) -> bool:
        return isinstance(geom, shapely.geometry.LineString)

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
                if not cls.validation_function(geom)
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

    @classmethod
    def fix_function(cls, row: pd.Series) -> pd.Series:
        if isinstance(row[cls.GEOMETRY_COLUMN], shapely.geometry.LineString):
            return row
        elif row[cls.GEOMETRY_COLUMN] is None:
            logging.error("row[cls.GEOMETRY_COLUMN] is None")
            return row
        # Fix will not throw error if merge cannot happen. It will simply
        # return a MultiLineString instead of a LineString.
        fixed_geom = shapely.ops.linemerge(row[cls.GEOMETRY_COLUMN])
        # If linemerge results in LineString:
        if isinstance(fixed_geom, shapely.geometry.LineString):
            try:
                removed_error = row[cls.ERROR_COLUMN].remove(cls.ERROR)
            except ValueError:
                # Error not in row for some reason... TODO
                removed_error = row[cls.ERROR_COLUMN]
            removed_error = [] if removed_error is None else removed_error
            # Update input row with fixed geometry and with error removed
            # from list
            row[cls.GEOMETRY_COLUMN] = fixed_geom
            row[cls.ERROR_COLUMN] = removed_error
            return row
        else:
            # Fix was not succesful, keep error message in column.
            logging.info(
                "Unable to convert MultiLineString to LineString. "
                "MultiLineString probably consists of disjointed segments."
            )
            return row

        return row

    @classmethod
    def fix(cls, trace_geodataframe: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        trace_geodataframe = trace_geodataframe.apply(cls.fix_function, axis=1)
        return trace_geodataframe


class MultiJunctionValidator(BaseValidator):
    """
    Validates that junctions consists of a maximum of two lines crossing,
    never more.
    """

    ERROR = "MULTI JUNCTION"

    @classmethod
    def validation_function(
        cls,
        point_ids: Tuple[int],
        faulty_junctions: gpd.GeoSeries,
    ) -> bool:
        """
        Returns False if point ids match faulty junctions.
        """
        assert isinstance(faulty_junctions, gpd.GeoSeries)
        if not isinstance(point_ids, tuple):
            return False
        try:
            return (
                False
                if len(set(point_ids) & set(list(faulty_junctions.index))) > 0
                else True
            )
        except ValueError:
            breakpoint()
            return False

    @classmethod
    def determine_faulty_junctions(
        cls, nodes_of_interaction: List[shapely.geometry.Point], parallel=False
    ) -> gpd.GeoSeries:
        """
        Determines when a point of interest represents a multi junction i.e.
        when there are more than 2 points within the buffer distance of each other.

        Two points is the limit because an intersection point between two traces
        is added to the list of interest points twice, once for each trace.

        Faulty junction can also represent a slightly overlapping trace i.e.
        a snapping error.
        """
        nodes_of_interaction_geoseries = gpd.GeoSeries(nodes_of_interaction)
        # Now all points that are not erronous are removed from the
        # nodes_of_interaction_geoseries
        indexes_not_to_remove: List[bool]
        indexes_not_to_remove = []
        for idx, point in enumerate(nodes_of_interaction_geoseries):
            points_intersecting_point = nodes_of_interaction_geoseries.drop(
                idx
            ).intersects(
                point.buffer(cls.SNAP_THRESHOLD * cls.SNAP_THRESHOLD_ERROR_MULTIPLIER)
            )
            if len([p for p in points_intersecting_point if p]) >= 2:
                # The junction is erronous when there are more than 2 points
                # within the buffer distance of each other.
                # >= 2 because current 'point' is + 1
                indexes_not_to_remove.append(True)
            else:
                indexes_not_to_remove.append(False)

        faulty_junctions = nodes_of_interaction_geoseries.loc[indexes_not_to_remove]
        return faulty_junctions

    @classmethod
    def validate(
        cls,
        trace_geodataframe: gpd.GeoDataFrame,
        area_geodataframe=None,
        parallel=False,
    ) -> gpd.GeoDataFrame:
        nodes_of_interaction, node_id_data = cls.get_nodes(
            trace_geodataframe, parallel=parallel
        )
        assert len(node_id_data) == len(trace_geodataframe)
        assert all([isinstance(x, tuple) for x in node_id_data])
        # Inplace insertion of new column that contains point ids that
        # correspond to points of intersection with other traces.
        trace_geodataframe.insert(
            loc=len(trace_geodataframe.columns),
            column=cls.INTERACTION_NODES_COLUMN,
            value=pd.Series(node_id_data, dtype=object),
        )
        faulty_junctions = cls.determine_faulty_junctions(
            nodes_of_interaction, parallel
        )

        trace_geodataframe = cls.update_error_column(
            trace_geodataframe, faulty_junctions
        )

        trace_geodataframe = trace_geodataframe.drop(
            columns=cls.INTERACTION_NODES_COLUMN, inplace=False
        )
        trace_geodataframe = cls.handle_error_column(trace_geodataframe)
        return trace_geodataframe

    @classmethod
    def update_error_column(
        cls, trace_geodataframe: gpd.GeoDataFrame, validation_data: gpd.GeoSeries
    ) -> gpd.GeoDataFrame:
        trace_geodataframe[cls.ERROR_COLUMN] = pd.Series(
            [
                cls.error_test(error) + [cls.ERROR]
                if not cls.validation_function(point_ids, validation_data)
                and cls.ERROR not in cls.error_test(error)
                else cls.error_test(error)
                for point_ids, error in cls.zip_equal(
                    trace_geodataframe[cls.INTERACTION_NODES_COLUMN],
                    trace_geodataframe[cls.ERROR_COLUMN],
                )
            ],
            dtype=object,
        )
        return trace_geodataframe


class VNodeValidator(MultiJunctionValidator):
    """
    Finds V-nodes within trace data. Inherits from MultiJunctionValidator
    because it uses the same classmethods: update_error_column and
    validation_function
    """

    ERROR = "V NODE"

    @classmethod
    def determine_v_nodes(
        cls, endpoints: List[shapely.geometry.Point]
    ) -> gpd.GeoSeries:
        endpoints_geoseries = gpd.GeoSeries(endpoints)
        indexes_not_to_remove: List[bool]
        indexes_not_to_remove = []
        for idx, point in enumerate(endpoints_geoseries):
            points_intersecting_point = endpoints_geoseries.drop(idx).intersects(
                point.buffer(cls.SNAP_THRESHOLD * cls.SNAP_THRESHOLD_ERROR_MULTIPLIER)
            )
            if len([p for p in points_intersecting_point if p]) >= 1:
                # The junction is erronous when there are any points
                # within the buffer distance of each other.
                indexes_not_to_remove.append(True)
            else:
                indexes_not_to_remove.append(False)

        assert len(indexes_not_to_remove) == len(endpoints_geoseries)
        v_nodes = endpoints_geoseries.loc[indexes_not_to_remove]
        return v_nodes

    @classmethod
    def validate(
        cls,
        trace_geodataframe: gpd.GeoDataFrame,
        area_geodataframe=None,
        parallel=False,
    ) -> gpd.GeoDataFrame:
        # determine_nodes returns only the endpoints
        endpoints, node_id_data = cls.get_nodes(
            trace_geodataframe, interactions=False, parallel=parallel
        )
        trace_geodataframe.insert(
            loc=len(trace_geodataframe.columns),
            column=cls.INTERACTION_NODES_COLUMN,
            value=pd.Series(node_id_data, dtype=object),
        )
        v_nodes = cls.determine_v_nodes(endpoints)

        trace_geodataframe = cls.update_error_column(trace_geodataframe, v_nodes)

        trace_geodataframe = trace_geodataframe.drop(
            columns=cls.INTERACTION_NODES_COLUMN, inplace=False
        )
        trace_geodataframe = cls.handle_error_column(trace_geodataframe)
        return trace_geodataframe


class MultipleCrosscutValidator(BaseValidator):
    """
    Finds traces that cross-cut each other multiple times. This indicates the
    possibility of duplicate traces.
    """

    ERROR = "MULTIPLE CROSSCUTS"

    @classmethod
    def validation_function(cls, idx, rows_with_stacked):
        return idx not in rows_with_stacked

    @classmethod
    def validate(
        cls,
        trace_geodataframe: gpd.GeoDataFrame,
        area_geodataframe=None,
        parallel=False,
    ) -> gpd.GeoDataFrame:
        rows_with_stacked = cls.determine_stacked_traces(trace_geodataframe)

        trace_geodataframe[cls.ERROR_COLUMN] = pd.Series(
            [
                cls.error_test(error) + [cls.ERROR]
                if not cls.validation_function(idx, rows_with_stacked)
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
    def determine_stacked_traces_old(
        cls,
        trace_geodataframe: gpd.GeoDataFrame,
    ) -> list:
        rows_with_stacked = []
        for idx, row in trace_geodataframe.iterrows():
            intersection_geoms = trace_geodataframe.geometry.drop(idx).intersection(
                row.geometry
            )
            # Stacked traces are defined as traces with more than two intersections
            # with each other.
            # the if-statement checks for MultiPoints in intersection_geoms
            # which indicates atleast two intersection between traces.
            # if there are more than two points in a MultiPoint geometry
            # -> stacked traces
            if any(
                [
                    len(list(geom.geoms)) > 2
                    for geom in intersection_geoms
                    if isinstance(geom, shapely.geometry.MultiPoint)
                ]
            ):
                rows_with_stacked.append(idx)
        return rows_with_stacked

    @classmethod
    def determine_stacked_traces(
        cls,
        trace_geodataframe: gpd.GeoDataFrame,
    ) -> List[int]:
        """
        Determines row indexes in the trace_geodataframe that have stacked
        traces. Stackes is defined as having 3 or more intersections with
        another trace.
        """
        rows_with_stacked = []
        trace_geodataframe.reset_index(drop=True, inplace=True)
        spatial_index = trace_geodataframe.geometry.sindex
        for idx, trace in enumerate(trace_geodataframe.geometry):
            # for idx, row in trace_geodataframe.iterrows():
            trace_candidates_idx = list(spatial_index.intersection(trace.bounds))
            trace_candidates_idx.remove(idx)
            trace_candidates = trace_geodataframe.geometry.iloc[trace_candidates_idx]

            intersection_geoms = trace_candidates.intersection(trace)
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
                    if isinstance(geom, shapely.geometry.MultiPoint)
                ]
            ):
                rows_with_stacked.append(idx)
        return rows_with_stacked


class UnderlappingSnapValidator(MultipleCrosscutValidator):
    """
    Finds snapping errors of
    underlapping traces by using a multiple of the given snap_threshold

    Uses validation_function from MultipleCrosscutValidator
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
                if not cls.validation_function(idx, rows_with_underlapping)
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
                row_endpoints = cls.get_trace_endpoints(row.geometry)
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
                if not cls.validation_function(idx, rows_with_underlapping)
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
                row_endpoints = cls.get_trace_endpoints(row.geometry)
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
    Validates the geometry for NULL errors. These cannot be handled and
    the rows are consequently removed.
    """

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
                if not cls.validation_function(idx, rows_with_overlapping)
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
    def split_to_determine_triangle_errors(cls, trace, splitter_trace):
        assert isinstance(trace, LineString)
        assert isinstance(splitter_trace, LineString)
        try:
            segments = split(trace, splitter_trace)
        except ValueError:
            # split not possible, the traces overlap
            return True
        if len(segments) > 2:
            seg_lengths: List[float]
            seg_lengths = [seg.length for seg in segments]
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
    ) -> Union[List[LineString], LineString]:
        assert isinstance(linestring, LineString)
        points = [Point(c) for c in linestring.coords]
        segmentized: List[Point]
        segmentized = []
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
    def validation_function(
        cls, geom: Union[shapely.geometry.LineString, shapely.geometry.MultiLineString]
    ) -> bool:
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
                if not cls.validation_function(geom)
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
    def validation_function(
        cls, geom: Union[shapely.geometry.LineString, shapely.geometry.MultiLineString]
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
                if not cls.validation_function(geom)
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


# def calc_azimu(line):
#     """
#     Calculates azimuth of given line.

#     e.g.:
#     Accepts LineString

#     >>> calc_azimu(shapely.geometry.LineString([(0, 0), (1, 1)]))
#     45.0

#     Accepts mergeable MultiLineString

#     >>> calc_azimu(shapely.geometry.MultiLineString([((0, 0), (1, 1)), ((1, 1), (2, 2))]))
#     45.0

#     Returns np.nan when the line cannot be merged into one continuous line.

#     >>> calc_azimu(shapely.geometry.MultiLineString([((0, 0), (1, 1)), ((1.5, 1), (2, 2))]))
#     nan

#     :param line: Continous line feature (trace, branch, etc.)
#     :type line: shapely.geometry.LineString | shapely.geometry.MultiLineString
#     :return: Azimuth of line.
#     :rtype: float | np.nan
#     """
#     try:
#         coord_list = list(line.coords)
#     except NotImplementedError:
#         # TODO: Needs more testing?
#         line = linemerge(line)
#         try:
#             coord_list = list(line.coords)
#         except NotImplementedError:
#             return np.NaN
#     start_x = coord_list[0][0]
#     start_y = coord_list[0][1]
#     end_x = coord_list[-1][0]
#     end_y = coord_list[-1][1]
#     azimu = 90 - math.degrees(math.atan2((end_y - start_y), (end_x - start_x)))
#     if azimu < 0:
#         azimu = azimu + 360
#     return azimu


# def azimu_half(degrees):
#     """
#     Transforms azimuths from 180-360 range to range 0-180

#     :param degrees: Degrees in range 0 - 360
#     :type degrees: float
#     :return: Degrees in range 0 - 180
#     :rtype: float
#     """
#     if degrees >= 180:
#         degrees = degrees - 180
#     return degrees


def get_trace_coord_points(trace: LineString) -> List[Point]:
    assert isinstance(trace, LineString)
    return [Point(xy) for xy in trace.coords]


def point_to_xy(point: Point) -> Tuple[float, float]:
    x, y = point.xy
    x, y = [val[0] for val in (x, y)]
    return (x, y)


class SharpCornerValidator(BaseValidator):
    """
    Finds sharp cornered traces.
    """

    ERROR = "SHARP TURNS"

    @classmethod
    def validation_function(
        cls, trace: Union[shapely.geometry.LineString, shapely.geometry.MultiLineString]
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
                if not cls.validation_function(geom)
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
        if 1 < dot_product or dot_product < -1 or np.isnan(dot_product):
            rad_angle = np.nan
        else:
            rad_angle = np.arccos(dot_product)
        deg_angle = np.rad2deg(rad_angle)
        if deg_angle > threshold_angle:
            # If angle between more than threshold_angle -> False
            return False
        return True
