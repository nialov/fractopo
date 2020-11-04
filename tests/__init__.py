import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Point, LineString, MultiLineString, Polygon
import numpy as np
import hypothesis
from hypothesis.strategies import (
    booleans,
    floats,
    sets,
    lists,
    tuples,
    one_of,
    text,
    integers,
)
from hypothesis import given
from hypothesis_geometry import planar

from fractopo.tval import trace_validator
from fractopo.tval.trace_validator import (
    BaseValidator,
    GeomTypeValidator,
    MultiJunctionValidator,
    VNodeValidator,
    MultipleCrosscutValidator,
    TargetAreaSnapValidator,
    UnderlappingSnapValidator,
    GeomNullValidator,
    StackedTracesValidator,
    EmptyGeometryValidator,
    SimpleGeometryValidator,
    SharpCornerValidator,
)
from fractopo.tval import trace_builder

GEOMETRY_COLUMN = BaseValidator.GEOMETRY_COLUMN
ERROR_COLUMN = BaseValidator.ERROR_COLUMN

SNAP_THRESHOLD = 0.001
SNAP_THRESHOLD_ERROR_MULTIPLIER = 1.1
AREA_EDGE_SNAP_MULTIPLIER = 5


class Helpers:
    valid_geom = shapely.geometry.LineString(((0, 0), (1, 1)))
    invalid_geom_empty = shapely.geometry.LineString()
    invalid_geom_none = None
    invalid_geom_multilinestring = shapely.geometry.MultiLineString(
        [((0, 0), (1, 1)), ((-1, 0), (1, 0))]
    )
    (
        valid_traces,
        invalid_traces,
        valid_areas_geoseries,
        invalid_areas_geoseries,
    ) = trace_builder.main(False, SNAP_THRESHOLD, SNAP_THRESHOLD_ERROR_MULTIPLIER)
    valid_error_srs = pd.Series([[] for _ in valid_traces])
    invalid_error_srs = pd.Series([[] for _ in invalid_traces])
    random_data_column = lambda i: ["aaa" for _ in i]
    # geoms are all LineStrings and no errors
    @classmethod
    def valid_gdf_get(cls):
        return gpd.GeoDataFrame(
            {
                GEOMETRY_COLUMN: Helpers.valid_traces,
                ERROR_COLUMN: Helpers.valid_error_srs,
                "random_col": cls.random_data_column(Helpers.valid_traces),
                "random_col2": cls.random_data_column(Helpers.valid_traces),
                "random_col3": cls.random_data_column(Helpers.valid_traces),
                "random_col4": cls.random_data_column(Helpers.valid_traces),
            }
        )

    @classmethod
    def invalid_gdf_get(cls):
        return gpd.GeoDataFrame(
            {
                GEOMETRY_COLUMN: Helpers.invalid_traces,
                ERROR_COLUMN: Helpers.invalid_error_srs,
                "random_col": cls.random_data_column(Helpers.invalid_traces),
            }
        )

    @classmethod
    def invalid_gdf_null_get(cls):
        return gpd.GeoDataFrame(
            {
                GEOMETRY_COLUMN: [None, shapely.geometry.LineString()],
                ERROR_COLUMN: [[], []],
                "random_col": cls.random_data_column(range(2)),
            }
        )

    @staticmethod
    def valid_area_gdf_get():
        return gpd.GeoDataFrame({GEOMETRY_COLUMN: Helpers.valid_areas_geoseries})

    @staticmethod
    def invalid_area_gdf_get():
        return gpd.GeoDataFrame({GEOMETRY_COLUMN: Helpers.invalid_areas_geoseries})

    faulty_error_srs = pd.Series([[] for _ in valid_traces])
    faulty_error_srs[0] = np.nan
    faulty_error_srs[1] = "this cannot be transformed to list?"
    faulty_error_srs[2] = (1, 2, 3, "hello?")
    faulty_error_srs[5] = 5.12315235

    @classmethod
    def valid_gdf_with_faulty_error_col_get(cls):
        return gpd.GeoDataFrame(
            {
                GEOMETRY_COLUMN: Helpers.valid_traces,
                ERROR_COLUMN: Helpers.faulty_error_srs,
                "random_col": cls.random_data_column(Helpers.valid_traces),
            }
        )

    @staticmethod
    def iterate_validators():
        for validator in (
            GeomNullValidator,
            GeomTypeValidator,
            MultiJunctionValidator,
            VNodeValidator,
            MultipleCrosscutValidator,
            UnderlappingSnapValidator,
            TargetAreaSnapValidator,
        ):
            yield validator

    nice_integer_coordinates = integers(-10, 10)
    nice_float = floats(
        allow_nan=False, allow_infinity=False, min_value=-1e5, max_value=1e5
    )
    nice_tuple = tuples(
        nice_float,
        nice_float,
    )
    triple_tuples = tuples(
        nice_tuple,
        nice_tuple,
        nice_tuple,
    )

    @classmethod
    def get_multi_polyline_strategy(cls):
        multi_polyline_strategy = tuples(
            *tuple(
                [
                    planar.polylines(
                        x_coordinates=cls.nice_integer_coordinates,
                        min_size=2,
                        max_size=5,
                    )
                    for _ in range(5)
                ]
            )
        )
        return multi_polyline_strategy

    snap_threshold = 0.001
    geosrs_identicals = gpd.GeoSeries(
        [Point(1, 1), Point(1, 1), Point(2, 1), Point(2, 1), Point(3, 1), Point(2, 3)]
    )

    traces_geosrs = gpd.GeoSeries(
        [
            LineString([(-1, 0), (1, 0)]),
            LineString([(0, -1), (0, 1)]),
            LineString(
                [(-1.0 - snap_threshold * 0.99, -1), (-1.0 - snap_threshold * 0.99, 1)]
            ),
        ]
    )
    areas_geosrs = gpd.GeoSeries([Polygon([(5, 5), (-5, 5), (-5, -5), (5, -5)])])

    nice_traces = gpd.GeoSeries(
        [
            # Horizontal
            LineString([(-10, 0), (10, 0)]),
            # Underlapping
            LineString([(-5, 2), (-5, 0 + snap_threshold * 0.01)]),
            LineString([(-4, 2), (-4, 0 + snap_threshold * 0.5)]),
            LineString([(-3, 2), (-3, 0 + snap_threshold * 0.7)]),
            LineString([(-2, 2), (-2, 0 + snap_threshold * 0.9)]),
            LineString([(-1, 2), (-1, 0 + snap_threshold * 1.1)]),
            # Overlapping
            LineString([(1, 2), (1, 0 - snap_threshold * 1.1)]),
            LineString([(2, 2), (2, 0 - snap_threshold * 0.9)]),
            LineString([(3, 2), (3, 0 - snap_threshold * 0.7)]),
            LineString([(4, 2), (4, 0 - snap_threshold * 0.5)]),
            LineString([(5, 2), (5, 0 - snap_threshold * 0.01)]),
        ]
    )
    nice_integer_coordinates = integers(-10, 10)
    nice_float = floats(
        allow_nan=False, allow_infinity=False, min_value=-1e5, max_value=1e5
    )
    nice_tuple = tuples(
        nice_float,
        nice_float,
    )
    triple_tuples = tuples(
        nice_tuple,
        nice_tuple,
        nice_tuple,
    )
    nice_point = planar.points(nice_integer_coordinates)
    # TODO: Is not really nice...
    nice_polyline = planar.polylines(nice_integer_coordinates).filter(
        lambda x: LineString(x).is_valid
    )

    @classmethod
    def get_nice_traces(cls):
        return cls.nice_traces.copy()

    @classmethod
    def get_traces_geosrs(cls):
        return cls.traces_geosrs.copy()

    @classmethod
    def get_areas_geosrs(cls):
        return cls.areas_geosrs.copy()

    @classmethod
    def get_geosrs_identicals(cls):
        return cls.geosrs_identicals.copy()
