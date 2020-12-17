import pandas as pd
import geopandas as gpd
import shapely
from shapely.ops import linemerge
from shapely.wkt import loads
from shapely.geometry import Point, LineString, MultiLineString, Polygon
import numpy as np
import hypothesis
from pathlib import Path
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
from fractopo.analysis import tools, parameters
from fractopo.general import CC_branch, CI_branch, II_branch, X_node, Y_node, I_node
import fractopo.tval.trace_validation as trace_validation
from fractopo.tval.executor import Validation


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
    mergeable_geom_multilinestring = shapely.geometry.MultiLineString(
        [((0, 0), (1, 1)), ((1, 1), (2, 2))]
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

    line_1 = LineString([(0, 0), (0.5, 0.5)])
    line_2 = LineString([(0, 0), (0.5, -0.5)])
    line_3 = LineString([(0, 0), (1, 0)])
    line_1_sp = Point(list(line_1.coords)[0])
    line_2_sp = Point(list(line_2.coords)[0])
    line_1_ep = Point(list(line_1.coords)[-1])
    line_2_ep = Point(list(line_2.coords)[-1])
    halved_azimuths = [
        tools.azimu_half(tools.calc_azimu(l))
        for l in (
            line_1,
            line_2,
            line_3,
        )
    ]
    branch_frame = gpd.GeoDataFrame(
        {
            "geometry": [line_1, line_2, line_3],
            "Connection": ["C - C", "C - I", "I - I"],
            "Class": ["X - I", "Y - Y", "I - I"],
            "halved": halved_azimuths,
            "length": [line.length for line in [line_1, line_2, line_3]],
        }
    )

    trace_frame = gpd.GeoDataFrame(
        {
            "geometry": [line_1, line_2],
            "length": [line_1.length, line_2.length],
            "startpoint": [line_1_sp, line_2_sp],
            "endpoint": [line_1_ep, line_2_ep],
        }
    )
    point_1 = Point(0.5, 0.5)
    point_2 = Point(1, 1)
    point_3 = Point(10, 10)
    node_frame = gpd.GeoDataFrame(
        {"geometry": [point_1, point_2, point_3], "Class": ["X", "Y", "I"]}
    )
    node_frame["c"] = node_frame["Class"]
    area_1 = Polygon([(0, 0), (1, 1), (1, 0)])
    area_frame = gpd.GeoDataFrame({"geometry": [area_1]})

    sample_trace_data = Path("tests/sample_data/KB11_traces.shp")
    sample_branch_data = Path("tests/sample_data/KB11_branches.shp")
    sample_area_data = Path("tests/sample_data/KB11_area.shp")

    kb7_trace_path = Path("tests/sample_data/KB7/KB7_tulkinta_50.shp")
    kb7_area_path = Path("tests/sample_data/KB7/KB7_tulkinta_alue.shp")

    kb7_traces = gpd.read_file(kb7_trace_path)
    kb7_area = gpd.read_file(kb7_area_path)
    test_tracevalidate_params = [
        (
            Path("tests/sample_data/KB7/KB7_tulkinta_50.shp"),  # cut 0-50
            Path("tests/sample_data/KB7/KB7_tulkinta_alue.shp"),
            "--fix",
        ),
        (
            Path("tests/sample_data/KB7/KB7_tulkinta_100.shp"),  # cut 50-100
            Path("tests/sample_data/KB7/KB7_tulkinta_alue.shp"),
            "--fix",
        ),
    ]

    test_match_crs_params = [
        (
            gpd.GeoSeries([Point(1, 1)]).set_crs(3067),  # first
            gpd.GeoSeries([Point(1, 1)]),  # second
            True,  # same
            True,  # from_first
            False,  # is_none
        ),
        (
            gpd.GeoSeries([Point(1, 1)]),  # first
            gpd.GeoSeries([Point(1, 1)]),  # second
            True,  # same
            True,  # from_first
            True,  # is_none
        ),
        (
            gpd.GeoSeries([Point(1, 1)]).set_crs(3067),  # first
            gpd.GeoSeries([Point(1, 1)]).set_crs(3066),  # second
            False,  # same
            True,  # from_first
            False,  # is_none
        ),
    ]
    test_is_within_buffer_distance_params = [
        (nice_traces, 0.5, 25),
        (nice_traces, 1, 35),
    ]

    test_plot_xyi_plot_params = [
        ([{X_node: 0, Y_node: 0, I_node: 50}], ["title"]),
        ([{X_node: 0, Y_node: 0, I_node: 0}], ["title"]),
        ([{X_node: 0, Y_node: 10, I_node: 25}], [""]),
    ]

    test_plot_branch_plot_params = [
        ([{CC_branch: 30, CI_branch: 15, II_branch: 50}], ["title"]),
        ([{CC_branch: 0, CI_branch: 0, II_branch: 50}], ["title"]),
        ([{CC_branch: 0, CI_branch: 0, II_branch: 0}], ["title"]),
    ]

    test_determine_topology_parameters_params = [
        (
            np.array([10, 10, 10, 10]),  # trace_length_array
            np.array([5, 5, 5, 5, 5, 5, 5, 5]),  # branch_length_array
            {X_node: 3, Y_node: 5, I_node: 8},  # node_counts dict
            10.0,  # area
        ),
        (
            np.array([1, 1, 1, 1]),  # trace_length_array
            np.array([1, 1, 1, 1, 1, 1, 1, 1]),  # branch_length_array
            {X_node: 3, Y_node: 5, I_node: 8},  # node_counts dict
            1.0,  # area
        ),
    ]

    test_plot_topology_params = [
        (
            [
                parameters.determine_topology_parameters(  # topology_parameters_list
                    *test_determine_topology_parameters_params[0]
                )
            ],
            ["title"],  # labels
            ["black"],  # colors
        )
    ]
    test_determine_nodes_intersecting_sets_params = [
        (
            (
                gpd.GeoSeries([LineString([(0, 0), (1, 1)])]),
                gpd.GeoSeries([LineString([(0, 1), (0, -1)])]),
            ),  # trace_series_two_sets
            np.array(["1", "2"]),  # set_array
            ("1", "2"),  # set_names_two_sets
            gpd.GeoSeries(
                [Point(0, 0), Point(1, 1), Point(0, 1), Point(0, -1)]
            ),  # node_series_xy
            0.001,  # buffer_value
            [True, False, False, False],  # assumed_intersections
        ),
        (
            (
                gpd.GeoSeries([LineString([(0.5, 0.5), (1, 1)])]),
                gpd.GeoSeries([LineString([(0, 1), (0, -1)])]),
            ),  # trace_series_two_sets
            np.array(["1", "2"]),  # set_array
            ("1", "2"),  # set_names_two_sets
            gpd.GeoSeries(
                [Point(0.5, 0.5), Point(1, 1), Point(0, 1), Point(0, -1)]
            ),  # node_series_xy
            0.001,  # buffer_value
            [False, False, False, False],  # assumed_intersections
        ),
    ]

    test_prepare_geometry_traces_params = [
        (
            gpd.GeoSeries(
                [LineString([(0.5, 0.5), (1, 1)]), LineString([(0, 1), (0, -1)])]
            )
        ),
        (
            gpd.GeoSeries(
                [
                    LineString([(0.5, 0.5), (1, 1)]),
                    LineString([(0, 1), (0, -1)]),
                    LineString([(0, 100), (0, -15)]),
                    LineString([(5, 100), (67, -15), (67, -150)]),
                ]
            )
        ),
    ]
    test_determine_intersects_params = [
        (
            (
                gpd.GeoSeries([LineString([(0, 0), (1, 1)])]),
                gpd.GeoSeries([LineString([(0, 1), (0, -1)])]),
            ),  # trace_series_two_sets
            ("1", "2"),  # set_names_two_sets
            gpd.GeoSeries([Point(0, 0)]),  # node_series_xy_intersects
            np.array(["Y"]),  # node_types_xy_intersects
            # assumed_intersections
            0.001,  # buffer_value
        ),
        (
            (
                gpd.GeoSeries([LineString([(0.5, 0.5), (1, 1)])]),
                gpd.GeoSeries([LineString([(0, 1), (0, -1)])]),
            ),  # trace_series_two_sets
            ("1", "2"),  # set_names_two_sets
            gpd.GeoSeries([]),  # node_series_xy_intersects
            np.array([]),  # node_types_xy_intersects
            # assumed_intersections
            0.001,  # buffer_value
        ),
    ]
    test_determine_crosscut_abutting_relationships_params = [
        (
            gpd.GeoSeries(
                [LineString([(0, 0), (1, 0)]), LineString([(0, 1), (0, -1)])]
            ),  # trace_series
            gpd.GeoSeries(
                [Point(0, 0), Point(1, 0), Point(0, 1), Point(0, -1)]
            ),  # node_series
            np.array(["Y", "I", "I", "I"]),  # node_types
            np.array(["1", "2"]),  # set_array
            ("1", "2"),  # set_names
            0.001,  # buffer_value
            "title",  # label
        ),
    ]

    test__validate_params = [
        (
            trace_validation.GeomNullValidator,  # validator
            None,  # geom
            [],  # current_errors
            True,  # allow_fix
            [None, [trace_validation.GeomNullValidator.ERROR], True],  # assumed_result
        ),
        (
            trace_validation.GeomTypeValidator,  # validator
            invalid_geom_multilinestring,  # geom
            [],  # current_errors
            True,  # allow_fix
            [
                invalid_geom_multilinestring,
                [trace_validation.GeomTypeValidator.ERROR],
                True,
            ],  # assumed_result
        ),
        (
            trace_validation.GeomTypeValidator,  # validator
            mergeable_geom_multilinestring,  # geom
            [],  # current_errors
            True,  # allow_fix
            [
                loads("LINESTRING (0 0, 1 1, 2 2)"),
                [],
                False,
            ],  # assumed_result
        ),
    ]
