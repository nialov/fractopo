"""
Test parameters i.e. sample data, known past errors, etc.
"""
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis.strategies import (
    booleans,
    floats,
    integers,
    lists,
    one_of,
    sets,
    text,
    tuples,
)
from hypothesis_geometry import planar
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import linemerge
from shapely.wkt import loads

from fractopo.analysis import parameters, tools
from fractopo.general import (
    CC_branch,
    CI_branch,
    E_node,
    I_node,
    II_branch,
    X_node,
    Y_node,
    bounding_polygon,
)
from fractopo.tval import trace_builder, trace_validation
from fractopo.tval.trace_validators import (
    BaseValidator,
    EmptyTargetAreaValidator,
    GeomNullValidator,
    GeomTypeValidator,
    MultiJunctionValidator,
    MultipleCrosscutValidator,
    SharpCornerValidator,
    SimpleGeometryValidator,
    StackedTracesValidator,
    TargetAreaSnapValidator,
    UnderlappingSnapValidator,
    VNodeValidator,
)
from tests.sample_data.py_samples.samples import (
    results_in_false_positive_stacked_traces_list,
    results_in_false_positive_underlapping_ls,
    results_in_multijunction_why_ls_list,
    results_in_multijunction_why_ls_list_2,
    results_in_overlapping_ls_list,
    should_result_in_multij_ls_list,
    should_result_in_some_error_ls_list,
    should_result_in_target_area_underlapping_ls,
    should_result_in_target_area_underlapping_poly,
    should_result_in_vnode_ls_list,
)
from tests.sample_data.py_samples.stacked_traces_sample import non_stacked_traces_ls

GEOMETRY_COLUMN = trace_validation.Validation.GEOMETRY_COLUMN
ERROR_COLUMN = trace_validation.Validation.ERROR_COLUMN

SNAP_THRESHOLD = 0.001
SNAP_THRESHOLD_ERROR_MULTIPLIER = 1.1
AREA_EDGE_SNAP_MULTIPLIER = 5


class Helpers:
    valid_geom = LineString(((0, 0), (1, 1)))

    invalid_geom_empty = LineString()
    invalid_geom_none = None
    invalid_geom_multilinestring = MultiLineString(
        [((0, 0), (1, 1)), ((-1, 0), (1, 0))]
    )
    mergeable_geom_multilinestring = MultiLineString(
        [((0, 0), (1, 1)), ((1, 1), (2, 2))]
    )
    (
        valid_traces,
        invalid_traces,
        valid_areas_geoseries,
        invalid_areas_geoseries,
    ) = trace_builder.main(False, SNAP_THRESHOLD, SNAP_THRESHOLD_ERROR_MULTIPLIER)
    valid_error_srs = pd.Series([[] for _ in valid_traces.geometry.values])
    invalid_error_srs = pd.Series([[] for _ in invalid_traces.geometry.values])
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
                GEOMETRY_COLUMN: [None, LineString()],
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

    faulty_error_srs = pd.Series([[] for _ in valid_traces.geometry.values])
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
    kb11_traces = gpd.read_file(sample_trace_data)
    kb11_area = gpd.read_file(sample_area_data)

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
            {X_node: 3, Y_node: 5, I_node: 8, E_node: 0},  # node_counts dict
            10.0,  # area
        ),
        (
            np.array([1, 1, 1, 1]),  # trace_length_array
            {X_node: 3, Y_node: 5, I_node: 8, E_node: 0},  # node_counts dict
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
            GeomNullValidator,  # validator
            None,  # geom
            [],  # current_errors
            True,  # allow_fix
            [None, [GeomNullValidator.ERROR], True],  # assumed_result
        ),
        (
            GeomTypeValidator,  # validator
            invalid_geom_multilinestring,  # geom
            [],  # current_errors
            True,  # allow_fix
            [
                invalid_geom_multilinestring,
                [GeomTypeValidator.ERROR],
                True,
            ],  # assumed_result
        ),
        (
            GeomTypeValidator,  # validator
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
    intersect_nodes = [
        (Point(0, 0), Point(1, 1)),
        (Point(1, 1),),
        (Point(5, 5),),
        (Point(0, 0), Point(1, 1)),
    ]

    # Intersects next trace three times
    intersects_next_trace_3_times = LineString(
        [Point(-4, -3), Point(-2, -3), Point(-4, -2), Point(-2, -1)]
    )

    # Straight line which is intersected twice by same line
    intersected_3_times = LineString([Point(-3, -4), Point(-3, -1)])
    test_validation_params = [
        (
            kb7_traces,  # traces
            kb7_area,  # area
            "kb7",  # name
            True,  # auto_fix
            [SharpCornerValidator.ERROR],  # assume_errors
        ),
        # (
        #     kb11_traces,  # traces
        #     kb11_area,  # area
        #     "kb11",  # name
        #     True,  # auto_fix
        #     None,  # assume_errors
        # ),
        (
            gpd.GeoDataFrame(
                geometry=trace_builder.make_invalid_traces(
                    snap_threshold=0.01, snap_threshold_error_multiplier=1.1
                )
            ),  # traces
            gpd.GeoDataFrame(
                geometry=trace_builder.make_invalid_target_areas()
            ),  # area
            "invalid_traces",  # name
            True,  # auto_fix
            None,  # assume_errors
        ),
        (
            gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 1)])]),  # traces
            gpd.GeoDataFrame(
                geometry=[
                    Polygon(
                        [
                            Point(-1, -1),
                            Point(-1, 1.011),
                            Point(1, 1.011),
                            Point(1, -1),
                        ]
                    )
                ]
            ),  # area
            "TargetAreaSnapValidator error",  # name
            True,  # auto_fix
            [TargetAreaSnapValidator.ERROR],  # assume_errors
        ),
        (
            gpd.GeoDataFrame(
                geometry=[LineString([(0, 0), (0, 1)]), LineString([(5, 5), (5, 6)])]
            ),  # traces
            gpd.GeoDataFrame(
                geometry=[
                    Polygon(
                        [
                            Point(-1, -1),
                            Point(-1, 1.011),
                            Point(1, 1.011),
                            Point(1, -1),
                        ]
                    ),
                    Polygon(
                        [
                            Point(2, 2),
                            Point(2, 6.011),
                            Point(6, 6.011),
                            Point(6, 2),
                        ]
                    ),
                ]
            ),  # area
            "TargetAreaSnapValidator error",  # name
            True,  # auto_fix
            [TargetAreaSnapValidator.ERROR],  # assume_errors
        ),
        (
            gpd.GeoDataFrame(
                geometry=[LineString([(0, 0), (0, 1)]), LineString([(5, 5), (5, 6)])]
            ),  # traces
            gpd.GeoDataFrame(
                geometry=[
                    MultiPolygon(
                        [
                            Polygon(
                                [
                                    Point(-1, -1),
                                    Point(-1, 1.011),
                                    Point(1, 1.011),
                                    Point(1, -1),
                                ]
                            ),
                            Polygon(
                                [
                                    Point(2, 2),
                                    Point(2, 6.011),
                                    Point(6, 6.011),
                                    Point(6, 2),
                                ]
                            ),
                        ]
                    )
                ]
            ),  # area
            "TargetAreaSnapValidator error",  # name
            True,  # auto_fix
            [TargetAreaSnapValidator.ERROR],  # assume_errors
        ),
    ]

    test_determine_v_nodes_params = [
        (
            [(Point(1, 1),), (Point(1, 1),)],  # endpoint_nodes
            0.01,  # snap_threshold
            1.1,  # snap_threshold_error_multiplier
            {0, 1},  # assumed_result
        ),
        (
            [(Point(1, 1),), (Point(1, 1),)],  # endpoint_nodes
            0.01,  # snap_threshold
            1.1,  # snap_threshold_error_multiplier
            {0, 1},  # assumed_result
        ),
    ]

    test_determine_node_junctions_params = [
        (
            [
                (Point(0, 0), Point(1, 1)),
                (Point(1, 1),),
                (Point(5, 5),),
                (Point(0, 0), Point(1, 1)),
            ],  # nodes
            0.01,  # snap_threshold
            1.1,  # snap_threshold_error_multiplier
            2,  # error_threshold
        )
    ]

    test_bounding_polygon_params = [
        (gpd.GeoSeries([line_1, line_2, line_3])),
        (gpd.GeoSeries([line_1])),
    ]

    test_testtargetareasnapvalidator_validation_method = [
        (
            LineString([(0.5, 0), (0.5, 0.5)]),  # geom: LineString,
            gpd.GeoDataFrame(
                geometry=[
                    MultiPolygon(
                        [
                            Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
                            Polygon([(10, 10), (10, 11), (11, 11), (11, 10)]),
                        ]
                    )
                ]
            ),  # area:gpd.GeoDataFrame
            0.01,  # snap_threshold: float,
            1.1,  # snap_threshold_error_multiplier: float,
            2.5,  # area_edge_snap_multiplier: float,
            True,  # assumed_result: bool,
        ),
        (
            LineString([(0.5, 0.01 * 1.05), (0.5, 0.5)]),  # geom: LineString,
            gpd.GeoDataFrame(
                geometry=[
                    MultiPolygon(
                        [
                            Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
                            Polygon([(10, 10), (10, 11), (11, 11), (11, 10)]),
                        ]
                    )
                ]
            ),  # area:gpd.GeoDataFrame
            0.01,  # snap_threshold: float,
            1.1,  # snap_threshold_error_multiplier: float,
            2.5,  # area_edge_snap_multiplier: float,
            False,  # assumed_result: bool,
        ),
        (
            LineString([(0.5, 0), (0.5, 0.5)]),  # geom: LineString,
            gpd.GeoDataFrame(
                geometry=[
                    Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
                    Polygon([(10, 10), (10, 11), (11, 11), (11, 10)]),
                ]
            ),  # area:gpd.GeoDataFrame
            0.01,  # snap_threshold: float,
            1.1,  # snap_threshold_error_multiplier: float,
            2.5,  # area_edge_snap_multiplier: float,
            True,  # assumed_result: bool,
        ),
        (
            LineString([(10, 0), (4.991, 0)]),  # geom: LineString,
            gpd.GeoDataFrame(
                geometry=[Polygon([(5, 5), (-5, 5), (-5, -5), (5, -5)])]
            ),  # area:gpd.GeoDataFrame
            0.01,  # snap_threshold: float,
            1.1,  # snap_threshold_error_multiplier: float,
            1.5,  # area_edge_snap_multiplier: float,
            True,  # assumed_result: bool,
        ),  # Test that traces coming from outside area are not marked as underlapping
        (
            LineString([(10, 0), (5.011, 0)]),  # geom: LineString,
            gpd.GeoDataFrame(
                geometry=[Polygon([(5, 5), (-5, 5), (-5, -5), (5, -5)])]
            ),  # area:gpd.GeoDataFrame
            0.01,  # snap_threshold: float,
            1.1,  # snap_threshold_error_multiplier: float,
            1.5,  # area_edge_snap_multiplier: float,
            True,  # assumed_result: bool,
        ),  # Test that traces coming from outside area are not marked as underlapping
        (
            should_result_in_target_area_underlapping_ls,  # geom: LineString,
            gpd.GeoDataFrame(
                geometry=[should_result_in_target_area_underlapping_poly]
            ),  # area:gpd.GeoDataFrame
            0.01,  # snap_threshold: float,
            1.1,  # snap_threshold_error_multiplier: float,
            2.5,  # area_edge_snap_multiplier: float,
            False,  # assumed_result: bool,
        ),  # Test that traces coming from outside area are not marked as underlapping
    ]

    test_tracevalidate_only_area_params = [
        (
            [
                "tests/sample_data/KB7/KB7_tulkinta_50.shp",  # cut 0-50
                "tests/sample_data/KB7/KB7_tulkinta_alue.shp",
                "--fix",
                "--only-area-validation",
            ]  # args
        )
    ]

    geta_1_traces = gpd.read_file(
        "tests/sample_data/geta1/Getaberget_20m_1_traces.gpkg"
    )
    geta_1_1_area = gpd.read_file(
        "tests/sample_data/geta1/Getaberget_20m_1_1_area.gpkg"
    )

    test_network_params = [
        (
            geta_1_traces,  # traces
            geta_1_1_area,  # area
            "Geta1_1",  # name
            True,  # determine_branches_nodes
            True,  # truncate_traces
            0.001,  # snap_threshold
        ),
    ]

    test_network_random_sampler_params = [
        (
            geta_1_traces,  # trace_gdf
            geta_1_1_area,  # area_gdf
            10,  # min_radius
            0.001,  # snap_threshold
            1,  # samples
        )
    ]


class ValidationHelpers:

    # Known Errors
    # ============

    known_errors = dict()

    known_multi_junction_gdfs = [
        gpd.GeoDataFrame(
            geometry=[
                LineString([Point(0, -3), Point(2, -3)]),
                LineString([Point(1, -4), Point(1, -2)]),
                LineString([Point(2, -4), Point(0.5, -2.50001)]),
            ]
        ),
        gpd.GeoDataFrame(
            geometry=[
                LineString([Point(-2, 0), Point(2, 0)]),
                LineString([Point(0, -2), Point(0, 4)]),
                LineString([Point(1, -1), Point(-1, 1)]),
            ]
        ),
        gpd.GeoDataFrame(
            geometry=[
                LineString([Point(-2, 4), Point(-3, 4)]),
                LineString([Point(-2.5, 3.5), Point(-3.5, 4.5)]),
                LineString([Point(-3.5, 3.5), Point(-2.5, 4.5)]),
            ]
        ),
        gpd.GeoDataFrame(
            geometry=[
                LineString([Point(-2, 2), Point(-4, 2)]),
                LineString(
                    [
                        Point(-3, 1),
                        Point(-3, 2 + 0.01 + 0.0001),
                    ]
                ),
            ]
        ),
        gpd.GeoDataFrame(geometry=should_result_in_some_error_ls_list),
        gpd.GeoDataFrame(geometry=should_result_in_multij_ls_list),
        # TODO: Is this a validation or snapping error.
        # gpd.read_file(Path("tests/sample_data/KB11/KB11_last_validation.geojson")),
    ]

    known_multilinestring_gdfs = [
        gpd.GeoDataFrame(
            geometry=[
                MultiLineString(
                    [
                        LineString([Point(3, -4), Point(3, -1)]),
                        LineString([Point(3, 0), Point(3, 4)]),
                    ]
                )
            ],
        )
    ]
    known_vnode_gdfs = [
        gpd.GeoDataFrame(
            geometry=[
                LineString([Point(0, 0), Point(1.0001, 1)]),
                LineString([Point(1, 0), Point(1.0001, 0.9999)]),
            ]
        ),
        gpd.GeoDataFrame(geometry=should_result_in_vnode_ls_list),
    ]
    known_stacked_gdfs = [
        gpd.GeoDataFrame(
            geometry=[
                LineString([Point(0, -7), Point(0, -5)]),
                LineString([Point(-1, -7), Point(0 + 0.01, -6), Point(-1, -5)]),
            ]
        ),
    ]

    known_non_underlaping_gdfs_but_overlapping = [
        gpd.GeoDataFrame(geometry=results_in_false_positive_underlapping_ls)
    ]

    known_null_gdfs = [gpd.GeoDataFrame(geometry=[None, LineString()])]

    known_errors[MultiJunctionValidator.ERROR] = known_multi_junction_gdfs

    known_errors[GeomTypeValidator.ERROR] = known_multilinestring_gdfs
    known_errors[VNodeValidator.ERROR] = known_vnode_gdfs
    known_errors[StackedTracesValidator.ERROR] = known_stacked_gdfs
    known_errors[GeomNullValidator.ERROR] = known_null_gdfs
    known_errors[
        UnderlappingSnapValidator._OVERLAPPING
    ] = known_non_underlaping_gdfs_but_overlapping

    # False Positives
    # ===============

    known_false_positives = dict()

    known_non_stacked_gdfs = [
        gpd.GeoDataFrame(geometry=non_stacked_traces_ls),
        gpd.GeoDataFrame(geometry=results_in_false_positive_stacked_traces_list),
    ]

    known_non_overlapping_gdfs = [
        gpd.GeoDataFrame(geometry=results_in_overlapping_ls_list)
    ]

    known_non_multijunction_gdfs = [
        gpd.GeoDataFrame(geometry=results_in_multijunction_why_ls_list),
        gpd.GeoDataFrame(geometry=results_in_multijunction_why_ls_list_2),
    ]

    known_false_positives[StackedTracesValidator.ERROR] = known_non_stacked_gdfs
    known_false_positives[
        UnderlappingSnapValidator._UNDERLAPPING
    ] = known_non_underlaping_gdfs_but_overlapping
    known_false_positives[
        UnderlappingSnapValidator._OVERLAPPING
    ] = known_non_overlapping_gdfs
    known_false_positives[MultiJunctionValidator.ERROR] = known_non_multijunction_gdfs

    # Class methods to generate pytest params for parametrization
    # ===========================================================

    @classmethod
    def generate_known_params(cls, error, false_positive):
        knowns: List[gpd.GeoDataFrame] = (
            cls.known_errors[error]
            if not false_positive
            else cls.known_false_positives[error]
        )
        amounts = [
            gdf.shape[0]
            if error
            not in (
                UnderlappingSnapValidator._UNDERLAPPING,
                UnderlappingSnapValidator._OVERLAPPING,
            )
            else 1
            for gdf in knowns
        ]
        try:
            areas = [
                gpd.GeoDataFrame(geometry=[bounding_polygon(gdf)]) for gdf in knowns
            ]
        except:
            areas = [
                gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 1), (1, 0)])])
                for _ in knowns
            ]
        assert len(knowns) == len(areas) == len(amounts)
        return [
            pytest.param(
                known,
                area,
                f"{error}, {amount}",
                True,
                [error],
                amount,
                false_positive,
                id=f"{error}_{amount}".replace(" ", "_"),
            )
            for known, area, amount in zip(knowns, areas, amounts)
        ]

    @classmethod
    def get_all_errors(cls):
        # TODO: UnderlappingSnapValidator doesn't follow protocol
        all_error_types = set(
            [validator.ERROR for validator in trace_validation.ALL_VALIDATORS]
            + [
                UnderlappingSnapValidator._OVERLAPPING,
                UnderlappingSnapValidator._UNDERLAPPING,
            ]
        )
        all_errs = []
        for err in all_error_types:
            try:
                all_errs.extend(cls.generate_known_params(err, false_positive=False))
            except KeyError:
                pass
            try:
                all_errs.extend(cls.generate_known_params(err, false_positive=True))
            except KeyError:
                pass

        assert len(all_errs) > 0
        return all_errs
