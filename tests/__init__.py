"""
Test parameters i.e. sample data, known past errors, etc.
"""
import logging
import os
from functools import lru_cache
from pathlib import Path
from traceback import print_tb
from typing import Any, List

# Make GeoPandas use shapely 2.0 instead of pygeos
os.environ["USE_PYGEOS"] = "0"

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from click.testing import Result
from hypothesis.strategies import floats, integers, tuples
from shapely import wkt
from shapely.errors import GEOSException
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

from fractopo import general
from fractopo.analysis import length_distributions, parameters
from fractopo.general import (
    CC_branch,
    CI_branch,
    E_node,
    I_node,
    II_branch,
    X_node,
    Y_node,
    bounding_polygon,
    determine_azimuth,
    read_geofile,
)
from fractopo.tval import trace_validation
from fractopo.tval.trace_validators import (  # MultipleCrosscutValidator,
    GeomNullValidator,
    GeomTypeValidator,
    MultiJunctionValidator,
    SharpCornerValidator,
    StackedTracesValidator,
    TargetAreaSnapValidator,
    UnderlappingSnapValidator,
    VNodeValidator,
)
from tests.sample_data.py_samples.samples import (
    results_in_false_pos_stacked_traces_list,
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
    v_node_network_error_ls_list,
)

GEOMETRY_COLUMN = trace_validation.Validation.GEOMETRY_COLUMN
ERROR_COLUMN = trace_validation.Validation.ERROR_COLUMN

SNAP_THRESHOLD = 0.001
SNAP_THRESHOLD_ERROR_MULTIPLIER = 1.1
AREA_EDGE_SNAP_MULTIPLIER = 5


def click_error_print(result: Result):
    """
    Print click result traceback.
    """
    if result.exit_code == 0:
        return
    assert result.exc_info is not None
    _, _, tb = result.exc_info
    # print(err_class, err)
    print_tb(tb)
    print(result.output)
    raise Exception(result.exception)


valid_geom = LineString(((0, 0), (1, 1)))

invalid_geom_empty = LineString()
invalid_geom_none = None
invalid_geom_multilinestring = MultiLineString([((0, 0), (1, 1)), ((-1, 0), (1, 0))])
mergeable_geom_multilinestring = MultiLineString([((0, 0), (1, 1)), ((1, 1), (2, 2))])


def trace_builder(
    plot_figs=False, snap_threshold=0.001, snap_threshold_error_multiplier=1.1
):
    """
    Create two GeoSeries of traces.

    Saves plots of these traces into ./figs.

    Returns a list of traces, where:
        1. Valid traces
        2. Invalid traces
        3. Valid areas
        4. Invalid areas

    """
    valid_traces = make_valid_traces()
    valid_geoseries = gpd.GeoSeries(valid_traces)
    valid_areas = make_valid_target_areas()
    valid_areas_geoseries = gpd.GeoSeries(valid_areas)
    # valid_savepath = Path("figs/valid_geoseries.png")

    invalid_traces = make_invalid_traces(
        snap_threshold, snap_threshold_error_multiplier
    )
    invalid_geoseries = gpd.GeoSeries(invalid_traces)
    invalid_areas = make_invalid_target_areas()
    invalid_areas_geoseries = gpd.GeoSeries(invalid_areas)
    # invalid_savepath = Path("figs/invalid_geoseries.png")

    # if plot_figs:
    #     plot_geoseries(valid_geoseries, valid_areas_geoseries, valid_savepath)
    #     plot_geoseries(invalid_geoseries, invalid_areas_geoseries, invalid_savepath)
    return [
        valid_geoseries,
        invalid_geoseries,
        valid_areas_geoseries,
        invalid_areas_geoseries,
    ]


def line_generator(points: list):
    """
    Create a .LineString from a list of Points.
    """
    assert all(isinstance(point, Point) for point in points)
    linestring = LineString(points)
    return linestring


def multi_line_generator(point_lists: list):
    """
    Create a MultiLineString from a list of list of Points.
    """
    line_list = []
    for point_list in point_lists:
        assert all(isinstance(point, Point) for point in point_list)
        line_list.append(LineString(point_list))
    multilinestring = MultiLineString(line_list)
    return multilinestring


def make_valid_traces() -> List[LineString]:
    """
    Make bunch of valid traces.
    """
    traces = [
        line_generator([Point(-1, 1), Point(3, 1)]),
        line_generator([Point(-2, 0), Point(0, 0)]),
        line_generator([Point(-1, 4), Point(-1, -2)]),
        line_generator([Point(-2, -2), Point(2, -2)]),
        line_generator([Point(1, 0), Point(1, -3)]),
        line_generator([Point(2, 0), Point(-2, 4)]),
        line_generator([Point(-3, 2), Point(3, 2)]),
        line_generator([Point(2, 4), Point(2, 2)]),
        # Next is in target area 2
        # Overlaps on right end, and snaps on left
        line_generator([Point(-3, -4), Point(4, -4)]),
    ]
    return traces


def make_valid_target_areas():
    """
    Make valid target areas.
    """
    areas = [
        box(-3, -3, 3, 4),
        # Polygon([(-3, 4), (3, 4), (3, -3), (-3, -3)]),
        box(-3, -5, 3, -3.5),
        # Polygon([(-3, -3.5), (3, -3.5), (3, -5), (-3, -5)]),
    ]
    return areas


def make_invalid_traces(snap_threshold, snap_threshold_error_multiplier):
    """
    Make invalid traces.
    """
    traces = [
        # Horizontal (triple junction)
        line_generator([Point(-2, 0), Point(2, 0)]),
        # Vertical (triple junction)
        line_generator([Point(0, -2), Point(0, 4)]),
        # Crosses two above to form a triple junction
        line_generator([Point(1, -1), Point(-1, 1)]),
        # Triple junction with non-identical cross-points but within
        # snap threshold (next 3 traces):
        line_generator([Point(0, -3), Point(2, -3)]),
        line_generator([Point(1, -4), Point(1, -2)]),
        line_generator([Point(2, -4), Point(0.5, -2.50001)]),
        # V-node
        line_generator([Point(1, 2), Point(0, 4)]),
        # Intersects next trace three times
        line_generator([Point(-4, -3), Point(-2, -3), Point(-4, -2), Point(-2, -1)]),
        # Straight line which is intersected twice by same line
        line_generator([Point(-3, -4), Point(-3, -1)]),
        # Y-node error. Not snapped with error of 0.005
        # snap_threshold is the snapping threshold.
        # (Distances lower than it are snapped -> shouldnt cause errors)
        # Snap error in next trace is:
        # snap_threshold_error_multiplier * 0.5 * snap_threshold
        line_generator(
            [
                Point(2, 1),
                Point(0 + snap_threshold * 0.95 * snap_threshold_error_multiplier, 1),
            ]
        ),
        # Geometry type is multilinestring but can be merged
        multi_line_generator([(Point(4, -4), Point(4, 0)), (Point(4, 0), Point(4, 4))]),
        # Geometry type is multilinestring but cannot be merged
        multi_line_generator(
            [(Point(3, -4), Point(3, -1)), (Point(3, 0), Point(3, 4))]
        ),
        # Y-node right next to X-node -> triple junction (next 3):
        line_generator([Point(-2, 4), Point(-3, 4)]),
        line_generator([Point(-2.5, 3.5), Point(-3.5, 4.5)]),
        line_generator([Point(-3.5, 3.5), Point(-2.5, 4.5)]),
        # Overlapping snap. Should be caught by MultiJunctionValidator
        line_generator([Point(-2, 2), Point(-4, 2)]),
        line_generator(
            [
                Point(-3, 1),
                Point(-3, 2 + snap_threshold * 0.5 * snap_threshold_error_multiplier),
            ]
        ),
        # Target area underlapping traces (next 2)
        line_generator(
            [
                Point(-6, 4 - snap_threshold * snap_threshold_error_multiplier * 0.95),
                Point(-6, 2),
            ]
        ),
        line_generator(
            [
                Point(-6, -5 + snap_threshold * snap_threshold_error_multiplier * 0.95),
                Point(-6, -2),
            ]
        ),
        # Partially overlapping traces
        line_generator([Point(6, -4), Point(6, 4)]),
        line_generator(
            [
                Point(6, -4),
                Point(6, -3),
            ]
        ),
        # Overlapping within buffer distance
        line_generator([Point(7, -4), Point(7, 4)]),
        line_generator(
            [
                Point(7 + snap_threshold * 0.5, -3.5),
                Point(7 + snap_threshold * 0.5, -3),
            ]
        ),
        # Triangle intersection
        line_generator(
            [
                Point(0, -7),
                Point(0, -5),
            ]
        ),
        line_generator(
            [
                Point(-1, -7),
                Point(0 + snap_threshold, -6),
                Point(-1, -5),
            ]
        ),
    ]
    return traces


def make_invalid_target_areas():
    """
    Make invalid target areas for invalid traces.
    """
    # Two target areas, both contain a trace that underlaps
    areas = [
        box(-8, 0, -5, 4),
        # Polygon([(-8, 4), (-5, 4), (-5, 0), (-8, 0)]),
        box(-8, -5, -5, -1),
        # Polygon([(-8, -1), (-5, -1), (-5, -5), (-8, -5)]),
    ]
    return areas


(
    valid_traces,
    invalid_traces,
    valid_areas_geoseries,
    invalid_areas_geoseries,
) = trace_builder(False, SNAP_THRESHOLD, SNAP_THRESHOLD_ERROR_MULTIPLIER)

valid_error_srs = pd.Series([[] for _ in valid_traces.geometry.values])
invalid_error_srs = pd.Series([[] for _ in invalid_traces.geometry.values])

multilinestring_critical_err_in_validation_gdf = read_geofile(
    Path("tests/sample_data/validation_errror_08042021/Circle3_fractures.shp")
)
multilinestring_critical_err_in_validation_gdf.error_amount = 1

v_node_network_error_gdf = gpd.GeoDataFrame(geometry=v_node_network_error_ls_list)
v_node_network_error_area_gdf = gpd.GeoDataFrame(
    geometry=[bounding_polygon(v_node_network_error_gdf)]
)

hastholmen_traces = read_geofile(Path("tests/sample_data/hastholmen_traces.geojson"))
hastholmen_area = read_geofile(Path("tests/sample_data/hastholmen_area.geojson"))
traces_200k = read_geofile(Path("tests/sample_data/traces_200k.geojson"))
area_200k = read_geofile(Path("tests/sample_data/area_200k.geojson"))
hastholmen_traces_validated = read_geofile(
    Path("tests/sample_data/hastholmen_traces_validated.geojson")
)

geta_lidar_inf_valid_traces = read_geofile(
    Path("tests/sample_data/geta_lidar_lineaments_infinity_traces.geojson")
)
geta_lidar_inf_valid_area = read_geofile(
    Path("tests/sample_data/geta_lidar_lineaments_inf_circular_area.geojson")
)


# def random_data_column(iterable):
#     """
#     Make random data column contents.
#     """
#     return ["aaa" for _ in iterable]


# geoms are all LineStrings and no errors


# def valid_gdf_get():
#     """
#     Get valid gdf.
#     """
#     return gpd.GeoDataFrame(
#         {
#             GEOMETRY_COLUMN: valid_traces,
#             ERROR_COLUMN: valid_error_srs,
#             "random_col": random_data_column(valid_traces),
#             "random_col2": random_data_column(valid_traces),
#             "random_col3": random_data_column(valid_traces),
#             "random_col4": random_data_column(valid_traces),
#         }
#     )


# def invalid_gdf_get():
#     """
#     Get invalid gdf.
#     """
#     return gpd.GeoDataFrame(
#         {
#             GEOMETRY_COLUMN: invalid_traces,
#             ERROR_COLUMN: invalid_error_srs,
#             "random_col": random_data_column(invalid_traces),
#         }
#     )


# def invalid_gdf_null_get():
#     """
#     Get gdf with None and empty geometries.
#     """
#     return gpd.GeoDataFrame(
#         {
#             GEOMETRY_COLUMN: [None, LineString()],
#             ERROR_COLUMN: [[], []],
#             "random_col": random_data_column(range(2)),
#         }
#     )


# def valid_area_gdf_get():
#     """
#     Get a valid area gdf.
#     """
#     return gpd.GeoDataFrame({GEOMETRY_COLUMN: valid_areas_geoseries})


# def invalid_area_gdf_get():
#     """
#     Get an invalid area gdf.
#     """
#     return gpd.GeoDataFrame({GEOMETRY_COLUMN: invalid_areas_geoseries})


faulty_error_srs = pd.Series([[] for _ in valid_traces.geometry.values])
faulty_error_srs[0] = np.nan
faulty_error_srs[1] = "this cannot be transformed to list?"
faulty_error_srs[2] = (1, 2, 3, "hello?")
faulty_error_srs[5] = 5.12315235


# def valid_gdf_with_faulty_error_col_get():
#     """
#     Get valid gdf with faulty error column.
#     """
#     return gpd.GeoDataFrame(
#         {
#             GEOMETRY_COLUMN: valid_traces,
#             ERROR_COLUMN: faulty_error_srs,
#             "random_col": random_data_column(valid_traces),
#         }
#     )


# def iterate_validators():
#     """
#     Iterate over validators.
#     """
#     for validator in (
#         GeomNullValidator,
#         GeomTypeValidator,
#         MultiJunctionValidator,
#         VNodeValidator,
#         MultipleCrosscutValidator,
#         UnderlappingSnapValidator,
#         TargetAreaSnapValidator,
#     ):
#         yield validator


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
# areas_geosrs = gpd.GeoSeries([Polygon([(5, 5), (-5, 5), (-5, -5), (5, -5)])])
areas_geosrs = gpd.GeoSeries([box(-5, -5, 5, 5)])

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


def get_nice_traces():
    """
    Get nice traces GeoSeries.
    """
    return nice_traces.copy()


# def get_traces_geosrs():
#     """
#     Get traces GeoSeries.
#     """
#     return traces_geosrs.copy()


# def get_areas_geosrs():
#     """
#     Get areas GeoSeries.
#     """
#     return areas_geosrs.copy()


def get_geosrs_identicals():
    """
    Get GeoSeries with identical geometries.
    """
    return geosrs_identicals.copy()


line_1 = LineString([(0, 0), (0.5, 0.5)])
line_2 = LineString([(0, 0), (0.5, -0.5)])
line_3 = LineString([(0, 0), (1, 0)])
line_1_sp = Point(list(line_1.coords)[0])
line_2_sp = Point(list(line_2.coords)[0])
line_1_ep = Point(list(line_1.coords)[-1])
line_2_ep = Point(list(line_2.coords)[-1])
halved_azimuths = [
    determine_azimuth(line, halved=True)
    for line in (
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
# area_1 = Polygon([(0, 0), (1, 1), (1, 0)])
area_1 = Polygon([(0, 0), (1, 1), (1, 0), (0, 0)])
# area_frame = gpd.GeoDataFrame({"geometry": [area_1]})

# sample_trace_data = Path("tests/sample_data/KB11_traces.shp")
sample_trace_data = Path("tests/sample_data/KB11/KB11_traces.geojson")
sample_trace_100_data = Path("tests/sample_data/KB11/KB11_traces_100.geojson")
# sample_area_data = Path("tests/sample_data/KB11_area.shp")
sample_area_data = Path("tests/sample_data/KB11/KB11_area.geojson")
kb11_traces = read_geofile(sample_trace_data)
kb11_area = read_geofile(sample_area_data)

kb10_unfit_traces = read_geofile(
    Path("tests/sample_data/kb10_validation_error/kb10_unfit_traces.geojson")
)
kb10_unfit_area = read_geofile(
    Path("tests/sample_data/kb10_validation_error/kb10_unfit_area.geojson")
)
kl2_2_traces = read_geofile(Path("tests/sample_data/kl2_2/kl2_2_traces.geojson"))
kl2_2_area = read_geofile(Path("tests/sample_data/kl2_2/kl2_2_area.geojson"))
multipolygon_traces = read_geofile(
    Path(
        "tests/sample_data/multipolygon_traces_area/"
        "traces_within_multipolygon_target_area.geojson"
    )
)
multipolygon_area = read_geofile(
    Path(
        "tests/sample_data/multipolygon_traces_area/" "multipolygon_target_area.geojson"
    )
)
manypolygon_area = read_geofile(
    Path(
        "tests/sample_data/multipolygon_traces_area/" "many_polygon_target_area.geojson"
    )
)
assert isinstance(kb11_traces, gpd.GeoDataFrame)
assert isinstance(kb11_area, gpd.GeoDataFrame)

kb7_trace_100_path = Path("tests/sample_data/KB7/KB7_tulkinta_100.geojson")
kb7_trace_50_path = Path("tests/sample_data/KB7/KB7_tulkinta_50.geojson")
kb7_area_path = Path("tests/sample_data/KB7/KB7_tulkinta_alue.geojson")

kb7_traces = read_geofile(kb7_trace_50_path)
kb7_area = read_geofile(kb7_area_path)

test_tracevalidate_params = [
    (
        kb7_trace_50_path,  # cut 0-50
        kb7_area_path,
        "--allow-fix",
    ),
    (
        kb7_trace_100_path,  # cut 50-100
        kb7_area_path,
        "--allow-fix",
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
test_determine_proximal_traces_params = [
    (nice_traces, 0.5, 25),
    (nice_traces, 1, 35),
]

test_plot_xyi_plot_params = [
    ([{X_node: 0, Y_node: 0, I_node: 50}], ["title"]),
    ([{X_node: 0, Y_node: 0, I_node: 0}], ["title"]),
    ([{X_node: 0, Y_node: 10, I_node: 25}], [""]),
]

test_plot_branch_plot_params = [
    ([{CC_branch: 30, CI_branch: 15, II_branch: 50}], ["title"], ["black"]),
    ([{CC_branch: 0, CI_branch: 0, II_branch: 50}], ["title"], ["white"]),
    ([{CC_branch: 0, CI_branch: 0, II_branch: 0}], ["title"], None),
]

test_determine_topology_parameters_params_dict = [
    dict(
        trace_length_array=np.array([10, 10, 10, 10]),  # trace_length_array
        node_counts={
            X_node: 3,
            Y_node: 5,
            I_node: 8,
            E_node: 0,
        },  # node_counts dict
        area=10.0,  # area
        branches_defined=True,  # branches_defined
        correct_mauldon=True,  # correct_mauldon
        branch_length_array=np.array([5, 5, 5, 5]),  # branch_length_array
    ),
    dict(
        trace_length_array=np.array([1, 1, 1, 1]),  # trace_length_array
        node_counts={
            X_node: 3,
            Y_node: 5,
            I_node: 8,
            E_node: 0,
        },  # node_counts dict
        area=1.0,  # area
        branches_defined=True,  # branches_defined
        correct_mauldon=True,  # correct_mauldon
        branch_length_array=np.array([0.1, 0.1, 0.1, 0.1]),  # branch_length_array
    ),
]

test_determine_topology_parameters_params = [
    kwargs.values() for kwargs in test_determine_topology_parameters_params_dict
]

test_plot_topology_params = [
    (
        [
            parameters.determine_topology_parameters(  # topology_parameters_list
                **test_determine_topology_parameters_params_dict[0]  # type: ignore
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
    (gpd.GeoSeries([LineString([(0.5, 0.5), (1, 1)]), LineString([(0, 1), (0, -1)])])),
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
            wkt.loads("LINESTRING (0 0, 1 1, 2 2)"),
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
    (
        hastholmen_traces,  # traces
        hastholmen_area,  # area
        "hastholmen_traces",  # name
        True,  # auto_fix
        [],  # assume_errors
    ),
    (
        gpd.GeoDataFrame(
            geometry=make_invalid_traces(
                snap_threshold=0.01, snap_threshold_error_multiplier=1.1
            )
        ),  # traces
        gpd.GeoDataFrame(geometry=make_invalid_target_areas()),  # area
        "invalid_traces",  # name
        True,  # auto_fix
        None,  # assume_errors
    ),
    (
        gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 1)])]),  # traces
        gpd.GeoDataFrame(
            geometry=[
                box(-1, -1, 1, 1.011)
                # Polygon(
                #     [
                #         Point(-1, -1),
                #         Point(-1, 1.011),
                #         Point(1, 1.011),
                #         Point(1, -1),
                #     ]
                # )
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
                box(-1, -1, 1, 1.011),
                # Polygon(
                #     [
                #         Point(-1, -1),
                #         Point(-1, 1.011),
                #         Point(1, 1.011),
                #         Point(1, -1),
                #     ]
                # ),
                box(2, 2, 6, 6.011),
                # Polygon(
                #     [
                #         Point(2, 2),
                #         Point(2, 6.011),
                #         Point(6, 6.011),
                #         Point(6, 2),
                #     ]
                # ),
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
                        box(-1, -1, 1, 1.011),
                        # Polygon(
                        #     [
                        #         Point(-1, -1),
                        #         Point(-1, 1.011),
                        #         Point(1, 1.011),
                        #         Point(1, -1),
                        #     ]
                        # ),
                        box(2, 2, 6, 6.011),
                        # Polygon(
                        #     [
                        #         Point(2, 2),
                        #         Point(2, 6.011),
                        #         Point(6, 6.011),
                        #         Point(6, 2),
                        #     ]
                        # ),
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
                        box(0, 0, 1, 1),
                        # Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
                        box(10, 10, 11, 11),
                        # Polygon([(10, 10), (10, 11), (11, 11), (11, 10)]),
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
                        box(0, 0, 1, 1),
                        # Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
                        box(10, 10, 11, 11),
                        # Polygon([(10, 10), (10, 11), (11, 11), (11, 10)]),
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
                box(0, 0, 1, 1),
                # Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
                box(10, 10, 11, 11),
                # Polygon([(10, 10), (10, 11), (11, 11), (11, 10)]),
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
            geometry=[
                # Polygon([(5, 5), (-5, 5), (-5, -5), (5, -5)])
                box(-5, -5, 5, 5),
            ]
        ),  # area:gpd.GeoDataFrame
        0.01,  # snap_threshold: float,
        1.1,  # snap_threshold_error_multiplier: float,
        1.5,  # area_edge_snap_multiplier: float,
        True,  # assumed_result: bool,
    ),  # Test that traces coming from outside area are not marked as underlapping
    (
        LineString([(10, 0), (5.011, 0)]),  # geom: LineString,
        gpd.GeoDataFrame(
            geometry=[
                box(-5, -5, 5, 5),
                # Polygon([(5, 5), (-5, 5), (-5, -5), (5, -5)])
            ]
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
            # "tests/sample_data/KB7/KB7_tulkinta_50.shp",  # cut 0-50
            # "tests/sample_data/KB7/KB7_tulkinta_alue.shp",
            str(kb7_trace_50_path),  # cut 0-50
            str(kb7_area_path),
            "--allow-fix",
            "--only-area-validation",
        ]  # args
    )
]

geta_1_traces = read_geofile(
    Path("tests/sample_data/geta1/Getaberget_20m_1_traces.gpkg")
)
geta_1_1_area = read_geofile(
    Path("tests/sample_data/geta1/Getaberget_20m_1_1_area.gpkg")
)

geta_1_traces_1000_n = geta_1_traces.iloc[0:1000]

test_network_params = [
    (
        geta_1_traces,  # traces
        geta_1_1_area,  # area
        "Geta1_1",  # name
        True,  # determine_branches_nodes
        True,  # truncate_traces
        0.001,  # snap_threshold
        True,  # circular_target_area
        False,  # try_export_of_data
    ),
    (
        kb11_traces,  # traces
        kb11_area,  # area
        "KB11",  # name
        True,  # determine_branches_nodes
        True,  # truncate_traces
        0.001,  # snap_threshold
        False,  # circular_target_area
        False,  # try_export_of_data
    ),
    (
        kb11_traces.iloc[0:100],  # traces
        kb11_area,  # area
        "KB11_0_100",  # name
        True,  # determine_branches_nodes
        True,  # truncate_traces
        0.001,  # snap_threshold
        False,  # circular_target_area
        True,  # try_export_of_data
    ),
    (
        v_node_network_error_gdf,  # traces
        v_node_network_error_area_gdf,  # area
        "v-node-error-network",  # name
        True,  # determine_branches_nodes
        True,  # truncate_traces
        0.001,  # snap_threshold
        False,  # circular_target_area
        False,  # try_export_of_data
    ),
    (
        multipolygon_traces,  # traces
        multipolygon_area,  # area
        "MultiPolygon_target_area",  # name
        True,  # determine_branches_nodes
        True,  # truncate_traces
        0.001,  # snap_threshold
        False,  # circular_target_area
        False,  # try_export_of_data
    ),
    (
        multipolygon_traces,  # traces
        multipolygon_area,  # area
        "MultiPolygon_target_area",  # name
        True,  # determine_branches_nodes
        False,  # truncate_traces
        0.001,  # snap_threshold
        False,  # circular_target_area
        False,  # try_export_of_data
    ),
    (
        multipolygon_traces,  # traces
        manypolygon_area,  # area
        "MultiPolygon_target_area",  # name
        True,  # determine_branches_nodes
        True,  # truncate_traces
        0.001,  # snap_threshold
        False,  # circular_target_area
        False,  # try_export_of_data
    ),
    (
        kl2_2_traces.iloc[0:1500],  # traces
        kl2_2_area,  # area
        "kl_2_2",  # name
        True,  # determine_branches_nodes
        True,  # truncate_traces
        0.001,  # snap_threshold
        False,  # circular_target_area
        False,  # try_export_of_data
    ),
]

test_network_random_sampler_params = [
    (
        geta_1_traces,  # trace_gdf
        geta_1_1_area,  # area_gdf
        10,  # min_radius
        0.001,  # snap_threshold
        1,  # samples
        "area",  # random_choice
    ),
    (
        geta_1_traces,  # trace_gdf
        geta_1_1_area,  # area_gdf
        10,  # min_radius
        0.001,  # snap_threshold
        1,  # samples
        "radius",  # random_choice
    ),
]

test_describe_powerlaw_fit_params = [
    (
        geta_1_traces.geometry.length.values,  # lengths
        "traces",  # label
    )
]

test_determine_boundary_intersecting_lines_params = [
    (
        gpd.GeoDataFrame(
            geometry=[
                LineString([(0, 0), (0, 5)]),
                LineString([(0, 0), (0, 3)]),
                LineString([(0, 0), (0, 10)]),
            ]
        ),  # line_gdf
        gpd.GeoDataFrame(
            geometry=[
                box(-5, -5, 5, 5),
                # Polygon([(-5, 5), (5, 5), (5, -5), (-5, -5)]),
            ]
        ),  # area_gdf
        0.01,  # snap_threshold
        np.array([True, False, True]),  # assumed_result_inter
        np.array([False, False, False]),  # assumed_result_cuts
    ),
    (
        gpd.GeoDataFrame(
            geometry=[
                LineString([(0, -50), (0, 5)]),
                LineString([(0, 0), (0, 3)]),
                LineString([(0, 0), (0, 10)]),
            ]
        ),  # line_gdf
        gpd.GeoDataFrame(
            geometry=[
                box(-5, -5, 5, 5),
                # Polygon([(-5, 5), (5, 5), (5, -5), (-5, -5)]),
            ]
        ),  # area_gdf
        0.01,  # snap_threshold
        np.array([True, False, True]),  # assumed_result_inter
        np.array([True, False, False]),  # assumed_result_cuts
    ),
    (
        gpd.GeoDataFrame(
            geometry=[
                LineString([(0, -50), (0, -5.00001)]),
            ]
        ),  # line_gdf
        gpd.GeoDataFrame(
            geometry=[
                # Polygon([(-5, 5), (5, 5), (5, -5), (-5, -5)]),
                box(-5, -5, 5, 5),
            ]
        ),  # area_gdf
        0.01,  # snap_threshold
        np.array([True]),  # assumed_result_inter
        np.array([False]),  # assumed_result_cuts
    ),
    (
        gpd.GeoDataFrame(
            geometry=[
                LineString([(0, -4.999), (0, 4.999)]),
            ]
        ),  # line_gdf
        gpd.GeoDataFrame(
            geometry=[
                # Polygon([(-5, 5), (5, 5), (5, -5), (-5, -5)]),
                box(-5, -5, 5, 5),
            ]
        ),  # area_gdf
        0.01,  # snap_threshold
        np.array([True]),  # assumed_result_inter
        np.array([True]),  # assumed_result_cuts
    ),
    (
        gpd.GeoDataFrame(
            geometry=[
                LineString([(0, 0), (0, 1)]),
                LineString([(10, 0), (10, 1)]),
            ]
        ),  # line_gdf
        gpd.GeoDataFrame(
            geometry=[
                box(-1, -1, 1, 1),
                # Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)]),
                box(9, -1, 11, 1),
                # Polygon([(9, -1), (11, -1), (11, 1), (9, 1)]),
            ]
        ),  # area_gdf
        0.01,  # snap_threshold
        np.array([True, True]),  # assumed_result_inter
        np.array([False, False]),  # assumed_result_cuts
    ),
    (
        gpd.GeoDataFrame(
            geometry=[
                LineString([(-2, 0), (2, 1)]),
                LineString([(8, 0), (12, 0)]),
            ]
        ),  # line_gdf
        gpd.GeoDataFrame(
            geometry=[
                box(-1, -1, 1, 1),
                # Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)]),
                box(9, -1, 11, 1),
                # Polygon([(9, -1), (11, -1), (11, 1), (9, 1)]),
            ]
        ),  # area_gdf
        0.01,  # snap_threshold
        np.array([True, True]),  # assumed_result_inter
        np.array([True, True]),  # assumed_result_cuts
    ),
]

test_network_circular_target_area_params = [
    (
        gpd.GeoDataFrame(
            geometry=[
                LineString([(0, 0), (0, 10)]),  # one intersect
                LineString([(1, 0), (1, 3)]),  # no intersect
                LineString([(4, -10), (4, 10)]),  # two intersect
            ]
        ),  # trace_gdf
        gpd.GeoDataFrame(
            geometry=[
                # Polygon([(-5, -5), (-5, 5), (5, 5), (5, -5)]),
                box(-5, -5, 5, 5),
            ]
        ),  # area_gdf
        "circular_target_area_param_test",  # name
    )
]

test_snap_trace_to_another_params = [
    (
        [Point(0, 0), Point(0, 4)],  # trace_endpoints
        LineString([(-5, 5), (5, 5)]),  # another
        1.1,  # snap_threshold
        LineString([(-5, 5), (0, 4), (5, 5)]),  # another
    ),
    (
        [Point(0, 0), Point(0, 4)],  # trace_endpoints
        LineString([(-5, 5), (5, 5)]),  # another
        0.1,  # snap_threshold
        LineString([(-5, 5), (5, 5)]),  # another
    ),
]

test_insert_point_to_linestring_params = [
    (
        LineString([(0, 0), (1, 1), (2, 2)]),
        Point(0.5, 0.5),
        0.01,  # snap_threshold
        None,
    ),
    (
        LineString([(0, 0), (-1, -1), (-2, -2)]),
        Point(-0.5, -0.5),
        0.01,  # snap_threshold
        None,
    ),
    (
        LineString([(0, 0), (-1, -1), (-2, -2)]),
        Point(-1.0, -1.1),
        0.11,  # snap_threshold
        LineString([(0, 0), (-1.0, -1.1), (-2, -2)]),
    ),
    (
        LineString([(0, 0), (-1, -1), (-1.5, -1.5), (-2, -2)]),
        Point(-1.0, -1.1),
        0.11,  # snap_threshold
        LineString([(0, 0), (-1.0, -1.1), (-1.5, -1.5), (-2, -2)]),
    ),
    (
        LineString([(0, 0), (-1.5, -1.5), (-2, -2)]),
        Point(-1.5, -1.5),
        0.11,  # snap_threshold
        LineString([(0, 0), (-1.5, -1.5), (-2, -2)]),
    ),
]

sample_traces_path = Path("tests/sample_data/branches_and_nodes/traces.gpkg")
sample_areas_path = Path("tests/sample_data/branches_and_nodes/areas.gpkg")
sample_traces = read_geofile(sample_traces_path)
sample_areas = read_geofile(sample_areas_path)

test_branches_and_nodes_regression_params = [
    (
        sample_traces,  # traces
        sample_areas,  # areas
        0.001,  # snap_threshold
        10,  # allowed_loops
        False,  # already_clipped
    )
]

troubling_traces_path = Path(
    "tests/sample_data/branches_and_nodes/traces_troubling.gpkg"
)
troubling_traces = read_geofile(troubling_traces_path)

troubling_upper = troubling_traces.geometry.values[1]
troubling_middle = troubling_traces.geometry.values[0]
troubling_lower = troubling_traces.geometry.values[2]

test_simple_snap_params = [
    (
        troubling_upper,  # trace
        gpd.GeoSeries([troubling_middle, troubling_lower]),  # trace_candidates
        0.001,  # snap_threshold
        0,  # intersects_idx
    ),
    (
        troubling_lower,  # trace
        gpd.GeoSeries([troubling_middle, troubling_upper]),  # trace_candidates
        0.001,  # snap_threshold
        0,  # intersects_idx
    ),
]

test_snap_trace_simple_params = [
    (
        0,  # idx
        troubling_upper,  # trace
        0.001,  # snap_threshold
        [troubling_upper, troubling_middle, troubling_upper],  # traces
        1,  # intersects_idx
    )
]

unary_err_traces_path = Path("tests/sample_data/unary_error_data/err_traces.shp")
unary_err_areas_path = Path("tests/sample_data/unary_error_data/err_area.shp")
unary_err_traces = gpd.read_file(unary_err_traces_path).iloc[5500:8000]
unary_err_areas = gpd.read_file(unary_err_areas_path)
assert isinstance(unary_err_traces, gpd.GeoDataFrame)
assert isinstance(unary_err_areas, gpd.GeoDataFrame)

test_safer_unary_union_params = [
    (
        unary_err_traces.geometry,  # traces_geosrs
        0.001,  # snap_threshold
        13000,  # size_threshold
    ),
    (
        unary_err_traces.geometry,  # traces_geosrs
        0.001,  # snap_threshold
        50,  # size_threshold
    ),
]

test_segment_within_buffer_params = [
    (valid_geom, invalid_geom_multilinestring, 0.001, 1.1, 50, True),
    (valid_geom, mergeable_geom_multilinestring, 0.001, 1.1, 50, True),
    (
        valid_geom,
        MultiLineString([LineString([(10, 10), (50, 50)])]),
        0.001,
        1.1,
        50,
        False,
    ),
]

test_segmentize_linestring_params = [
    (LineString(((0, 0), (0, 1))), 0.1, 10),
    (LineString(((0, 0), (1, 1))), 0.1, 15),
    (LineString(((0, 0), (0, 1))), 1, 1),
]

test_split_to_determine_triangle_errors_params = [
    (
        LineString([(-1, 0), (0, 2), (1, 0)]),  # trace
        LineString([(-1, 1.99), (0, 1.99), (1, 1.99)]),  # splitter_trace
        0.001,  # snap_threshold
        50,  # triangle_error_snap_multiplier
        True,  # assumed_result
    ),
    (
        LineString([(-1, 0), (0, 5), (1, 0)]),  # trace
        LineString([(-1, 1.99), (0, 1.99), (1, 1.99)]),  # splitter_trace
        0.001,  # snap_threshold
        50,  # triangle_error_snap_multiplier
        False,  # assumed_result
    ),
    (
        LineString([(-1, 0), (0, 1.98), (1, 0)]),  # trace
        LineString([(-1, 1.99), (0, 1.99), (1, 1.99)]),  # splitter_trace
        0.001,  # snap_threshold
        50,  # triangle_error_snap_multiplier
        False,  # assumed_result
    ),
]

test_determine_middle_in_triangle_params = [
    (
        [
            LineString([(0, 0), (0, 1)]),
            LineString([(0, 1), (0, 2)]),
            LineString([(0, 2), (0, 3)]),
        ],  # segments
        0.001,  # snap_threshold
        1.1,  # snap_threshold_error_multiplier
        [
            LineString([(0, 1), (0, 2)]),
        ],  # assumed_result
    ),
    (
        [
            LineString([(0, 0), (0, 1)]),
            LineString([(0, 1), (0, 2)]),
            LineString([(0, 2), (0, 3)]),
            LineString([(0, 3), (0, 4)]),
        ],  # segments
        0.001,  # snap_threshold
        1.1,  # snap_threshold_error_multiplier
        [
            LineString([(0, 1), (0, 2)]),
            LineString([(0, 2), (0, 3)]),
        ],  # assumed_result
    ),
    (
        [
            LineString([(0, 0), (0, 1)]),
            LineString([(0, 2), (0, 3)]),
        ],  # segments
        0.001,  # snap_threshold
        1.1,  # snap_threshold_error_multiplier
        [],  # assumed_result
    ),
]

test_network_contour_grid_params = [
    (
        kb11_traces,  # traces
        kb11_area,  # areas
        0.001,  # snap_threshold
        "kb11_traces",
    ),
]

test_report_snapping_loop_params = [
    (
        5,  # loop
        10,  # allowed_loops
        False,  # will_error
    ),
    (
        11,  # loop
        10,  # allowed_loops
        True,  # will_error
    ),
]


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
    ),
    multilinestring_critical_err_in_validation_gdf,
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
    gpd.GeoDataFrame(geometry=results_in_false_positive_stacked_traces_list),
    gpd.GeoDataFrame(geometry=results_in_false_pos_stacked_traces_list),
    kb10_unfit_traces,
]

known_non_overlapping_gdfs = [gpd.GeoDataFrame(geometry=results_in_overlapping_ls_list)]

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


def generate_known_params(error, false_positive):
    """
    Generate pytest.params.
    """
    knowns: List[gpd.GeoDataFrame] = (
        known_errors[error] if not false_positive else known_false_positives[error]
    )
    amounts = [
        (gdf.shape[0] if not hasattr(gdf, "error_amount") else gdf.error_amount)
        if error
        not in (
            UnderlappingSnapValidator._UNDERLAPPING,
            UnderlappingSnapValidator._OVERLAPPING,
        )
        else 1
        for gdf in knowns
    ]
    try:
        areas = [gpd.GeoDataFrame(geometry=[bounding_polygon(gdf)]) for gdf in knowns]
    except (ValueError, AttributeError, GEOSException):
        areas = [
            gpd.GeoDataFrame(
                geometry=[
                    Polygon([(0, 0), (1, 1), (1, 0), (0, 0)])
                    # box(0,0,1,1)
                ]
            )
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


def get_all_errors():
    """
    Get the defined errors.
    """
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
            all_errs.extend(generate_known_params(err, false_positive=False))
        except KeyError:
            pass
        try:
            all_errs.extend(generate_known_params(err, false_positive=True))
        except KeyError:
            pass

    assert len(all_errs) > 0
    return all_errs


@lru_cache(maxsize=None)
def kb11_traces_lengths():
    """
    Get trace lengths of KB11.
    """
    return kb11_traces.geometry.length.values


@lru_cache(maxsize=None)
def kb11_area_value():
    """
    Get area value of KB11.
    """
    return sum(kb11_area.geometry.area)


@lru_cache(maxsize=None)
def hastholmen_traces_lengths():
    """
    Get trace lengths of hastholmen infinity lineaments.
    """
    return hastholmen_traces.geometry.length.values


@lru_cache(maxsize=None)
def hastholmen_area_value():
    """
    Get area value of hastholmen.
    """
    return sum(hastholmen_area.geometry.area)


@lru_cache(maxsize=None)
def traces_200k_lengths():
    """
    Get trace lengths of 200k lineaments.
    """
    return traces_200k.geometry.length.values


@lru_cache(maxsize=None)
def area_200k_value():
    """
    Get area value of hastholmen.
    """
    return sum(area_200k.geometry.area)


@lru_cache(maxsize=None)
def test_populate_sample_cell_new_params():
    """
    Params for test_populate_sample_cell_new.
    """
    return [
        (
            box(0, 0, 2, 2),
            gpd.GeoDataFrame(geometry=[LineString([(-5, 1), (5, 1)])]),
            0.001,
            gpd.GeoDataFrame(geometry=[LineString([(-5, 1), (5, 1)])]),  # branches
        )
    ]


@lru_cache(maxsize=None)
def test_multinetwork_params():
    """
    Params for test_multinetwork.
    """
    return [
        (
            (
                dict(
                    trace_gdf=geta_1_traces_1000_n,
                    area_gdf=geta_1_1_area,
                    name="geta1_1",
                    circular_target_area=True,
                    snap_threshold=0.001,
                ),
                dict(
                    trace_gdf=geta_1_traces_1000_n,
                    area_gdf=geta_1_1_area,
                    name="geta1_2",
                    circular_target_area=True,
                    snap_threshold=0.001,
                ),
            ),
            1,
            5.0,
        ),
    ]


def test_ternary_heatmapping_params():
    """
    Params for test_ternary_heatmapping.
    """
    return [
        (
            np.array([0.2, 0.8, 0.1]),
            np.array([0.4, 0.1, 0.4]),
            np.array([0.4, 0.1, 0.5]),
            15,
        ),
        (
            np.array([0.4, 0.1, 0.4]),
            np.array([0.2, 0.8, 0.1]),
            np.array([0.4, 0.1, 0.5]),
            15,
        ),
    ]


def test_normalize_fit_to_area_params():
    """
    Params for test_normalize_fit_to_area.
    """
    return [
        length_distributions.LengthDistribution(
            name="kb11",
            lengths=kb11_traces_lengths(),
            area_value=kb11_area_value(),
            using_branches=False,
        ),
        length_distributions.LengthDistribution(
            name="kb11_50",
            lengths=kb11_traces_lengths()[0:50],
            area_value=kb11_area_value(),
            using_branches=False,
        ),
    ]


# def test_concat_length_distributions_params():
#     """
#     Params for test_concat_length_distributions.
#     """
#     return [
#         ([kb11_traces_lengths()], [kb11_area_value()], ["kb11_full"]),
#         ([kb11_traces_lengths()[0:50]], [kb11_area_value()], ["kb11_50"]),
#         (
#             [kb11_traces_lengths(), hastholmen_traces_lengths()],
#             [kb11_area_value(), hastholmen_area_value()],
#             ["kb11_full", "hastholmen_full"],
#         ),
#     ]


def test_fit_to_multi_scale_lengths_params():
    """
    Params for test_fit_to_multi_scale_lengths.
    """
    return [
        [
            length_distributions.LengthDistribution(
                name="kb11",
                lengths=kb11_traces_lengths(),
                area_value=kb11_area_value(),
                using_branches=False,
            ),
            length_distributions.LengthDistribution(
                name="hastholmen",
                lengths=hastholmen_traces_lengths(),
                area_value=hastholmen_area_value(),
                using_branches=False,
            ),
            length_distributions.LengthDistribution(
                name="200k",
                lengths=traces_200k_lengths(),
                area_value=area_200k_value(),
                using_branches=False,
            ),
        ]
    ]


def test_aggregate_chosen_params():
    """
    Params for test_aggregate_chosen.
    """
    return [
        (
            [
                {
                    general.Param.AREA.value.name: 2000.0,
                    general.Param.CIRCLE_COUNT.value.name: 20.0,
                    general.Param.FRACTURE_INTENSITY_P21.value.name: 2.0,
                    general.Param.CONNECTIONS_PER_BRANCH.value.name: 1.2,
                },
                {
                    general.Param.AREA.value.name: 4000.0,
                    general.Param.CIRCLE_COUNT.value.name: 40.0,
                    general.Param.FRACTURE_INTENSITY_P21.value.name: 2.0,
                    general.Param.CONNECTIONS_PER_BRANCH.value.name: 1.4,
                },
            ],
            float,
            {
                general.Param.AREA.value.name: 6000.0,
                general.Param.CIRCLE_COUNT.value.name: 60.0,
                general.Param.FRACTURE_INTENSITY_P21.value.name: 2.0,
                general.Param.CONNECTIONS_PER_BRANCH.value.name: 1.3333333333333333,
            },
        ),
        (
            [
                {
                    general.Param.AREA.value.name: 2000.0,
                    "random-column": "Some name?",
                },
                {
                    general.Param.AREA.value.name: 4000.0,
                    "random-column": "Some other name?",
                },
            ],
            (float, str),
            {
                general.Param.AREA.value.name: 6000.0,
                "random-column": str(["Some name?", "Some other name?"]),
            },
        ),
    ]


def test_random_sample_of_circles_params():
    """
    Params for test_random_sample_of_circles.
    """
    return [
        (
            {
                "geta7": [
                    {
                        general.Param.AREA.value.name: 2000.0,
                        general.Param.CIRCLE_COUNT.value.name: 20.0,
                        general.Param.FRACTURE_INTENSITY_P21.value.name: 2.0,
                        general.Param.CONNECTIONS_PER_BRANCH.value.name: 1.2,
                    },
                    {
                        general.Param.AREA.value.name: 4000.0,
                        general.Param.CIRCLE_COUNT.value.name: 40.0,
                        general.Param.FRACTURE_INTENSITY_P21.value.name: 2.0,
                        general.Param.CONNECTIONS_PER_BRANCH.value.name: 1.4,
                    },
                ]
            },
            {"geta7": 12.5},
            1,
            None,
        ),
        (
            {
                "geta7": [
                    {
                        general.Param.AREA.value.name: 2000.0,
                        general.Param.CIRCLE_COUNT.value.name: 20.0,
                        general.Param.FRACTURE_INTENSITY_P21.value.name: 2.0,
                        general.Param.CONNECTIONS_PER_BRANCH.value.name: 1.2,
                    },
                    {
                        general.Param.AREA.value.name: 4000.0,
                        general.Param.CIRCLE_COUNT.value.name: 40.0,
                        general.Param.FRACTURE_INTENSITY_P21.value.name: 2.0,
                        general.Param.CONNECTIONS_PER_BRANCH.value.name: 1.4,
                    },
                ],
                "geta6": [
                    {
                        general.Param.AREA.value.name: 2000.0,
                        general.Param.CIRCLE_COUNT.value.name: 20.0,
                        general.Param.FRACTURE_INTENSITY_P21.value.name: 2.0,
                        general.Param.CONNECTIONS_PER_BRANCH.value.name: 1.2,
                    },
                    {
                        general.Param.AREA.value.name: 4000.0,
                        general.Param.CIRCLE_COUNT.value.name: 40.0,
                        general.Param.FRACTURE_INTENSITY_P21.value.name: 2.0,
                        general.Param.CONNECTIONS_PER_BRANCH.value.name: 1.4,
                    },
                ],
            },
            {"geta7": 12.5, "geta6": 5.0},
            2,
            None,
        ),
    ]


def test_collect_indexes_of_base_circles_params():
    """
    Params for test_collect_indexes_of_base_circles.
    """
    return [
        ([1, 2, 3], 1, [10, 10, 20]),
        ([1, 2, 4], 1, [10, 10, 30]),
        ([100, 2323, 10000], 2, [10, 10, 30]),
        ([100, 2323, 10000], 3, [10, 10, 30]),
    ]


@lru_cache(maxsize=None)
def test_multinetwork_plot_multi_length_distribution_slow_params():
    """
    Params for test_multinetwork_plot_multi_length_distribution_slow.
    """
    return [
        [
            dict(
                trace_gdf=geta_1_traces,
                area_gdf=geta_1_1_area,
                name="geta_1_1",
                circular_target_area=True,
                snap_threshold=0.001,
            ),
            dict(
                trace_gdf=geta_lidar_inf_valid_traces,
                area_gdf=geta_lidar_inf_valid_area,
                name="geta_lidar_inf",
                circular_target_area=True,
                snap_threshold=0.001,
            ),
        ],
    ]


KB11_NETWORK_PARAMS = dict(
    trace_gdf=kb11_traces,
    area_gdf=kb11_area,
    name="kb11",
    circular_target_area=False,
    snap_threshold=0.001,
)
KB7_NETWORK_PARAMS = dict(
    trace_gdf=kb7_traces,
    area_gdf=kb7_area,
    name="kb7",
    circular_target_area=False,
    snap_threshold=0.001,
)
HASTHOLMEN_VALID_NETWORK_PARAMS = dict(
    trace_gdf=hastholmen_traces_validated,
    area_gdf=hastholmen_area,
    name="hastholmen",
    circular_target_area=False,
    snap_threshold=0.001,
)


@lru_cache(maxsize=None)
def test_multinetwork_plot_multi_length_distribution_fast_params():
    """
    Params for test_multinetwork_plot_multi_length_distribution_fast.
    """
    return [
        [KB11_NETWORK_PARAMS, KB7_NETWORK_PARAMS],
    ]


@lru_cache(maxsize=None)
def test_multinetwork_plot_azimuth_set_lengths_params():
    """
    Params for test_multinetwork_plot_multi_length_distribution_fast.
    """
    return [
        [KB11_NETWORK_PARAMS, HASTHOLMEN_VALID_NETWORK_PARAMS],
    ]


# @lru_cache(maxsize=None)
# def test_plot_mld_optimized_params():
#     """
#     Params for test_plot_mld_optimized.
#     """
#     return [
#         (
#             [kb11_traces_lengths(), hastholmen_traces_lengths()],
#             [kb11_area_value(), hastholmen_area_value()],
#             ["kb11", "hastholmen"],
#         )
#     ]


def test_plot_azimuth_plot_params():
    """
    Params for test_plot_azimuth_plot.
    """
    return [
        (
            # np.linspace(0, 360, 100),
            np.random.normal(180, 90, 100),
            np.ones(100),
            np.ones(100),
            np.ones(100).astype(str),
            "PlotWithNonAxialAzimuthData",
            False,
            False,
            False,
            False,
            "white",
        ),
        (
            np.random.normal(90, 45, 100),
            # np.linspace(0, 180, 100),
            np.ones(100),
            np.ones(100),
            np.ones(100).astype(str),
            "PlotWithAxialAzimuthData",
            False,
            False,
            True,
            True,
            "red",
        ),
        (
            np.random.normal(90, 45, 100),
            # np.linspace(0, 180, 100),
            np.ones(100),
            np.ones(100),
            np.ones(100).astype(str),
            "PlotWithAxialAzimuthData",
            True,
            True,
            True,
            True,
            "red",
        ),
    ]


def round_geometry_coordinates(geom: Any) -> Any:
    """
    Round shapely geometry coordinates using wkt.dumps and wkt.loads.
    """
    rounded = geom
    try:
        rounded = wkt.loads(wkt.dumps(geom, rounding_precision=6))
    except ValueError:
        logging.error(f"Expected for wkt to be able to parse geom: {geom}")
    return rounded
