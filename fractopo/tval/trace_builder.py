import geopandas as gpd
import shapely
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from typing import List

from shapely.geometry import LineString, MultiLineString, Point


def main(plot_figs=False, snap_threshold=0.001, snap_threshold_error_multiplier=1.1):
    """
    Creates two GeoSeries of traces. Saves plots of these traces into ./figs.

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
    valid_savepath = Path("figs/valid_geoseries.png")

    invalid_traces = make_invalid_traces(
        snap_threshold, snap_threshold_error_multiplier
    )
    invalid_geoseries = gpd.GeoSeries(invalid_traces)
    invalid_areas = make_invalid_target_areas()
    invalid_areas_geoseries = gpd.GeoSeries(invalid_areas)
    invalid_savepath = Path("figs/invalid_geoseries.png")

    if plot_figs:
        plot_geoseries(valid_geoseries, valid_areas_geoseries, valid_savepath)
        plot_geoseries(invalid_geoseries, invalid_areas_geoseries, invalid_savepath)
    return [
        valid_geoseries,
        invalid_geoseries,
        valid_areas_geoseries,
        invalid_areas_geoseries,
    ]


def line_generator(points: list):
    """
    Creates a shapely.geometry.LineString from a list of
    shapely.geometry.Point objects
    """
    assert all([isinstance(point, Point) for point in points])
    linestring = LineString(points)
    return linestring


def multi_line_generator(point_lists: list):
    """
    Creates a shapely.geometry.MultiLineString from a list of
    shapely.geometry.Point in another list
    """
    line_list = []
    for point_list in point_lists:
        assert all([isinstance(point, Point) for point in point_list])
        line_list.append(LineString(point_list))
    multilinestring = MultiLineString(line_list)
    return multilinestring


def plot_geoseries(
    geoseries: gpd.GeoSeries, area_geoseries: gpd.GeoSeries, savepath: Path
):
    fig, ax = plt.subplots()
    geoseries.plot(ax=ax)
    area_geoseries.plot(ax=ax, edgecolor="red", facecolor="none", linewidth=2)
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(base=1.0))

    plt.grid()
    plt.savefig(savepath)


def make_valid_traces() -> List[LineString]:
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
    areas = [
        shapely.geometry.Polygon([(-3, 4), (3, 4), (3, -3), (-3, -3)]),
        shapely.geometry.Polygon([(-3, -3.5), (3, -3.5), (3, -5), (-3, -5)]),
    ]
    return areas


def make_invalid_traces(snap_threshold, snap_threshold_error_multiplier):
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
    # Two target areas, both contain a trace that underlaps
    areas = [
        shapely.geometry.Polygon([(-8, 4), (-5, 4), (-5, 0), (-8, 0)]),
        shapely.geometry.Polygon([(-8, -1), (-5, -1), (-5, -5), (-8, -5)]),
    ]
    return areas


if __name__ == "__main__":
    main(plot_figs=True)
