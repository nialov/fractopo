"""
Interoperability with ``porepy``
"""

import logging
from itertools import starmap

import geopandas as gpd
import numpy as np
import shapely
from beartype import beartype
from beartype.typing import NamedTuple, Optional, SupportsFloat, Union
from numpy.typing import NDArray
from shapely.geometry import LineString, Point

from fractopo.general import (
    calc_strike,
    determine_azimuth,
    get_trace_endpoints,
    zip_equal,
)
from fractopo.typing import (
    GeoDataFrameWithLineStrings,
    GeoDataFrameWithPoints,
    NDArrayWithDipDirections,
    NDArrayWithDips,
    NDArrayWithPositives,
)

log = logging.getLogger(__name__)

POREPY_2D_CSV_LINES_HEADER = "# OBJECTID,START_X,START_Y,END_X,END_Y"
POREPY_2D_CSV_DOMAIN_HEADER = "# Domain X_MIN, Y_MIN, X_MAX, Y_MAX"


class EllipticalFracture(NamedTuple):
    center_x: SupportsFloat
    center_y: SupportsFloat
    center_z: SupportsFloat
    major_axis: SupportsFloat
    minor_axis: SupportsFloat
    major_axis_angle: SupportsFloat
    strike_angle_rad: SupportsFloat
    dip_angle_rad: SupportsFloat

    @classmethod
    def from_linestring_and_orientation(
        cls, ls: LineString, center_z: SupportsFloat, dip: SupportsFloat
    ) -> "EllipticalFracture":
        bary = ls.interpolate(0.5, normalized=True)
        azimuth = determine_azimuth(ls, halved=True)
        strike_angle = convert_azimuth_to_strike(azimuth)
        center_x, center_y = bary.x, bary.y
        axis = ls.length
        major_axis = minor_axis = axis
        major_axis_angle = 0.0
        strike_angle_rad = np.deg2rad(strike_angle)
        dip_angle_rad = np.deg2rad(float(dip))
        return cls(
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
            major_axis=major_axis,
            minor_axis=minor_axis,
            major_axis_angle=major_axis_angle,
            strike_angle_rad=strike_angle_rad,
            dip_angle_rad=dip_angle_rad,
        )

    @classmethod
    def from_structural_measurement(
        cls,
        point: Point,
        center_z: SupportsFloat,
        dip: SupportsFloat,
        dip_direction: SupportsFloat,
        length: SupportsFloat,
    ) -> "EllipticalFracture":
        strike_angle = calc_strike(dip_direction=dip_direction)
        center_x, center_y = point.x, point.y
        axis = length
        major_axis = minor_axis = axis
        major_axis_angle = 0.0
        strike_angle_rad = np.deg2rad(strike_angle)
        dip_angle_rad = np.deg2rad(float(dip))
        return cls(
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
            major_axis=major_axis,
            minor_axis=minor_axis,
            major_axis_angle=major_axis_angle,
            strike_angle_rad=strike_angle_rad,
            dip_angle_rad=dip_angle_rad,
        )

    def to_csv_row(self) -> str:
        return ",".join(
            map(
                str,
                (
                    self.center_x,
                    self.center_y,
                    self.center_z,
                    self.major_axis,
                    self.minor_axis,
                    self.major_axis_angle,
                    self.strike_angle_rad,
                    self.dip_angle_rad,
                ),
            )
        )


@beartype
def get_geometry_bounds(geoms: list) -> tuple[float, float, float, float]:
    """
    Returns (min_x, min_y, max_x, max_y) bounds for a list of shapely geometries using geopandas fast routines.
    """
    if not geoms:
        raise ValueError("Empty geometry list.")
    bounds = gpd.array.from_shapely(geoms).total_bounds
    x_min, y_min, x_max, y_max, *_ = bounds
    return x_min, y_min, x_max, y_max


@beartype
def scale_geometries_to_local(
    geometries: list, y_scale: SupportsFloat
) -> tuple[
    list, SupportsFloat, SupportsFloat, SupportsFloat, SupportsFloat, SupportsFloat
]:
    """
    Scales a list of shapely geometries so that the y-axis spans y_scale units,
    preserving the aspect ratio.

    >>> ls1 = LineString([(0, 0), (10, 10)])
    >>> ls2 = LineString([(0, 5), (10, 5)])
    >>> scale_geometries_to_local([ls1, ls2], y_scale=100)
    [<LINESTRING (0 0, 100 100)>, <LINESTRING (0 50, 100 50)>]

    >>> ls1 = LineString([(25500000, 6670000), (25500100, 6670200)])
    >>> ls2 = LineString([(25500050, 6670100), (25500150, 6670300)])
    >>> scale_geometries_to_local([ls1, ls2], y_scale=100)
    [<LINESTRING (0 0, 33.333 66.667)>, <LINESTRING (16.667 33.333, 50 100)>]

    >>> p1 = Point(0, 0)
    >>> p2 = Point(0, 5)
    >>> scale_geometries_to_local([p1, p2], y_scale=100)
    [<POINT (0 0)>, <POINT (0 100)>]
    """
    x_min, y_min, x_max, y_max, *_ = gpd.array.from_shapely(geometries).total_bounds
    y_range = y_max - y_min
    if y_range == 0:
        log.error("All y values are the same; cannot scale.")
        raise ValueError("All y values are the same; cannot scale.")
    scale = y_scale / y_range

    def _scaler(x, y):
        return (x - x_min) * scale, (y - y_min) * scale

    scaled_geometries = [
        shapely.transform(geom, _scaler, interleaved=False) for geom in geometries
    ]
    return scaled_geometries, x_min, y_min, x_max, y_max, scale


@beartype
def check_porepy_2d_csv_format(csv_text: str) -> bool:
    """
    Check that input csv file uses porepy 2d csv format.

    Returns True if format matches, raises ValueError if not.
    """

    def _is_float(val: str) -> bool:
        try:
            float(val)
            return True
        except ValueError:
            return False

    lines = [line.strip() for line in csv_text.splitlines() if not line.startswith("#")]

    # Domain line (skip comments)
    try:
        domain_idx = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    except StopIteration:
        raise ValueError("No domain line found after comments.")

    first_fields = [x.strip() for x in lines[domain_idx].split(",")]
    log.info("Domain line fields: %s", first_fields)

    if len(first_fields) != 4 or not all(_is_float(x) for x in first_fields):
        raise ValueError(f"Domain line must have 4 float values, got: {first_fields}")

    # Data lines: all remaining non-comment lines after header
    for i, line in enumerate(lines[domain_idx + 1 :]):
        if line.startswith("#"):
            continue
        fields = [x.strip() for x in line.split(",")]
        if len(fields) != 4 or not all(_is_float(x) for x in fields):
            raise ValueError(f"Row {i} must have 4 float values, got: {fields}")

    return True


@beartype
def determine_scale(ys, y_scale) -> float:
    min_y, max_y = min(ys), max(ys)
    y_range = max_y - min_y
    if y_range == 0:
        log.error("All y values are the same; cannot scale.")
        raise ValueError("All y values are the same; cannot scale.")
    scale = y_scale / y_range
    return scale


@beartype
def prepare_geometries_for_export(
    geometries: Union[list, gpd.GeoDataFrame],
    y_scale: Optional[SupportsFloat],
) -> tuple[list, float, float, float, float, float]:
    """
    Standardizes geometry preparation (scaling and bounds) for PorePy export routines.

    Returns a 6-tuple:
      (geoms, x_min, y_min, x_max, y_max, scale)

    >>> from shapely.geometry import Point, LineString
    >>> prepare_geometries_for_export([Point(1, 2), Point(2, 5)], y_scale=None)
    ([<POINT (1 2)>, <POINT (2 5)>], 1.0, 2.0, 2.0, 5.0, 1.0)
    >>> prepare_geometries_for_export([Point(1, 2), Point(2, 5)], y_scale=10)
    ([<POINT (0 0)>, <POINT (10 10)>], 1.0, 2.0, 2.0, 5.0, 3.3333333333333335)

    >>> l = [LineString([(0, 0), (2, 2)]), LineString([(1, 5), (5, 6)])]
    >>> r = prepare_geometries_for_export(l, y_scale=10)
    >>> all(g.__class__.__name__.startswith('LineString') for g in r[0])
    True
    """
    # Convert GeoDataFrame to geometry list
    if isinstance(geometries, gpd.GeoDataFrame):
        geoms_list = list(geometries.geometry)
    else:
        geoms_list = list(geometries)

    # Optionally scale
    if y_scale is not None:
        scaled, x_min, y_min, x_max, y_max, scale = scale_geometries_to_local(
            geoms_list, y_scale
        )
    else:
        scaled = geoms_list
        x_min, y_min, x_max, y_max = get_geometry_bounds(geoms_list)
        scale = 1.0

    return scaled, x_min, y_min, x_max, y_max, scale


@beartype
def export_traces_to_porepy_2d_csv_format(
    traces: GeoDataFrameWithLineStrings,
    y_scale: Optional[SupportsFloat] = None,
    include_domain: bool = True,
) -> str:
    """
    Export traces to a 2D CSV format compatible with PorePy's `network_from_csv`.

    >>> traces = gpd.GeoDataFrame(geometry=[
    ...     LineString([(0, 0), (1, 1)]),
    ...     LineString([(1, 0), (0, 1)])
    ... ])
    >>> print(export_traces_to_porepy_2d_csv_format(traces))
    0.0,0.0,1.0,1.0
    0.0,0.0,1.0,1.0
    1.0,0.0,0.0,1.0
    """
    linestrings, x_min, y_min, x_max, y_max, scale = prepare_geometries_for_export(
        geometries=traces, y_scale=y_scale
    )

    # Collect endpoints only (no IDs)
    entries = list(
        starmap(
            lambda start_point, end_point: (
                start_point.x,
                start_point.y,
                end_point.x,
                end_point.y,
            ),
            map(get_trace_endpoints, linestrings),
        )
    )
    # Build CSV lines to satisfy PorePy's importer requirements
    out_lines = [f"{x_min},{y_min},{x_max},{y_max}"] if include_domain else []
    for x0, y0, x1, y1 in entries:
        out_lines.append(f"{x0},{y0},{x1},{y1}")

    output_csv = "\n".join(out_lines)

    check_porepy_2d_csv_format(output_csv)
    return output_csv


def convert_azimuth_to_strike(azimuth: float) -> float:
    if azimuth <= 90:
        return 90 - azimuth
    return 360 - (azimuth - 90)


@beartype
def export_structural_measurements_to_porepy_3d_csv_format(
    dip_values: NDArrayWithDips,
    dip_direction_values: NDArrayWithDipDirections,
    length_values: NDArrayWithPositives,
    measurement_points: Union[GeoDataFrameWithPoints, list[Point]],
    y_scale: Optional[SupportsFloat],
    z_values: Optional[NDArray[np.floating]] = None,
) -> str:
    """
    Export structural measurements to a 3D CSV format compatible with PorePy's `network_from_csv`.
    """

    points, *_ = prepare_geometries_for_export(measurement_points, y_scale=y_scale)

    # Assume a constant Z=0 for each point's center unless a z column exists
    if z_values is None:
        log.info(
            "No z_values passed, assuming constant z coordinate of 0.0 for all points"
        )
        z_values = np.zeros(len(points))

    log.info(
        "Exporting %d structural measurement points as 3D elliptic fractures",
        len(points),
    )

    elliptical_fractures = starmap(
        EllipticalFracture.from_structural_measurement,
        zip_equal(points, z_values, dip_values, dip_direction_values, length_values),
    )
    csv_lines = map(EllipticalFracture.to_csv_row, elliptical_fractures)

    output_csv = "\n".join(csv_lines)
    return output_csv


@beartype
def export_traces_to_porepy_3d_csv_format(
    traces: Union[GeoDataFrameWithLineStrings, list[LineString]],
    dip_values: NDArrayWithDips,
    y_scale: Optional[SupportsFloat],
    # TODO: Domain, if wanted, should be calculated from 3D ellipse extents?
    # include_domain: bool = True,
    z_values: Optional[NDArray[np.floating]] = None,
) -> str:
    """
    Export traces to a 3D CSV format compatible with PorePy's `network_from_csv`.
    """
    linestrings, x_min, y_min, x_max, y_max, scale = prepare_geometries_for_export(
        geometries=traces,
        y_scale=y_scale,
    )

    # Assume a constant Z=0 for each line's center unless a z column exists
    if z_values is None:
        log.info(
            "No z_values passed, assuming constant z coordinate of 0.0 for all traces"
        )
        z_values = np.zeros(len(linestrings))

    log.info("Exporting %d traces as 3D elliptic fractures", len(traces))

    elliptical_fractures = starmap(
        EllipticalFracture.from_linestring_and_orientation,
        zip_equal(linestrings, z_values, dip_values),
    )
    csv_lines = map(EllipticalFracture.to_csv_row, elliptical_fractures)

    output_csv = "\n".join(csv_lines)
    return output_csv
