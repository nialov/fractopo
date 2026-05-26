import logging
from itertools import starmap

import geopandas as gpd
import numpy as np
import shapely
from beartype import beartype
from beartype.typing import Optional, SupportsFloat, Union
from numpy.typing import NDArray
from shapely.geometry import LineString

from fractopo.general import get_trace_endpoints
from fractopo.typing import (
    GeoDataFrameWithLineStrings,
    NDArrayWithDipDirections,
    NDArrayWithDips,
)

log = logging.getLogger(__name__)

POREPY_2D_CSV_LINES_HEADER = "# OBJECTID,START_X,START_Y,END_X,END_Y"
POREPY_2D_CSV_DOMAIN_HEADER = "# Domain X_MIN, Y_MIN, X_MAX, Y_MAX"


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
def scale_linestrings_to_local(
    linestrings: list[LineString], y_scale: SupportsFloat
) -> list[LineString]:
    """
    Scales a list of LineStrings so that the y-axis spans y_scale units,
    preserving the aspect ratio. The minimum x and y are mapped to (0, 0).

    >>> ls1 = LineString([(0, 0), (10, 10)])
    >>> ls2 = LineString([(0, 5), (10, 5)])
    >>> scale_linestrings_to_local([ls1, ls2], y_scale=100)
    [<LINESTRING (0 0, 100 100)>, <LINESTRING (0 50, 100 50)>]

    >>> ls1 = LineString([(25500000, 6670000), (25500100, 6670200)])
    >>> ls2 = LineString([(25500050, 6670100), (25500150, 6670300)])
    >>> scale_linestrings_to_local([ls1, ls2], y_scale=100)
    [<LINESTRING (0 0, 33.333 66.667)>, <LINESTRING (16.667 33.333, 50 100)>]

    """
    all_coords = [coord for ls in linestrings for coord in ls.coords]
    xs, ys, *_ = zip(*all_coords)
    min_x = min(xs)
    min_y, max_y = min(ys), max(ys)
    y_range = max_y - min_y
    if y_range == 0:
        log.error("All y values are the same; cannot scale.")
        raise ValueError("All y values are the same; cannot scale.")
    scale = y_scale / y_range

    log.info(
        "Scaling %d LineStrings: min_x=%.3f, min_y=%.3f, max_y=%.3f, y_range=%.3f, scale=%.3f",
        len(linestrings),
        min_x,
        min_y,
        max_y,
        y_range,
        scale,
    )

    def _scaler(x, y):
        return (x - min_x) * scale, (y - min_y) * scale

    scaled_linestrings = [
        shapely.transform(linestring, _scaler, interleaved=False)
        for linestring in linestrings
    ]
    log.info("Scaling complete. Output %d LineStrings.", len(scaled_linestrings))
    return scaled_linestrings


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
    original_linestrings: list[LineString] = traces.geometry.tolist()

    if y_scale is not None:
        log.info("Scaling linestrings to local coordinates with y_scale: %s", y_scale)
        linestrings = scale_linestrings_to_local(original_linestrings, y_scale=y_scale)
    else:
        linestrings = original_linestrings

    # Compute bounds (domain)
    x_min, y_min, x_max, y_max = gpd.array.from_shapely(linestrings).total_bounds
    log.info("Computed domain: %s", (x_min, y_min, x_max, y_max))

    # When scaling, set y_min/y_max exactly as in the normalization
    if y_scale is not None:
        y_max = y_scale
        y_min = 0

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


@beartype
def export_traces_to_porepy_3d_csv_format(
    traces: Union[GeoDataFrameWithLineStrings, list[LineString]],
    dip_values: NDArrayWithDips,
    y_scale: Optional[SupportsFloat],
    dip_direction_values: Optional[NDArrayWithDipDirections] = None,
    # TODO: Domain, if wanted, should be calculated from 3D ellipse extents?
    # include_domain: bool = True,
    z_values: Optional[NDArray[np.floating]] = None,
) -> str:
    """
    Export traces to a 3D CSV format compatible with PorePy's `network_from_csv`.
    """
    if isinstance(traces, gpd.GeoDataFrame):
        original_linestrings = traces.geometry.tolist()
    else:
        original_linestrings = traces

    if y_scale is not None:
        log.info("Scaling linestrings to local coordinates with y_scale: %s", y_scale)
        linestrings = scale_linestrings_to_local(original_linestrings, y_scale=y_scale)
    else:
        linestrings = original_linestrings

    # Assume a constant Z=0 for each line's center unless a z column exists
    if z_values is None:
        log.info(
            "No z_values passed, assuming constant z coordinate of 0.0 for all traces"
        )
        z_values = np.zeros(len(linestrings))

    log.info("Exporting %d traces as 3D elliptic fractures", len(traces))

    csv_lines = []

    for idx, (ls, dip, dip_dir) in enumerate(
        zip(linestrings, traces["dip"], traces["dip_direction"])
    ):
        bary = ls.interpolate(0.5, normalized=True)
        azimuth = fractopo.general.determine_azimuth(ls, halved=True)
        strike_angle = convert_azimuth_to_strike(azimuth)
        center_x, center_y = bary.x, bary.y
        center_z = z_vals.iloc[idx] if hasattr(z_vals, "iloc") else z_vals[idx]
        axis = ls.length
        major_axis = minor_axis = axis
        major_axis_angle = 0.0
        strike_angle_rad = np.deg2rad(strike_angle)
        dip_angle_rad = np.deg2rad(dip)
        line = f"{center_x},{center_y},{center_z},{major_axis},{minor_axis},{major_axis_angle},{strike_angle_rad},{dip_angle_rad}"
        csv_lines.append(line)

    output_csv = "\n".join(csv_lines)
    log.info("3D CSV contains %d lines", len(csv_lines))
    return output_csv
