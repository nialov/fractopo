"""
PorePy interoperability helpers.

This module covers using fractopo functionality to create PorePy-compatible
input files and input parameters. Model building in PorePy is not covered;
PorePy is not a fractopo dependency.
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
    NULL_SET,
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


class _FractureSetSamples(NamedTuple):
    """Aligned usable observations grouped by azimuth set."""

    azimuth: dict[str, NDArray[np.floating]]
    dip: dict[str, NDArray[np.floating]]
    length: dict[str, NDArray[np.floating]]
    proportions: dict[str, float]


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
        axis = ls.length / 2
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
        axis = length / 2
        major_axis = minor_axis = axis
        major_axis_angle = 0.0
        strike_angle_rad = np.deg2rad(strike_angle)
        dip_angle_rad = np.deg2rad(dip)
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
def _extract_network_fracture_set_samples(network) -> _FractureSetSamples:
    """Extract usable, aligned trace observations from a ``Network``.

    Only the calculated trace-data arrays and the configured set names are
    used for set membership.  Dip is the sole orientation value read from
    ``trace_gdf``; dip direction is intentionally not part of this boundary.
    """
    trace_data = network.trace_data
    labels = np.asarray(trace_data.azimuth_set_array)
    azimuth = np.asarray(trace_data.azimuth_array, dtype=float)
    length = np.asarray(trace_data.length_array, dtype=float)

    if "dip" not in network.trace_gdf:
        raise ValueError("no usable fracture set remains: trace_gdf is missing dip")
    try:
        dip = np.asarray(network.trace_gdf["dip"], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "no usable fracture set remains: dip is not numeric"
        ) from error

    if not (len(labels) == len(azimuth) == len(length) == len(dip)):
        raise ValueError(
            "no usable fracture set remains: trace data arrays are misaligned"
        )

    arrays = {name: ([], [], []) for name in network.azimuth_set_names}
    for label, azi, dip_value, trace_length in zip(labels, azimuth, dip, length):
        if (
            label == NULL_SET
            or label not in arrays
            or not np.isfinite(azi)
            or not np.isfinite(dip_value)
            or not 0 <= dip_value <= 90
            or not np.isfinite(trace_length)
            or trace_length <= 0
        ):
            continue
        arrays[label][0].append(azi)
        arrays[label][1].append(dip_value)
        arrays[label][2].append(trace_length)

    usable = {
        name: tuple(np.asarray(values, dtype=float) for values in grouped)
        for name, grouped in arrays.items()
        if grouped[0]
    }
    if not usable:
        raise ValueError("no usable named fracture set remains")
    count = sum(len(values[0]) for values in usable.values())

    return _FractureSetSamples(
        azimuth={name: values[0] for name, values in usable.items()},
        dip={name: values[1] for name, values in usable.items()},
        length={name: values[2] for name, values in usable.items()},
        proportions={name: len(values[0]) / count for name, values in usable.items()},
    )


@beartype
def export_network_to_porepy_3d_csv_format(
    network,
    fracture_count: int,
    bounds: tuple[
        SupportsFloat,
        SupportsFloat,
        SupportsFloat,
        SupportsFloat,
        SupportsFloat,
        SupportsFloat,
    ],
    rng: Optional[np.random.Generator] = None,
) -> str:
    """Export a Network's empirical fracture population for PorePy 3D loading.

    The v1 source is a ``Network``.  Trace observations are grouped by the
    named sets in ``network.azimuth_set_names``; set quotas are proportional
    to usable-row counts, and rows are sampled with replacement.  Pass a
    caller-owned ``numpy.random.Generator`` for reproducible output::

        csv_text = export_network_to_porepy_3d_csv_format(
            network,
            fracture_count=100,
            bounds=(0, 0, 0, 10, 10, 10),
            rng=np.random.default_rng(42),
        )
        Path("fractures.csv").write_text(csv_text)

    ``bounds`` are ordered ``(xmin, ymin, zmin, xmax, ymax, zmax)``.  The
    first output row is the PorePy cuboid domain; subsequent rows have the
    eight fields ``CENTER_X,CENTER_Y,CENTER_Z,MAJOR_AXIS,MINOR_AXIS,
    MAJOR_AXIS_ANGLE,STRIKE_ANGLE,DIP_ANGLE``.  Centres are sampled inside
    the cuboid independently, but fracture disks are not guaranteed to fit
    inside it.  Circular semi-axes are half the sampled observed trace
    length.  The trace azimuth is used as strike (in radians); PorePy's
    right-hand rule implies the dip direction, so no dip-direction field is
    read or emitted.

    Fitted powerlaw distributions are intentionally not sampled in v1:
    empirical resampling is reproducible from the supplied Generator, while
    powerlaw ``generate_random`` uses global NumPy randomness and does not
    honour ``xmax``.  PorePy is not imported by this function.  Load the
    resulting file in PorePy with, for example (not executed here)::

        pp.network_from_csv(path, has_domain=True)
    """
    if (
        isinstance(fracture_count, bool)
        or not isinstance(fracture_count, (int, np.integer))
        or fracture_count <= 0
    ):
        raise ValueError("fracture_count must be a positive integer")
    if len(bounds) != 6:
        raise ValueError("bounds must contain xmin, ymin, zmin, xmax, ymax, zmax")
    bounds = tuple(float(value) for value in bounds)
    if not all(np.isfinite(value) for value in bounds):
        raise ValueError("bounds must be finite")
    xmin, ymin, zmin, xmax, ymax, zmax = bounds
    if not (xmin < xmax and ymin < ymax and zmin < zmax):
        raise ValueError("bounds must be strictly increasing on every axis")

    samples = _extract_network_fracture_set_samples(network)
    generator = np.random.default_rng() if rng is None else rng
    names = list(samples.proportions)
    raw = np.array([samples.proportions[name] * fracture_count for name in names])
    quotas = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - quotas), kind="stable")[
        : fracture_count - quotas.sum()
    ]:
        quotas[index] += 1

    rows = [f"{xmin},{ymin},{zmin},{xmax},{ymax},{zmax}"]
    for name, quota in zip(names, quotas):
        for _ in range(quota):
            source = int(generator.choice(len(samples.length[name])))
            center = (
                generator.uniform(xmin, xmax),
                generator.uniform(ymin, ymax),
                generator.uniform(zmin, zmax),
            )
            fracture = EllipticalFracture(
                center_x=center[0],
                center_y=center[1],
                center_z=center[2],
                major_axis=samples.length[name][source] / 2,
                minor_axis=samples.length[name][source] / 2,
                major_axis_angle=0.0,
                strike_angle_rad=np.deg2rad(samples.azimuth[name][source]),
                dip_angle_rad=np.deg2rad(samples.dip[name][source]),
            )
            rows.append(fracture.to_csv_row())
    return "\n".join(rows)


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
    >>> s_geoms, *_ = scale_geometries_to_local([ls1, ls2], y_scale=100)
    >>> s_geoms
    [<LINESTRING (0 0, 100 100)>, <LINESTRING (0 50, 100 50)>]

    >>> p1 = Point(0, 0)
    >>> p2 = Point(0, 5)
    >>> s_geoms, *_ = scale_geometries_to_local([p1, p2], y_scale=100)
    >>> s_geoms
    [<POINT (0 0)>, <POINT (0 100)>]
    """
    # TODO: Extract resolve_scale_factor as separate function for use in e.g. test_porepy.py
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
def prepare_geometries_for_export(
    geometries: Union[list, gpd.GeoDataFrame],
    y_scale: Optional[SupportsFloat],
) -> tuple[list, float, float, float, float, float]:
    """
    Standardizes geometry preparation (scaling and bounds) for PorePy export routines.

    >>> p_geoms, *_ = prepare_geometries_for_export([Point(1, 2), Point(2, 5)], y_scale=None)
    >>> p_geoms
    [<POINT (1 2)>, <POINT (2 5)>]

    >>> p_geoms, *_ = prepare_geometries_for_export([Point(1, 2), Point(2, 5)], y_scale=10)
    >>> p_geoms
    [<POINT (0 0)>, <POINT (3.333 10)>]

    >>> l = [LineString([(0, 0), (2, 2)]), LineString([(1, 5), (5, 6)])]
    >>> p_geoms, *_ = prepare_geometries_for_export(l, y_scale=10)
    >>> p_geoms
    [<LINESTRING (0 0, 3.333 3.333)>, <LINESTRING (1.667 8.333, 8.333 10)>]
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
        x_min, y_min, x_max, y_max = gpd.array.from_shapely(geoms_list).total_bounds
        scale = 1.0

    return scaled, x_min, y_min, x_max, y_max, scale


@beartype
def export_traces_to_porepy_2d_csv_format(
    traces: GeoDataFrameWithLineStrings,
    y_scale: Optional[SupportsFloat] = None,
    include_domain: bool = True,
) -> str:
    """
    Export traces to a 2D CSV format compatible with PorePy's
    ``network_from_csv``.

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
    log.info(
        "prepare_geometries_for_export output: geometries=%r, x_min=%.6f, y_min=%.6f, x_max=%.6f, y_max=%.6f, scale=%.6f",
        linestrings,
        x_min,
        y_min,
        x_max,
        y_max,
        scale,
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


def _elliptical_fractures_to_csv(elliptical_fractures) -> str:
    return "\n".join(map(EllipticalFracture.to_csv_row, elliptical_fractures))


def _default_z_values(z_values, count: int) -> np.ndarray:
    return np.zeros(count) if z_values is None else z_values


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
    Export structural measurements to a 3D CSV format compatible with
    PorePy's ``network_from_csv``.
    """

    points, x_min, y_min, x_max, y_max, scale = prepare_geometries_for_export(
        measurement_points, y_scale=y_scale
    )
    log.info(
        "prepare_geometries_for_export output: geometries=%r, x_min=%.6f, y_min=%.6f, x_max=%.6f, y_max=%.6f, scale=%.6f",
        points,
        x_min,
        y_min,
        x_max,
        y_max,
        scale,
    )

    z_values = _default_z_values(z_values, len(points))

    log.info(
        "Exporting %d structural measurement points as 3D elliptic fractures",
        len(points),
    )

    elliptical_fractures = starmap(
        EllipticalFracture.from_structural_measurement,
        zip_equal(points, z_values, dip_values, dip_direction_values, length_values),
    )
    return _elliptical_fractures_to_csv(elliptical_fractures)


@beartype
def export_traces_to_porepy_3d_csv_format(
    traces: Union[GeoDataFrameWithLineStrings, list[LineString]],
    dip_values: NDArrayWithDips,
    y_scale: Optional[SupportsFloat],
    # TODO: Domain, if wanted, should be calculated from 3D ellipse extents?
    # include_domain: bool = True,
    z_values: Optional[NDArray] = None,
) -> str:
    """
    Export traces to a 3D CSV format compatible with PorePy's
    ``network_from_csv``.

    >>> traces = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])])
    >>> dip_values = np.array([45])
    >>> print(export_traces_to_porepy_3d_csv_format(traces, dip_values, y_scale=None))
    0.5,0.5,0.0,0.7071067811865476,0.7071067811865476,0.0,0.7853981633974483,0.7853981633974483
    """
    linestrings, x_min, y_min, x_max, y_max, scale = prepare_geometries_for_export(
        geometries=traces,
        y_scale=y_scale,
    )
    log.info(
        "Prepared geometries: geometries count=%s, x_min=%.6f, y_min=%.6f, x_max=%.6f, y_max=%.6f, scale=%.6f",
        len(linestrings),
        x_min,
        y_min,
        x_max,
        y_max,
        scale,
    )

    z_values = _default_z_values(z_values, len(linestrings))
    log.info("Exporting %d traces as 3D elliptic fractures", len(traces))

    elliptical_fractures = starmap(
        EllipticalFracture.from_linestring_and_orientation,
        zip_equal(linestrings, z_values, dip_values),
    )
    return _elliptical_fractures_to_csv(elliptical_fractures)
