import tempfile
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import geopandas as gpd
import marimo as mo
import pyogrio

from fractopo.analysis.length_distributions import Dist
from fractopo.analysis.network import Network
from fractopo.general import Number, write_geodataframe

IGNORED_LAYERS = frozenset({"layer_styles"})

ENABLE_VERBOSE_BASE = "Enable verbose debug output? {}"
UPLOAD_TRACE_DATA_BASE = "## Upload trace data: {}"
UPLOAD_AREA_DATA_BASE = UPLOAD_TRACE_DATA_BASE.replace("trace", "area")
TRACE_LAYER_NAME_BASE = "Trace layer name, if applicable: {}"
AREA_LAYER_NAME_BASE = TRACE_LAYER_NAME_BASE.replace("Trace", "Area")


def make_data_upload_prompts(
    input_traces_file, input_area_file, input_trace_layer_name, input_area_layer_name
):
    return map(
        mo.md,
        (
            UPLOAD_TRACE_DATA_BASE.format(input_traces_file),
            TRACE_LAYER_NAME_BASE.format(input_trace_layer_name),
            UPLOAD_AREA_DATA_BASE.format(input_area_file),
            AREA_LAYER_NAME_BASE.format(input_area_layer_name),
        ),
    )


def _check_empty_geodataframe(gdf: gpd.GeoDataFrame, label: str):
    """Raise an explicit error if a GeoDataFrame is empty."""
    if gdf.empty:
        raise ValueError(
            f"The provided {label} layer is empty (0 features). "
            f"Please check that you uploaded the correct file and specified the correct layer name."
        )


def _resolve_layer_name(path: Path, layer: Optional[str]) -> Optional[str]:
    """Resolve the layer name for a spatial file.

    If ``layer`` is provided, return it unchanged. Otherwise, enumerate layers
    with :func:`pyogrio.list_layers`, filter out non-spatial and ignored
    layers, and auto-select the layer when exactly one spatial layer exists.
    """
    if layer is not None:
        return layer

    layers = pyogrio.list_layers(path)
    # layers is an ndarray of (name, geometry_type) pairs
    spatial_layers = [
        name
        for name, geom_type in layers
        if geom_type is not None and name not in IGNORED_LAYERS
    ]

    if len(spatial_layers) == 1:
        return spatial_layers[0]
    if len(spatial_layers) == 0:
        raise ValueError(
            f"No spatial layers found in {path}. "
            f"Available layers: {[name for name, _ in layers]}"
        )
    raise ValueError(
        f"Multiple spatial layers found in {path}: {spatial_layers}. "
        f"Please specify a layer name."
    )


def read_spatial_app_input(
    input_spatial_file: mo.ui.file, input_spatial_layer_name: mo.ui.text
) -> Tuple[Optional[str], gpd.GeoDataFrame]:
    input_spatial_file_path = Path(input_spatial_file.name())
    # if input_spatial_file_path.name.endswith(".gdb.zip"):
    #     read_file = partial(gpd.read_file, driver="/vsizip/OpenFileGDB/")
    # elif input_spatial_file_path.suffix == ".zip":
    #     read_file = partial(gpd.read_file, driver="/vsizip/")
    # else:
    read_file = gpd.read_file
    layer_name = (
        input_spatial_layer_name.value if input_spatial_layer_name.value != "" else None
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        tmp_data_path = tmp_dir_path.joinpath(input_spatial_file_path)
        tmp_data_path.write_bytes(input_spatial_file.contents())
        layer_name = _resolve_layer_name(tmp_data_path, layer_name)
        gdf = read_file(
            tmp_data_path,
            layer=layer_name,
        )
    return layer_name, gdf


def _parse_app_base_args(
    input_traces_file: mo.ui.file,
    input_trace_layer_name: mo.ui.text,
    input_area_file: mo.ui.file,
    input_area_layer_name: mo.ui.text,
    input_snap_threshold: mo.ui.text,
) -> tuple[str, gpd.GeoDataFrame, gpd.GeoDataFrame, float, str]:
    """Base argument parsing (traces, area, name, snap_threshold, layer) for app UI functions."""
    (traces_gdf, trace_layer_name), (area_gdf, _) = read_traces_and_area(
        input_traces_file=input_traces_file,
        input_trace_layer_name=input_trace_layer_name,
        input_area_file=input_area_file,
        input_area_layer_name=input_area_layer_name,
    )
    snap_threshold = float(input_snap_threshold.value)
    name = resolve_name(
        input_traces_file=input_traces_file, trace_layer_name=trace_layer_name
    )
    return name, traces_gdf, area_gdf, snap_threshold, trace_layer_name


def parse_network_app_args(
    input_trace_layer_name: mo.ui.text,
    input_area_layer_name: mo.ui.text,
    input_traces_file: mo.ui.file,
    input_area_file: mo.ui.file,
    input_snap_threshold: mo.ui.text,
    input_contour_grid_cell_size: mo.ui.text,
    input_define_azimuth_sets: mo.ui.switch,
    input_azimuth_set_ranges: mo.ui.array,
    input_azimuth_set_names: mo.ui.array,
    input_fits_to_plot: mo.ui.multiselect,
) -> Tuple[
    str,
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
    float,
    Optional[float],
    Sequence[Tuple[Number, Number]],
    Sequence[str],
    Tuple[Dist, ...],
]:
    name, traces_gdf, area_gdf, snap_threshold, _ = _parse_app_base_args(
        input_traces_file,
        input_trace_layer_name,
        input_area_file,
        input_area_layer_name,
        input_snap_threshold,
    )
    contour_grid_cell_size = (
        None
        if input_contour_grid_cell_size.value == ""
        else float(input_contour_grid_cell_size.value)
    )
    if input_define_azimuth_sets.value:
        print("Using user-defined azimuth sets")
        azimuth_set_ranges: List[Tuple[Number, Number]] = [
            tuple(range_values) for range_values in input_azimuth_set_ranges.value
        ]
        azimuth_set_names = tuple(input_azimuth_set_names.value)
    else:
        print("Using default azimuth sets")
        azimuth_set_ranges = Network.azimuth_set_ranges
        azimuth_set_names = Network.azimuth_set_names
    fits_to_plot = tuple(map(Dist, input_fits_to_plot.value))
    print(f"Snap threshold: {snap_threshold}")
    if contour_grid_cell_size is not None:
        print(f"Contour grid cell size: {contour_grid_cell_size}")
    print(f"Name: {name}")
    print(f"Azimuth set ranges: {azimuth_set_ranges}")
    print(f"Azimuth set names: {azimuth_set_names}")
    print(f"Length distribution fits to plot: {fits_to_plot}")
    return (
        name,
        traces_gdf,
        area_gdf,
        snap_threshold,
        contour_grid_cell_size,
        azimuth_set_ranges,
        azimuth_set_names,
        fits_to_plot,
    )


def capture_function_outputs(
    func: Callable,
) -> Tuple[Optional[Any], Optional[Exception], str]:
    tmp_io_stdout_and_stderr = StringIO()
    exc = None
    results = None
    try:
        with (
            redirect_stdout(tmp_io_stdout_and_stderr),
            redirect_stderr(tmp_io_stdout_and_stderr),
        ):
            results = func()
    except Exception as exception:
        exc = exception
    stderr_and_stdout = tmp_io_stdout_and_stderr.getvalue()
    return results, exc, stderr_and_stdout


def _write_dir_contents_to_zip(tmp_dir_path: Path) -> BytesIO:
    zip_io = BytesIO()
    # Open an in-memory zip file
    with zipfile.ZipFile(zip_io, mode="a") as zip_file:
        for path in tmp_dir_path.rglob("*"):
            # Do not add directories, only files
            if path.is_dir():
                continue
            path_rel = path.relative_to(tmp_dir_path)

            # Write file in-memory to zip file
            zip_file.write(path, arcname=path_rel)
    # Move to start of file
    zip_io.seek(0)
    return zip_io


def network_results_to_download_element(
    network: Optional[Network],
    name: str,
    contour_grid_cell_size: float,
    fits_to_plot: Tuple[Dist, ...],
):
    if network is None:
        return None
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        # Create and write plots to tmp_dir
        network.export_network_analysis(
            output_path=tmp_dir_path,
            contour_grid_cell_size=contour_grid_cell_size,
            fits_to_plot=fits_to_plot,
        )

        zip_io = _write_dir_contents_to_zip(tmp_dir_path=tmp_dir_path)

    download_element = mo.download(
        data=zip_io,
        filename=f"{name}.zip",
        mimetype="application/octet-stream",
    )
    return download_element


def read_traces_and_area(
    input_traces_file: mo.ui.file,
    input_trace_layer_name: mo.ui.text,
    input_area_file: mo.ui.file,
    input_area_layer_name: mo.ui.text,
) -> Tuple[
    Tuple[gpd.GeoDataFrame, Optional[str]], Tuple[gpd.GeoDataFrame, Optional[str]]
]:
    trace_layer_name, traces_gdf = read_spatial_app_input(
        input_spatial_file=input_traces_file,
        input_spatial_layer_name=input_trace_layer_name,
    )
    area_layer_name, area_gdf = read_spatial_app_input(
        input_spatial_file=input_area_file,
        input_spatial_layer_name=input_area_layer_name,
    )
    _check_empty_geodataframe(traces_gdf, "traces")
    _check_empty_geodataframe(area_gdf, "area")
    print(
        f"Trace layer name: {trace_layer_name}"
        if trace_layer_name is not None
        else "No trace layer specified"
    )
    print(
        f"Area layer name: {area_layer_name}"
        if area_layer_name is not None
        else "No area layer specified"
    )

    return (traces_gdf, trace_layer_name), (area_gdf, area_layer_name)


def resolve_name(input_traces_file: mo.ui.file, trace_layer_name: str) -> str:
    return (
        input_traces_file.name().split(".")[0]
        if trace_layer_name is None
        else trace_layer_name
    )


def parse_validation_app_args(
    input_trace_layer_name: mo.ui.text,
    input_area_layer_name: mo.ui.text,
    input_traces_file: mo.ui.file,
    input_area_file: mo.ui.file,
    input_snap_threshold: mo.ui.text,
):
    name, traces_gdf, area_gdf, snap_threshold, _ = _parse_app_base_args(
        input_traces_file,
        input_trace_layer_name,
        input_area_file,
        input_area_layer_name,
        input_snap_threshold,
    )
    print(f"Snap threshold: {snap_threshold}")
    return name, traces_gdf, area_gdf, snap_threshold


def validated_clean_to_download_element(validated_clean, name):
    if validated_clean is None:
        return None
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        write_geodataframe(
            geodataframe=validated_clean, name=name, results_dir=tmp_dir_path
        )
        zip_io = _write_dir_contents_to_zip(tmp_dir_path=tmp_dir_path)
    download_element = mo.download(
        data=zip_io,
        filename=f"{name}_validated.zip",
        mimetype="application/octet-stream",
    )
    return download_element


def make_snap_threshold_hstack(input_snap_threshold: mo.ui.text) -> mo.Html:
    return mo.hstack(
        [
            mo.md(
                "[Snap threshold](https://nialov.github.io/fractopo/misc.html#snap-threshold-parameter):"
            ),
            input_snap_threshold,
            "{}".format(input_snap_threshold.value),
        ]
    )
