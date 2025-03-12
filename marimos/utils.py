import tempfile
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import geopandas as gpd
import marimo as mo

import fractopo.tval.trace_validation
from fractopo.analysis.network import Network


def parse_network_cli_args(cli_args):
    cli_traces_path = Path(cli_args.get("traces-path"))
    cli_area_path = Path(cli_args.get("area-path"))

    name = cli_args.get("name") or cli_traces_path.stem

    traces_gdf = gpd.read_file(cli_traces_path)
    area_gdf = gpd.read_file(cli_area_path)
    snap_threshold_str = cli_args.get("snap-threshold")
    contour_grid_cell_size_str = cli_args.get("contour-grid-cell-size")
    contour_grid_cell_size = (
        float(contour_grid_cell_size_str)
        if contour_grid_cell_size_str is not None
        else None
    )
    if snap_threshold_str is None:
        snap_threshold = fractopo.tval.trace_validation.Validation.SNAP_THRESHOLD
    else:
        snap_threshold = float(snap_threshold_str)
    return name, traces_gdf, area_gdf, snap_threshold, contour_grid_cell_size


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
        gdf = read_file(
            tmp_data_path,
            layer=layer_name,
        )
    return layer_name, gdf


def parse_network_app_args(
    input_trace_layer_name: mo.ui.text,
    input_area_layer_name: mo.ui.text,
    input_traces_file: mo.ui.file,
    input_area_file: mo.ui.file,
    input_snap_threshold: mo.ui.text,
    input_contour_grid_cell_size: mo.ui.text,
) -> Tuple[str, gpd.GeoDataFrame, gpd.GeoDataFrame, float, Optional[float]]:
    (traces_gdf, trace_layer_name), (area_gdf, _) = read_traces_and_area(
        input_traces_file=input_traces_file,
        input_trace_layer_name=input_trace_layer_name,
        input_area_file=input_area_file,
        input_area_layer_name=input_area_layer_name,
    )

    snap_threshold = float(input_snap_threshold.value)
    contour_grid_cell_size = (
        None
        if input_contour_grid_cell_size.value == ""
        else float(input_contour_grid_cell_size.value)
    )
    name = resolve_name(
        input_traces_file=input_traces_file, trace_layer_name=trace_layer_name
    )
    print(f"Snap threshold: {snap_threshold}")
    if contour_grid_cell_size is not None:
        print(f"Contour grid cell size: {contour_grid_cell_size}")
    print(f"Name: {name}")

    return name, traces_gdf, area_gdf, snap_threshold, contour_grid_cell_size


def capture_function_outputs(
    func: Callable,
) -> Tuple[Optional[Any], Optional[Exception], str]:
    tmp_io_stdout_and_stderr = StringIO()
    exc = None
    results = None
    try:
        with redirect_stdout(tmp_io_stdout_and_stderr):
            with redirect_stderr(tmp_io_stdout_and_stderr):
                results = func()
    except Exception as exception:
        exc = exception
    stderr_and_stdout = tmp_io_stdout_and_stderr.getvalue()
    return results, exc, stderr_and_stdout


def network_results_to_download_element(
    network: Optional[Network],
    determine_branches_nodes: bool,
    name: str,
    contour_grid_cell_size: float,
):
    if network is None:
        return None
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        # Create and write plots to tmp_dir
        if determine_branches_nodes:
            network.export_network_analysis(
                output_path=tmp_dir_path, contour_grid_cell_size=contour_grid_cell_size
            )
        else:
            _, fig, _ = network.plot_trace_azimuth()
            fig.savefig(tmp_dir_path / "trace_azimuth.png", bbox_inches="tight")
            _, fig, _ = network.plot_trace_lengths()
            fig.savefig(tmp_dir_path / "trace_lengths.png", bbox_inches="tight")

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

    download_element = mo.download(
        data=zip_io,
        filename=f"{name}.zip",
        mimetype="application/octet-stream",
    )
    return download_element


def parse_validation_cli_args(cli_args):
    cli_traces_path = Path(cli_args.get("traces-path"))
    cli_area_path = Path(cli_args.get("area-path"))

    name = cli_args.get("name") or cli_traces_path.stem

    traces_gdf = gpd.read_file(cli_traces_path)
    area_gdf = gpd.read_file(cli_area_path)
    snap_threshold_str = cli_args.get("snap-threshold")
    if snap_threshold_str is None:
        snap_threshold = fractopo.tval.trace_validation.Validation.SNAP_THRESHOLD
    else:
        snap_threshold = float(snap_threshold_str)

    return name, traces_gdf, area_gdf, snap_threshold


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
    (traces_gdf, trace_layer_name), (area_gdf, _) = read_traces_and_area(
        input_traces_file=input_traces_file,
        input_trace_layer_name=input_trace_layer_name,
        input_area_file=input_area_file,
        input_area_layer_name=input_area_layer_name,
    )
    snap_threshold = float(input_snap_threshold.value)
    print(f"Snap threshold: {snap_threshold}")

    name = resolve_name(
        input_traces_file=input_traces_file, trace_layer_name=trace_layer_name
    )
    return name, traces_gdf, area_gdf, snap_threshold


def validated_clean_to_download_element(validated_clean, name):
    if validated_clean is None:
        return None
    validated_clean_file = BytesIO()
    validated_clean.to_file(validated_clean_file, driver="GPKG", layer=name)
    validated_clean_file.seek(0)
    download_element = mo.download(
        data=validated_clean_file,
        filename=f"{name}_validated.gpkg",
        mimetype="application/octet-stream",
    )
    return download_element
