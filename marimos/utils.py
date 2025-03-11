import tempfile
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import geopandas as gpd
import marimo as mo
import pyogrio

import fractopo.tval.trace_validation
from fractopo.analysis.network import Network


def parse_network_cli_args(cli_args):
    cli_traces_path = Path(cli_args.get("traces-path"))
    cli_area_path = Path(cli_args.get("area-path"))

    name = cli_args.get("name") or cli_traces_path.stem

    driver = pyogrio.detect_write_driver(cli_traces_path.name)

    traces_gdf = gpd.read_file(cli_traces_path, driver=driver)
    area_gdf = gpd.read_file(cli_area_path, driver=driver)
    snap_threshold_str = cli_args.get("snap-threshold")
    if snap_threshold_str is None:
        snap_threshold = fractopo.tval.trace_validation.Validation.SNAP_THRESHOLD
    else:
        snap_threshold = float(snap_threshold_str)
    return name, driver, traces_gdf, area_gdf, snap_threshold


def parse_network_app_args(
    input_trace_layer_name,
    input_area_layer_name,
    input_traces_file,
    input_area_file,
    input_snap_threshold,
):
    trace_layer_name = (
        input_trace_layer_name.value if input_trace_layer_name.value != "" else None
    )
    area_layer_name = (
        input_area_layer_name.value if input_area_layer_name.value != "" else None
    )

    driver = pyogrio.detect_write_driver(input_traces_file.name())
    print(f"Detected driver: {driver}")

    print(
        f"Trace layer name: {trace_layer_name}"
        if trace_layer_name is not None
        else "No layer specified"
    )
    traces_gdf = gpd.read_file(
        input_traces_file.contents(),
        layer=trace_layer_name,
        # , driver=driver
    )
    print(
        f"Area layer name: {area_layer_name}"
        if area_layer_name is not None
        else "No layer specified"
    )
    area_gdf = gpd.read_file(
        input_area_file.contents(),
        layer=area_layer_name,
        # , driver=driver
    )

    snap_threshold = float(input_snap_threshold.value)
    name = (
        Path(input_traces_file.name()).stem
        if trace_layer_name is None
        else trace_layer_name
    )
    print(f"Snap threshold: {snap_threshold}")
    print(f"Name: {name}")

    return name, driver, traces_gdf, area_gdf, snap_threshold


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
    network: Optional[Network], determine_branches_nodes: bool, name: str
):
    if network is None:
        return None
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        # Create and write plots to tmp_dir
        if determine_branches_nodes:
            network.export_network_analysis(output_path=tmp_dir_path)
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

    driver = pyogrio.detect_write_driver(cli_traces_path.name)

    traces_gdf = gpd.read_file(cli_traces_path, driver=driver)
    area_gdf = gpd.read_file(cli_area_path, driver=driver)
    snap_threshold_str = cli_args.get("snap-threshold")
    if snap_threshold_str is None:
        snap_threshold = fractopo.tval.trace_validation.Validation.SNAP_THRESHOLD
    else:
        snap_threshold = float(snap_threshold_str)

    return name, traces_gdf, area_gdf, snap_threshold


def parse_validation_app_args(
    input_trace_layer_name,
    input_area_layer_name,
    input_traces_file,
    input_area_file,
    input_snap_threshold,
):
    trace_layer_name = (
        input_trace_layer_name.value if input_trace_layer_name.value != "" else None
    )
    area_layer_name = (
        input_area_layer_name.value if input_area_layer_name.value != "" else None
    )

    driver = pyogrio.detect_write_driver(input_traces_file.name())
    print(f"Detected driver: {driver}")

    print(
        f"Trace layer name: {trace_layer_name}"
        if trace_layer_name is not None
        else "No layer specified"
    )
    traces_gdf = gpd.read_file(
        input_traces_file.contents(),
        layer=trace_layer_name,
        # , driver=driver
    )
    print(
        f"Area layer name: {area_layer_name}"
        if area_layer_name is not None
        else "No layer specified"
    )
    area_gdf = gpd.read_file(
        input_area_file.contents(),
        layer=area_layer_name,
        # , driver=driver
    )

    snap_threshold = float(input_snap_threshold.value)
    print(f"Snap threshold: {snap_threshold}")

    name = (
        Path(input_traces_file.name()).stem
        if trace_layer_name is None
        else trace_layer_name
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
