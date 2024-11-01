"""
Command-line integration of fractopo with click.
"""

import json
import logging
import sys
import time
from enum import Enum, unique
from itertools import chain
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import click
import geopandas as gpd
import pandas as pd
import pyogrio
import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text
from typer import Typer

from fractopo import __version__
from fractopo.analysis.network import Network
from fractopo.general import check_for_wrong_geometries, read_geofile, save_fig
from fractopo.tval.trace_validation import Validation
from fractopo.tval.trace_validators import SharpCornerValidator, TargetAreaSnapValidator

log = logging.getLogger(__name__)

# Initialize typer command-line app object
APP = Typer()

# Use minimum console width of 80
CONSOLE = Console(width=min([80, Console().width]))

SNAP_THRESHOLD_HELP = (
    "Distance threshold used to estimate whether e.g. a trace abuts another."
)
TRACE_FILE_HELP = "Path to lineament or fracture trace data."
AREA_FILE_HELP = "Path to target area data that delineates trace data."


@unique
class LogLevel(Enum):
    """
    Enums for log levels.
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def get_click_path_args(exists=True, **kwargs):
    """
    Get basic click path args.
    """
    path_arguments = dict(
        type=click.Path(
            exists=exists, file_okay=True, dir_okay=False, resolve_path=True, **kwargs
        ),
        nargs=1,
    )
    return path_arguments


def _check_for_wrong_geometries_cli(traces: gpd.GeoDataFrame, area: gpd.GeoDataFrame):
    try:
        check_for_wrong_geometries(traces=traces, area=area)
    except TypeError:
        CONSOLE.print(
            Text.assemble(
                "Check that traces and target area arguments are passed in"
                " the correct order in the command-line (Check with --help)."
            )
        )
        raise


def describe_results(
    validated: gpd.GeoDataFrame, error_column: str, console: Console = CONSOLE
):
    """
    Describe validation results to stdout.
    """
    error_count = sum(len(val) != 0 for val in validated[error_column].values)
    error_types = {
        c for c in chain(*validated[error_column].to_list()) if isinstance(c, str)
    }
    trace_count = validated.shape[0]
    count_string = f"Out of {trace_count} traces, {error_count} were invalid."
    type_string = f"There were {len(error_types)} error types. These were:\n"
    type_color = "yellow"
    for error_type in error_types:
        type_string += error_type + "\n"
        if SharpCornerValidator.ERROR not in error_type:
            type_color = "bold red"
    count_color = "bold red" if error_count / trace_count > 0.05 else "yellow"
    console.print(Text.assemble((count_string, count_color)))
    if len(error_types) > 0:
        console.print(Text.assemble((type_string, type_color)))


def make_output_dir(base_path: Path) -> Path:
    """
    Make timestamped output dir.
    """
    localtime = time.localtime()
    tm_min = localtime.tm_min
    hour = localtime.tm_hour
    day = localtime.tm_mday
    month = localtime.tm_mon
    year = localtime.tm_year
    timestr = "_".join(map(str, [day, month, year, hour, tm_min]))
    output_dir = base_path / f"validated_{timestr}"
    if not output_dir.exists():
        output_dir.mkdir()
    return output_dir


def rich_table_from_parameters(parameters: Dict[str, float]) -> Table:
    """
    Generate ``rich`` ``Table`` from network parameters.
    """
    param_table = Table(
        title="Selected Network Parameters",
    )
    param_table.add_column("Parameter", header_style="bold", style="bold green")
    param_table.add_column("Value", header_style="bold", style="blue")

    for key, value in parameters.items():
        param_table.add_row(key, f"{value:.4f}")
    return param_table


@APP.callback()
def fractopo_callback(
    log_level: LogLevel = typer.Option(LogLevel.WARNING.value),
):
    """
    Use fractopo command-line utilities.
    """
    log_level_int = int(getattr(logging, log_level.value))
    log.info("Setting up log with basicConfig.")
    logging.basicConfig(level=log_level_int, force=True)


@APP.command()
def tracevalidate(
    trace_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help=TRACE_FILE_HELP,
    ),
    area_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help=AREA_FILE_HELP,
    ),
    allow_fix: bool = typer.Option(
        True,
        help=(
            "Enable the direct modification of output trace file to fix errors. "
            "Input files will not be modified unless specified as the --output target."
        ),
    ),
    summary: bool = typer.Option(True, help="Print summary of validation results."),
    snap_threshold: float = typer.Option(
        0.001,
        help="Distance threshold used to estimate whether e.g. a trace abuts another.",
    ),
    output: Optional[Path] = typer.Option(
        None, help="Where to save validated output trace data."
    ),
    only_area_validation: bool = typer.Option(
        False, help="Only validate the area boundary snapping."
    ),
    allow_empty_area: bool = typer.Option(
        True, help="Allow empty areas to validation."
    ),
):
    """
    Validate trace data delineated by target area data.

    If allow_fix is True, some automatic fixing will be done to e.g. convert
    MultiLineStrings to LineStrings.
    """
    # Assert that read files result in GeoDataFrames
    traces: gpd.GeoDataFrame = read_geofile(trace_file)
    areas: gpd.GeoDataFrame = read_geofile(area_file)
    if not all(isinstance(val, gpd.GeoDataFrame) for val in (traces, areas)):
        raise TypeError(
            "Expected trace and area files to be readable as GeoDataFrames."
        )

    _check_for_wrong_geometries_cli(traces=traces, area=areas)
    log.info(f"Validating traces: {trace_file} area: {area_file}.")
    # Get input crs
    input_crs = traces.crs

    # Validate
    validation = Validation(
        traces,
        areas,
        trace_file.stem,
        allow_fix,
        SNAP_THRESHOLD=snap_threshold,
    )
    if only_area_validation:
        CONSOLE.print(Text.assemble(("Only performing area validation.", "yellow")))
        choose_validators: Optional[Tuple[Type[TargetAreaSnapValidator]]] = (
            TargetAreaSnapValidator,
        )
    else:
        choose_validators = None
    CONSOLE.print(
        Text.assemble("Performing validation of ", (trace_file.name, "blue"), ".")
    )
    validated_trace = validation.run_validation(
        choose_validators=choose_validators, allow_empty_area=allow_empty_area
    )

    if validated_trace.shape[0] == 0:
        CONSOLE.print(
            Text.assemble(
                ("Validation returned a GeoDataFrame with no traces.", "red"), ""
            )
        )

    # Set same crs as input if input had crs
    if input_crs is not None:
        validated_trace = validated_trace.set_crs(input_crs)

    # Get input driver to use as save driver

    save_driver = pyogrio.detect_write_driver(trace_file)

    # Resolve output if not explicitly given
    if output is None:
        output_dir = make_output_dir(Path(".")).resolve()
        output_path = output_dir / f"{trace_file.stem}_validated{trace_file.suffix}"
        CONSOLE.print(
            Text.assemble(
                (
                    f"Generated output directory at {output_dir} "
                    f"where validated output will be saved in file {output_path.name}.",
                    "blue",
                )
            )
        )
    else:
        output_path = output

    # Remove file if one exists at output_path
    if output_path.exists():
        CONSOLE.print(
            Text.assemble(("Overwriting old file at given output path.", "yellow"))
        )
        output_path.unlink()

    # Change validation_error column to type: str and consequently save
    # the GeoDataFrame.
    if validated_trace.shape[0] != 0:
        assert not isinstance(validated_trace[validation.ERROR_COLUMN].iloc[0], list)
    validated_trace.astype({validation.ERROR_COLUMN: str}).to_file(
        output_path, driver=save_driver
    )
    if summary:
        describe_results(validated_trace, validation.ERROR_COLUMN, console=CONSOLE)


@APP.command()
def network(
    trace_file: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help=TRACE_FILE_HELP
    ),
    area_file: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help=AREA_FILE_HELP
    ),
    snap_threshold: float = typer.Option(0.001, help=SNAP_THRESHOLD_HELP),
    determine_branches_nodes: bool = typer.Option(
        True,
        help="Whether to determine branches and nodes as part of analysis. Recommended.",
    ),
    name: Optional[str] = typer.Option(
        None, help="Name for Network. Used when saving outputs and as plot titles."
    ),
    circular_target_area: bool = typer.Option(
        False, help="Is/are target area(s) circles?"
    ),
    truncate_traces: bool = typer.Option(
        True, help="Whether to cut traces at target area boundary. Recommended."
    ),
    censoring_area: Optional[Path] = typer.Option(
        None,
        help="Path to area data that delineates censored areas within target areas.",
    ),
    branches_output: Optional[Path] = typer.Option(
        None, help="Where to save branch data."
    ),
    nodes_output: Optional[Path] = typer.Option(None, help="Where to save node data."),
    general_output: Optional[Path] = typer.Option(
        None, help="Where to save general network analysis outputs e.g. plots."
    ),
    parameters_output: Optional[Path] = typer.Option(
        None, help="Where to save numerical parameter data from analysis."
    ),
):
    """
    Analyze the geometry and topology of trace network.
    """
    network_name = name if name is not None else area_file.stem

    CONSOLE.print(
        Text.assemble(
            "Performing network analysis of ", (network_name, "bold green"), "."
        )
    )
    traces = gpd.read_file(trace_file)
    areas = gpd.read_file(area_file)
    _check_for_wrong_geometries_cli(traces=traces, area=areas)

    network = Network(
        trace_gdf=traces,
        area_gdf=areas,
        snap_threshold=snap_threshold,
        determine_branches_nodes=determine_branches_nodes,
        name=network_name,
        circular_target_area=circular_target_area,
        truncate_traces=truncate_traces,
        censoring_area=(
            read_geofile(censoring_area)
            if censoring_area is not None
            else gpd.GeoDataFrame()
        ),
    )
    (
        general_output_path,
        branches_output_path,
        nodes_output_path,
        parameters_output_path,
    ) = default_network_output_paths(
        network_name=network_name,
        general_output=general_output,
        branches_output=branches_output,
        nodes_output=nodes_output,
        parameters_output=parameters_output,
    )
    if determine_branches_nodes:
        # Save branches and nodes
        CONSOLE.print(
            Text.assemble(
                "Saving branches to ",
                (str(branches_output_path), "bold green"),
                " and nodes to ",
                (str(nodes_output_path), "bold green"),
                ".",
            )
        )
        network.branch_gdf.to_file(branches_output_path, driver="GPKG")
        network.node_gdf.to_file(nodes_output_path, driver="GPKG")

    base_parameters = network.parameters

    # Print pretty table of basic network parameters
    CONSOLE.print(rich_table_from_parameters(base_parameters))

    parameters = (
        network.numerical_network_description()
        if determine_branches_nodes
        else base_parameters
    )

    pd.DataFrame([parameters]).to_csv(parameters_output_path)

    CONSOLE.print(
        Text.assemble(
            "Saving network parameter csv to path:\n",
            (str(parameters_output_path), "bold green"),
        )
    )

    if determine_branches_nodes:
        # Plot ternary XYI-node proportion plot
        fig, _, _ = network.plot_xyi()
        save_fig(fig=fig, results_dir=general_output_path, name="xyi_ternary_plot")

        # Plot ternary branch proportion plot
        fig, _, _ = network.plot_branch()
        save_fig(fig=fig, results_dir=general_output_path, name="branch_ternary_plot")

    # Plot trace azimuth rose plot
    _, fig, _ = network.plot_trace_azimuth()
    save_fig(fig=fig, results_dir=general_output_path, name="trace_azimuth")

    # Plot trace length distribution plot
    _, fig, _ = network.plot_trace_lengths()
    save_fig(fig=fig, results_dir=general_output_path, name="trace_length_distribution")

    # Save additional numerical data for possible post-processing needs in a
    # json file
    json_data_arrays = {
        "trace_length_array": network.trace_length_array,
        "branch_length_array": (
            network.branch_length_array if determine_branches_nodes else None
        ),
        "trace_azimuth_array": network.trace_azimuth_array,
        "branch_azimuth_array": (
            network.branch_azimuth_array if determine_branches_nodes else None
        ),
        "trace_azimuth_set_array": network.trace_azimuth_set_array,
        "branch_azimuth_set_array": (
            network.branch_azimuth_set_array if determine_branches_nodes else None
        ),
    }
    json_data_lists = {
        key: (item.tolist() if item is not None else None)
        for key, item in json_data_arrays.items()
    }
    json_data_path = general_output_path / "additional_numerical_data.json"

    json_data_path.write_text(json.dumps(json_data_lists, sort_keys=True))


def default_network_output_paths(
    network_name: str,
    general_output: Optional[Path],
    branches_output: Optional[Path],
    nodes_output: Optional[Path],
    parameters_output: Optional[Path],
):
    """
    Determine default network output paths.
    """
    general_output_path = (
        Path(f"{network_name}_outputs") if general_output is None else general_output
    )
    general_output_path.mkdir(exist_ok=True)

    branches_output_path = (
        general_output_path / f"{network_name}_branches.gpkg"
        if branches_output is None
        else branches_output
    )
    nodes_output_path = (
        general_output_path / f"{network_name}_nodes.gpkg"
        if nodes_output is None
        else nodes_output
    )
    parameters_output_path = (
        general_output_path / f"{network_name}_parameters.csv"
        if parameters_output is None
        else parameters_output
    )

    return (
        general_output_path,
        branches_output_path,
        nodes_output_path,
        parameters_output_path,
    )


@APP.command()
def info():
    """
    Print out information about fractopo installation and python environment.
    """
    information = dict(
        fractopo_version=__version__,
        geopandas_version=gpd.__version__,
        package_location=str(Path(__file__).parent.absolute()),
        python_location=str(Path(sys.executable).absolute()),
    )
    CONSOLE.print_json(json.dumps(information, sort_keys=True))
