"""
Command-line integration of fractopo with click.
"""
import time
from itertools import chain
from pathlib import Path
from typing import Optional, Tuple, Type, Union

import click
from rich.table import Table
from rich.console import Console
from rich.text import Text
import fiona
import geopandas as gpd
import pandas as pd
from typer import Typer
import typer

from fractopo.general import read_geofile
from fractopo.tval.trace_validation import Validation
from fractopo.tval.trace_validators import TargetAreaSnapValidator
from fractopo.analysis.network import Network

app = Typer()


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


def describe_results(validated: gpd.GeoDataFrame, error_column: str):
    """
    Describe validation results to stdout.
    """
    error_count = sum([len(val) != 0 for val in validated[error_column].values])
    error_types = {
        c for c in chain(*validated[error_column].to_list()) if isinstance(c, str)
    }
    count_string = f"Out of {validated.shape[0]} traces, {error_count} were invalid."
    type_string = f"There were {len(error_types)} error types. These were:\n"
    for error_type in error_types:
        type_string += error_type + "\n"
    print(count_string)
    print(type_string)


def make_output_dir(trace_path: Path) -> Path:
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
    output_dir = trace_path.parent / f"validated_{timestr}"
    if not output_dir.exists():
        output_dir.mkdir()
    return output_dir


@click.command()
@click.argument("trace_file", **get_click_path_args())  # type: ignore
@click.argument("area_file", **get_click_path_args())  # type: ignore
@click.option(
    "output_path",
    "--output",
    **get_click_path_args(exists=False, writable=True),
    default=None,
)  # type: ignore
@click.option("allow_fix", "--fix", is_flag=True, help="Allow automatic fixing.")
@click.option(
    "summary", "--summary", is_flag=True, help="Print summary of validation results"
)
@click.option("snap_threshold", "--snap-threshold", type=float, default=0.01)
@click.option(
    "only_area_validation", "--only-area-validation", is_flag=True, default=False
)
@click.option("allow_empty_area", "--no-empty-area", is_flag=True, default=True)
def tracevalidate(
    trace_file: str,
    area_file: str,
    allow_fix: bool,
    summary: bool,
    snap_threshold: float,
    output_path: Union[Path, None],
    only_area_validation: bool,
    allow_empty_area: bool,
):
    """
    Validate trace data delineated by target area data.

    If allow_fix is True, some automatic fixing will be done to e.g. convert
    MultiLineStrings to LineStrings.
    """
    trace_path = Path(trace_file)
    area_path = Path(area_file)

    # Resolve output_path if not explicitly given
    if output_path is None:
        output_dir = make_output_dir(trace_path)
        output_path = (
            trace_path.parent
            / output_dir
            / f"{trace_path.stem}_validated{trace_path.suffix}"
        )
    print(f"Validating with snap threshold of {snap_threshold}.")

    # Assert that read files result in GeoDataFrames
    traces: gpd.GeoDataFrame = read_geofile(trace_path)
    areas: gpd.GeoDataFrame = read_geofile(area_path)
    if not all(isinstance(val, gpd.GeoDataFrame) for val in (traces, areas)):
        raise TypeError(
            "Expected trace and area data to be resolvable as GeoDataFrames."
        )

    # Get input crs
    input_crs = traces.crs

    # Validate
    validation = Validation(
        traces,
        areas,
        trace_path.stem,
        allow_fix,
        SNAP_THRESHOLD=snap_threshold,
    )
    if only_area_validation:
        choose_validators: Optional[Tuple[Type[TargetAreaSnapValidator]]] = (
            TargetAreaSnapValidator,
        )
    else:
        choose_validators = None
    validated_trace = validation.run_validation(
        choose_validators=choose_validators, allow_empty_area=allow_empty_area
    )

    # Set same crs as input if input had crs
    if input_crs is not None:
        validated_trace.crs = input_crs

    # Get input driver to use as save driver
    with fiona.open(trace_path) as open_trace_file:
        assert open_trace_file is not None
        save_driver = open_trace_file.driver

    # Remove file if one exists at output_path
    if Path(output_path).exists():
        Path(output_path).unlink()

    # Change validation_error column to type: `string` and consequently save
    # the GeoDataFrame.
    validated_trace.astype({validation.ERROR_COLUMN: str}).to_file(
        output_path, driver=save_driver
    )
    if summary:
        describe_results(validated_trace, validation.ERROR_COLUMN)


@app.command()
def network(
    traces: Path = typer.Option("", exists=True, dir_okay=False),
    area: Path = typer.Option("", exists=True, dir_okay=False),
    snap_threshold: float = typer.Option(0.001),
    determine_branches_nodes: bool = typer.Option(True),
    name: Optional[str] = typer.Option(None),
    circular_target_area: bool = typer.Option(False),
    truncate_traces: bool = typer.Option(True),
    censoring_area: Optional[Path] = typer.Option(None),
    branches_output: Optional[Path] = typer.Option(None),
    nodes_output: Optional[Path] = typer.Option(None),
    general_output: Optional[Path] = typer.Option(None),
    parameters_output: Optional[Path] = typer.Option(None),
):
    """
    Analyze geometry and topology of trace network.
    """
    network_name = name if name is not None else area.stem
    console = Console()

    console.print(
        Text.assemble(
            "Performing network analysis of ", (network_name, "bold green"), "."
        )
    )

    network = Network(
        trace_gdf=read_geofile(traces),
        area_gdf=read_geofile(area),
        snap_threshold=snap_threshold,
        determine_branches_nodes=determine_branches_nodes,
        name=network_name,
        circular_target_area=circular_target_area,
        truncate_traces=truncate_traces,
        censoring_area=read_geofile(censoring_area)
        if censoring_area is not None
        else None,
    )

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
    console.print(
        Text.assemble(
            "Saving branches to ",
            (str(branches_output_path), "bold blue"),
            " and nodes to ",
            (str(nodes_output_path), "bold blue"),
            ".",
        )
    )
    network.get_branch_gdf().to_file(branches_output_path, driver="GPKG")
    network.get_node_gdf().to_file(nodes_output_path, driver="GPKG")

    param_table = Table(
        title="Network Parameters",
    )
    param_table.add_column("Parameter", header_style="bold", style="bold green")
    param_table.add_column("Value", header_style="bold", style="blue")

    for key, value in network.parameters.items():
        param_table.add_row(key, f"{value:.4f}")
    console.print(param_table)

    pd.DataFrame([network.parameters]).to_csv(parameters_output_path)

    console.print(
        Text.assemble(
            "Saving parameter csv to ",
            (str(parameters_output_path), "bold blue"),
            ".",
        )
    )
