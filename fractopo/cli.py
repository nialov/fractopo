import logging
from itertools import chain
from pathlib import Path
from typing import Dict, Union

import click

import fiona
import geopandas as gpd
from fractopo.tval.trace_validation import Validation


def get_click_path_args(exists=True, **kwargs):
    path_args = dict(
        type=click.Path(
            exists=exists, file_okay=True, dir_okay=False, resolve_path=True, **kwargs
        ),
        nargs=1,
    )
    return path_args


def describe_results(validated: gpd.GeoDataFrame, error_column: str):
    error_count = sum([len(val) != 0 for val in validated[error_column]])  # type: ignore
    error_types = set([c for c in chain(*validated[error_column].to_list())])  # type: ignore
    count_string = f"Out of {validated.shape[0]} traces, {error_count} were invalid."
    type_string = f"There were {len(error_types)} error types. These were:\n"
    for error_type in error_types:
        type_string += error_type + "\n"
    print(count_string)
    print(type_string)


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
def tracevalidate(
    trace_file: str,
    area_file: str,
    allow_fix: bool,
    summary: bool,
    snap_threshold: float,
    output_path: Union[Path, None],
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
        output_path = (
            trace_path.parent / f"{trace_path.stem}_validated{trace_path.suffix}"
        )
    print(f"Validating with snap threshold of {snap_threshold}.")

    # Assert that read files result in GeoDataFrames
    traces: gpd.GeoDataFrame = gpd.read_file(trace_path)  # type: ignore
    areas: gpd.GeoDataFrame = gpd.read_file(area_path)  # type: ignore
    if not all([isinstance(val, gpd.GeoDataFrame) for val in (traces, areas)]):
        raise TypeError(
            "Expected trace and area data to be resolvable as GeoDataFrame."
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
    validated_trace = validation.run_validation()

    # Set same crs as input if input had crs
    if input_crs is not None:
        validated_trace.crs = input_crs

    # Get input driver to use as save driver
    with fiona.open(trace_path) as open_trace_file:
        save_driver = open_trace_file.driver

    # Change validation_error column to type: `string` and consequently save
    # the GeoDataFrame.
    validated_trace.astype({validation.ERROR_COLUMN: str}).to_file(
        output_path, driver=save_driver
    )
    if summary:
        describe_results(validated_trace, validation.ERROR_COLUMN)

