from typing import Union, Dict
import logging
from pathlib import Path
from itertools import chain

import click
import geopandas as gpd
import fiona

from fractopo.tval.executor import main
from fractopo.tval.trace_validator import BaseValidator
from fractopo.tval.executor_v2 import Validation


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
@click.argument("trace_path", **get_click_path_args())  # type: ignore
@click.argument("area_path", **get_click_path_args())  # type: ignore
@click.option(
    "output_path", "--output", **get_click_path_args(exists=False, writable=True)
)  # type: ignore
@click.option("auto_fix", "--fix", is_flag=True)
def tracevalidate(
    trace_path: Union[Path, str],
    area_path: Union[Path, str],
    auto_fix: bool,
    output_path: Union[Path, None] = None,
):
    trace_path = Path(trace_path)
    # Get input crs
    input_crs = gpd.read_file(trace_path).crs
    area_path = Path(area_path)
    if output_path is None:
        output_path = (
            trace_path.parent / f"{trace_path.stem}_validated{trace_path.suffix}"
        )
    # Sensible defaults
    # TODO: Refactor tval
    snap_threshold = 0.001
    snap_threshold_error_multiplier = 1.1
    area_edge_snap_multiplier = 5
    BaseValidator.set_snap_threshold_and_multipliers(
        snap_threshold=snap_threshold,
        snap_threshold_error_multiplier=snap_threshold_error_multiplier,
        area_edge_snap_multiplier=area_edge_snap_multiplier,
    )
    # Validate
    validated: Dict[str, gpd.GeoDataFrame] = main(
        [gpd.read_file(trace_path)],  # type: ignore
        [gpd.read_file(area_path)],  # type: ignore
        [trace_path.stem],
        auto_fix=auto_fix,
    )
    # Get result from validated dict
    validated_trace: gpd.GeoDataFrame = validated[trace_path.stem]
    # Set same crs as input if input had crs
    if input_crs is not None:
        validated_trace.crs = input_crs
    # Get input driver to use as save driver
    with fiona.open(trace_path) as trace_file:
        save_driver = trace_file.driver
    # Change validation_error column to type: `string` and consequently save
    # the GeoDataFrame.
    validated_trace.astype({BaseValidator.ERROR_COLUMN: str}).to_file(
        output_path, driver=save_driver
    )


@click.command()
@click.argument("trace_path", **get_click_path_args())  # type: ignore
@click.argument("area_path", **get_click_path_args())  # type: ignore
@click.option(
    "output_path", "--output", **get_click_path_args(exists=False, writable=True)
)  # type: ignore
@click.option("allow_fix", "--fix", is_flag=True, help="Allow automatic fixing.")
@click.option(
    "summary", "--summary", is_flag=True, help="Print summary of validation results"
)
def tracevalidatev2(
    trace_path: Union[Path, str],
    area_path: Union[Path, str],
    allow_fix: bool,
    summary: bool,
    output_path: Union[Path, None] = None,
):
    """
    Validate trace data delineated by target area data.

    If allow_fix is True, some automatic fixing will be done to e.g. convert
    MultiLineStrings to LineStrings.
    """
    trace_path = Path(trace_path)
    # Get input crs
    input_crs = gpd.read_file(trace_path).crs
    area_path = Path(area_path)
    if output_path is None:
        output_path = (
            trace_path.parent / f"{trace_path.stem}_validated{trace_path.suffix}"
        )
    print(f"Validating with snap threshold of {Validation.SNAP_THRESHOLD}.")
    # Validate
    validation = Validation(
        gpd.read_file(trace_path), gpd.read_file(area_path), trace_path.stem, allow_fix
    )
    validated_trace = validation.run_validation()
    # Set same crs as input if input had crs
    if input_crs is not None:
        validated_trace.crs = input_crs
    # Get input driver to use as save driver
    with fiona.open(trace_path) as trace_file:
        save_driver = trace_file.driver
    # Change validation_error column to type: `string` and consequently save
    # the GeoDataFrame.
    validated_trace.astype({validation.ERROR_COLUMN: str}).to_file(
        output_path, driver=save_driver
    )
    if summary:
        describe_results(validated_trace, validation.ERROR_COLUMN)
