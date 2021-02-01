"""
Command-line integration of fractopo with click.
"""
import time
from itertools import chain
from pathlib import Path
from typing import Union

import click
import fiona
import geopandas as gpd

from fractopo.tval.trace_validation import Validation
from fractopo.tval.trace_validators import TargetAreaSnapValidator


def get_click_path_args(exists=True, **kwargs):
    """
    Get basic click path args.
    """
    path_args = dict(
        type=click.Path(
            exists=exists, file_okay=True, dir_okay=False, resolve_path=True, **kwargs
        ),
        nargs=1,
    )
    return path_args


def describe_results(validated: gpd.GeoDataFrame, error_column: str):
    """
    Describe validation results to stdout.
    """
    error_count = sum([len(val) != 0 for val in validated[error_column]])  # type: ignore
    error_types = set([c for c in chain(*validated[error_column].to_list())])  # type: ignore
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
    min = localtime.tm_min
    hour = localtime.tm_hour
    day = localtime.tm_mday
    month = localtime.tm_mon
    year = localtime.tm_year
    timestr = "_".join(map(str, [day, month, year, hour, min]))
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
    traces: gpd.GeoDataFrame = gpd.read_file(trace_path)  # type: ignore
    areas: gpd.GeoDataFrame = gpd.read_file(area_path)  # type: ignore
    if not all([isinstance(val, gpd.GeoDataFrame) for val in (traces, areas)]):
        raise TypeError(
            "Expected trace and area data to be resolvable as GeoDataFrames."
        )

    # Get input crs
    input_crs = traces.crs

    # Validate
    validation = Validation(
        traces, areas, trace_path.stem, allow_fix, SNAP_THRESHOLD=snap_threshold,
    )
    if only_area_validation:
        choose_validators = [TargetAreaSnapValidator]
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

