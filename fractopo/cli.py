import click
from typing import Union, Dict
import geopandas as gpd
import fiona

from fractopo.tval.executor import main
from fractopo.tval.trace_validator import BaseValidator
from pathlib import Path


def get_click_path_args(exists=True, **kwargs):
    path_args = dict(
        type=click.Path(
            exists=exists, file_okay=True, dir_okay=False, resolve_path=True, **kwargs
        ),
        nargs=1,
    )
    return path_args


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
