"""
Script for profiling ``fractopo`` performance.
"""
import logging
from pathlib import Path

import click

from fractopo import MultiNetwork, Network, Validation
from fractopo.general import read_geofile

SAMPLE_TRACES_PATH = Path("tests/sample_data/geta1/Getaberget_20m_1_traces.gpkg")
SAMPLE_AREA_PATH = Path("tests/sample_data/geta1/Getaberget_20m_1_1_area.gpkg")


@click.command()
@click.option("do_validation", "--do-validation", is_flag=True, default=True)
@click.option("do_network", "--do-network", is_flag=True, default=True)
@click.option("do_grid", "--do-grid", is_flag=True, default=True)
@click.option("do_subsampling", "--do-subsampling", is_flag=True, default=False)
def perf_profile(
    do_validation: bool = True,
    do_network: bool = True,
    do_grid: bool = True,
    do_subsampling: bool = False,
):
    """
    Profile ``fractopo`` performance.
    """
    if not all(path.exists() for path in (SAMPLE_TRACES_PATH, SAMPLE_AREA_PATH)):
        logging.error(f"Current path: {Path.cwd().resolve()}")
        raise FileNotFoundError(
            "Expected sample to exist at:" f" {(SAMPLE_TRACES_PATH, SAMPLE_AREA_PATH)}"
        )

    traces = read_geofile(SAMPLE_TRACES_PATH)
    area = read_geofile(SAMPLE_AREA_PATH)
    name = SAMPLE_AREA_PATH.stem
    snap_threshold = 0.001

    if do_validation:
        validator = Validation(
            traces=traces,
            area=area,
            SNAP_THRESHOLD=snap_threshold,
            name=name,
            allow_fix=True,
        )
        validator.run_validation()

    if do_network:
        network = Network(
            trace_gdf=traces,
            area_gdf=area,
            snap_threshold=snap_threshold,
            name=name,
            circular_target_area=True,
            truncate_traces=True,
            determine_branches_nodes=True,
        )

        network.numerical_network_description()
        if do_grid:
            network.contour_grid(cell_width=2.0)

        if do_subsampling:
            multi_network = MultiNetwork([network])
            multi_network.subsample(min_radius=5.0, samples=5)


if __name__ == "__main__":
    perf_profile()
