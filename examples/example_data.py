"""
Module for creating ``fractopo.Network`` objects for examples
=============================================================

The module loads fracture datasets, e.g. one named KB11, from ``fractopo``
GitHub repository and creates a ``fractopo.Network`` object of it. Along with
the traces the target area in which the traces have been determined in is
required (it is similarly loaded from ``fractopo`` GitHub page).
"""
# %%
# Setup matplotlib rcParams for better plotting in the examples
# -------------------------------------------------------------

# import matplotlib as mpl

# mpl.rcParams["figure.autolayout"] = True
# mpl.rcParams["figure.constrained_layout.use"] = True
# mpl.rcParams["savefig.bbox"] = "tight"

# %%
# Initialize ``fractopo.Network``
# -------------------------------
# Note that azimuth sets are explicitly set here for both networks. They are
# user-defined though the Network will use default azimuth sets if not set by
# the user.

from pathlib import Path

import geopandas as gpd

from fractopo import Network

# Root fractopo repository directory path
PROJECT_BASE_PATH = Path("..")


def read_path_from_base(path: Path, project_base_path: Path = PROJECT_BASE_PATH):
    """
    Prefix path with project base path.
    """
    return gpd.read_file(project_base_path / path)


KB11_TRACE_GDF = read_path_from_base(
    path=Path("tests/sample_data/KB11/KB11_traces.geojson"),
)

KB11_AREA_GDF = read_path_from_base(
    path=Path("tests/sample_data/KB11/KB11_area.geojson"),
)
KB7_TRACE_GDF = read_path_from_base(
    path=Path("tests/sample_data/KB7/KB7_traces.geojson"),
)
KB7_AREA_GDF = read_path_from_base(
    path=Path("tests/sample_data/KB7/KB7_area.geojson"),
)

HASTHOLMEN_TRACE_GDF = read_path_from_base(
    path=Path("tests/sample_data/hastholmen_traces_validated.geojson"),
)
HASTHOLMEN_AREA_GDF = read_path_from_base(
    path=Path("tests/sample_data/hastholmen_area.geojson"),
)

LIDAR_200K_TRACE_GDF = read_path_from_base(
    path=Path("tests/sample_data/traces_200k.geojson"),
)

LIDAR_200K_AREA_GDF = read_path_from_base(
    path=Path("tests/sample_data/area_200k.geojson"),
)


KB11_NETWORK = Network(
    name="KB11",
    trace_gdf=KB11_TRACE_GDF,
    area_gdf=KB11_AREA_GDF,
    truncate_traces=True,
    circular_target_area=False,
    determine_branches_nodes=True,
    snap_threshold=0.001,
    # Explicitly set azimuth sets
    azimuth_set_names=("N-S", "E-W"),
    azimuth_set_ranges=((135, 45), (45, 135)),
)

KB7_NETWORK = Network(
    name="KB7",
    trace_gdf=KB7_TRACE_GDF,
    area_gdf=KB7_AREA_GDF,
    truncate_traces=True,
    circular_target_area=False,
    determine_branches_nodes=True,
    snap_threshold=0.001,
)

HASTHOLMEN_NETWORK = Network(
    name="Hastholmen",
    trace_gdf=HASTHOLMEN_TRACE_GDF,
    area_gdf=HASTHOLMEN_AREA_GDF,
    truncate_traces=True,
    circular_target_area=False,
    determine_branches_nodes=True,
    snap_threshold=0.001,
    # Explicitly set azimuth sets (same as for KB11)
    azimuth_set_names=("N-S", "E-W"),
    azimuth_set_ranges=((135, 45), (45, 135)),
)

LIDAR_200K_NETOWORK = Network(
    name="1_200_000",
    trace_gdf=LIDAR_200K_TRACE_GDF,
    area_gdf=LIDAR_200K_AREA_GDF,
    truncate_traces=True,
    circular_target_area=False,
    determine_branches_nodes=True,
    snap_threshold=0.001,
    # Explicitly set azimuth sets (same as for KB11)
    azimuth_set_names=("N-S", "E-W"),
    azimuth_set_ranges=((135, 45), (45, 135)),
)

KB11_IMAGE_PATH = PROJECT_BASE_PATH / "docs_src/imgs/kb11_orthomosaic.jpg"
