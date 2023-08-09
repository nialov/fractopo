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
from urllib.error import URLError

import geopandas as gpd

from fractopo import Network


def remote_or_local_read_file(url: str, path: Path):
    """
    Try to read remotely and fallback to local without internet.
    """
    try:
        return gpd.read_file(url)
    except URLError:
        try:
            return gpd.read_file(Path(__file__).parent.parent / path)
        except Exception:
            message = f"""
            { Path(__file__) }
            { Path(__file__).parent }
            { Path(__file__).parent.parent }
            { Path(__file__).parent.parent / path }
            { (Path(__file__).parent.parent / path).parent.glob("*") }
            """
            raise FileNotFoundError(message)


kb11_trace_gdf = remote_or_local_read_file(
    url=(
        "https://raw.githubusercontent.com/nialov/"
        "fractopo/master/tests/sample_data/KB11/KB11_traces.geojson"
    ),
    path=Path("tests/sample_data/KB11/KB11_traces.geojson"),
)

kb11_area_gdf = remote_or_local_read_file(
    url=(
        "https://raw.githubusercontent.com/nialov/"
        "fractopo/master/tests/sample_data/KB11/KB11_area.geojson"
    ),
    path=Path("tests/sample_data/KB11/KB11_area.geojson"),
)
kb7_trace_gdf = remote_or_local_read_file(
    url=(
        "https://raw.githubusercontent.com/nialov/"
        "fractopo/master/tests/sample_data/KB7/KB7_traces.geojson"
    ),
    path=Path("tests/sample_data/KB7/KB7_traces.geojson"),
)
kb7_area_gdf = remote_or_local_read_file(
    url=(
        "https://raw.githubusercontent.com/nialov/"
        "fractopo/master/tests/sample_data/KB7/KB7_area.geojson"
    ),
    path=Path("tests/sample_data/KB7/KB7_area.geojson"),
)

hastholmen_trace_gdf = remote_or_local_read_file(
    url=(
        "https://raw.githubusercontent.com/nialov/"
        "fractopo/master/tests/sample_data/hastholmen_traces_validated.geojson"
    ),
    path=Path("tests/sample_data/hastholmen_traces_validated.geojson"),
)
hastholmen_area_gdf = remote_or_local_read_file(
    url=(
        "https://raw.githubusercontent.com/nialov/"
        "fractopo/master/tests/sample_data/hastholmen_area.geojson"
    ),
    path=Path("tests/sample_data/hastholmen_area.geojson"),
)

lidar200k_trace_gdf = remote_or_local_read_file(
    url=(
        "https://raw.githubusercontent.com/nialov/"
        "fractopo/master/tests/sample_data/traces_200k.geojson"
    ),
    path=Path("tests/sample_data/traces_200k.geojson"),
)

lidar200k_area_gdf = remote_or_local_read_file(
    url=(
        "https://raw.githubusercontent.com/nialov/"
        "fractopo/master/tests/sample_data/area_200k.geojson"
    ),
    path=Path("tests/sample_data/area_200k.geojson"),
)


kb11_network = Network(
    name="KB11",
    trace_gdf=kb11_trace_gdf,
    area_gdf=kb11_area_gdf,
    truncate_traces=True,
    circular_target_area=False,
    determine_branches_nodes=True,
    snap_threshold=0.001,
    # Explicitly set azimuth sets
    azimuth_set_names=("N-S", "E-W"),
    azimuth_set_ranges=((135, 45), (45, 135)),
)

kb7_network = Network(
    name="KB7",
    trace_gdf=kb7_trace_gdf,
    area_gdf=kb7_area_gdf,
    truncate_traces=True,
    circular_target_area=False,
    determine_branches_nodes=True,
    snap_threshold=0.001,
)

hastholmen_network = Network(
    name="Hastholmen",
    trace_gdf=hastholmen_trace_gdf,
    area_gdf=hastholmen_area_gdf,
    truncate_traces=True,
    circular_target_area=False,
    determine_branches_nodes=True,
    snap_threshold=0.001,
    # Explicitly set azimuth sets (same as for KB11)
    azimuth_set_names=("N-S", "E-W"),
    azimuth_set_ranges=((135, 45), (45, 135)),
)

lidar_200k_network = Network(
    name="1_200_000",
    trace_gdf=lidar200k_trace_gdf,
    area_gdf=lidar200k_area_gdf,
    truncate_traces=True,
    circular_target_area=False,
    determine_branches_nodes=True,
    snap_threshold=0.001,
    # Explicitly set azimuth sets (same as for KB11)
    azimuth_set_names=("N-S", "E-W"),
    azimuth_set_ranges=((135, 45), (45, 135)),
)
