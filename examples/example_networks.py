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

import geopandas as gpd

from fractopo import Network

kb11_network = Network(
    name="KB11",
    trace_gdf=gpd.read_file(
        "https://raw.githubusercontent.com/nialov/"
        "fractopo/master/tests/sample_data/KB11/KB11_traces.geojson"
    ),
    area_gdf=gpd.read_file(
        "https://raw.githubusercontent.com/nialov/"
        "fractopo/master/tests/sample_data/KB11/KB11_area.geojson"
    ),
    truncate_traces=True,
    circular_target_area=False,
    determine_branches_nodes=True,
    snap_threshold=0.001,
    # Explicitly set azimuth sets
    azimuth_set_names=("N-S", "E-W"),
    azimuth_set_ranges=((135, 45), (45, 135)),
)

hastholmen_network = Network(
    name="Hastholmen",
    trace_gdf=gpd.read_file(
        "https://raw.githubusercontent.com/nialov/"
        "fractopo/master/tests/sample_data/hastholmen_traces_validated.geojson"
    ),
    area_gdf=gpd.read_file(
        "https://raw.githubusercontent.com/nialov/"
        "fractopo/master/tests/sample_data/hastholmen_area.geojson"
    ),
    truncate_traces=True,
    circular_target_area=False,
    determine_branches_nodes=True,
    snap_threshold=0.001,
    # Explicitly set azimuth sets (same as for KB11)
    azimuth_set_names=("N-S", "E-W"),
    azimuth_set_ranges=((135, 45), (45, 135)),
)
