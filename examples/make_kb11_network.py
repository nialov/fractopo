"""
Module for creating a ``fractopo.Network``
==========================================

The module loads a fracture dataset, named KB11, from ``fractopo`` GitHub
repository and creates a ``fractopo.Network`` object of it. Along
with the traces the target area in which the traces have been determined
in is required (it is similarly loaded from ``fractopo`` GitHub page).
"""
# %%
# Initialize ``fractopo.Network``
# -------------------------------

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
)
