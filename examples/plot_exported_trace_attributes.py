"""
Exporting trace data with, e.g., length and azimuth attributes included
=======================================================================

``fractopo`` provides ready-to-use calculations for attributes such as the
length and orientation of fracture traces or branches. These attributes,
calculated with ``fractopo``, might be useful for use elsewhere.

Two examples are provided:

1. Call of ``fractopo`` provided functions to calculate length and
   orientation into new columns in a GeoDataFrame.
2. Use of builtin ``Network`` functionality to realize these attributes
   into the ``trace_gdf`` used by the network and then exporting it
   afterwards.
"""

# %%
# Initializing and loading of example trace and area data
# -------------------------------------------------------

from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import matplotlib.pyplot as plt

from fractopo.analysis.network import Network
from fractopo.general import Col, determine_azimuth

BASE_DIR_PATH = Path("..")

KB11_TRACE_GDF = gpd.read_file(
    BASE_DIR_PATH.joinpath(
        Path("tests/sample_data/KB11/KB11_traces.geojson")
    )
)
KB11_AREA_GDF = gpd.read_file(
    BASE_DIR_PATH.joinpath(
        Path("tests/sample_data/KB11/KB11_area.geojson")
    )
)

# %%
# 1. Call of ``fractopo`` provided functions
# ------------------------------------------------------------------
# This method will not modify the trace geometries in any way.
# Only the attributes are calculated.

# Create copy so that the original KB11_TRACE_GDF is not modified
kb11_trace_gdf_1 = KB11_TRACE_GDF.copy()

# %%
# Calculate length and azimuth
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ``geopandas``, through ``shapely``, already provides easy
# calculation of geometry lengths using the length attribute
kb11_trace_gdf_1["length"] = kb11_trace_gdf_1.geometry.length

# How to calculate the azimuth, i.e. orientation, is more subjective.
# ``fractopo`` defaults to a simple approach, where the orientation
# is defined solely by the start and end points of a trace.
kb11_trace_gdf_1["azimuth"] = [
    determine_azimuth(line=trace, halved=True)
    for trace in kb11_trace_gdf_1.geometry.values
]

# %%
# Export trace GeoDataFrame
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Do not use TemporaryDirectory yourself if you want to persist the data!
with TemporaryDirectory() as tmp_dir:
    output_path_1 = Path(tmp_dir).joinpath("KB11_traces_with_length_and_azimuth.gpkg")
    kb11_trace_gdf_1.to_file(output_path_1, driver="GPKG")


# %%
# 2. Use of builtin ``Network`` functionality
# ------------------------------------------------------------------
# This method can modify the traces by, e.g., truncating them to
# the target area (``truncate_traces=True``).


KB11_NETWORK = Network(
    name="KB11",
    trace_gdf=KB11_TRACE_GDF,
    area_gdf=KB11_AREA_GDF,
    truncate_traces=True,
    circular_target_area=False,
    determine_branches_nodes=True,
    snap_threshold=0.001,
    azimuth_set_names=("N-S", "E-W"),
    azimuth_set_ranges=((135, 45), (45, 135)),
)

# %%
# Check that length and azimuth columns do not pre-exist in network
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

assert (
    len(
        set(KB11_NETWORK.trace_gdf.columns).intersection(
            {Col.LENGTH.value, Col.AZIMUTH.value}
        )
    )
    == 0
)

# %%
# Access trace length and orientation arrays
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

KB11_NETWORK.trace_length_array
KB11_NETWORK.trace_azimuth_array

# Check that underlying GeoDataFrame now contains columns for length and
# azimuths
assert (
    len(
        set(KB11_NETWORK.trace_gdf.columns).intersection(
            {Col.LENGTH.value, Col.AZIMUTH.value}
        )
    )
    == 2
)

# %%
# Export trace GeoDataFrame
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Do not use TemporaryDirectory yourself if you want to persist the data!
with TemporaryDirectory() as tmp_dir:
    output_path_2 = Path(tmp_dir).joinpath("KB11_traces_with_network_attributes.gpkg")
    KB11_NETWORK.trace_gdf.to_file(output_path_2, driver="GPKG")

# %%
# Show a preview of the exported data as a table
# -----------------------------------------------------------

preview_df = KB11_NETWORK.trace_gdf[[Col.LENGTH.value, Col.AZIMUTH.value]].head(5)

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis("off")
tbl = ax.table(cellText=preview_df.values, colLabels=preview_df.columns, loc="center")
tbl.set_fontsize(16)
tbl.scale(1.2, 1.2)
ax.set_title("Preview of exported trace attributes")
