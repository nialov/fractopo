"""
Creating a fractopo network from trace data
===========================================

Data is loaded into ``fractopo`` using ``geopandas`` which can load from
a wide variety of sources.
"""

# %%
# Here we load data from the internet from the ``fractopo`` GitHub repository.

import geopandas as gpd

trace_data_url = (
    "https://raw.githubusercontent.com/nialov/"
    "fractopo/master/tests/sample_data/KB11/KB11_traces.geojson"
)
area_data_url = (
    "https://raw.githubusercontent.com/nialov/"
    "fractopo/master/tests/sample_data/KB11/KB11_area.geojson"
)

# Use geopandas to load data from urls
traces = gpd.read_file(trace_data_url)
area = gpd.read_file(area_data_url)

# Check that the type is GeoDataFrame
assert isinstance(traces, gpd.GeoDataFrame)
assert isinstance(area, gpd.GeoDataFrame)

# Name the dataset
name = "KB11"

# %%
# Plotting the loaded data
# ------------------------

# Import matplotlib
import matplotlib.pyplot as plt

# Initialize the figure and ax in which data is plotted
fig, ax = plt.subplots(figsize=(9, 9))

# Plot the loaded trace dataset consisting of fracture traces.
traces.plot(ax=ax, color="blue")

# Plot the loaded area dataset that consists of a single polygon
# that delineates the traces.
area.boundary.plot(ax=ax, color="red")

# Give the figure a title
ax.set_title(f"{name}, Coordinate Reference System = {traces.crs}")

# %%
# Creating a fractopo Network
# ---------------------------

# Import Network class from fractopo
from fractopo import Network

kb11_network = Network(
    trace_gdf=traces,
    area_gdf=area,
    truncate_traces=True,
    circular_target_area=False,
    determine_branches_nodes=True,
    snap_threshold=0.001,
)

# %%
# Plotting using the Network
# --------------------------

# Rose plot of network branch orientations
kb11_network.plot_branch_azimuth()

# XYI-plot of topological node counts
kb11_network.plot_xyi()
