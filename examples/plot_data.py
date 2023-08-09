"""
Plotting the trace data used as input in ``fractopo``
=====================================================

Data is loaded into ``fractopo`` using ``geopandas`` which can load from
a wide variety of sources.
"""

# %%
# Here we load data from the internet from the ``fractopo`` GitHub repository.

# Import matplotlib for plotting
import matplotlib.pyplot as plt
from example_networks import kb11_network

# Name the dataset
name = "KB11"

# %%
# Plotting the loaded data
# ------------------------


# Initialize the figure and ax in which data is plotted
fig, ax = plt.subplots(figsize=(9, 9))

# Plot the loaded trace dataset consisting of fracture traces.
kb11_network.trace_gdf.plot(ax=ax, color="blue")

# Plot the loaded area dataset that consists of a single polygon
# that delineates the traces.
kb11_network.area_gdf.boundary.plot(ax=ax, color="red")

# Give the figure a title
ax.set_title(f"{name}, Coordinate Reference System = {kb11_network.trace_gdf.crs}")
