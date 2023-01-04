"""
Plotting rose plots with ``fractopo``
=====================================
"""
# %%
# Initializing
# ------------

from pprint import pprint

# Load kb11_network network from examples/example_networks.py
from example_networks import kb11_network

# %%
# Plotting a rose plot of fracture network trace orientations
# -----------------------------------------------------------

# Rose plot of network trace orientations
azimuth_bins, fig, ax = kb11_network.plot_trace_azimuth()

# %%
# Plotting a rose plot of fracture network branch orientations
# ------------------------------------------------------------

# Rose plot of network branch orientations
kb11_network.plot_branch_azimuth()

# %%
# Numerical data is accessible with methods and class properties
# ---------------------------------------------------------------

pprint((kb11_network.branch_azimuth_set_counts, kb11_network.trace_azimuth_set_counts))

# %%
# The azimuth sets were not explicitly given during
# Network creation so they are set to defaults.

pprint((kb11_network.azimuth_set_names, kb11_network.azimuth_set_ranges))
