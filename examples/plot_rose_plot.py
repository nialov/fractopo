"""
Plotting rose plots with ``fractopo``
=====================================
"""

# %%
# Initializing
# ------------

from pprint import pprint

# Load kb11_network network from examples/example_data.py
from example_data import KB11_NETWORK

# %%
# Plotting a rose plot of fracture network trace orientations
# -----------------------------------------------------------

# Rose plot of network trace orientations
azimuth_bins, fig, ax = KB11_NETWORK.plot_trace_azimuth()

# %%
# Plotting a rose plot of fracture network branch orientations
# ------------------------------------------------------------

# Rose plot of network branch orientations
KB11_NETWORK.plot_branch_azimuth()

# %%
# Numerical data is accessible with methods and class properties
# ---------------------------------------------------------------

pprint((KB11_NETWORK.branch_azimuth_set_counts, KB11_NETWORK.trace_azimuth_set_counts))

# %%
# The azimuth sets were not explicitly given during
# Network creation so they are set to defaults.

pprint((KB11_NETWORK.azimuth_set_names, KB11_NETWORK.azimuth_set_ranges))
