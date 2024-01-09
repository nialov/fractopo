"""
Numerical network characteristics
=================================

Lineament & fracture networks can be characterized with numerous
geometric and topological parameters.
"""
# %%
# Initializing
# ------------
# Create two Networks so that their numerical attributes can be compared.

from pprint import pprint

import matplotlib.pyplot as plt

# Load kb11_network network from examples/example_data.py
from example_data import KB7_NETWORK, KB11_NETWORK

# Import Network class from fractopo
from fractopo.analysis.parameters import plot_parameters_plot

# %%
# Geometric and topological Network parameters
# --------------------------------------------
#
# All parameters are accessible through an attribute

kb11_parameters = KB11_NETWORK.parameters
kb7_parameters = KB7_NETWORK.parameters

# %%
pprint(kb11_parameters)

# %%
pprint(kb7_parameters)

# %%
# Compare KB11 and KB7 Networks selected parameter values

b22 = "Dimensionless Intensity B22"
cpb = "Connections per Branch"
selected = {b22, cpb}

# Filter to only selected parameters
kb11_network_selected_params = {
    param: value for param, value in kb11_parameters.items() if param in selected
}
kb7_network_selected_params = {
    param: value for param, value in kb7_parameters.items() if param in selected
}

# Compare parameters with a simple bar plot
figs, axes = plot_parameters_plot(
    topology_parameters_list=[
        kb11_network_selected_params,
        kb7_network_selected_params,
    ],
    labels=["KB11", "KB7"],
    colors=["red", "blue"],
)
plt.show()
