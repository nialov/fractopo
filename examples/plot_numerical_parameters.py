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

import geopandas as gpd
import matplotlib.pyplot as plt

# Load kb11_network network from examples/make_kb11_network.py
from make_kb11_network import kb11_network

# Import Network class from fractopo
from fractopo import Network
from fractopo.analysis.parameters import plot_parameters_plot

# Make kb7_network here
kb7_network = Network(
    name="KB7",
    trace_gdf=gpd.read_file(
        "https://raw.githubusercontent.com/nialov/"
        "fractopo/master/tests/sample_data/KB7/KB7_traces.geojson"
    ),
    area_gdf=gpd.read_file(
        "https://raw.githubusercontent.com/nialov/"
        "fractopo/master/tests/sample_data/KB7/KB7_area.geojson"
    ),
    truncate_traces=True,
    circular_target_area=False,
    determine_branches_nodes=True,
    snap_threshold=0.001,
)

# %%
# Geometric and topological Network parameters
# --------------------------------------------
#
# All parameters are accessible through an attribute

kb11_parameters = kb11_network.parameters
kb7_parameters = kb7_network.parameters

pprint(kb11_network)
pprint(kb7_network)

# %%
# Compare KB11 and KB7 Networks

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

# Compare parameters

figs, axes = plot_parameters_plot(
    topology_parameters_list=[
        kb11_network_selected_params,
        kb7_network_selected_params,
    ],
    labels=["KB11", "KB7"],
    colors=["red", "blue"],
)
plt.show()
