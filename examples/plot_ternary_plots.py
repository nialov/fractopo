"""
Plotting topological ternary plots with ``fractopo``
====================================================
"""
# %%
# Initializing
# ------------

from pprint import pprint

# Load kb11_network network from examples/example_networks.py
from example_networks import kb11_network

# %%
# Plotting a ternary plot of fracture network node counts
# -----------------------------------------------------------

xyi_fig, xyi_ax, xyi_tax = kb11_network.plot_xyi()

# %%
# Plotting a ternary plot of fracture network branch counts
# ------------------------------------------------------------

branch_fig, branch_ax, branch_tax = kb11_network.plot_branch()

# %%
# Numerical data is accessible
# ----------------------------

pprint(dict(node_counts=kb11_network.node_counts))

pprint(dict(branch_counts=kb11_network.branch_counts))
