"""
Plotting topological ternary plots with ``fractopo``
====================================================
"""

# %%
# Initializing
# ------------

from pprint import pprint

# Load kb11_network network from examples/example_data.py
from example_data import KB11_NETWORK

# %%
# Plotting a ternary plot of fracture network node counts
# -----------------------------------------------------------

xyi_fig, xyi_ax, xyi_tax = KB11_NETWORK.plot_xyi()

# %%
# Plotting a ternary plot of fracture network branch counts
# ------------------------------------------------------------

branch_fig, branch_ax, branch_tax = KB11_NETWORK.plot_branch()

# %%
# Numerical data is accessible
# ----------------------------

pprint(dict(node_counts=KB11_NETWORK.node_counts))

pprint(dict(branch_counts=KB11_NETWORK.branch_counts))
