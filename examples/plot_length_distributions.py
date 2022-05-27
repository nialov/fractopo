"""
Plotting length distributions with ``fractopo``
===============================================
"""
# %%
# Initializing
# ------------

from pprint import pprint

import matplotlib as mpl
import matplotlib.pyplot as plt

# Load kb11_network network from examples/example_networks.py
from example_networks import kb11_network

mpl.rcParams["figure.figsize"] = (5, 5)
mpl.rcParams["font.size"] = 8

# %%
# Plotting length distribution plots of fracture traces and branches
# ------------------------------------------------------------------

# Log-log plot of network trace length distribution
fit, fig, ax = kb11_network.plot_trace_lengths()

# Use matplotlib helpers to make sure plot fits in the gallery webpage!
# (Not required.)
plt.tight_layout()
plt.show()

# %%

# Log-log plot of network branch length distribution
fit, fig, ax = kb11_network.plot_branch_lengths()
plt.tight_layout()
plt.show()

# %%
# Numerical descriptions of fits are accessible as properties
# -----------------------------------------------------------

# Use pprint for printing with prettier output
pprint(kb11_network.trace_lengths_powerlaw_fit_description)

# %%

pprint(kb11_network.branch_lengths_powerlaw_fit_description)

# %%
# Set-wise length distribution plotting
# -----------------------------------------------------------

pprint(kb11_network.azimuth_set_names)
pprint(kb11_network.azimuth_set_ranges)

# %%

fits, figs, axes = kb11_network.plot_trace_azimuth_set_lengths()
