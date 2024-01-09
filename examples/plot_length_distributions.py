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

# Load kb11_network network from examples/example_data.py
from example_data import KB11_NETWORK

mpl.rcParams["figure.figsize"] = (5, 5)
mpl.rcParams["font.size"] = 8

# %%
# Plotting length distribution plots of fracture traces and branches
# ------------------------------------------------------------------

# %%
# Using Complementary Cumulative Number/Function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Log-log plot of network trace length distribution
fit, fig, ax = KB11_NETWORK.plot_trace_lengths()

# Use matplotlib helpers to make sure plot fits in the gallery webpage!
# (Not required.)
plt.tight_layout()
plt.show()

# %%

# Log-log plot of network branch length distribution
KB11_NETWORK.plot_branch_lengths()
plt.tight_layout()
plt.show()

# %%
# Using Probability Density Function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Log-log plot of network trace length distribution
KB11_NETWORK.plot_trace_lengths(use_probability_density_function=True)

# Use matplotlib helpers to make sure plot fits in the gallery webpage!
# (Not required.)
plt.tight_layout()
plt.show()

# %%

# Log-log plot of network branch length distribution
KB11_NETWORK.plot_branch_lengths(use_probability_density_function=True)
plt.tight_layout()
plt.show()

# %%
# Numerical descriptions of fits are accessible as properties
# -----------------------------------------------------------

# Use pprint for printing with prettier output
pprint(KB11_NETWORK.trace_lengths_powerlaw_fit_description)

# %%

pprint(KB11_NETWORK.branch_lengths_powerlaw_fit_description)

# %%
# Set-wise length distribution plotting
# -----------------------------------------------------------

pprint(KB11_NETWORK.azimuth_set_names)
pprint(KB11_NETWORK.azimuth_set_ranges)

# %%

fits, figs, axes = KB11_NETWORK.plot_trace_azimuth_set_lengths()
