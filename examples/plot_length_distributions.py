"""
Plotting length distributions with ``fractopo``
===============================================
"""
# %%
# Initializing
# ------------

from pprint import pprint

import matplotlib.pyplot as plt

# Load kb11_network network from examples/make_kb11_network.py
from make_kb11_network import kb11_network

# %%
# Plotting length distribution plots of fracture traces and branches
# ------------------------------------------------------------------

# Log-log plot of network trace length distribution
fit, fig, ax = kb11_network.plot_trace_lengths()

# Use matplotlib helpers to make sure plot fits in the gallery webpage!
# (Not required.)
fig.set_size_inches(12, 7)
plt.tight_layout()
plt.show()

# %%

# Log-log plot of network branch length distribution
fit, fig, ax = kb11_network.plot_branch_lengths()
fig.set_size_inches(12, 7)
plt.tight_layout()
plt.show()

# %%
# Numerical descriptions of fits are accessible as properties
# -----------------------------------------------------------

# Use pprint for printing with prettier output
pprint(kb11_network.trace_lengths_powerlaw_fit_description)

# %%

pprint(kb11_network.branch_lengths_powerlaw_fit_description)
