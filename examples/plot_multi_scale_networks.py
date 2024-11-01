"""
Plotting multi-scale fracture networks with ``fractopo``
========================================================
"""

# %%
# Initializing
# ------------

import matplotlib as mpl
import matplotlib.pyplot as plt

# Load kb11_network and hastholmen_network
from example_data import HASTHOLMEN_NETWORK, KB11_NETWORK

from fractopo import MultiNetwork

mpl.rcParams["figure.figsize"] = (5, 5)
mpl.rcParams["font.size"] = 8

# %%
# Create MultiNetwork object
# ------------------------------------------------------------------

multi_network = MultiNetwork((KB11_NETWORK, HASTHOLMEN_NETWORK))

# %%
# Plot automatically cut length distributions with a multi-scale fit
# ------------------------------------------------------------------
# Powerlaw exponent and fit 'score' are embedded into the plots.
# MSLE = Mean Squared Logarithmic Error

# Log-log plot of MultiNetwork trace length distribution
mld_traces, polyfit, fig, ax = multi_network.plot_multi_length_distribution(
    using_branches=False,
    automatic_cut_offs=True,
    plot_truncated_data=True,
)

# Visual plot setup
plt.tight_layout()

# %%

# Log-log plot of MultiNetwork branch length distribution
mld_branches, polyfit, fig, ax = multi_network.plot_multi_length_distribution(
    using_branches=True,
    automatic_cut_offs=True,
    plot_truncated_data=True,
)

# Visual plot setup
plt.tight_layout()

# %%
# Numerical details of multi-scale length distributions
# -----------------------------------------------------------

# The returned MultiLengthDistribution objects contain details
print(f"Exponent of traces fit: {polyfit.m_value}")

# %%

print(f"Exponent of branches fit: {polyfit.m_value}")

# %%
# Plot set-wise multi-scale distributions for traces
# -----------------------------------------------------------
# Requires that same azimuth sets are defined in all Networks in the
# MultiNetwork.

# Set names and ranges
# MultiNetwork.collective_azimuth_sets() will check that same azimuth sets are
# set in all Networks.
print(multi_network.collective_azimuth_sets())

# %%

mlds, polyfits, figs, axes = multi_network.plot_trace_azimuth_set_lengths(
    automatic_cut_offs=True,
    plot_truncated_data=True,
)

# Just some visual plot setup...
for fig in figs:
    fig.tight_layout()

# %%
# Plot ternary plots of nodes and branches
# -----------------------------------------------------------

# %%
# Topological XYI-node plot
# ~~~~~~~~~~~~~~~~~~~~~~~~~

fig, ax, tax = multi_network.plot_xyi()

# Visual plot setup
plt.tight_layout()

# %%
# Topological branch type plot
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fig, ax, tax = multi_network.plot_branch()

# Visual plot setup
plt.tight_layout()
