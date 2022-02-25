"""
Optimizing multi-scale cut-offs with ``fractopo``
========================================================

This functionality is very much so a **work-in-progress**. Optimization of a
power-law fit for continuous, "single-scale", data is easily handled by the
functionality provided by ``powerlaw`` but it does not directly translate to
the required methods for multi-scale data where normalization of the
complementary cumulative number (=ccm) has been done.
"""
# %%
# Initializing
# ------------

import matplotlib.pyplot as plt

# Load three networks, each digitized from a different scale of observation
from example_networks import hastholmen_network, kb11_network, lidar_200k_network

from fractopo.analysis import length_distributions

# %%
# Collect LengthDistributions into MultiLengthDistribution
# ------------------------------------------------------------------

networks = [kb11_network, hastholmen_network, lidar_200k_network]

distributions = [netw.trace_length_distribution(azimuth_set=None) for netw in networks]

mld = length_distributions.MultiLengthDistribution(
    distributions=distributions,
    using_branches=False,
    fitter=length_distributions.scikit_linear_regression,
)

# %%
# Use scipy.optimize.shgo to optimize cut-off values
# ------------------------------------------------------------------

shgo_kwargs = dict(sampling_method="sobol")

# Returns new instance of MultiLengthDistribution
opt_result, opt_mld = mld.optimize_cut_offs()

# Use optimized MultiLengthDistribution to plot distributions and fit
polyfit, fig, ax = opt_mld.plot_multi_length_distributions(automatic_cut_offs=False)

# Print optimized cut-offs
print("Optimized cut-offs:")
print(opt_result.optimize_result.x)

# Print resulting exponent
print("Resulting power-law exponent:")
print(opt_result.polyfit.m_value)

# Visual plot setup
fig.set_size_inches(5, 5)
plt.tight_layout()
