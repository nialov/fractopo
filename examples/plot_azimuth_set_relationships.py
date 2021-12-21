"""
Plotting azimuth set relationships
==================================

The relationships i.e. crosscuts and abutments between lineament & fracture
traces can be determined with ``fractopo``.
"""
# %%

from pprint import pprint

import matplotlib.pyplot as plt

# Load kb11_network network from examples/make_kb11_network.py
from make_kb11_network import kb11_network

# %%
# Analyzing azimuth set relationships
# -----------------------------------
#
# Azimuth sets are set to defaults.

pprint((kb11_network.azimuth_set_names, kb11_network.azimuth_set_ranges))

# %%
# Visualize the relationships with a plot.

figs, _ = kb11_network.plot_azimuth_crosscut_abutting_relationships()

# Edit the figure to better fit the gallery webpage
figs[0].set_size_inches(12, 4)
figs[0].suptitle(
    kb11_network.name,
    fontsize=15,
    fontweight="bold",
    fontfamily="DejaVu Sans",
)
plt.tight_layout()
plt.show()

# %%
# The relationships are also accessible in numerical form as a ``pandas``
# DataFrame.

pprint(kb11_network.azimuth_set_relationships)
