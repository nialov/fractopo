"""
Plotting azimuth set relationships
==================================

The relationships i.e. crosscuts and abutments between lineament & fracture
traces can be determined with ``fractopo``.
"""
# %%

from pprint import pprint

import matplotlib as mpl
import matplotlib.pyplot as plt

# Load kb11_network network from examples/example_data.py
from example_data import KB11_NETWORK

mpl.rcParams["figure.figsize"] = (5, 5)
mpl.rcParams["font.size"] = 8

# %%
# Analyzing azimuth set relationships
# -----------------------------------
#
# Azimuth sets (set by user):

pprint((KB11_NETWORK.azimuth_set_names, KB11_NETWORK.azimuth_set_ranges))

# %%
# Visualize the relationships with a plot.

figs, _ = KB11_NETWORK.plot_azimuth_crosscut_abutting_relationships()

# Edit the figure to better fit the gallery webpage
figs[0].suptitle(
    KB11_NETWORK.name,
    fontsize="large",
    fontweight="bold",
    fontfamily="DejaVu Sans",
)
plt.tight_layout()
plt.show()

# %%
# The relationships are also accessible in numerical form as a ``pandas``
# DataFrame.

pprint(KB11_NETWORK.azimuth_set_relationships)
