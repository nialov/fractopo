"""
Automatic azimuth set detection
===============================

Detect axial azimuth set centers and ranges with
``fractopo.analysis.automatic_azimuth_sets`` and compare the detected centers
to a rose plot of the same network.
"""

# %%
# Initializing
# ------------

from pprint import pprint
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np

# Load kb11_network network from examples/example_data.py
from example_data import KB11_NETWORK

from fractopo import Network
from fractopo.analysis.automatic_azimuth_sets import automatic_azimuth_sets

# %%
# Input azimuths
# --------------
#
# Use trace azimuths from the KB11 example network. These are axial azimuths in
# the range [0, 180), so 0° and 180° represent the same direction.

azimuths = KB11_NETWORK.trace_azimuth_array
print(f"Number of trace azimuths: {azimuths.size}")
pprint(azimuths[:10])

# %%
# Detect sets automatically
# -------------------------
#
# Choose the number of sets to detect and keep the example deterministic by
# passing an explicit random state.

n_sets = 3
centers, ranges = automatic_azimuth_sets(azimuths, n_sets=n_sets, random_state=0)

print(f"Detected {n_sets} sets")
print("Detected center azimuths (degrees):")
pprint(np.round(np.sort(centers), 1))

print("Detected set ranges (degrees):")
pprint(tuple(tuple(np.round(range_tuple, 1)) for range_tuple in ranges))

# %%
# Create default set names from ranges
# ------------------------------------

azimuth_set_names = tuple(f"{start:.0f}-{end:.0f}" for start, end in ranges)
pprint(azimuth_set_names)

# %%
# Visualize the detected set centers together with a rose plot
# ------------------------------------------------------------

_, fig, ax = KB11_NETWORK.plot_trace_azimuth()
for center in centers:
    radians = np.deg2rad(center)
    ax.plot([radians, radians], [0, ax.get_ylim()[1]], linestyle="--", linewidth=2)
    ax.plot(
        [radians + np.pi, radians + np.pi],
        [0, ax.get_ylim()[1]],
        linestyle="--",
        linewidth=2,
    )

ax.set_title(
    fill("KB11 trace azimuths with automatically detected set centers", 30),
)
plt.show()

# %%
# Create a new ``Network`` using the automatically detected set ranges
# --------------------------------------------------------------------

kb11_network_automatic_sets = Network(
    trace_gdf=KB11_NETWORK.trace_gdf,
    area_gdf=KB11_NETWORK.area_gdf,
    name="KB11 automatic sets",
    truncate_traces=KB11_NETWORK.truncate_traces,
    circular_target_area=KB11_NETWORK.circular_target_area,
    determine_branches_nodes=KB11_NETWORK.determine_branches_nodes,
    snap_threshold=KB11_NETWORK.snap_threshold,
    azimuth_set_names=azimuth_set_names,
    azimuth_set_ranges=ranges,
)

pprint(
    (
        kb11_network_automatic_sets.azimuth_set_names,
        kb11_network_automatic_sets.azimuth_set_ranges,
    )
)
