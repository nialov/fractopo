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
from fractopo.analysis.automatic_azimuth_sets import (
    automatic_azimuth_sets,
    trim_azimuth_set_ranges,
)
from fractopo.general import determine_set

# %%
# Input azimuths
# --------------
#
# Use trace azimuths from the KB11 example network. These are axial azimuths in
# the range [0, 180), so 0° and 180° represent the same direction.

azimuths = KB11_NETWORK.trace_azimuth_array
lengths = KB11_NETWORK.trace_length_array
print(f"Number of trace azimuths: {azimuths.size}")
pprint(azimuths[:10])

# %%
# Detect sets automatically
# -------------------------
#
# Choose the number of sets to detect and keep the example deterministic by
# passing an explicit random state. Fracture lengths are used as weights, so
# longer traces influence the detected centers more strongly.

n_sets = 3
centers, ranges = automatic_azimuth_sets(
    azimuths,
    lengths,
    n_sets=n_sets,
    random_state=0,
)

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
# Trim detected ranges and label fractures outside them as background
# -----------------------------------------------------------------

trimmed_ranges, trimmed_labels = trim_azimuth_set_ranges(
    azimuths,
    lengths,
    ranges,
    retained_length_fraction=0.6,
)
trimmed_set_names = tuple(f"{start:.0f}-{end:.0f}" for start, end in trimmed_ranges)

background_labels = np.array(
    [
        determine_set(
            value=azimuth,
            value_ranges=trimmed_ranges,
            set_names=trimmed_set_names,
            loop_around=True,
            null_set="background",
        )
        for azimuth in azimuths
    ]
)

print("Trimmed set ranges (degrees):")
pprint(tuple(tuple(np.round(range_tuple, 1)) for range_tuple in trimmed_ranges))
print("Background-classified trace counts:")
pprint(dict(zip(*np.unique(background_labels, return_counts=True), strict=True)))

# %%
# Create a new ``Network`` using the trimmed set ranges
# -----------------------------------------------------

kb11_network_automatic_sets = Network(
    trace_gdf=KB11_NETWORK.trace_gdf[["geometry"]],
    area_gdf=KB11_NETWORK.area_gdf,
    name="KB11 automatic sets",
    truncate_traces=KB11_NETWORK.truncate_traces,
    circular_target_area=KB11_NETWORK.circular_target_area,
    determine_branches_nodes=KB11_NETWORK.determine_branches_nodes,
    snap_threshold=KB11_NETWORK.snap_threshold,
    azimuth_set_names=trimmed_set_names,
    azimuth_set_ranges=trimmed_ranges,
)

pprint(kb11_network_automatic_sets.trace_azimuth_set_counts)

kb11_network_automatic_sets.plot_trace_azimuth(
    visualize_sets=True, add_abundance_order=True
)
