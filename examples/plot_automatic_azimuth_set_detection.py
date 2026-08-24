"""
Automatic azimuth set detection
===============================

Detect axial azimuth set centers and ranges with
``fractopo.analysis.automatic_azimuth_sets`` and (optionally) trim them with
``trim_azimuth_set_ranges``. The example compares the detected centers with a
rose plot of the same network.

The definition of azimuth sets based only on the orientation and lengths of
fractures in an area might not reflect how they have actually been formed
geologically. Consequently, any automatic set detection algorithm result should
be critically evaluated during deeper analysis of fracturing in an area.

The example here tries to help with a case when fractures seem to have been,
within the whole target area, clustered in orientation to few specific
directions. However, not all fractures follow these specific directions
and they are, consequently, considered "background" fractures. This is
only one interpretation.
"""

# %%
# Initializing
# ------------

from pprint import pprint
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np

# Load the KB11 network from examples/example_data.py
from example_data import KB11_NETWORK

from fractopo import Network
from fractopo.analysis.automatic_azimuth_sets import (
    automatic_azimuth_sets,
    trim_azimuth_set_ranges,
)

# %%
# Automatic Network initialization
# ---------------------------------
# Network detects, trims, and labels the azimuth sets during initialization.
kb11_network_automatic_sets = Network(
    trace_gdf=KB11_NETWORK.trace_gdf[["geometry"]],
    area_gdf=KB11_NETWORK.area_gdf,
    name="KB11 automatic sets",
    truncate_traces=KB11_NETWORK.truncate_traces,
    circular_target_area=KB11_NETWORK.circular_target_area,
    determine_branches_nodes=KB11_NETWORK.determine_branches_nodes,
    snap_threshold=KB11_NETWORK.snap_threshold,
    n_azimuth_sets=2,
    random_state=0,
)
pprint(np.round(kb11_network_automatic_sets.azimuth_set_centers, 1))
pprint(kb11_network_automatic_sets.azimuth_set_ranges)
pprint(kb11_network_automatic_sets.trace_azimuth_set_counts)

# Plot the centers and the resolved ranges from the Network object.
_, fig, ax = kb11_network_automatic_sets.plot_trace_azimuth(
    visualize_sets=True,
    append_azimuth_set_text=True,
)
for center in kb11_network_automatic_sets.azimuth_set_centers:
    radians = np.deg2rad(center)
    ax.plot([radians, radians], [0, ax.get_ylim()[1]], linestyle="--", linewidth=2)
    ax.plot(
        [radians + np.pi, radians + np.pi],
        [0, ax.get_ylim()[1]],
        linestyle="--",
        linewidth=2,
    )
ax.set_title(fill("Automatic Network azimuth sets", 30))
plt.show()

# %%
# Input azimuths
# --------------
#
# The KB11 trace azimuths are axial values in the range [0, 180). Thus, 0°
# and 180° represent the same direction.

azimuths = KB11_NETWORK.trace_azimuth_array
lengths = KB11_NETWORK.trace_length_array
print(f"Number of trace azimuths: {azimuths.size}")
pprint(azimuths[:10])

# %%
# Detect sets automatically
# -------------------------
#
# Set the number of groups to find. The clustering weights each azimuth by
# fracture length, so longer traces have more influence on the centers.
# ``random_state`` keeps the example output reproducible; omit it otherwise.

n_sets = 2
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
# Plot the detected centers on a rose plot
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
# Narrow the detected ranges and label the remaining fractures as background
# -------------------------------------------------------------------

trimmed_ranges, trimmed_labels = trim_azimuth_set_ranges(
    azimuths,
    lengths,
    ranges,
    retained_length_fraction=0.8,
)
trimmed_set_names = tuple(f"{start:.0f}-{end:.0f}" for start, end in trimmed_ranges)

print("Trimmed set ranges (degrees):")
pprint(tuple(tuple(np.round(range_tuple, 1)) for range_tuple in trimmed_ranges))
print("Classified trace counts, including background fractures:")
pprint(
    {
        str(key): int(val)
        for key, val in zip(*np.unique(trimmed_labels, return_counts=True), strict=True)
    }
)

# %%
# Build a new ``Network`` with the trimmed set ranges
# -----------------------------------------------------

manual_network = Network(
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

pprint(manual_network.trace_azimuth_set_counts)

manual_network.plot_trace_azimuth(
    visualize_sets=True, add_abundance_order=True, append_azimuth_set_text=True
)
