"""
Automatic azimuth set detection
===============================

This example covers two automatic azimuth-set workflows.

First, :class:`fractopo.analysis.network.Network` detects and trims sets during initialization
when ``azimuth_set_ranges`` is omitted. It stores the resolved ranges and
centers for later analysis and plotting.

The second workflow calls
``fractopo.analysis.automatic_azimuth_sets.automatic_azimuth_sets`` and
``trim_azimuth_set_ranges`` separately. This lets you inspect the detected
centers and ranges, then try different values for ``n_sets`` and
``retained_length_fraction`` before deciding which ranges to use in a
``Network``.

Detection uses trace orientations and weighted trace lengths. Higher retained
fractions, up to ``1``, give tighter ranges. Lower fractions give wider ranges.
Check the results against the geology of the target area.
"""

# %%
# Imports and example data
# ------------------------

from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from example_data import KB11_NETWORK

from fractopo import Network
from fractopo.analysis.automatic_azimuth_sets import (
    automatic_azimuth_sets,
    trim_azimuth_set_ranges,
)

# %%
# Workflow 1: let ``Network`` resolve the azimuth sets
# ----------------------------------------------------
#
# ``Network`` detects and trims the sets during initialization. The resolved
# ranges and centers are available on the network for plotting and analysis.
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
    # Set ranges are not given by user so they are automatically detected
    azimuth_set_ranges=None,
    # Higher values retain tighter ranges; lower values retain wider ranges.
    retained_azimuth_length_fraction=0.8,
)
print("Resolved center azimuths (degrees):")
pprint(np.round(kb11_network_automatic_sets.azimuth_set_centers, 1))
print("Resolved trimmed set ranges (degrees):")
pprint(kb11_network_automatic_sets.azimuth_set_ranges)
print("Trace counts by resolved azimuth set:")
pprint(kb11_network_automatic_sets.trace_azimuth_set_counts)

# The detected centers and trimmed ranges are available on the initialized
# ``Network`` and can be plotted through the normal plotting API.
kb11_network_automatic_sets.plot_trace_azimuth(
    visualize_sets=True, append_azimuth_set_text=True
)
plt.show()

# %%
# Workflow 2: inspect the helper functions and iterate manually
# -------------------------------------------------------------
#
# Inspect the intermediate results and change the parameters before creating
# a Network with the ranges you want to use.

# %%
# Start from the trace azimuths
# -----------------------------
#
# The KB11 trace azimuths are axial values in the range [0, 180). Thus, 0°
# and 180° represent the same direction.

azimuths = KB11_NETWORK.trace_azimuth_array
lengths = KB11_NETWORK.trace_length_array
print(f"Number of trace azimuths: {azimuths.size}")
pprint(azimuths[:10])

# %%
# Detect candidate sets
# ---------------------
#
# Set how many groups to find. Longer traces have more influence on the
# centers because clustering weights each azimuth by trace length. The fixed
# random state makes this example reproducible. Try other ``n_sets`` values
# to compare the results.

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
# Review the detected ranges before trimming
# ------------------------------------------
#
# Plot the detected ranges on a temporary Network. The printed values help
# interpret broad ranges that may be hard to read from the rose plot.

inspection_network = Network(
    trace_gdf=KB11_NETWORK.trace_gdf[["geometry"]],
    area_gdf=KB11_NETWORK.area_gdf,
    name="KB11 detected ranges",
    truncate_traces=KB11_NETWORK.truncate_traces,
    circular_target_area=KB11_NETWORK.circular_target_area,
    determine_branches_nodes=KB11_NETWORK.determine_branches_nodes,
    snap_threshold=KB11_NETWORK.snap_threshold,
    azimuth_set_names=tuple(str(index) for index in range(n_sets)),
    azimuth_set_ranges=ranges,
)

print("Inspection network set ranges (degrees):")
pprint(tuple(tuple(np.round(range_tuple, 1)) for range_tuple in ranges))
inspection_network.plot_trace_azimuth(visualize_sets=True, append_azimuth_set_text=True)
plt.show()

# %%
# Trim the detected ranges
# ------------------------
#
# ``trim_azimuth_set_ranges`` narrows each detected range to retain the chosen
# fraction of its assigned weighted trace length. Traces outside the trimmed
# ranges are assigned to the null set. Adjust the fraction if the ranges look
# too broad or too narrow.

retained_length_fraction = 0.8
trimmed_ranges, trimmed_labels = trim_azimuth_set_ranges(
    azimuths,
    lengths,
    ranges,
    retained_length_fraction=retained_length_fraction,
)
trimmed_set_names = tuple(f"{start:.0f}-{end:.0f}" for start, end in trimmed_ranges)

print("Trimmed set ranges (degrees):")
pprint(tuple(tuple(np.round(range_tuple, 1)) for range_tuple in trimmed_ranges))
print("Classified trace counts, including traces in the null set:")
pprint(
    {
        str(key): int(val)
        for key, val in zip(*np.unique(trimmed_labels, return_counts=True), strict=True)
    }
)

# %%
# Build a final ``Network`` from the inspected result
# ---------------------------------------------------
#
# Pass the ranges you want to keep into ``Network`` as a manual definition.

tuned_network = Network(
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

print("Trace counts by tuned azimuth set:")
pprint(tuned_network.trace_azimuth_set_counts)

tuned_network.plot_trace_azimuth(
    visualize_sets=True, add_abundance_order=True, append_azimuth_set_text=True
)
plt.show()
