"""
Automatic azimuth set detection
===============================

This example shows two ways to use automatic azimuth set detection.

The first workflow uses :class:`fractopo.Network` directly. When
``azimuth_set_ranges=None`` (the default), ``Network`` detects azimuth sets
from the processed trace data during initialization, trims the detected ranges,
and stores the resolved ranges and centers for later plots and analyses.

The second workflow uses
``fractopo.analysis.automatic_azimuth_sets.automatic_azimuth_sets`` and
``trim_azimuth_set_ranges`` step by step. Use it to inspect the raw detected
centers and ranges, try parameters such as ``n_sets`` and
``retained_length_fraction``, and iterate before creating a ``Network`` with
manual ranges.

Automatic azimuth sets use only trace orientations and weighted trace lengths.
They are useful for exploration, but the result should still be checked
against the geology of the target area.
"""

# %%
# Imports and example data
# ------------------------
#
# Load the example network and the helper functions used below.

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
# This is the short path. ``Network`` handles the automatic detection and
# trimming during initialization. The resolved definitions are then available
# everywhere the network uses azimuth sets.
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
    # This controls the trimming of the sets (see below workflow about trimming)
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
# This workflow exposes the intermediate results. Use it to inspect what the
# detector found, try different parameters, or decide how tightly to trim the
# ranges before creating a final ``Network``.

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
# Set the number of groups to find. The clustering weights each azimuth by
# fracture length, so longer traces have more influence on the centers.
# ``random_state`` keeps the example output reproducible. In your own work,
# rerun this step with different ``n_sets`` values to compare the result.

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
# One way to inspect the helper output is to build a temporary ``Network``
# from the detected ranges and plot it with set visualization enabled. Print
# the ranges alongside the plot because broad detected ranges can be hard to
# read from the rose plot alone.

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
# The detector returns full set ranges. ``trim_azimuth_set_ranges`` narrows
# them so that each set retains a chosen fraction of its assigned weighted trace
# length. Traces outside the retained ranges fall into the null set. This is a
# useful parameter to adjust when the raw ranges look too broad or too narrow.

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
# After reviewing the helper output, pass the chosen ranges back into
# ``Network`` and continue with a manual azimuth set definition.

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
