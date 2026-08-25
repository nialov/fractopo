"""
Automatic azimuth set detection
===============================

This example shows two ways to work with automatic azimuth set detection.

The first workflow uses :class:`fractopo.Network` directly. When
``azimuth_set_ranges=None`` (the default), ``Network`` detects azimuth sets
from the processed trace data during initialization. It also trims the detected
ranges and stores the resolved ranges and centers for later plots and analyses.

The second workflow uses
``fractopo.analysis.automatic_azimuth_sets.automatic_azimuth_sets`` and
``trim_azimuth_set_ranges`` step by step. That is more useful when you want to
inspect the detected centers and ranges, adjust parameters such as ``n_sets``
and ``retained_length_fraction``, and iterate before creating a ``Network``
with manually chosen ranges.

Automatic azimuth sets are based only on trace orientations and weighted trace
lengths. They are useful for exploratory work, but the result should still be
checked against the geology of the target area.
"""

# %%
# Initializing
# ------------

from pprint import pprint

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
# Workflow 1: let ``Network`` resolve the azimuth sets
# ----------------------------------------------------
#
# This is the short path. ``Network`` handles the automatic detection during
# initialization. The resolved definitions are then available everywhere the
# network uses azimuth sets.
kb11_network_automatic_sets = Network(
    trace_gdf=KB11_NETWORK.trace_gdf[["geometry"]],
    area_gdf=KB11_NETWORK.area_gdf,
    name="KB11 automatic sets",
    truncate_traces=KB11_NETWORK.truncate_traces,
    circular_target_area=KB11_NETWORK.circular_target_area,
    determine_branches_nodes=KB11_NETWORK.determine_branches_nodes,
    snap_threshold=KB11_NETWORK.snap_threshold,
    azimuth_set_ranges=None,
    n_azimuth_sets=2,
    random_state=0,
)
pprint(np.round(kb11_network_automatic_sets.azimuth_set_centers, 1))
pprint(kb11_network_automatic_sets.azimuth_set_ranges)
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
# This workflow exposes the intermediate results. Use it when you want to see
# what the detector found, try different parameters, or decide how tightly to
# trim the detected ranges before creating a final ``Network``.

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
# ``random_state`` keeps the example output reproducible. In your own work you
# can rerun this step with different ``n_sets`` values to compare the result.

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
# One practical way to inspect the helper output is to build a temporary
# ``Network`` from the detected ranges and plot it with set visualization
# enabled. That lets you compare helper output with the higher-level workflow.

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

inspection_network.plot_trace_azimuth(visualize_sets=True, append_azimuth_set_text=True)
plt.show()

# %%
# Trim the detected ranges
# ------------------------
#
# The detector returns full set ranges. ``trim_azimuth_set_ranges`` narrows
# them so that each set retains a chosen fraction of its assigned weighted trace
# length. Traces outside the retained ranges fall into the null set. This is a
# good parameter to iterate when the raw ranges look too broad or too narrow.

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
# After reviewing the helper output, you can pass the chosen ranges back into
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

pprint(tuned_network.trace_azimuth_set_counts)

tuned_network.plot_trace_azimuth(
    visualize_sets=True, add_abundance_order=True, append_azimuth_set_text=True
)
plt.show()
