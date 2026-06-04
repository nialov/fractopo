"""
Automatic azimuth set detection
===============================

Detect axial azimuth sets with ``fractopo.analysis.automatic_azimuth_sets``
and compare the detected centers to a rose plot of the same network.
"""

# %%
# Initializing
# ------------

from pathlib import Path
from pprint import pprint

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from fractopo import Network
from fractopo.analysis.automatic_azimuth_sets import automatic_azimuth_sets

PROJECT_BASE_PATH = Path().resolve().parent

# %%
# Input azimuths
# --------------
#
# Use trace azimuths from the KB7 example network. These are axial azimuths in
# the range [0, 180), so 0° and 180° represent the same direction.

trace_gdf = gpd.read_file(PROJECT_BASE_PATH / "tests/sample_data/KB7/KB7_traces.geojson")
area_gdf = gpd.read_file(PROJECT_BASE_PATH / "tests/sample_data/KB7/KB7_area.geojson")
kb7_network = Network(
    name="KB7",
    trace_gdf=trace_gdf,
    area_gdf=area_gdf,
    truncate_traces=True,
    circular_target_area=False,
    determine_branches_nodes=True,
    snap_threshold=0.001,
)

azimuths = kb7_network.trace_azimuth_array
print(f"Number of trace azimuths: {azimuths.size}")
pprint(azimuths[:10])

# %%
# Detect sets automatically
# -------------------------
#
# Choose the number of sets to detect and keep the example deterministic by
# passing an explicit random state.

n_sets = 3
labels, centers = automatic_azimuth_sets(azimuths, n_sets=n_sets, random_state=0)

print(f"Detected {n_sets} sets")
print("Detected center azimuths (degrees):")
pprint(np.round(np.sort(centers), 1))

# %%
# Count how many traces were assigned to each detected set
# --------------------------------------------------------

unique_labels, counts = np.unique(labels, return_counts=True)
for label, center, count in sorted(zip(unique_labels, centers, counts), key=lambda row: row[1]):
    print(f"Set {label}: center={center:.1f}°, count={count}")

# %%
# Visualize the detected set centers together with a rose plot
# ------------------------------------------------------------

_, fig, ax = kb7_network.plot_trace_azimuth()
for center in centers:
    radians = np.deg2rad(center)
    ax.plot([radians, radians], [0, ax.get_ylim()[1]], linestyle="--", linewidth=2)
    ax.plot([radians + np.pi, radians + np.pi], [0, ax.get_ylim()[1]], linestyle="--", linewidth=2)

ax.set_title("KB7 trace azimuths with automatically detected set centers")
plt.show()

# %%
# Inspect the first few assignments directly
# ------------------------------------------

pprint(list(zip(np.round(azimuths[:10], 1), labels[:10])))
