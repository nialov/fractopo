"""
Automatic azimuth set detection
===============================

Detect axial azimuth sets with ``fractopo.analysis.automatic_azimuth_sets``
and compare the detected centers to a rose plot of the same network.
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
labels, centers = automatic_azimuth_sets(azimuths, n_sets=n_sets, random_state=0)

print(f"Detected {n_sets} sets")
print("Detected center azimuths (degrees):")
pprint(np.round(np.sort(centers), 1))

# %%
# Count how many traces were assigned to each detected set
# --------------------------------------------------------

unique_labels, counts = np.unique(labels, return_counts=True)
for label, center, count in sorted(
    zip(unique_labels, centers, counts), key=lambda row: row[1]
):
    print(f"Set {label}: center={center:.1f}°, count={count}")

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

fig.suptitle(fill("KB11 trace azimuths with automatically detected set centers", 40))
plt.show()

# %%
# Inspect the first few assignments directly
# ------------------------------------------

pprint(list(zip(np.round(azimuths[:10], 1), labels[:10])))
