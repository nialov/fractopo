"""
Visualizing azimuth sets
========================
"""
# %%
# Initializing
# ------------

from pprint import pprint

import matplotlib.pyplot as plt

# Load kb11_network network from examples/make_kb11_network.py
from make_kb11_network import kb11_network

# %%

pprint((kb11_network.azimuth_set_names, kb11_network.azimuth_set_ranges))

# %%

pprint(kb11_network.trace_azimuth_set_counts)

# %%
fig, ax = plt.subplots(figsize=(8, 8))
colors = ("red", "green", "blue")
assert len(colors) == len(kb11_network.azimuth_set_names)
for azimuth_set, set_range, color in zip(
    kb11_network.azimuth_set_names, kb11_network.azimuth_set_ranges, colors
):

    trace_gdf_set = kb11_network.trace_gdf.loc[
        kb11_network.trace_gdf["azimuth_set"] == azimuth_set
    ]

    trace_gdf_set.plot(color=color, label=f"{azimuth_set} - {set_range}", ax=ax)
plt.legend()
plt.show()
