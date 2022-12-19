"""
Visualization of ``fractopo`` workflow.
=======================================
"""
# %%
# Initializing
# ------------

from pathlib import Path
from pprint import pprint
from string import ascii_uppercase
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

# Load kb11_network network from examples/example_networks.py
from example_networks import kb11_network
from matplotlib.patheffects import Stroke, withStroke
from PIL import Image, ImageOps


def arrow_annotation(ax, start: Tuple[float, float], end: Tuple[float, float]):
    ax.annotate(
        "",
        xy=end,
        xycoords="axes fraction",
        xytext=start,
        textcoords="axes fraction",
        arrowprops=dict(facecolor="black", shrink=0.05),
        horizontalalignment="center",
        verticalalignment="top",
    )


# %%
# Plotting a rose plot of fracture network trace orientations
# -----------------------------------------------------------

fig, axes = plt.subplots(3, 3, figsize=(9, 9))

axes_flatten = axes.flatten("F")

area_ax = axes_flatten[0]
traces_and_area_ax = axes_flatten[1]
ortho_ax = axes_flatten[2]


# Target area
kb11_network.area_gdf.boundary.plot(ax=area_ax, color="red")
# Annotate A to B
area_ax = arrow_annotation(ax=area_ax, start=(0.5, 0.1), end=(0.5, -0.2))


# Traces and area
kb11_network.area_gdf.boundary.plot(ax=traces_and_area_ax, color="red")
kb11_network.trace_gdf.plot(ax=traces_and_area_ax, color="blue")


# Ortho
image_path = Path("..") / "docs_src/imgs/kb11_orthomosaic.jpg"
pprint(image_path)
with Image.open(image_path) as image:
    # image = image.convert("RGBA")
    # data = np.ones((image.size[1], image.size[0], 4), dtype=np.uint8) * 255
    # mask = Image.fromarray(data)
    # mask.paste(image, (0, 0))
    ortho_ax.imshow(image)

ortho_ax = arrow_annotation(ax=ortho_ax, start=(0.5, 0.9), end=(0.5, 1.2))

for idx, ax in enumerate(axes_flatten):
    text = ax.text(x=0.1, y=0.95, s=ascii_uppercase[idx], transform=ax.transAxes)
    ax.axis("equal")
    text.set_path_effects(
        [
            withStroke(linewidth=2, foreground="white"),
            # Stroke(linewidth=1, foreground="black"),
        ]
    )
    ax.axis("off")

# fig = plt.figure(figsize=(9, 6))

# width = 0.25
# bottom = 0.1
# height = 0.8

# start_left = width / 2
# end_of_second_column = 0.66

# # left, bottom, width, height
# axes1 = fig.add_axes([start_left, bottom, width, height])  # main axes

# axes2 = fig.add_axes([start_left + width, bottom, width, height / 3])  # inset axes

# plt.tight_layout()
