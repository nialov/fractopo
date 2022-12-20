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
from tempfile import TemporaryDirectory
from typing import Iterator, NamedTuple, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Load kb11_network network from examples/example_networks.py
from example_networks import kb11_network
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patheffects import Stroke, withStroke
from matplotlib.projections import PolarAxes
from PIL import Image, ImageOps

from fractopo.analysis.network import assign_branch_and_node_colors


def arrow_annotation(ax, start: Tuple[float, float], end: Tuple[float, float]):
    ax.annotate(
        "",
        xy=end,
        xycoords="axes fraction",
        xytext=start,
        textcoords="axes fraction",
        arrowprops=dict(facecolor="black", shrink=2.05, width=0.1),
        horizontalalignment="center",
        verticalalignment="top",
        zorder=100,
    )


# %%
# Plotting a rose plot of fracture network trace orientations
# -----------------------------------------------------------

figure = plt.figure(figsize=(9, 9))
gridspec = figure.add_gridspec(3, 3)

# fig, axes = plt.subplots(3, 3, figsize=(9, 9))

# axes_flatten = axes.flatten("F")

area_ax = figure.add_subplot(gridspec[0, 0])
traces_and_area_ax = figure.add_subplot(gridspec[1, 0])
ortho_ax = figure.add_subplot(gridspec[2, 0])
rose_ax = figure.add_subplot(gridspec[0, 1])
length_ax = figure.add_subplot(gridspec[1, 1])
branches_and_area_ax = figure.add_subplot(gridspec[2, 1])
relationships_ax = figure.add_subplot(gridspec[0, 2])
xyi_ax = figure.add_subplot(gridspec[1, 2])
branch_ax = figure.add_subplot(gridspec[2, 2])
# intensity_ax = figure.add_subplot(gridspec[2, 2])

axes_flatten = (
    area_ax,
    traces_and_area_ax,
    ortho_ax,
    rose_ax,
    length_ax,
    branches_and_area_ax,
    relationships_ax,
    xyi_ax,
    branch_ax,
)
assert len(axes_flatten) == 3 * 3
# skip_axes = (length_ax, rose_ax)


def plot_area() -> Figure:
    area_fig, area_ax = plt.subplots()
    # Target area
    kb11_network.area_gdf.boundary.plot(ax=area_ax, color="red")
    # Annotate A to B
    area_ax.axis("off")
    return area_fig


def plot_traces_and_area() -> Figure:
    area_fig, traces_and_area_ax = plt.subplots()
    # Target area

    # Traces and area
    kb11_network.area_gdf.boundary.plot(ax=traces_and_area_ax, color="red")
    kb11_network.trace_gdf.plot(ax=traces_and_area_ax, color="blue")
    traces_and_area_ax.axis("off")
    return area_fig


def plot_branches_and_area() -> Figure:
    area_fig, branches_and_area_ax = plt.subplots()
    # Target area

    # Traces and area
    kb11_network.area_gdf.boundary.plot(ax=branches_and_area_ax, color="red")
    kb11_network.branch_gdf.plot(
        colors=[assign_branch_and_node_colors(bt) for bt in kb11_network.branch_types],
        ax=branches_and_area_ax,
    )
    branches_and_area_ax.axis("off")
    return area_fig


def plot_rose() -> Figure:
    import plot_rose_plot

    return plot_rose_plot.fig


def add_plot_image_to_ax(figure: Figure, ax: Axes):
    with TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "plot.png"
        figure.savefig(save_path, bbox_inches="tight")
        with Image.open(save_path) as image:
            ax.imshow(image)


def plot_ortho():
    # Ortho
    image_path = Path("..") / "docs_src/imgs/kb11_orthomosaic.jpg"
    with Image.open(image_path) as image:
        # image = image.convert("RGBA")
        # data = np.ones((image.size[1], image.size[0], 4), dtype=np.uint8) * 255
        # mask = Image.fromarray(data)
        # mask.paste(image, (0, 0))
        ortho_ax.imshow(image)


def plot_lengths():
    import plot_length_distributions

    return plot_length_distributions.fig


def plot_xyi():
    import plot_ternary_plots

    return plot_ternary_plots.xyi_fig


def plot_branches():
    import plot_ternary_plots

    return plot_ternary_plots.branch_fig


def plot_relationships():
    import plot_azimuth_set_relationships

    return plot_azimuth_set_relationships.figs[0]


add_plot_image_to_ax(ax=area_ax, figure=plot_area())
add_plot_image_to_ax(ax=traces_and_area_ax, figure=plot_traces_and_area())
plot_ortho()
add_plot_image_to_ax(ax=rose_ax, figure=plot_rose())
add_plot_image_to_ax(ax=length_ax, figure=plot_lengths())
add_plot_image_to_ax(ax=relationships_ax, figure=plot_relationships())
add_plot_image_to_ax(ax=branches_and_area_ax, figure=plot_branches_and_area())
add_plot_image_to_ax(ax=branch_ax, figure=plot_branches())
add_plot_image_to_ax(ax=xyi_ax, figure=plot_xyi())

# Final annotations and setup

## Arrows
arrow_annotation(ax=ortho_ax, start=(0.5, 0.9), end=(0.5, 1.2))
arrow_annotation(ax=area_ax, start=(0.5, 0.1), end=(0.5, -0.2))

# traces and area to data
arrow_annotation(ax=traces_and_area_ax, start=(0.9, 0.5), end=(1.2, 1.5))
arrow_annotation(ax=traces_and_area_ax, start=(0.9, 0.5), end=(1.2, 0.5))
arrow_annotation(ax=traces_and_area_ax, start=(0.9, 0.5), end=(1.2, -0.5))

# Branches to xyi and branch ternary
arrow_annotation(ax=branches_and_area_ax, start=(0.9, 0.5), end=(1.2, 0.5))
arrow_annotation(ax=branches_and_area_ax, start=(0.9, 0.5), end=(1.2, 1.35))

# Rose to relationships
arrow_annotation(ax=rose_ax, start=(0.9, 0.5), end=(1.1, 0.5))

## Labels
for idx, ax in enumerate(axes_flatten):
    text = ax.text(
        x=0.1,
        y=0.95,
        s=ascii_uppercase[idx],
        transform=ax.transAxes,
        fontsize="x-large",
    )
    # try:
    #     ax.axis("equal")
    # except ValueError:
    #     # PolarAxes cannot be set to equal
    #     pass
    text.set_path_effects(
        [
            withStroke(linewidth=2, foreground="white"),
        ]
    )
    ax.axis("off")


figure.savefig("tmp.png")
