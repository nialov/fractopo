"""
Script to create a workflow visualization of ``fractopo``.
"""

import sys
from pathlib import Path
from string import ascii_uppercase
from tempfile import TemporaryDirectory
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
from example_networks import kb11_network
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patheffects import withStroke
from PIL import Image

from fractopo.analysis.network import Network, assign_branch_and_node_colors


def close_fig(func: Callable):
    """
    Wrap function to close any ``matplotlib`` plots after call.
    """

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        plt.close()
        return result

    return wrapper


def plot_area(kb11_network: Network = kb11_network) -> Figure:
    """
    Plot area boundary.
    """
    area_fig, area_ax = plt.subplots()
    # Target area
    kb11_network.area_gdf.boundary.plot(ax=area_ax, color="red")
    # Annotate A to B
    area_ax.axis("off")
    return area_fig


def plot_traces_and_area(kb11_network: Network = kb11_network) -> Figure:
    """
    Plot area boundary along with the traces.
    """
    area_fig, traces_and_area_ax = plt.subplots()
    # Target area

    # Traces and area
    kb11_network.area_gdf.boundary.plot(ax=traces_and_area_ax, color="red")
    kb11_network.trace_gdf.plot(ax=traces_and_area_ax, color="blue")
    traces_and_area_ax.axis("off")
    return area_fig


def plot_branches_and_area(kb11_network: Network = kb11_network) -> Figure:
    """
    Plot area boundary along with the branches.
    """
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
    """
    Plot rose plot.
    """
    import plot_rose_plot

    return plot_rose_plot.fig


def plot_ortho(ax: Axes):
    """
    Plot drone orthomosaic.
    """
    # Ortho
    image_path = Path(__file__).parent.parent / "docs_src/imgs/kb11_orthomosaic.jpg"
    with Image.open(image_path) as image:
        # image = image.convert("RGBA")
        # data = np.ones((image.size[1], image.size[0], 4), dtype=np.uint8) * 255
        # mask = Image.fromarray(data)
        # mask.paste(image, (0, 0))
        ax.imshow(image)


def plot_lengths():
    """
    Plot length distribution plot.
    """
    import plot_length_distributions

    return plot_length_distributions.fig


def plot_xyi():
    """
    Plot XYI node count ternary plot.
    """
    import plot_ternary_plots

    return plot_ternary_plots.xyi_fig


def plot_branches():
    """
    Plot branch count ternary plot.
    """
    import plot_ternary_plots

    return plot_ternary_plots.branch_fig


def plot_relationships():
    """
    Plot azimuth set relationships plot.
    """
    import plot_azimuth_set_relationships

    return plot_azimuth_set_relationships.figs[0]


def add_plot_image_to_ax(figure: Figure, ax: Axes):
    """
    Add a disk-saved plot to an ax.
    """
    with TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "plot.png"
        figure.savefig(save_path, bbox_inches="tight")
        with Image.open(save_path) as image:
            ax.imshow(image)


def arrow_annotation(ax: Axes, start: Tuple[float, float], end: Tuple[float, float]):
    """
    Annotate ax with arrow.
    """
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


def main(output_path: Optional[Path] = None):
    # Initialize figure with 3x3 grid
    figure, axes = plt.subplots(3, 3, figsize=(9, 9))
    assert isinstance(figure, Figure)
    axes_flatten = axes.flatten(order="F")

    # Give explicit names to axes
    (
        area_ax,
        traces_and_area_ax,
        ortho_ax,
        rose_ax,
        length_ax,
        branches_and_area_ax,
        relationships_ax,
        xyi_ax,
        branch_ax,
    ) = axes_flatten

    # Compose the image by adding disk-written images of plots with imshow to all the axes
    add_plot_image_to_ax(ax=area_ax, figure=plot_area())
    add_plot_image_to_ax(ax=traces_and_area_ax, figure=plot_traces_and_area())
    # Orthomosaic is already an image on the disk
    plot_ortho(ax=ortho_ax)
    add_plot_image_to_ax(ax=rose_ax, figure=plot_rose())
    add_plot_image_to_ax(ax=length_ax, figure=plot_lengths())
    add_plot_image_to_ax(ax=relationships_ax, figure=plot_relationships())
    add_plot_image_to_ax(ax=branches_and_area_ax, figure=plot_branches_and_area())
    add_plot_image_to_ax(ax=branch_ax, figure=plot_branches())
    add_plot_image_to_ax(ax=xyi_ax, figure=plot_xyi())

    # Final annotations and setup

    # Arrows
    arrow_annotation(ax=ortho_ax, start=(0.5, 0.9), end=(0.5, 1.2))
    arrow_annotation(ax=area_ax, start=(0.5, 0.1), end=(0.5, -0.2))

    # Traces and area to data
    arrow_annotation(ax=traces_and_area_ax, start=(0.95, 0.5), end=(1.2, 1.5))
    arrow_annotation(ax=traces_and_area_ax, start=(0.95, 0.5), end=(1.2, 0.5))
    arrow_annotation(ax=traces_and_area_ax, start=(0.95, 0.5), end=(1.2, -0.5))

    # Branches to xyi and branch ternary
    arrow_annotation(ax=branches_and_area_ax, start=(0.9, 0.5), end=(1.2, 0.5))
    arrow_annotation(ax=branches_and_area_ax, start=(0.9, 0.5), end=(1.2, 1.35))

    # Rose to relationships
    arrow_annotation(ax=rose_ax, start=(0.9, 0.5), end=(1.1, 0.5))

    area_ax.text(
        x=0.37,
        y=-0.04,
        s="Define target area",
        rotation=90,
        transform=area_ax.transAxes,
        va="center",
    )
    ortho_ax.text(
        x=0.55,
        y=1.1,
        s="Digitize traces",
        rotation=90,
        transform=ortho_ax.transAxes,
        va="center",
    )
    traces_and_area_ax.text(
        x=1.05,
        y=1.1,
        s="Orientation/Azimuth",
        rotation=75,
        transform=traces_and_area_ax.transAxes,
        va="center",
        ha="center",
    )
    traces_and_area_ax.text(
        x=1.12,
        y=0.05,
        s="Topology",
        rotation=-75,
        transform=traces_and_area_ax.transAxes,
        va="center",
        ha="center",
    )
    branches_and_area_ax.text(
        x=1.07,
        y=0.4,
        s="Branches",
        # rotation=90,
        transform=branches_and_area_ax.transAxes,
        va="center",
        ha="center",
    )
    branches_and_area_ax.text(
        x=1.0,
        y=1.0,
        s="Nodes",
        rotation=70,
        transform=branches_and_area_ax.transAxes,
        va="center",
        ha="center",
    )

    # Labels
    for idx, ax in enumerate(axes_flatten):
        assert isinstance(ax, Axes)
        text = ax.text(
            x=0.1,
            y=0.95,
            s=f"{ascii_uppercase[idx]}.",
            transform=ax.transAxes,
            fontsize="xx-large",
        )
        ax.axis("equal")
        text.set_path_effects(
            [
                withStroke(linewidth=2, foreground="white"),
            ]
        )
        ax.axis("off")

    plt.close("all")
    if output_path is not None:
        print(f"Saving workflow plot to {output_path}")
        figure.savefig(output_path, bbox_inches="tight", dpi=150)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    else:
        output_path = None
    main(output_path=output_path)
