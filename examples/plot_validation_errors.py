"""
Visualize different types of validatation errors
================================================

A network consists of the geometrical traces and their interactions with
each other.
"""

# %%
# Imports
# -------

import logging
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure

# Import the geometries used to create traces and target areas.
from shapely.geometry import LineString, Point

# %%
# Traces are explicitly defined in code and plotted
# -------------------------------------------------

# Initialization
fig: Figure
fig, axes = plt.subplots(2, 2, figsize=(7, 7))
fig.tight_layout(h_pad=1.5)

axes_flat: Sequence[Axes] = axes.flatten()


def label_gen():
    for label in ("A.", "B.", "C.", "D."):
        yield label


labeler = label_gen()
for ax in axes_flat:
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.axis("off")
    ax.text(x=-5, y=5, s=next(labeler), fontsize="x-large", fontweight="bold")

# Axis 1: MULTI JUNCTION

axis_1 = axes_flat[0]

traces_1 = gpd.GeoDataFrame(
    geometry=[
        LineString([(-5, 0), (5, 0)]),
        LineString([(0, 5), (0, -5)]),
        LineString([(-2.5, 2.5), (2.5, -2.5)]),
    ]
)
errors_1 = gpd.GeoDataFrame(geometry=[Point(0, 0)])
traces_1.plot(ax=axis_1, color="black")
errors_1.plot(ax=axis_1, marker="X", color="red", zorder=10)
axis_1.text(
    x=0, y=-7, s="More than two traces" "\n" "in the same intersection.", ha="center"
)

# Axis 2: MULTI JUNCTION

axis_2 = axes_flat[1]

traces_2 = gpd.GeoDataFrame(
    geometry=[
        LineString([(-2.5, 2.5), (0.5, -0.5)]),
        LineString([(-5, -5), (5, 5)]),
    ]
)
traces_2.plot(ax=axis_2, color="black")
axis_2.annotate(
    "Overlap distance higher\n than defined snap threshold.",
    xy=(0.5, -0.5),
    xytext=(-0.5, -4),
    arrowprops=dict(arrowstyle="->", color="red"),
    fontstyle="italic",
    fontsize="small",
)
# axis_2.set_title("Trace overlaps another trace.")
axis_2.text(x=0, y=-7, s="Trace overlaps another trace.", ha="center")

# Axis 3: V NODE

axis_3 = axes_flat[2]

traces_3 = gpd.GeoDataFrame(
    geometry=[
        LineString([(-2.5, 2.5), (0, 0)]),
        LineString([(-2.5, -5), (0, 0)]),
    ]
)
errors_3 = gpd.GeoDataFrame(geometry=[Point(0, 0)])
traces_3.plot(ax=axis_3, color="black")
errors_3.plot(ax=axis_3, marker="X", color="red", zorder=10)
# axis_3.set_title("Two traces end in a\nV-node formation.")
axis_3.text(x=0, y=-7, s="Two traces end in a\nV-node formation.", ha="center")

# Axis 4: MULTIPLE CROSSCUTS

axis_4 = axes_flat[3]

traces_4 = gpd.GeoDataFrame(
    geometry=[
        LineString([(-5, 0), (5, 0)]),
        LineString([(-5, 1), (-2, -1), (1, 1), (5, -1)]),
    ]
)
intersections = traces_4.geometry.values[0].intersection(traces_4.geometry.values[1])

errors_4 = gpd.GeoDataFrame(geometry=list(intersections.geoms))

traces_4.plot(ax=axis_4, color="black")
errors_4.plot(ax=axis_4, marker="X", color="red", zorder=10)
# axis_4.set_title("Two traces cross each\nother more than two times.")
axis_4.text(
    x=0, y=-7, s="Two traces cross each\nother more than two times.", ha="center"
)

plt.subplots_adjust(wspace=0.01)

if __name__ == "__main__":
    # Save plot for usage outside sphinx
    # This section can be ignored if looking at the documentation
    try:
        output_path = Path(__file__).parent / "validation_errors.svg"
        fig.savefig(output_path, bbox_inches="tight")
    except Exception:
        # Log error due to e.g. execution as jupyter notebook
        logging.info(f"Failed to save validation_errors.svg plot.", exc_info=True)
