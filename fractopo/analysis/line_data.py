from pathlib import Path
from textwrap import wrap
from dataclasses import dataclass, field
from enum import Enum, unique
import logging

import geopandas as gpd
import matplotlib.patches as patches
import matplotlib

# Math and analysis imports
# Plotting imports
# DataFrame analysis imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ternary
import powerlaw

from scipy.interpolate import CubicSpline

# Own code imports
import fractopo.analysis.tools as tools
import fractopo.analysis.config as config
from fractopo.general import determine_azimuth, determine_set, Col
from fractopo.branches_and_nodes import branches_and_nodes
from fractopo.analysis import length_distributions, azimuth, parameters

from fractopo.general import POWERLAW, LOGNORMAL, EXPONENTIAL
from typing import Dict, Tuple, Union, List, Optional, Literal, Callable, Any
import logging


def _column_array_property(
    column: Literal[Col.AZIMUTH, Col.LENGTH, Col.AZIMUTH_SET, Col.LENGTH_SET],
    gdf: gpd.GeoDataFrame,
) -> Optional[np.ndarray]:
    if column.value in gdf:
        return gdf[column.value].to_numpy()
    else:
        return None


@dataclass
class LineData:
    """
    Wrapper around the given line_gdf (trace or branch data).

    The line_gdf reference is passed and LineData will modify the input
    line_gdf instead of copying the input frame. This means line_gdf
    columns are accesible in the passed input reference.
    """

    line_gdf: gpd.GeoDataFrame

    azimuth_set_ranges: Tuple[Tuple[float, float], ...]
    azimuth_set_names: Tuple[str, ...]

    length_set_ranges: Optional[Tuple[Tuple[float, float], ...]]
    length_set_names: Optional[Tuple[str, ...]]

    # def __post_init__(self):
    #     self.line_gdf = self.line_gdf.copy()

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, (gpd.GeoSeries, gpd.GeoDataFrame)):
            self.__dict__[name] = value.copy()
        else:
            self.__dict__[name] = value

    @property
    def azimuth_array(self):
        column_array = _column_array_property(column=Col.AZIMUTH, gdf=self.line_gdf)
        if column_array is None:
            column_array = np.array(
                [
                    determine_azimuth(line, halved=True)
                    for line in self.line_gdf.geometry
                ]
            )
            self.line_gdf[Col.AZIMUTH.value] = column_array
        return column_array

    @property
    def azimuth_set_array(self):
        column_array = _column_array_property(column=Col.AZIMUTH_SET, gdf=self.line_gdf)
        if column_array is None:
            column_array = np.array(
                [
                    determine_set(
                        azimuth,
                        self.azimuth_set_ranges,
                        self.azimuth_set_names,
                        loop_around=True,
                    )
                    for azimuth in self.azimuth_array
                ]
            )
            self.line_gdf[Col.AZIMUTH_SET.value] = column_array
        return column_array

    @property
    def length_array(self):
        column_array = _column_array_property(column=Col.LENGTH, gdf=self.line_gdf)
        if column_array is None:
            column_array = self.line_gdf.geometry.length.to_numpy()
            self.line_gdf[Col.LENGTH.value] = column_array
        return column_array

    @property
    def length_set_array(self) -> Optional[np.ndarray]:
        if self.length_set_names is None or self.length_set_ranges is None:
            logging.error("Expected length_set_names and _ranges to be defined.")
            return None
        column_array = _column_array_property(column=Col.LENGTH_SET, gdf=self.line_gdf)
        if column_array is None:
            column_array = np.array(
                [
                    determine_set(
                        length,
                        self.length_set_ranges,
                        self.length_set_names,
                        loop_around=False,
                    )
                    for length in self.length_array
                ]
            )
            self.line_gdf[Col.LENGTH_SET.value] = column_array
        return column_array

    @property
    def azimuth_set_counts(self) -> Dict[str, int]:
        return parameters.determine_set_counts(
            self.azimuth_set_names, self.azimuth_set_array
        )

    @property
    def length_set_counts(self) -> Dict[str, int]:
        return parameters.determine_set_counts(
            self.length_set_names, self.length_set_array
        )

    def plot_lengths(
        self, label: str, cut_off: Optional[float] = None
    ) -> Tuple[powerlaw.Fit, matplotlib.figure.Figure, matplotlib.axes.Axes]:  # type: ignore
        return length_distributions.plot_distribution_fits(
            self.length_array, label=label, cut_off=cut_off
        )

    def plot_azimuth(self, label: str) -> Tuple[Dict[str, np.ndarray], matplotlib.figure.Figure, matplotlib.axes.Axes]:  # type: ignore
        return azimuth.plot_azimuth_plot(
            self.azimuth_array,
            self.length_array,
            self.azimuth_set_array,
            self.azimuth_set_names,
            label=label,
        )

    def plot_azimuth_set_count(
        self, label: str
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        return parameters.plot_set_count(self.azimuth_set_counts, label=label)

    def plot_length_set_count(
        self, label: str
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        return parameters.plot_set_count(self.length_set_counts, label=label)
