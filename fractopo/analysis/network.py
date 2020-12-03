"""
Handles a single target area or a single grouped area. Does not discriminate between a single target area and
grouped target areas.
"""

# Python Windows co-operation imports
from pathlib import Path
from textwrap import wrap
from dataclasses import dataclass, field
from enum import Enum, unique

import geopandas as gpd
import matplotlib.patches as patches

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
from fractopo.general import determine_azimuth, determine_set
from fractopo.branches_and_nodes import branches_and_nodes

from fractopo.analysis.config import POWERLAW, LOGNORMAL, EXPONENTIAL
from typing import Dict, Tuple, Union, List, Optional, Literal
import logging


@unique
class Col(Enum):

    LENGTH_COLUMN = "length"
    AZIMUTH_COLUMN = "azimuth"
    AZIMUTH_SET_COLUMN = "set"


@dataclass
class Network:
    """
    Trace network.

    Consists of at its simplest of validated traces. All other datasets are optional
    but most analyses are locked behind the target area dataset.
    """

    # Base data
    trace_gdf: Union[gpd.GeoSeries, gpd.GeoDataFrame]
    area_geoseries: Optional[Union[gpd.GeoSeries, gpd.GeoDataFrame]] = None

    # Azimuth sets
    azimuth_set_ranges: List[Tuple[float, float]] = [
        (0, 60),
        (60, 120),
        (120, 180),
    ]
    azimuth_set_names: List[str] = ["1", "2", "3"]

    # Branches and nodes
    branch_gdf: Optional[gpd.GeoDataFrame] = None
    node_gdf: Optional[gpd.GeoDataFrame] = None
    determine_branches_nodes: bool = False
    snap_threshold: float = 0.001

    # Length distributions
    length_cut_off: Optional[float] = None

    def __post_init__(self):
        if self.determine_branches_nodes:
            self.assign_branches_nodes()
        if isinstance(self.trace_gdf, gpd.GeoSeries):
            # Transform to GeoDataFrame
            self.trace_gdf = gpd.GeoDataFrame(geometry=self.trace_gdf).set_crs(
                self.trace_gdf.crs
            )

    @property
    def azimuth_array(self) -> np.ndarray:
        column_array = self._column_array_property(column=Col.AZIMUTH_COLUMN)
        if column_array is None:
            column_array = self.trace_gdf.geometry.apply(
                lambda trace: determine_azimuth(trace, halved=True)
            ).to_numpy()
            self.trace_gdf[Col.AZIMUTH_COLUMN] = column_array
        return column_array

    @property
    def length_array(self) -> np.ndarray:
        column_array = self._column_array_property(column=Col.LENGTH_COLUMN)
        if column_array is None:
            column_array = self.trace_gdf.geometry.length.to_numpy()
            self.trace_gdf[Col.LENGTH_COLUMN] = column_array
        return column_array

    @property
    def azimuth_set_array(self) -> np.ndarray:
        column_array = self._column_array_property(column=Col.AZIMUTH_SET_COLUMN)
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
            self.trace_gdf[Col.AZIMUTH_SET_COLUMN] = column_array
        return column_array

    def _column_array_property(
        self,
        column: Literal[Col.AZIMUTH_COLUMN, Col.LENGTH_COLUMN, Col.AZIMUTH_SET_COLUMN],
    ) -> Optional[np.ndarray]:
        if column in self.trace_gdf:
            return self.trace_gdf[column].to_numpy()
        else:
            return None

    def assign_branches_nodes(self):
        if self.area_geoseries is not None:
            branches, nodes = branches_and_nodes(
                self.trace_gdf, self.area_geoseries, self.snap_threshold
            )
            if self.trace_gdf.crs is not None:
                branches.set_crs(self.trace_gdf.crs, inplace=True)
                nodes.set_crs(self.trace_gdf.crs, inplace=True)
            self.branch_gdf = branches
            self.node_gdf = nodes
        else:
            print("Expected area_geoseries to be defined to assign branches and nodes.")
