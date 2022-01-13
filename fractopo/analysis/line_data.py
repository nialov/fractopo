"""
Trace and branch data analysis with LineData class abstraction.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import powerlaw
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.projections import PolarAxes

from fractopo.analysis import azimuth, length_distributions, parameters
from fractopo.general import (
    BOUNDARY_INTERSECT_KEYS,
    Col,
    SetRangeTuple,
    determine_azimuth,
    determine_set,
    intersection_count_to_boundary_weight,
    numpy_to_python_type,
    raise_determination_error,
)


def _column_array_property(
    column: Col,
    gdf: gpd.GeoDataFrame,
) -> Optional[np.ndarray]:
    """
    Return column values from GeoDataFrame if it exists in the GeoDataFrame.
    """
    if column.value in gdf:
        values = gdf[column.value]
        assert isinstance(values, pd.Series)
        return values.to_numpy()
    return None


@dataclass
class LineData:

    """
    Wrapper around the given GeoDataFrame with trace or branch data.

    The line_gdf reference is passed and LineData will modify the input
    line_gdf instead of copying the input frame. This means line_gdf columns
    are accessible in the passed input reference upstream.
    """

    _line_gdf: gpd.GeoDataFrame

    azimuth_set_ranges: SetRangeTuple
    azimuth_set_names: Tuple[str, ...]

    length_set_ranges: SetRangeTuple = ()
    length_set_names: Tuple[str, ...] = ()

    area_boundary_intersects: np.ndarray = np.array([])

    _automatic_fit: Optional[powerlaw.Fit] = None

    def __getattr__(self, __name: str) -> Any:
        """
        Overwrite __getattr__ to warn about accessing _line_gdf.
        """
        if __name == "_line_gdf":
            logging.error(
                "Output line_gdf might not have all column attributes defined.\n"
                "Use LineData attributes instead of getting the GeoDataFrame."
            )
        return getattr(self, __name)

    @property
    def azimuth_array(self):
        """
        Array of trace or branch azimuths.
        """
        column_array = _column_array_property(column=Col.AZIMUTH, gdf=self._line_gdf)
        if column_array is None:
            column_array = np.array(
                [determine_azimuth(line, halved=True) for line in self.geometry]
            )
            self._line_gdf[Col.AZIMUTH.value] = column_array
        return column_array

    @property
    def azimuth_set_array(self):
        """
        Array of trace or branch azimuth set ids.
        """
        column_array = _column_array_property(
            column=Col.AZIMUTH_SET, gdf=self._line_gdf
        )
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
            self._line_gdf[Col.AZIMUTH_SET.value] = column_array
        return column_array

    @property
    def length_boundary_weights(self):
        """
        Array of weights for lines based on intersection count with boundary.
        """
        column_array = _column_array_property(
            column=Col.LENGTH_WEIGHTS, gdf=self._line_gdf
        )
        if column_array is None:
            assert self.area_boundary_intersects.dtype in ("int64", "float64")
            column_array = np.array(
                [
                    intersection_count_to_boundary_weight(int(inter_count))
                    for inter_count in self.area_boundary_intersects
                ]
            )
            self._line_gdf[Col.LENGTH_WEIGHTS.value] = column_array
        return column_array

    @property
    def length_array_non_weighted(self):
        """
        Array of trace or branch lengths not weighted by boundary conditions.
        """
        column_array = _column_array_property(
            column=Col.LENGTH_NON_WEIGHTED, gdf=self._line_gdf
        )
        if column_array is None:
            column_array = self.geometry.length.to_numpy()
            self._line_gdf[Col.LENGTH_NON_WEIGHTED.value] = column_array
        return column_array

    @property
    def length_array(self) -> np.ndarray:
        """
        Array of trace or branch lengths.

        Note: lengths can be 0.0 due to boundary weighting.
        """
        column_array = _column_array_property(column=Col.LENGTH, gdf=self._line_gdf)
        if column_array is None:
            new_column_array = (
                self.geometry.length.to_numpy() * self.length_boundary_weights
                if len(self.area_boundary_intersects) > 0
                else 1.0
            )
            self._line_gdf[Col.LENGTH.value] = new_column_array

        else:
            new_column_array = column_array
        assert isinstance(new_column_array, np.ndarray)
        return new_column_array

    @property
    def length_set_array(self) -> np.ndarray:
        """
        Array of trace or branch length set ids.
        """
        if len(self.length_set_names) == 0 or len(self.length_set_ranges) == 0:
            logging.error("Expected length_set_names and _ranges to be non-empty.")
            raise_determination_error(
                "length_set_array",
                determine_target="length set attributes",
                verb="initializing",
            )
        column_array = _column_array_property(column=Col.LENGTH_SET, gdf=self._line_gdf)
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
            self._line_gdf[Col.LENGTH_SET.value] = column_array
        return column_array

    @property
    def azimuth_set_counts(self) -> Dict[str, int]:
        """
        Get dictionary of azimuth set counts.
        """
        return parameters.determine_set_counts(
            self.azimuth_set_names, self.azimuth_set_array
        )

    @property
    def length_set_counts(self) -> Dict[str, int]:
        """
        Get dictionary of length set counts.
        """
        return parameters.determine_set_counts(
            self.length_set_names, self.length_set_array
        )

    @property
    def automatic_fit(self) -> powerlaw.Fit:
        """
        Get automatic powerlaw Fit.
        """
        if self._automatic_fit is None:
            self._automatic_fit = length_distributions.determine_fit(self.length_array)
        return self._automatic_fit

    @property
    def boundary_intersect_count(self) -> Dict[str, int]:
        """
        Get counts of line intersects with boundary.
        """
        assert len(self.area_boundary_intersects) > 0
        keys, counts = np.unique(self.area_boundary_intersects, return_counts=True)
        keys = list(map(str, keys))
        counts = list(map(numpy_to_python_type, counts))
        key_counts = dict(zip(keys, counts))
        for default_key in BOUNDARY_INTERSECT_KEYS:
            if default_key not in key_counts:
                key_counts[default_key] = 0
        assert len(key_counts) == 3
        assert all(np.isin(BOUNDARY_INTERSECT_KEYS, list(key_counts)))
        return key_counts

    @property
    def geometry(self) -> gpd.GeoSeries:
        """
        Get line geometries.
        """
        return self._line_gdf.geometry

    def determine_manual_fit(self, cut_off: float) -> powerlaw.Fit:
        """
        Get manually determined Fit with set cut off.
        """
        return length_distributions.determine_fit(self.length_array, cut_off=cut_off)

    # def cut_off_proportion_of_data(self, fit: Optional[powerlaw.Fit] = None) -> float:
    #     """
    #     Get the proportion of data cut off by `powerlaw` cut off.

    #     If no fit is passed the cut off is the one used in `automatic_fit`.
    #     """
    #     fit = self.automatic_fit if fit is None else fit
    #     return (
    #         sum(self.length_array < fit.xmin) / len(self.length_array)
    #         if len(self.length_array) > 0
    #         else 0.0
    #     )

    def describe_fit(
        self, label: Optional[str] = None, cut_off: Optional[float] = None
    ):
        """
        Return short description of automatic powerlaw fit.
        """
        fit = (
            self.automatic_fit
            if cut_off is None
            else self.determine_manual_fit(cut_off=cut_off)
        )
        return length_distributions.describe_powerlaw_fit(
            fit, label=label, length_array=self.length_array
        )

    def plot_lengths(
        self,
        label: str,
        fit: Optional[powerlaw.Fit] = None,
    ) -> Tuple[powerlaw.Fit, Figure, Axes]:
        """
        Plot length data with powerlaw fit.
        """
        return length_distributions.plot_distribution_fits(
            self.length_array,
            label=label,
            fit=self.automatic_fit if fit is None else fit,
        )

    def plot_azimuth(
        self, label: str, append_azimuth_set_text: bool = False
    ) -> Tuple[azimuth.AzimuthBins, Figure, PolarAxes]:
        """
        Plot azimuth data in rose plot.
        """
        return azimuth.plot_azimuth_plot(
            self.azimuth_array,
            self.length_array,
            self.azimuth_set_array,
            self.azimuth_set_names,
            label=label,
            append_azimuth_set_text=append_azimuth_set_text,
        )

    def plot_azimuth_set_count(self, label: str) -> Tuple[Figure, Axes]:
        """
        Plot azimuth set counts.
        """
        return parameters.plot_set_count(self.azimuth_set_counts, label=label)

    def plot_length_set_count(self, label: str) -> Tuple[Figure, Axes]:
        """
        Plot length set counts.
        """
        return parameters.plot_set_count(self.length_set_counts, label=label)

    def boundary_intersect_count_desc(self, label: str) -> Dict[str, int]:
        """
        Get counts of line intersects with boundary.
        """
        key_counts = self.boundary_intersect_count
        intersect_key_counts = dict()
        for key, item in key_counts.items():
            # trace_key_counts[f"Trace Boundary {key} Intersect Count"] = item
            intersect_key_counts[f"{label} Boundary {key} Intersect Count"] = item
        return intersect_key_counts
