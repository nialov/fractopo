"""
Utilities for randomly Network sampling traces.
"""
from typing import Optional, Tuple, Union

import geopandas as gpd
import numpy as np
from pydantic import BaseModel, validator
from shapely.geometry import LineString, Point, Polygon

from fractopo.analysis.network import Network
from fractopo.general import (
    GEOMETRY_COLUMN,
    random_points_within,
    safe_buffer,
    calc_circle_area,
    calc_circle_radius,
)
from enum import Enum, unique


@unique
class RandomChoice(Enum):

    """
    Choose between random area or radius.

    The choise is relevant because area is polynomially correlated with radius.
    """

    area = "area"
    radius = "radius"


class NetworkRandomSampler(BaseModel):

    """
    Randomly sample traces inside given target area.
    """

    trace_gdf: gpd.GeoDataFrame
    area_gdf: gpd.GeoDataFrame
    min_radius: float
    snap_threshold: float
    random_choice: Union[RandomChoice, str]

    class Config:

        """
        Configure NetworkRandomSampler.
        """

        arbitrary_types_allowed = True

    @validator("random_choice")
    def random_choice_should_be_enum(cls, value):
        """
        Check that random_choice is valid.
        """
        # Convert strings if they match any enum values
        if isinstance(value, str):
            for choice_enum in RandomChoice:
                if value == choice_enum.value:
                    return choice_enum
        # Type should be checked by pydantic already
        else:
            return value

    @validator("trace_gdf")
    def trace_gdf_should_contain_traces(cls, value):
        """
        Check that trace_gdf contains LineString traces.
        """
        if not value.shape[0] > 0:
            raise ValueError("Expected non-empty trace_gdf.")
        if not all([isinstance(val, LineString) for val in value.geometry.values]):
            raise TypeError("Expected only LineStrings in trace_gdf.")
        return value

    @validator("area_gdf")
    def area_gdf_should_contain_polygon(cls, value):
        """
        Check that area_gdf contains one Polygon.
        """
        if not value.shape[0] == 1:
            raise ValueError("Expected area_gdf with one geometry.")
        geom = value.geometry.values[0]
        if not isinstance(geom, Polygon):
            raise TypeError("Expected one Polygon in area_gdf.")
        return value

    @validator("min_radius", "snap_threshold")
    def value_should_be_positive(cls, value):
        """
        Check that value is positive.
        """
        if not value > 0:
            raise ValueError("Expected positive non-zero value.")
        return value

    @property
    def target_circle(self) -> Polygon:
        """
        Target circle Polygon from area_gdf.
        """
        return self.area_gdf.geometry.values[0]

    @property
    def max_radius(self) -> float:
        """
        Calculate max radius from given area_gdf.
        """
        radius = np.sqrt(self.target_circle.area / np.pi)
        if self.min_radius > radius:
            raise ValueError("Expected min_radius smaller than max_radius.")
        return radius

    @property
    def min_area(self) -> float:
        """
        Calculate minimum area from min_radius.
        """
        return calc_circle_area(self.min_radius)

    @property
    def max_area(self) -> float:
        """
        Calculate maximum area from max_radius.
        """
        return calc_circle_area(self.max_radius)

    def random_radius(self) -> float:
        """
        Calculate random radius in range [min_radius, max_radius[.
        """
        radius_range = self.max_radius - self.min_radius
        radius = self.min_radius + np.random.random_sample() * radius_range
        return radius

    def random_area(self) -> float:
        """
        Calculate random area in area range.

        Range is calculated from [min_radius, max_radius[.
        """
        area_range = self.max_area - self.min_area
        area = self.min_area + np.random.random_sample() * area_range
        return area

    @property
    def target_area_centroid(self) -> Point:
        """
        Get target area centroid.
        """
        centroid = self.target_circle.centroid
        if not isinstance(centroid, Point):
            raise TypeError("Expected Point as centroid.")
        return centroid

    def random_target_circle(self) -> Tuple[Polygon, Point, float]:
        """
        Get random target area and its centroid and radius.

        The target area is always within the original target area.
        """
        radius = (
            self.random_radius()
            if self.random_choice == RandomChoice.radius
            else calc_circle_radius(self.random_area())
        )
        possible_buffer: Polygon = safe_buffer(
            self.target_area_centroid, self.max_radius - radius
        )
        random_target_centroid = random_points_within(possible_buffer, 1)[0]
        random_target_circle: Polygon = safe_buffer(random_target_centroid, radius)

        return random_target_circle, random_target_centroid, radius

    def random_network_sample(self) -> Tuple[Optional[Network], Point, float]:
        """
        Get random Network sample with a random target area.

        Returns the network, the sample circle centroid and circle radius.
        """
        target_circle, target_centroid, radius = self.random_target_circle()
        area_gdf = gpd.GeoDataFrame({GEOMETRY_COLUMN: [target_circle]})
        if self.trace_gdf.crs is not None:
            area_gdf = area_gdf.set_crs(self.trace_gdf.crs)
        try:
            network = Network(
                trace_gdf=self.trace_gdf,
                area_gdf=area_gdf,
                name=target_centroid.wkt,
                determine_branches_nodes=True,
                snap_threshold=self.snap_threshold,
            )
        except ValueError:
            network = None

        return network, target_centroid, radius
