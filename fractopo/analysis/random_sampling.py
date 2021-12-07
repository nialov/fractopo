"""
Utilities for randomly Network sampling traces.
"""
import logging
from dataclasses import dataclass
from enum import Enum, unique
from typing import Optional, Tuple, Union

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from fractopo.analysis.network import Network
from fractopo.general import (
    GEOMETRY_COLUMN,
    calc_circle_area,
    calc_circle_radius,
    numpy_to_python_type,
    random_points_within,
    safe_buffer,
)


@unique
class RandomChoice(Enum):

    """
    Choose between random area or radius.

    The choise is relevant because area is polynomially correlated with radius.
    """

    area = "area"
    radius = "radius"


@dataclass
class RandomSample:

    """
    Dataclass for sampling results.
    """

    network_maybe: Optional[Network]
    target_centroid: Point
    radius: float
    name: str


@dataclass
class NetworkRandomSampler:

    """
    Randomly sample traces inside given target area.
    """

    trace_gdf: gpd.GeoDataFrame
    area_gdf: gpd.GeoDataFrame
    min_radius: float
    snap_threshold: float
    random_choice: Union[RandomChoice, str]
    name: str

    def __post_init__(self):
        """
        Validate inputs post-initialization.
        """
        self.random_choice = self.random_choice_should_be_enum(self.random_choice)
        self.trace_gdf = self.trace_gdf_should_contain_traces(self.trace_gdf)
        self.area_gdf = self.area_gdf_should_contain_polygon(self.area_gdf)
        self.min_radius = self.value_should_be_positive(self.min_radius)

    @staticmethod
    def random_choice_should_be_enum(
        random_choice: Union[RandomChoice, str]
    ) -> RandomChoice:
        """
        Check that random_choice is valid.
        """
        # Convert strings if they match any enum values
        if isinstance(random_choice, str):
            for choice_enum in RandomChoice:
                if random_choice == choice_enum.value:
                    return choice_enum

            raise TypeError(
                "Expected random_choice to be"
                f" convertable to RandomChoice enum: {random_choice}."
            )
        return random_choice

    @staticmethod
    def trace_gdf_should_contain_traces(
        trace_gdf: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """
        Check that trace_gdf contains LineString traces.
        """
        if not trace_gdf.shape[0] > 0:
            raise ValueError("Expected non-empty trace_gdf.")
        if not all(isinstance(val, LineString) for val in trace_gdf.geometry.values):
            raise TypeError("Expected only LineStrings in trace_gdf.")
        return trace_gdf

    @staticmethod
    def area_gdf_should_contain_polygon(area_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Check that area_gdf contains one Polygon.
        """
        if not area_gdf.shape[0] == 1:
            raise ValueError("Expected area_gdf with one geometry.")
        geom = area_gdf.geometry.values[0]
        if not isinstance(geom, Polygon):
            raise TypeError("Expected one Polygon in area_gdf.")
        return area_gdf

    @staticmethod
    def value_should_be_positive(min_radius: float) -> float:
        """
        Check that value is positive.
        """
        if not min_radius > 0:
            raise ValueError(f"Expected positive non-zero min_radius: {min_radius}.")
        return min_radius

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
        radius = numpy_to_python_type(np.sqrt(self.target_circle.area / np.pi))
        assert isinstance(radius, float)
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

    def random_network_sample(self, determine_branches_nodes=True) -> RandomSample:
        """
        Get random Network sample with a random target area.

        Returns the network, the sample circle centroid and circle radius.
        """
        # Create random Polygon circle within area
        target_circle, target_centroid, radius = self.random_target_circle()

        # Collect into GeoDataFrame and set crs if it exists in input frame
        area_gdf = gpd.GeoDataFrame({GEOMETRY_COLUMN: [target_circle]})
        if self.trace_gdf.crs is not None:
            area_gdf = area_gdf.set_crs(self.trace_gdf.crs)

        try:
            network_maybe: Optional[Network] = Network(
                trace_gdf=self.trace_gdf,
                area_gdf=area_gdf,
                name=self.name,
                determine_branches_nodes=determine_branches_nodes,
                snap_threshold=self.snap_threshold,
                circular_target_area=True,
                truncate_traces=True,
            )
        except ValueError:
            logging.error(
                "Error occurred during creation of random_network_sample.",
                exc_info=True,
                extra=dict(
                    sampler_name=self.name,
                    target_centroid=target_centroid,
                    radius=radius,
                ),
            )
            network_maybe = None

        return RandomSample(
            network_maybe=network_maybe,
            target_centroid=target_centroid,
            radius=radius,
            name=self.name,
        )

    @classmethod
    def random_network_sampler(
        cls,
        network: Network,
        min_radius: float,
        random_choice: RandomChoice = RandomChoice.radius,
    ):
        """
        Initialize ``NetworkRandomSampler`` for random sampling.

        Assumes that ``Network`` target area is a single ``Polygon`` circle.
        """
        if not network.circular_target_area:
            raise ValueError(
                "Expected passed ``Network`` to be initialized with"
                " circular_target_area as ``True``."
            )
        return NetworkRandomSampler(
            trace_gdf=network.trace_gdf,
            area_gdf=network.area_gdf,
            min_radius=min_radius,
            snap_threshold=network.snap_threshold,
            random_choice=random_choice,
            name=network.name,
        )
