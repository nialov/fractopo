import powerlaw
import geopandas as gpd
import pandas as pd
import math
from typing import Tuple, Dict, List, Union

from shapely.geometry.linestring import LineString


def determine_set(
    value: Union[float, int], value_range: Tuple[float, float], loop_around: bool
):
    """
    Determines if value fits within the given value_range.

    If the value range has the possibility of looping around loop_around can be
    set to true.

    >>> determine_set(5, (0, 10), False)
    True

    >>> determine_set(5, (175, 15), True)
    True
    """
    if loop_around:
        if value_range[0] > value_range[1]:
            # Loops around case
            if (value >= value_range[0]) | (value <= value_range[1]):
                return True
    if value_range[0] <= value <= value_range[1]:
        return True


def determine_azimuth(line: LineString, halved: bool) -> float:
    """
    Calculates azimuth of given line.

    If halved -> return is in range [0, 180]
    Else -> [0, 360]

    e.g.:
    Accepts LineString

    >>> determine_azimuth(LineString([(0, 0), (1, 1)]), True)
    45.0

    >>> determine_azimuth(LineString([(0, 0), (0, 1)]), True)
    0.0

    >>> determine_azimuth(LineString([(0, 0), (-1, -1)]), False)
    225.0

    >>> determine_azimuth(LineString([(0, 0), (-1, -1)]), True)
    45.0
    """
    coord_list = list(line.coords)
    start_x = coord_list[0][0]
    start_y = coord_list[0][1]
    end_x = coord_list[-1][0]
    end_y = coord_list[-1][1]
    azimuth = 90 - math.degrees(math.atan2((end_y - start_y), (end_x - start_x)))
    if azimuth < 0:
        azimuth = azimuth + 360
    if azimuth > 360:
        azimuth -= 360
    if halved:
        azimuth = azimuth if not azimuth >= 180 else azimuth - 180
    return azimuth
