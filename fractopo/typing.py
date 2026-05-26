"""
Python types used across fractopo.
"""

import geopandas as gpd
import numpy as np
from beartype.typing import Annotated
from beartype.vale import Is
from numpy.typing import NDArray
from shapely.geometry import LineString

GeoDataFrameWithLineStrings = Annotated[
    gpd.GeoDataFrame,
    Is[lambda gdf: all(isinstance(geom, LineString) for geom in gdf.geometry.values)],
]


NDArrayWithDips = Annotated[
    NDArray[np.floating],
    Is[lambda arr: all(0.0 <= arr <= 90.0)],
]

NDArrayWithDipDirections = Annotated[
    NDArray[np.floating],
    Is[lambda arr: all(0.0 <= arr <= 360.0)],
]
