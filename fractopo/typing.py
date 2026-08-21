"""
Python types used across fractopo.
"""

import geopandas as gpd
import numpy as np
from beartype.typing import Annotated
from beartype.vale import Is
from numpy.typing import NDArray
from shapely.geometry import LineString, Point

GeoDataFrameWithLineStrings = Annotated[
    gpd.GeoDataFrame,
    Is[lambda gdf: all(isinstance(geom, LineString) for geom in gdf.geometry.values)],
]

GeoDataFrameWithPoints = Annotated[
    gpd.GeoDataFrame,
    Is[lambda gdf: all(isinstance(geom, Point) for geom in gdf.geometry.values)],
]


NDArrayWithDips = Annotated[
    NDArray,
    Is[lambda arr: all((arr >= 0.0) & (arr <= 90.0))],
]

NDArrayWithDipDirections = Annotated[
    NDArray,
    Is[lambda arr: all((arr >= 0.0) & (arr <= 360.0)) and arr.ndim == 1],
]

NDArrayWithAxialAzimuths = Annotated[
    NDArray,
    Is[lambda arr: arr.ndim == 1 and np.all((arr >= 0.0) & (arr <= 180.0))],
]

NDArrayWithPositives = Annotated[
    NDArray,
    Is[lambda arr: arr.ndim == 1 and np.all(arr > 0.0)],
]

NDArray1D = Annotated[NDArray, Is[lambda arr: arr.ndim == 1]]

NDArray1DNotEmpty = Annotated[NDArray1D, Is[lambda arr: arr.size > 0]]

NDArray1DFinite = Annotated[NDArray1D, Is[lambda arr: np.all(np.isfinite(arr))]]
