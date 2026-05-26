"""
Python types used across fractopo.
"""

import geopandas as gpd
from beartype.typing import Annotated
from beartype.vale import Is
from shapely.geometry import LineString

GeoDataFrameWithLineStrings = Annotated[
    gpd.GeoDataFrame,
    Is[lambda gdf: all(isinstance(geom, LineString) for geom in gdf.geometry.values)],
]
