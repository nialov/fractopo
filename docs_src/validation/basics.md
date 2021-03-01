# Basics of validating fracture trace data

## Prerequisites

Fracture or lineament trace data is typically digitized in a GIS
environment. The basemap on which fractures are digitized could be an
orthomosaic or a Light Detection and Ranging digital elevation model for
lineaments. If topology of the fracture network is to be analyzed the
constraints it poses on digitization must be taken into account. E.g.
traces must be accurately snapped to end to other traces to form a
Y-node.

Alongside trace data a defined target/sample area (or areas) must be
supplied.

All spatial data types supported by [geopandas](https://geopandas.org/)
can be validated as trace data. The validation tool along with all other
fractopo-2D modules only accept geopandas GeoDataFrames as inputs
(geopandas easily handles transformation of spatial data types --
shapefiles, geopackages, etc. -- to GeoDataFrames and back).

## Validation

Validation consists of finding errors in digitization and then fixing
them. However currently only very few error types are automatically
fixed. Instead it is recommended to use the validation tool to find the
errors and then fixing them manually. The tool creates a new column,
*VALIDATION ERRORS*, in the GeoDataFrame (visible in the attribute table
in GIS-software). Currently very few types of errors can be
automatically fixed and e.g. conversion from LineString to
MultiLineString has to be done to allow further validation. Therefore I
currently recommend allowing automatic fixes when prompted.

Page links below explain how to use the validation tool in Python, the
validation error types and how manually fix the validation errors.

-   [Usage](usage.md)
-   [Validation Error Types](errors.md)
