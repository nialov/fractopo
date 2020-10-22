# Using geotransform

## Command line

Run 

~~~bash
geotrans --help
~~~

to print the command line help for the utility.

To transform from a geopackage file with a single layer to an ESRI shapefile:

~~~bash
geotrans input_file.gpkg --to_type shp --output output_file.shp
~~~

To transform from a geopackage file with multiple layers to multiple ESRI
shapefiles into a given directory:

~~~bash
geotrans input_file.gpkg --to_type shp --output output_dir
~~~

## Python

All main functions in charge of loading and saving geodata files are
exposed in the transform.py file in the geotrans package.

~~~python
from geotrans.transform import load_file, save_files, SHAPEFILE_DRIVER
from pathlib import Path

# Your geodata file
filepath = Path("input_file.gpkg")

# load_file returns a single or multiple geodataframes depending
# on how many layers are in the file.
geodataframes, layer_names = load_file(filepath)

# Assuming geopackage contained only one layer ->
# Save acquired geodataframe and layer
save_files(geodataframes, layer_names, [Path("output_file.shp")], SHAPEFILE_DRIVER)
~~~
