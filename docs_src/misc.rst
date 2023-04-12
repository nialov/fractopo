Miscellanous/FAQ
================

Geometries with Z-coordinates
-----------------------------

Currently ``fractopo`` has only partly been tested with geometries that
have Z-coordinates (e.g. ``LineString([(0, 0, 0), (1, 1, 1)])``. Any
issues you believe might be related to Z-coordinates should be reported
on *GitHub*. To test for such issues using your data, try removing the
Z-coordinates with e.g. the following code:

   .. code:: python

      from shapely.geometry import Polygon, MultiPolygon, LineString


      def remove_z_coordinates(geometries):
          """
          :param geometries: Shapely geometry iterable
          :return: a list of Shapely geometries without z coordinates
          """
          return [
              shapely.wkb.loads(shapely.wkb.dumps(geometry, output_dimension=2))
              for geometry in geometries
          ]


      trace_data.geometry = remove_z_coordinates(trace_data.geometry)

Thanks to *Siyh* for the issue report and above code example
(https://github.com/nialov/fractopo/issues/21).

Furthermore, some validation and analysis remove Z-coordinates from
geometric results. Specifically, use of ``Validation`` will remove
Z-coordinates to avoid unexpected behaviour. Use of ``Network`` will by
default remove z-coordinates but this can be disabled (See ``Network``
input arguments)..

Snap threshold parameter
------------------------

Both in validation and analysis a ``snap_threshold`` parameter is used. The
``snap_threshold`` is for the most part designed to handle the (unavoidable)
errors in topological snapping. If the traces in the map are snapped using the
snapping functionality of e.g. QGIS or ArcGIS, the distance between endpoints
and the segments they should be snapped to be should be well below the
threshold of 0.001, which is the default value used in ``fractopo``. This
threshold during digitising can probably be changed in the settings of these
software but for the most part the snapping error should not be connected to
e.g. the lengths of the traces you are digitising.

I would recommend, rather than using a higher ``snap_threshold`` in
``fractopo``, to make sure that the snapping is done properly within the
GIS software used in digitising.

Coordinate systems
------------------

``fractopo`` for the most part assumes a metric coordinate system and
has mostly been tested with data which has the coordinate system of EPSG
3057. Issues with data from other coordinate systems should be posted on
*GitHub*. Using a coordinate system with metric units should be the
first step in debugging if you believe an issue is caused by
``fractopo`` not handling other units correctly. Especially the
``snap_threshold`` value should be adjusted based on the units used.
